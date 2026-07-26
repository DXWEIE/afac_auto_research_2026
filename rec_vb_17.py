
from helper_rec_vb_17 import *

# 全局运行状态（线程安全）
global_success_times = 0           # 全局训练成功总次数
_counter_lock = threading.Lock()   # 全局计数器锁
_gpu_lock = threading.Lock()       # GPU资源串行锁（保证运行时串行）

def get_gpu_lock():
    """非阻塞申请GPU锁
    返回True表示申请成功，False表示锁被占用
    """
    return _gpu_lock.acquire(blocking=False)

def release_gpu_lock():
    """安全释放GPU锁
    重复释放、未持有就释放都不会抛异常崩溃
    """
    try:
        _gpu_lock.release()
    except RuntimeError:
        # 捕获「释放未持有锁」的异常，兜底容错
        pass

# ==================================================
# 基类：通用Worker能力封装
# ==================================================
class BaseWorker:
    """所有工作器的基类，封装线程管理、LLM调用、状态管理通用逻辑"""
    def __init__(self, worker_id=0):
        self.worker_id = worker_id
        self.model = default_model
        self.task_type = "idle"  # 当前任务阶段，由子类定义具体枚举
        
        # 代码与路径状态
        self.current_code = ""
        self.current_code_path = ""
        
        # 错误上下文（生成下一轮代码时带入）
        self.last_error_info = ""
        
        # 异步线程管理
        self._thread = None
        self._result = None
        self._status = "idle"  # idle / running / finished / failed

    def _safe_llm_call(self, prompt, model=None):
        """安全调用LLM，自动捕获异常并记录错误"""
        model = model or self.model
        try:
            resp = call_llm_with_think(prompt, model=model)
            return filter_cdot(resp)
        except Exception as e:
            dual_print(f"[Worker-{self.worker_id}] LLM调用异常: {str(e)}")
            self.last_error_info = f"LLM调用报错: {str(e)}"
            return None

    def is_finished(self):
        """检查当前异步任务是否结束"""
        if self._thread is None:
            return True
        return not self._thread.is_alive()

    def get_result(self):
        """获取任务结果并重置线程状态"""
        if self.is_finished():
            res = self._result
            self._thread = None
            self._result = None
            return res
        return None

    def _run_async(self, target_func):
        """通用异步线程执行器，所有IO密集型任务都走这里"""
        if not self.is_finished():
            dual_print(f"[Worker-{self.worker_id}] 上一任务未完成，跳过启动")
            return False
        
        self._status = "running"
        def _wrapper():
            try:
                self._result = target_func()
                self._status = "finished"
            except Exception as e:
                dual_print(f"[Worker-{self.worker_id}] 异步任务异常:\n{traceback.format_exc()}")
                self.last_error_info = f"任务执行报错: {str(e)}"
                self._status = "failed"
                self._result = False
        
        self._thread = threading.Thread(target=_wrapper, daemon=True)
        self._thread.start()
        return True

    def kill_task(self):
        """标记任务终止（Python线程无法强制杀死，仅做状态标记）"""
        if self._thread and self._thread.is_alive():
            dual_print(f"[Worker-{self.worker_id}] 终止当前任务: {self.task_type}")
            self._status = "failed"
            self.task_type = "idle"


# ==================================================
# 数据分析工作器
# 触发规则：初始执行1次，之后每完成4次训练触发1次
# 流程：生成探查方向 → 生成分析代码 → 申请资源运行 → 写入全局MD
# ==================================================
class DataAnalysisWorker(BaseWorker):
    def __init__(self, worker_id=0):
        super().__init__(worker_id)
        # 状态枚举：idle / plan_gen / code_gen / running
        self.task_type = "idle"
        
        # 探查上下文
        self.current_inspect_plan = ""
        self._last_trigger_count = -1  # 记录上次触发的训练次数，防止重复触发
        
        # 重试次数限制（避免无限重试浪费资源）
        self._plan_retry = 0
        self._code_retry = 0
        self._max_retry = 3  # 单阶段最多重试3次

    # -------------------------- 阶段1：生成探查方向 --------------------------
    def _generate_inspect_plan(self):
        dual_print(f"[Worker-{self.worker_id}] 📊 生成数据探查方向...")
        history_md = read_file_safe(global_history_acc_md_path)
        
        prompt = data_analyse_plan_template.format(
            task_description=rec_descrition_template,
            dataset_description=rec_data_explanation_template,
            history_train_summary=history_md if history_md else "暂无历史训练结果"
        )
        plan = self._safe_llm_call(prompt)
        if plan:
            self.current_inspect_plan = plan
            self.last_error_info = ""  # 成功则清空错误
            return True
        return False

    def start_plan_gen(self):
        self.task_type = "plan_gen"
        self._run_async(self._generate_inspect_plan)

    # -------------------------- 阶段2：生成分析代码 --------------------------
    def _generate_analysis_code(self):
        dual_print(f"[Worker-{self.worker_id}] 💻 生成数据分析代码...")
        
        prompt = data_analyse_template.format(
            dataset_description=rec_data_explanation_template,
            inspect_plan=self.current_inspect_plan,
            last_error=self.last_error_info  # 带入运行报错/生成报错
        )
        code = self._safe_llm_call(prompt)
        if code:
            self.current_code = code
            self.current_code_path = f"./data_analysis_w{self.worker_id}.py"
            with open(self.current_code_path, "w", encoding="utf-8") as f:
                f.write(SAFE_EXEC_PREFIX +'\n' + code)
            self.last_error_info = ""
            return True
        return False

    def start_code_gen(self):
        self.task_type = "code_gen"
        self._run_async(self._generate_analysis_code)

    # -------------------------- 阶段3：运行分析脚本 --------------------------
    def _run_analysis(self):
        dual_print(f"[Worker-{self.worker_id}] ▶️ 运行数据分析脚本...")
        while not get_gpu_lock():
            time.sleep(1)
        
        try:
            # 代码运行
            success, result = run_data_analysis(
                script_path=self.current_code_path,
                output_md_path=global_data_inspect_md_path
            )
            if success:
                dual_print(f"[Worker-{self.worker_id}] ✅ 数据分析完成，结果已写入全局报告")
                self.last_error_info = ""
                return True
            else:
                self.last_error_info = f"数据分析代码运行报错: {result}"
                dual_print(f"[Worker-{self.worker_id}] ❌ 数据分析失败: {result[:200]}")
                return False
        finally:
            release_gpu_lock()

    def start_run_analysis(self):
        self.task_type = "running"
        self._run_async(self._run_analysis)

    # -------------------------- 触发逻辑 --------------------------
    def should_trigger_new_round(self):
        """判断是否需要开启【新一轮】数据分析（仅idle状态下判断）"""
        global global_success_times
        with _counter_lock:
            current_count = global_success_times
        
        # 初始触发（0次训练时执行第一次）
        if current_count == 0 and self._last_trigger_count == -1:
            self._last_trigger_count = 0
            return True
        
        # 每3次训练触发1次，避免重复触发
        if current_count > 0 and current_count % 10 == 0 and current_count != self._last_trigger_count:
            self._last_trigger_count = current_count
            # 新一轮开始前重置重试计数
            self._plan_retry = 0
            self._code_retry = 0
            return True
        
        return False

    # -------------------------- 主状态机驱动 --------------------------
    def main_loop_step(self):
        """单步生命周期推进，由外部主循环调用"""
        # 情况1：空闲状态 → 判断是否要开启新一轮
        if self.task_type == "idle":
            if self.should_trigger_new_round():
                self.start_plan_gen()
            return

        # 情况2：异步任务还在运行 → 不推进
        if not self.is_finished():
            return

        # 情况3：任务执行完成，根据结果流转状态
        result = self.get_result()

        if self.task_type == "plan_gen":
            if result:
                # 生成方向成功 → 进入代码生成
                self._code_retry = 0
                self.start_code_gen()
            else:
                # 生成失败 → 重试，超过次数则放弃本轮
                self._plan_retry += 1
                if self._plan_retry >= self._max_retry:
                    dual_print(f"[Worker-{self.worker_id}] 探查方向生成重试{self._max_retry}次均失败，放弃本轮分析")
                    self.task_type = "idle"
                else:
                    dual_print(f"[Worker-{self.worker_id}] 探查方向生成失败，第{self._plan_retry}次重试")
                    self.start_plan_gen()

        elif self.task_type == "code_gen":
            if result:
                # 代码生成成功 → 进入运行阶段
                self.start_run_analysis()
            else:
                # 代码生成失败 → 重试
                self._code_retry += 1
                if self._code_retry >= self._max_retry:
                    # 代码生成多次失败，回退到重新生成方向
                    dual_print(f"[Worker-{self.worker_id}] 分析代码生成重试{self._max_retry}次均失败，回退到重新生成探查方向")
                    self._plan_retry = 0
                    self.start_plan_gen()
                else:
                    dual_print(f"[Worker-{self.worker_id}] 分析代码生成失败，第{self._code_retry}次重试")
                    self.start_code_gen()

        elif self.task_type == "running":
            if result:
                # 运行成功 → 本轮结束，回到idle
                self.task_type = "idle"
            else:
                # 运行报错 → 带着错误信息重新生成代码
                self._code_retry += 1
                if self._code_retry >= self._max_retry:
                    dual_print(f"[Worker-{self.worker_id}] 分析脚本运行重试{self._max_retry}次均失败，放弃本轮分析")
                    self.task_type = "idle"
                else:
                    dual_print(f"[Worker-{self.worker_id}] 分析脚本运行失败，带着报错重新生成代码（第{self._code_retry}次）")
                    self.start_code_gen()


# ==================================================
# 训练代码工作器
# 状态机：接口生成 → 冒烟测试 → 正式训练 → 生成优化方向 → 生成代码 → 循环
# ==================================================
class TrainingWorker(BaseWorker):
    def __init__(self, worker_id=0, max_rounds=6):
        super().__init__(worker_id)
        self.max_rounds = max_rounds  # 最大训练轮次
        
        # 私有最佳结果
        self.private_best_code = ""
        self.private_best_acc = 0.0
        
        # 优化上下文
        self.current_optimize_direction = ""
        self.last_train_summary = ""
        
        # 状态枚举：idle / interface_gen / smoke_test / full_train / direction_gen / code_gen
        self.task_type = "idle"
        self._finished_rounds = 0  # 已完成的训练轮次

        self.current_model_save_dir = os.path.join(model_save_dir, f"worker_{self.worker_id}/round_{self._finished_rounds}")
        os.makedirs(self.current_model_save_dir, exist_ok=True)

        self.code_edit_times = 0 # 代码增量生成次数

    # -------------------------- 阶段1：初始接口代码生成 --------------------------
    def _interface_gen_task(self):
        """生成初始符合接口模板的训练代码"""
        dual_print(f"[Worker-{self.worker_id}] 🚀 生成初始训练接口代码...")
        # prompt = rec_gen_module_template.replace('{task_description}',rec_descrition_template).replace('{dataset_description}',rec_data_explanation_template).replace('{last_error}',self.last_error_info)
        
        last_edit_error = ""
        edit_success = False
        max_edit_retry = 3
        retry = 0
        if (self.private_best_acc>0 and self.current_code!="" and self.code_edit_times<max_edit_retry) or \
        (self.last_error_info!="" and self.current_code!="" and self.code_edit_times<max_edit_retry): # 这里增量生成一次，每次修改self.current_code
            while not edit_success and retry < max_edit_retry:
                prompt = train_code_gen_edit_based_template.format(
                    reference_code=self.current_code,
                    optimize_direction=self.current_optimize_direction,
                    last_error=self.last_error_info,
                    last_edit_error=last_edit_error
                )
                try:
                    edits = json.loads(filter_cdot(call_llm_with_think(prompt)))
                    edit_success, edit_error = apply_code_edits(
                        self.current_code_path, edits, self.current_code_path
                    )
                    if not edit_success:
                        retry += 1
                        last_edit_error = f"代码编辑失败: {edit_error}"
                except Exception as e:
                    retry += 1
                    last_edit_error = f"编辑结果解析异常: {str(e)}"
            if edit_success: # 加载一下当前代码
                self.code_edit_times += 1
                with open(self.current_code_path, "r", encoding="utf-8") as f:
                    self.current_code = f.read()
                dual_print(f"[Worker-{self.worker_id}] ✅ 训练代码增量编辑成功，进入冒烟测试")
                self.last_error_info = ""
                return True
        
        self.code_edit_times = 0 # 设置为0
        prompt = train_code_gen_template.format(
            task_description=rec_descrition_template,
            dataset_description=rec_data_explanation_template,
            reference_code="",
            optimize_direction="",
            last_error=self.last_error_info,
            model_save_dir=self.current_model_save_dir
        )
        
        code = self._safe_llm_call(prompt,model='qwen3.6-max-preview')
        dual_print(f"[Worker-{self.worker_id}] 生成的训练代码为:\n{code}")
        if code:
            self.current_code = code
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_code_path = os.path.join(self.current_model_save_dir, f"gen_{ts}_w{self.worker_id}.py")
            with open(self.current_code_path, "w", encoding="utf-8") as f:
                f.write(SAFE_EXEC_PREFIX +'\n' + code)
            return True
        return False

    def start_interface_gen(self):
        self.task_type = "interface_gen"
        self._run_async(self._interface_gen_task)

    # -------------------------- 阶段2：冒烟测试（同步执行，含资源申请） --------------------------
    def smoke_test(self):
        """冒烟测试：修改epoch=10、训练集缩为1000样本，运行验证是否报错"""
        dual_print(f"[Worker-{self.worker_id}] 🔥 执行冒烟测试...")
        smoke_code_path = f"./smoke_test_w{self.worker_id}.py"
        last_edit_error = ""
        edit_success = False
        max_edit_retry = 8
        retry = 0

        while not edit_success and retry < max_edit_retry:
            edit_prompt = smoking_test_edit_template.format(
                code_piece=self.current_code,
                last_edit_error=last_edit_error
            )
            try:
                edits = json.loads(filter_cdot(call_llm_with_think(edit_prompt)))
                edit_success, edit_error = apply_code_edits(
                    self.current_code_path, edits, smoke_code_path
                )
                if not edit_success:
                    retry += 1
                    last_edit_error = f"代码编辑失败: {edit_error}"
            except Exception as e:
                retry += 1
                last_edit_error = f"编辑结果解析异常: {str(e)}"
                dual_print(f"[Worker-{self.worker_id}] ❌ 冒烟测试代码编辑重试第{retry}次失败: {last_edit_error}")

        if not edit_success:
            self.last_error_info = f"冒烟测试代码编辑失败，生成的代码不够规范，需要重新生成"
            dual_print(f"[Worker-{self.worker_id}] ❌ 冒烟测试代码编辑重试8次仍然失败")
            return False

        # 2. 申请资源运行冒烟测试
        while not get_gpu_lock():
            time.sleep(1)
        
        try:
            smoke_pass, smoke_err = run_smoking_test(
                    data_path=fixed_data_path,
                    script_path=smoke_code_path,
                    submission_csv_path=os.path.join(self.current_model_save_dir, "B2.csv"),
                    topk=10
                )

        finally:
            release_gpu_lock()

        if not smoke_pass:
            self.last_error_info = f"生成的代码冒烟测试时运行失败，报错信息为: {smoke_err}"
            dual_print(f"[Worker-{self.worker_id}] ❌ 冒烟测试不通过: {smoke_err[:200]}")
            return False
        else:
            self.last_error_info = ""
            dual_print(f"[Worker-{self.worker_id}] ✅ 冒烟测试通过")
            return True

    # -------------------------- 阶段3：正式训练 --------------------------
    def _full_train_task(self):
        global global_best_acc, global_best_code, global_best_code_path, global_best_model_summary
        """运行完整训练流程"""
        dual_print(f"[Worker-{self.worker_id}] ⏳ 开始正式训练...")
        
        # 申请GPU资源（串行运行）
        while not get_gpu_lock():
            time.sleep(1)
        
        try:
            train_log, error_info, result_summary, is_killed = code_run(
                script_path=self.current_code_path,
                train_code=self.current_code,
                model_save_dir=self.current_model_save_dir,
                timeout=1200
            )
        finally:
            release_gpu_lock()

        # 运行失败处理
        if is_killed or not result_summary:
            self.last_error_info = f"训练运行失败，报错信息为: {error_info}"
            dual_print(f"[Worker-{self.worker_id}] ❌ 训练失败: {error_info[:200]}")
            return False

        self.last_train_summary = result_summary
        self.last_error_info = ""

        dual_print(f"[Worker-{self.worker_id}] ✅ 训练完成，结果总结:\n{result_summary[:2000]}")
        
        # 解析准确率，更新私有最佳
        try:
            result_data = json.loads(filter_cdot(result_summary))
            best_acc = float(result_data.get("best_ndcg", 0.0))
            # 进行模型集成
            run_ensemble()
            if best_acc > self.private_best_acc:
                self.private_best_acc = best_acc
                self.private_best_code = self.current_code
                dual_print(f"[Worker-{self.worker_id}] 🎉 刷新私有最佳 NDCG@10: {best_acc:.4f}")
            if best_acc > global_best_acc:
                global_best_acc = best_acc
                global_best_code = self.current_code
                global_best_code_path = self.current_code_path
                global_best_model_summary = self.last_train_summary
        except Exception as e:
            dual_print(f"[Worker-{self.worker_id}] 训练结果解析失败: {str(e)}")
            best_acc = 0.0

        # 写入全局历史总结MD（线程安全）
        with _counter_lock:
            global global_success_times
            global_success_times += 1
            round_idx = global_success_times
        
        with open(global_history_acc_md_path, "a", encoding="utf-8") as f:
            f.write(f"\n## 第{round_idx}轮 | Worker-{self.worker_id} | NDCG@10: {best_acc:.4f}\n")
            f.write(f"### 训练日志摘要\n```\n{train_log[-500:]}\n```\n")
            f.write(f"### 结果总结\n{result_summary}\n")
        
        # 更新
        self._finished_rounds += 1
        self.current_model_save_dir = os.path.join(model_save_dir, f"worker_{self.worker_id}/round_{self._finished_rounds}")
        os.makedirs(self.current_model_save_dir, exist_ok=True)
        return True

    def start_full_train(self):
        self.task_type = "full_train"
        self._run_async(self._full_train_task)

    # -------------------------- 阶段4：生成优化方向 --------------------------
    def _direction_gen_task(self):
        """基于历史训练结果和数据探查结果生成优化方向"""
        dual_print(f"[Worker-{self.worker_id}] 🧠 生成优化方向...")
        history_md = read_file_safe(global_history_acc_md_path)
        data_inspect_md = read_file_safe(global_data_inspect_md_path)

        prompt = train_history_analyse_template.format(
            data_inspect_result=data_inspect_md if data_inspect_md else "暂无数据探查结果",
            history_train_summary=history_md if history_md else "暂无历史训练结果"
        )
        direction = self._safe_llm_call(prompt,model='qwen3.6-max-preview')
        dual_print(f"[Worker-{self.worker_id}] 生成的优化方向为: {direction}")
        if direction:
            self.current_optimize_direction = direction
            self.last_error_info = ""
            dual_print(f"[Worker-{self.worker_id}] 🎯 生成优化方向成功: {direction}")
            return True
        return False

    def start_direction_gen(self):
        self.task_type = "direction_gen"
        self._run_async(self._direction_gen_task)

    # -------------------------- 阶段5：生成新一轮训练代码 --------------------------
    def _code_gen_task(self):
        """基于优化方向和最佳基线生成训练代码"""
        dual_print(f"[Worker-{self.worker_id}] 💻 生成优化后训练代码...")

        ref_code = global_best_code if global_best_code and len(global_best_code)>0 else self.current_code
        with open(self.current_code_path, "w", encoding="utf-8") as f:
            f.write(ref_code)
        last_edit_error = ""
        edit_success = False
        max_edit_retry = 3
        max_edit_try = 2
        retry = 0
        if self.last_error_info!="" and self.current_code!="" and self.code_edit_times<max_edit_try:
            while not edit_success and retry < max_edit_retry:
                # 每次写入一次
                with open(self.current_code_path, "w", encoding="utf-8") as f:
                    f.write(ref_code)
                prompt = train_code_gen_edit_based_template.format(
                    reference_code=ref_code,
                    optimize_direction=self.current_optimize_direction,
                    last_error=self.last_error_info,
                    last_edit_error=last_edit_error
                )
                try:
                    edits = json.loads(filter_cdot(call_llm_with_think(prompt)))
                    edit_success, edit_error = apply_code_edits(
                        self.current_code_path, edits, self.current_code_path
                    )
                    if not edit_success:
                        retry += 1
                        last_edit_error = f"代码编辑失败: {edit_error}"
                except Exception as e:
                    retry += 1
                    last_edit_error = f"编辑结果解析异常: {str(e)}"
            if edit_success: # 加载一下当前代码
                self.code_edit_times += 1
                with open(self.current_code_path, "r", encoding="utf-8") as f:
                    self.current_code = f.read()
                dual_print(f"[Worker-{self.worker_id}] ✅ 训练代码增量编辑成功，进入冒烟测试")
                self.last_error_info = ""
                return True
            
        self.code_edit_times = 0 # 设置为0
        dual_print(f"[Worker-{self.worker_id}] 🔄 进入全量生成阶段")
        # 全量生成
        prompt = train_code_gen_template.format(
            task_description=rec_descrition_template,
            dataset_description=rec_data_explanation_template,
            reference_code=ref_code,
            optimize_direction=self.current_optimize_direction,
            last_error=self.last_error_info,
            model_save_dir=self.current_model_save_dir
        )
        code = self._safe_llm_call(prompt)
        dual_print(f"[Worker-{self.worker_id}] 生成的训练代码为:\n{code}")
        if code:
            self.current_code = code
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_code_path = os.path.join(self.current_model_save_dir, f"gen_{ts}_w{self.worker_id}.py")
            os.makedirs(os.path.dirname(self.current_code_path), exist_ok=True)
            with open(self.current_code_path, "w", encoding="utf-8") as f:
                f.write(SAFE_EXEC_PREFIX + '\n' + code)
            self.last_error_info = ""
            return True
        return False

    def start_code_gen(self):
        self.task_type = "code_gen"
        self._run_async(self._code_gen_task)

    # -------------------------- 主状态机控制 --------------------------
    def main_loop_step(self):
        """单步状态推进，由外部主循环定期调用"""
        # 达到最大轮次则停止
        if self._finished_rounds >= self.max_rounds:
            self.task_type = "idle"
            return

        # 初始启动：生成初始接口代码
        if self.task_type == "idle" and self._finished_rounds == 0:
            self.start_interface_gen()
            return

        # 异步任务未完成时，不推进状态
        if not self.is_finished():
            return

        # 状态流转逻辑
        if self.task_type == "interface_gen":
            # 接口生成完成 → 进入冒烟测试
            if self.get_result():
                self.task_type = "smoke_test"
            else:
                dual_print(f"[Worker-{self.worker_id}] 接口生成失败，重试")
                self.start_interface_gen()

        elif self.task_type == "smoke_test":
            # 冒烟测试同步执行，通过则进入正式训练，失败则回退到代码生成
            if self.smoke_test():
                self.start_full_train()
            else:
                if self._finished_rounds == 0:
                    self.start_interface_gen()  # 第一轮失败，重新生成接口
                else:
                    self.start_code_gen()       # 后续轮次失败，重新生成代码

        elif self.task_type == "full_train":
            # 训练完成 → 生成优化方向
            if self.get_result():
                self.start_direction_gen()
            else:
                dual_print(f"[Worker-{self.worker_id}] 训练失败，重新生成代码")
                self.start_code_gen()

        elif self.task_type == "direction_gen":
            # 优化方向生成完成 → 生成训练代码
            if self.get_result():
                self.start_code_gen()
            else:
                self.start_direction_gen()

        elif self.task_type == "code_gen":
            # 代码生成完成 → 进入冒烟测试
            if self.get_result():
                self.task_type = "smoke_test"
            else:
                self.start_code_gen()


# # ==================================================
# # 主入口示例
# # ==================================================
# def main():
#     start_time = time.time()
#     # 初始化工作器（可扩展多个训练Worker并发生成代码）
#     #data_worker = DataAnalysisWorker(worker_id=0)
#     train_worker = TrainingWorker(worker_id=0, max_rounds=6)
    
#     dual_print("🚀 Auto Research Agent 脚手架启动...")
    
#     try:
#         while True:
#             # 推进各工作器状态
#             #data_worker.main_loop_step()
#             train_worker.main_loop_step()
            
#             # 终止条件：达到最大训练轮次且所有任务结束
#             if train_worker._finished_rounds >= train_worker.max_rounds and train_worker.is_finished():
#                 dual_print("✅ 所有训练轮次执行完成，退出")
#                 break
            
#             time.sleep(2)  # 轮询间隔，避免CPU空转
#             now_time = time.time()
#             # 是否超过最大超时时间2小时
#             if now_time - start_time > 1 * 60 * 60:
#                 dual_print("🛑 超过最大超时时间，终止所有任务")
#                 #data_worker.kill_task()
#                 train_worker.kill_task()
#                 break
#     except Exception as e:
#         dual_print("\n中断，退出运行")
#         #data_worker.kill_task()
#         train_worker.kill_task()




def main():
    # ========== 配置Worker数量 ==========
    # 1个数据分析Worker（负责数据探查，1个足够，多了反而重复写结果文件）
    start_time = time.time()
    data_workers = [
        DataAnalysisWorker(worker_id=0)
    ]
    
    # 2个训练Worker（并发生成代码/优化方向，运行时自动通过GPU锁串行）
    train_workers = [
        TrainingWorker(worker_id=0, max_rounds=15),
        TrainingWorker(worker_id=1, max_rounds=15),
        TrainingWorker(worker_id=2, max_rounds=15)
    ]
    
    all_workers = data_workers + train_workers
    dual_print(f"🚀 Auto Research Agent 启动 | 数据分析Worker: {len(data_workers)}个 | 训练Worker: {len(train_workers)}个")

    try:
        while True:
            # 1. 统一推进所有Worker的状态机
            for worker in all_workers:
                worker.main_loop_step()

            # 2. 终止条件：所有训练Worker都完成了目标轮次且处于空闲
            all_train_done = all(
                w._finished_rounds >= w.max_rounds and w.is_finished()
                for w in train_workers
            )
            if all_train_done:
                dual_print("✅ 所有训练轮次执行完成，退出主循环")
                break
            now_time = time.time()
            if now_time - start_time > 2 * 60 * 60:
                dual_print("🛑 超过最大超时时间，终止所有任务")
                for worker in all_workers:
                    worker.kill_task()
                break
            time.sleep(2)  # 轮询间隔，避免CPU空转

    except KeyboardInterrupt:
        dual_print("\n🛑 用户中断，终止所有任务")
        for worker in all_workers:
            worker.kill_task()

if __name__ == "__main__":
    main()


