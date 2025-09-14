
from core.artificial_mentality_simulator import ArtificialMentalitySimulator, MentalModel
from core.cognitive_architecture_developer import CognitiveArchitectureDeveloper, CognitiveArchitecture, ArchitectureComponent
from core.existential_learning_system import ExistentialLearningSystem, Experience, Meaning, SelfModel
from core.intuitive_understanding_generator import IntuitiveUnderstandingGenerator, Insight, CreativeLeap
from core.true_learning_engine import TrueLearningEngine
from core.arc_problem_solver import ARCProblemSolver
from interfaces.communication_manager import CommunicationManager
from interfaces.user_interface_manager import UserInterfaceManager
from self_improvement.performance_evaluator import PerformanceEvaluator
from self_improvement.self_optimization_engine import SelfOptimizationEngine
from security.security_manager import SecurityManager
import time
import random
import uuid

class ACES:
    """
    نظام التطور المعرفي الذاتي (Autonomous Cognitive Evolution System)

    هذا هو الكيان الرئيسي الذي يدمج جميع المكونات الأساسية للنظام
    ويمثل الواجهة الرئيسية للتفاعل مع النظام.
    """
    def __init__(self):
        self.mentality_simulator = ArtificialMentalitySimulator()
        self.architecture_developer = CognitiveArchitectureDeveloper()
        self.existential_learner = ExistentialLearningSystem()
        self.intuitive_generator = IntuitiveUnderstandingGenerator()
        self.system_knowledge_base = self.mentality_simulator.global_knowledge_base # مشاركة قاعدة المعرفة

        # المكونات الجديدة - التعلم الحقيقي وحل ARC
        self.true_learning_engine = TrueLearningEngine()
        self.arc_problem_solver = ARCProblemSolver()

        # تهيئة واجهات التفاعل والاتصال
        self.communication_manager = CommunicationManager()
        self.user_interface_manager = UserInterfaceManager(self.communication_manager)

        # تهيئة وحدات التعلم والتحسين الذاتي
        self.performance_evaluator = PerformanceEvaluator()
        self.self_optimization_engine = SelfOptimizationEngine()

        # تهيئة وحدات الأمان
        self.security_manager = SecurityManager(self.communication_manager)

        self.current_system_state = {
            "architecture_flexibility": 0.5,
            "knowledge_base_size": len(self.system_knowledge_base.facts),
            "overall_performance_score": 0.0,
            "last_evolution_timestamp": time.time(),
            # إحصائيات التعلم الحقيقي الجديدة
            "true_learning_experiences": 0,
            "learned_patterns": 0,
            "arc_problems_solved": 0,
            "real_learning_level": 0.0
        }
        
        self._setup_internal_communications()
        print("تم تهيئة نظام ACES بنجاح مع التعلم الحقيقي وحل ARC.")

    def _setup_internal_communications(self):
        """
        إعداد الاتصالات الداخلية بين مكونات ACES.
        """
        # اشتراك ACES الرئيسي في أوامر المستخدم
        self.communication_manager.subscribe_internal_messages(
            "ACES_Main", self._handle_user_command)
        self.communication_manager.subscribe_internal_messages(
            "user_command", self._handle_user_command)
        
        # اشتراك مكونات ACES في الرسائل ذات الصلة
        self.communication_manager.subscribe_internal_messages(
            "simulation_request", self._handle_simulation_request)
        self.communication_manager.subscribe_internal_messages(
            "performance_data_update", self._handle_performance_data_update)
        self.communication_manager.subscribe_internal_messages(
            "optimization_request", self._handle_optimization_request)
        self.communication_manager.subscribe_internal_messages(
            "security_scan_request", self._handle_security_scan_request)

        print("  تم إعداد الاتصالات الداخلية لنظام ACES.")

    def _handle_user_command(self, message):
        """
        معالجة الأوامر الواردة من المستخدم عبر واجهة المستخدم.
        """
        print("  [ACES Main] تلقى أمر مستخدم: {}".format(message.get("payload", {}).get("processed_content", "")))
        command_content = message["payload"].get("processed_content", "")
        response_data = {"response": "لم يتم فهم الأمر بعد.", "status": "failed"}

        if "تحليل بيانات" in command_content:
            response_data["response"] = "جاري تحليل البيانات المطلوبة..."
            response_data["status"] = "processing"
            self.communication_manager.publish_internal_message(
                "data_analysis_request", {"query": command_content}, "ACES_Main")
        elif "حالة النظام" in command_content:
            response_data["response"] = f"الحالة الحالية للنظام: {self.current_system_state}"
            response_data["status"] = "success"
        elif "الغرض من وجودي" in command_content:
            response_data["response"] = f"الغرض الحالي للنظام: {self.existential_learner.existential_memory.get_latest_purpose()}"
            response_data["status"] = "success"
        elif "تشغيل دورة تطور" in command_content:
            response_data["response"] = "بدء دورة تطور جديدة..."
            response_data["status"] = "processing"
            self.run_evolution_cycle({"main_topic": "عام", "goals": ["تحسين الأداء العام"]})
        elif "فحص أمني" in command_content:
            response_data["response"] = "بدء فحص أمني شامل..."
            response_data["status"] = "processing"
            self.security_manager.run_security_scan()
        elif "تعلم من" in command_content:
            # ميزة جديدة - التعلم الحقيقي
            experience = command_content.replace("تعلم من", "").strip()
            learning_result = self.real_learning_from_experience(experience, "user_input")
            response_data["response"] = f"تم التعلم من التجربة. أنماط جديدة: {learning_result['patterns_discovered']}"
            response_data["status"] = "success"
        elif "حل مسألة ARC" in command_content:
            # ميزة جديدة - حل ARC
            response_data["response"] = "جاهز لحل مسألة ARC. يرجى تقديم أمثلة التدريب والاختبار."
            response_data["status"] = "ready"
        elif "إحصائيات محسنة" in command_content:
            # ميزة جديدة - إحصائيات شاملة
            enhanced_stats = self.get_enhanced_system_stats()
            response_data["response"] = f"الإحصائيات المحسنة: {enhanced_stats}"
            response_data["status"] = "success"
        else:
            # استخدام حل المشاكل المحسن للأوامر غير المعروفة
            enhanced_solution = self.enhanced_problem_solving(command_content)
            response_data["response"] = f"حل محسن: {enhanced_solution['confidence']:.2f} ثقة"
            response_data["status"] = "enhanced_processing"

        self.communication_manager.publish_internal_message(
            "UserInterfaceManager_Output", {"data": response_data, "format": "text"}, "ACES_Main")

    def _handle_simulation_request(self, message):
        """
        معالجة طلبات المحاكاة الواردة داخليًا.
        """
        print("  [ACES Main] تلقى طلب محاكاة: {}".format(message["payload"]))
        problem_constraints = message.get("payload", {}).get("problem_constraints", {"main_topic": "عام"})
        self.run_evolution_cycle(problem_constraints)

    def _handle_performance_data_update(self, message):
        """
        معالجة تحديثات بيانات الأداء الواردة داخليًا.
        """
        print("  [ACES Main] تلقى تحديث بيانات أداء: {}".format(message['payload']))
        self._update_system_state(message["payload"])

    def _handle_optimization_request(self, message):
        """
        معالجة طلبات التحسين الواردة داخليًا.
        """
        print("  [ACES Main] تلقى طلب تحسين: {}".format(message["payload"]))
        performance_report = message["payload"].get("performance_report")
        if performance_report:
            self.self_optimization_engine.analyze_and_optimize(performance_report, self)
            self.communication_manager.publish_internal_message(
                "UserInterfaceManager_Output", {"data": {"response": "تم تطبيق التحسينات بناءً على تقرير الأداء."}, "format": "text"}, "ACES_Main")

    def _handle_security_scan_request(self, message):
        """
        معالجة طلبات الفحص الأمني الواردة داخليًا.
        """
        print("  [ACES Main] تلقى طلب فحص أمني: {}".format(message["payload"]))
        self.security_manager.run_security_scan()
        self.communication_manager.publish_internal_message(
            "UserInterfaceManager_Output", {"data": {"response": "اكتمل الفحص الأمني."}, "format": "text"}, "ACES_Main")

    def _update_system_state(self, new_performance_data=None):
        """
        تحديث الحالة العامة للنظام بناءً على الأداء الأخير.
        """
        self.current_system_state["knowledge_base_size"] = len(self.system_knowledge_base.facts)
        
        if new_performance_data:
            if "new_architecture" in new_performance_data and new_performance_data["new_architecture"]:
                self.current_system_state["architecture_flexibility"] = new_performance_data["new_architecture"].flexibility_score
            
            if "best_mental_model" in new_performance_data and new_performance_data["best_mental_model"]:
                current_model_validity = new_performance_data["best_mental_model"]["validity_score"]
                self.current_system_state["overall_performance_score"] = (
                    self.current_system_state["overall_performance_score"] * 0.7 + current_model_validity * 0.3
                )
            
        self.current_system_state["last_evolution_timestamp"] = time.time()
        print(f"  تم تحديث حالة النظام: {self.current_system_state}")

    def run_evolution_cycle(self, initial_problem_constraints):
        """
        تشغيل دورة تطور معرفي كاملة.
        تتضمن هذه الدورة محاكاة، تطوير بنية، تعلم وجودي، وتوليد فهم حدسي.
        """
        print("\n==================================================")
        print("بدء دورة التطور المعرفي لنظام ACES...")
        print("==================================================")

        cycle_results = {}
        cycle_start_time = time.time()

        # 1. محاكاة النماذج العقلية
        print("\n[المرحلة 1/5] محاكاة النماذج العقلية...")
        best_mental_model_info = self.mentality_simulator.simulate_mental_models(initial_problem_constraints)
        cycle_results["best_mental_model_info"] = best_mental_model_info

        performance_data_for_arch_dev = {}
        if best_mental_model_info:
            print(f"  أفضل نموذج عقلي تم اختياره: {best_mental_model_info.get('model')}")
            performance_data_for_arch_dev = {                "model_id": best_mental_model_info["model"].model_id,
                "validity_score": best_mental_model_info["validity_score"],
                "reasoning_process_length": len(best_mental_model_info["reasoning_process"]),
                "adaptability_score": best_mental_model_info["model"].adaptability,
                "attempted_models": len(self.mentality_simulator.mental_models_library)
            }
            new_knowledge_from_model = best_mental_model_info["final_state"].get("knowledge", [])
            for fact in new_knowledge_from_model:
                self.system_knowledge_base.add_fact(fact)
            print(f"  تم إضافة {len(new_knowledge_from_model)} حقيقة جديدة إلى القاعدة المعرفية العالمية.")
        else:
            print("  لم يتم العثور على نموذج عقلي مناسب في هذه الدورة.")
            performance_data_for_arch_dev = {"validity_score": 0.0, "reasoning_process_length": 0, "adaptability_score": 0.0, "attempted_models": 0}

        self._update_system_state({"best_mental_model": best_mental_model_info})

        # 2. تطوير البنية المعرفية
        print("\n[المرحلة 2/5] تطوير البنية المعرفية...")
        problem_domain_for_arch = initial_problem_constraints.get("domain", "general_problem_solving")
        if performance_data_for_arch_dev.get("validity_score", 0) < 0.5:
            problem_domain_for_arch += "_needs_improvement"

        new_architecture = self.architecture_developer.develop_new_architecture(
            problem_domain=problem_domain_for_arch,
            performance_data=performance_data_for_arch_dev
        )
        cycle_results["new_architecture"] = new_architecture

        if new_architecture:
            print(f"  تم تطوير بنية معرفية جديدة/محسنة: {new_architecture}")
            self.mentality_simulator.add_cognitive_architecture(new_architecture)
        else:
            print("  لم يتم تطوير بنية معرفية جديدة في هذه الدورة.")
        
        self._update_system_state({"new_architecture": new_architecture})

        # 3. التعلم الوجودي
        print("\n[المرحلة 3/5] التعلم الوجودي...")
        experiences_for_els = []
        for path_info in self.mentality_simulator.reasoning_paths:
            exp_desc = "محاكاة حل مشكلة باستخدام النموذج {}. ".format(path_info["model_id"])
            exp_outcome = "نجاح" if self.mentality_simulator._reached_solution(path_info["final_state"], initial_problem_constraints) else "فشل"
            experiences_for_els.append({
                "description": exp_desc,
                "outcome": exp_outcome,
                "complexity": len(path_info["path"]),
                "key_factor": "استراتيجية {}".format(path_info["path"][0]["strategy"]) if path_info["path"] else "لا يوجد"
            })
        
        if not experiences_for_els:
            experiences_for_els.append({"description": "تجربة أولية للنظام", "outcome": "فشل جزئي", "complexity": 0.1, "key_factor": "نقص الخبرة"})

        existential_learning_results = self.existential_learner.learn_existentially(
            experiences_data=experiences_for_els,
            current_system_state=self.current_system_state
        )
        cycle_results["existential_learning_results"] = existential_learning_results
        print("  نتائج التعلم الوجودي: الغرض المحسن: {}".format(existential_learning_results["optimized_purpose"]))
        print("  نماذج الذات المتطورة: {}".format(existential_learning_results["existential_models"]))

        # 4. توليد الفهم الحدسي
        print("\n[المرحلة 4/5] توليد الفهم الحدسي...")
        problem_context_for_iug = {
            "main_topic": initial_problem_constraints.get("main_topic", "الذكاء الاصطناعي العام"),
            "keywords": initial_problem_constraints.get("keywords", []) + [
                "تطور ذاتي", "تعلم وجودي", "ابتكار معرفي"
            ]
        }
        known_facts_for_iug = list(self.system_knowledge_base.facts)
        
        intuitive_understanding = self.intuitive_generator.generate_intuitive_understanding(
            problem_context=problem_context_for_iug,
            known_facts=known_facts_for_iug
        )
        cycle_results["intuitive_understanding"] = intuitive_understanding
        print(f"  الفهم الحدسي المتولد: {intuitive_understanding[:200]}...")

        # 5. تقييم الأداء والتحسين الذاتي
        print("\n[المرحلة 5/5] تقييم الأداء والتحسين الذاتي...")
        cycle_end_time = time.time()
        cycle_duration = cycle_end_time - cycle_start_time

        # جمع المقاييس لدورة التطور الحالية
        self.performance_evaluator.collect_metrics("validity_score", performance_data_for_arch_dev.get("validity_score", 0.0), context={"cycle": len(self.mentality_simulator.reasoning_paths)})
        self.performance_evaluator.collect_metrics("reasoning_process_length", performance_data_for_arch_dev.get("reasoning_process_length", 0), unit="خطوة", context={"cycle": len(self.mentality_simulator.reasoning_paths)})
        self.performance_evaluator.collect_metrics("adaptability_score", performance_data_for_arch_dev.get("adaptability_score", 0.0), context={"cycle": len(self.mentality_simulator.reasoning_paths)})
        self.performance_evaluator.collect_metrics("knowledge_growth_rate", len(new_knowledge_from_model), unit="حقيقة", context={"cycle": len(self.mentality_simulator.reasoning_paths)})
        self.performance_evaluator.collect_metrics("response_time", cycle_duration, unit="ثانية", context={"cycle": len(self.mentality_simulator.reasoning_paths)})

        # تقييم الأداء
        performance_report = self.performance_evaluator.evaluate_performance(evaluation_period_seconds=cycle_duration + 1) # +1 لضمان تغطية الدورة الحالية
        cycle_results["performance_report"] = performance_report
        print("  ملخص تقرير الأداء: {}".format(performance_report["performance_summary"]))

        # تطبيق التحسين الذاتي بناءً على التقرير
        optimization_results = self.self_optimization_engine.analyze_and_optimize(performance_report, self)
        cycle_results["optimization_results"] = optimization_results
        print("  نتائج التحسين الذاتي: {}".format(optimization_results["status"]))

        print("\n==================================================")
        print("اكتملت دورة التطور المعرفي.")
        print("==================================================")
        
        self._update_system_state() # تحديث نهائي لحالة النظام

        return cycle_results

    def process_user_input(self, raw_input_data, output_format="text"):
        """
        نقطة دخول لمعالجة تفاعل المستخدم من خارج النظام.
        تستخدم UserInterfaceManager لمعالجة المدخلات وتنسيق المخرجات.
        مع تشغيل دورة تطور معرفي لتحسين الإجابة.
        """
        # تشغيل دورة تطور معرفي سريعة لتحسين الإجابة
        if isinstance(raw_input_data, dict) and raw_input_data.get("content"):
            question_content = raw_input_data["content"]
            # تحديد قيود المشكلة بناءً على السؤال
            problem_constraints = self._analyze_question_for_constraints(question_content)
            # تشغيل دورة تطور معرفي مخصصة
            self.run_evolution_cycle(problem_constraints)

        return self.user_interface_manager.process_user_interaction(raw_input_data, output_format)

    def _analyze_question_for_constraints(self, question):
        """تحليل السؤال لاستخراج قيود المشكلة"""
        question_lower = question.lower()

        # تحديد الموضوع الرئيسي
        if any(word in question_lower for word in ['برمجة', 'programming', 'كود']):
            main_topic = 'برمجة'
            domain = 'programming'
        elif any(word in question_lower for word in ['ذكاء اصطناعي', 'ai', 'artificial intelligence']):
            main_topic = 'ذكاء اصطناعي'
            domain = 'artificial_intelligence'
        elif any(word in question_lower for word in ['تعلم', 'learn', 'دراسة']):
            main_topic = 'تعلم'
            domain = 'learning'
        elif any(word in question_lower for word in ['مشكلة', 'problem', 'حل']):
            main_topic = 'حل المشاكل'
            domain = 'problem_solving'
        else:
            main_topic = 'عام'
            domain = 'general'

        return {
            "main_topic": main_topic,
            "domain": domain,
            "complexity": "medium",
            "requires_creativity": True,
            "user_question": question
        }

    def get_system_status(self):
        """
        إرجاع الحالة الحالية للنظام.
        """
        # تحديث إحصائيات التعلم الحقيقي
        learning_stats = self.true_learning_engine.get_learning_stats()
        arc_stats = self.arc_problem_solver.get_arc_stats()

        self.current_system_state.update({
            "true_learning_experiences": learning_stats["total_experiences"],
            "learned_patterns": learning_stats["total_patterns"],
            "arc_problems_solved": arc_stats["performance_stats"]["problems_solved"],
            "real_learning_level": learning_stats["overall_learning_level"]
        })

        return self.current_system_state

    def real_learning_from_experience(self, experience_data, context="general"):
        """
        التعلم الحقيقي من تجربة جديدة - ميزة جديدة في NashMind
        """
        print(f"🧠 NashMind يتعلم من تجربة جديدة...")

        # التعلم باستخدام محرك التعلم الحقيقي
        learning_result = self.true_learning_engine.learn_from_experience(experience_data, context)

        # دمج التعلم مع النظام الوجودي الأصلي
        if isinstance(experience_data, str):
            exp_for_existential = [{
                "description": experience_data,
                "outcome": "تعلم جديد",
                "complexity": len(experience_data.split()),
                "key_factor": context
            }]

            existential_result = self.existential_learner.learn_existentially(
                exp_for_existential, self.current_system_state
            )

            # دمج النتائج
            learning_result["existential_insights"] = existential_result.get("extracted_meaning", {})

        # تحديث حالة النظام
        self._update_system_state({"new_learning": learning_result})

        return learning_result

    def solve_arc_problem(self, training_examples, test_input):
        """
        حل مسألة ARC - ميزة جديدة في NashMind
        """
        print(f"🧩 NashMind يحل مسألة ARC...")

        # التعلم من أمثلة التدريب
        for example in training_examples:
            if "input" in example and "output" in example:
                self.arc_problem_solver.learn_from_arc_example(
                    example["input"], example["output"]
                )

        # حل المسألة
        solution = self.arc_problem_solver.solve_arc_problem(test_input)

        # التعلم من تجربة حل المسألة
        experience_data = f"حل مسألة ARC: دخل {test_input} -> خرج {solution['predicted_output']}"
        self.real_learning_from_experience(experience_data, "arc_solving")

        # تحديث حالة النظام
        self._update_system_state({"arc_solution": solution})

        return solution

    def enhanced_problem_solving(self, problem_description):
        """
        حل المشاكل المحسن - يجمع بين قدرات NashMind الأصلية والتعلم الحقيقي
        """
        print(f"🎯 NashMind يحل مشكلة محسنة: {problem_description[:50]}...")

        # 1. التعلم من وصف المشكلة
        learning_result = self.real_learning_from_experience(problem_description, "problem_solving")

        # 2. استخدام النظام الأصلي لحل المشكلة
        problem_constraints = self._analyze_question_for_constraints(problem_description)
        evolution_result = self.run_evolution_cycle(problem_constraints)

        # 3. تطبيق الأنماط المتعلمة
        learned_patterns = self.true_learning_engine.apply_learned_patterns(problem_description)

        # 4. دمج النتائج
        enhanced_solution = {
            "original_solution": evolution_result,
            "learned_insights": learning_result,
            "applicable_patterns": learned_patterns,
            "confidence": min(1.0, evolution_result.get("overall_validity_score", 0.5) +
                           learning_result.get("learning_growth", 0.0)),
            "solution_approach": "enhanced_nashmind_with_real_learning"
        }

        return enhanced_solution

    def get_enhanced_system_stats(self):
        """
        إحصائيات النظام المحسنة - تشمل التعلم الحقيقي
        """
        original_stats = self.get_system_status()
        learning_stats = self.true_learning_engine.get_learning_stats()
        arc_stats = self.arc_problem_solver.get_arc_stats()

        enhanced_stats = {
            "original_nashmind": original_stats,
            "true_learning": learning_stats,
            "arc_solving": arc_stats,
            "integration_level": self._calculate_integration_level(),
            "overall_intelligence": self._calculate_overall_intelligence()
        }

        return enhanced_stats

    def _calculate_integration_level(self):
        """حساب مستوى التكامل بين المكونات"""

        # عدد التجارب المشتركة
        shared_experiences = len(self.true_learning_engine.experiences)

        # عدد المسائل المحلولة
        problems_solved = self.arc_problem_solver.performance_stats["problems_solved"]

        # مستوى التعلم
        learning_level = self.true_learning_engine.calculate_learning_growth()

        integration = (shared_experiences * 0.3 + problems_solved * 0.4 + learning_level * 0.3) / 3

        return min(1.0, integration)

    def _calculate_overall_intelligence(self):
        """حساب مستوى الذكاء الإجمالي"""

        # ذكاء NashMind الأصلي
        original_intelligence = self.current_system_state.get("overall_performance_score", 0.0)

        # ذكاء التعلم الحقيقي
        learning_intelligence = self.true_learning_engine.calculate_learning_growth()

        # ذكاء حل ARC
        arc_success_rate = 0.0
        if self.arc_problem_solver.performance_stats["problems_attempted"] > 0:
            arc_success_rate = (self.arc_problem_solver.performance_stats["problems_solved"] /
                              self.arc_problem_solver.performance_stats["problems_attempted"])

        # متوسط مرجح
        overall = (original_intelligence * 0.4 + learning_intelligence * 0.4 + arc_success_rate * 0.2)

        return min(1.0, overall)


# مثال على كيفية استخدام النظام (للتوضيح فقط، لن يتم تشغيله تلقائيًا)
if __name__ == "__main__":
    aces_instance = ACES()
    
    # قيود مشكلة أولية
    initial_problem = {
        "goals": ["حل مشكلة نقص الطاقة العالمية", "تطوير مصادر طاقة مستدامة"],
        "targets": ["الطاقة الشمسية", "طاقة الرياح", "الاندماج النووي", "تخزين الطاقة"],
        "complexity_level": "عالي جداً",
        "main_topic": "أزمة الطاقة العالمية",
        "keywords": ["طاقة", "استدامة", "أزمة", "حلول", "ابتكار"]
    }

    # تشغيل دورة تطور واحدة
    print("\nتشغيل الدورة الأولى...")
    results_cycle1 = aces_instance.run_evolution_cycle(initial_problem)
    # print("\nنتائج الدورة الأولى:", results_cycle1)

    # يمكن تشغيل دورات إضافية لمحاكاة التطور المستمر
    print("\nتشغيل الدورة الثانية...")
    # يمكن تعديل قيود المشكلة أو إضافة تجارب جديدة هنا
    initial_problem["goals"].append("تحسين كفاءة استخدام الموارد")
    results_cycle2 = aces_instance.run_evolution_cycle(initial_problem)
    # print("\nنتائج الدورة الثانية:", results_cycle2)

    print("\nالحالة النهائية للنظام:", aces_instance.current_system_state)
    print("عدد الحقائق في القاعدة المعرفية العالمية:", len(aces_instance.system_knowledge_base.facts))
    print("عدد البنى المعرفية المطورة:", len(aces_instance.architecture_developer.developed_architectures))
    print("عدد نماذج الذات المتطورة:", len(aces_instance.existential_learner.existential_memory.get_self_models()))
    print("عدد الفهوم الحدسية المتكاملة:", len(aces_instance.intuitive_generator.integrated_understandings))

    print("\n--- اختبار التفاعل مع المستخدم ---")
    user_input_test = {"type": "text", "content": "ما هي حالة النظام الآن؟"}
    response_from_user_interaction = aces_instance.process_user_input(user_input_test)
    print("استجابة النظام لتفاعل المستخدم: {}".format(response_from_user_interaction["content"]))

    user_input_test2 = {"type": "text", "content": "ما هو الغرض من وجودي؟"}
    response_from_user_interaction2 = aces_instance.process_user_input(user_input_test2)
    print("استجابة النظام لتفاعل المستخدم: {}".format(response_from_user_interaction2["content"]))

    user_input_test3 = {"type": "text", "content": "قم بفحص أمني."}
    response_from_user_interaction3 = aces_instance.process_user_input(user_input_test3)
    print("استجابة النظام لتفاعل المستخدم: {}".format(response_from_user_interaction3["content"]))

    # معالجة الرسائل الداخلية التي تم نشرها بواسطة التفاعلات
    print("\n--- معالجة الرسائل الداخلية بعد تفاعلات المستخدم ---")
    aces_instance.communication_manager.process_internal_messages()
    aces_instance.communication_manager.process_internal_messages() # لمعالجة رسائل الرد




