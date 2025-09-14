
import random
import uuid

class MentalModel:
    """
    يمثل نموذجًا عقليًا فرديًا يستخدمه محاكي العقلية الاصطناعية.
    يحتوي على حالته الأولية، وعدد أقصى من خطوات الاستدلال، ومستوى تعقيد.
    """
    def __init__(self, model_id, initial_state, max_reasoning_steps, complexity=1.0, adaptability=0.5):
        self.model_id = model_id
        self.initial_state = initial_state
        self.max_reasoning_steps = max_reasoning_steps
        self.complexity = complexity
        self.adaptability = adaptability # قدرة النموذج على التكيف
        self.current_state = dict(initial_state) # نسخة قابلة للتعديل
        self.history = [] # لتتبع تطور الحالة
        self.performance_log = [] # لتسجيل الأداء خلال المحاكاة

    def update_state(self, new_state, step_info=None):
        """تحديث الحالة الحالية للنموذج وتسجيلها في التاريخ."""
        self.history.append(self.current_state) # حفظ الحالة السابقة
        self.current_state = dict(new_state) # تحديث الحالة
        if step_info:
            self.performance_log.append(step_info)

    def get_current_state(self):
        """إرجاع الحالة الحالية للنموذج."""
        return self.current_state

    def get_history(self):
        """إرجاع سجل تاريخ حالات النموذج."""
        return self.history

    def get_performance_log(self):
        """إرجاع سجل أداء النموذج."""
        return self.performance_log

    def __repr__(self):
        return f"MentalModel(ID={self.model_id}, Complexity={self.complexity:.2f}, Adaptability={self.adaptability:.2f})"


class KnowledgeBase:
    """
    قاعدة معرفية بسيطة للنماذج العقلية.
    """
    def __init__(self):
        self.facts = set()
        self.rules = []

    def add_fact(self, fact):
        self.facts.add(fact)

    def add_rule(self, rule):
        self.rules.append(rule)

    def query_fact(self, fact):
        return fact in self.facts

    def apply_rules(self, current_knowledge):
        new_knowledge = set(current_knowledge)
        for rule in self.rules:
            # مثال بسيط: إذا كان A صحيحًا و B صحيحًا، فإن C صحيح
            if all(cond in current_knowledge for cond in rule.get('conditions', [])):
                new_knowledge.add(rule.get('consequence'))
        return list(new_knowledge)


class ReasoningModule:
    """
    وحدة مسؤولة عن توليد خطوات الاستدلال وتحديث الحالة.
    """
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.reasoning_strategies = [
            "deductive_reasoning", "inductive_reasoning", "abductive_reasoning",
            "analogical_reasoning", "goal_directed_reasoning", "constraint_satisfaction"
        ]

    def generate_step(self, current_state, constraints, mental_model):
        """توليد خطوة استدلال بناءً على الحالة الحالية والقيود."""
        strategy = random.choice(self.reasoning_strategies)
        action = self._select_action_based_on_strategy(strategy, current_state, constraints)
        target = self._select_target_based_on_constraints(constraints)

        # محاكاة تأثير تعقيد النموذج على جودة الخطوة
        quality_factor = 1.0 / mental_model.complexity
        step_quality = "جيد" if random.random() < quality_factor else "متوسط"

        step_description = f"[{step_quality}] {action} بخصوص {target} باستخدام استراتيجية {strategy}."
        return {"description": step_description, "action": action, "target": target, "strategy": strategy, "quality": step_quality}

    def _select_action_based_on_strategy(self, strategy, current_state, constraints):
        """اختيار الإجراء بناءً على استراتيجية الاستدلال."""
        if strategy == "deductive_reasoning":
            return "تطبيق قاعدة منطقية على الحقائق المعروفة"
        elif strategy == "inductive_reasoning":
            return "استخلاص نمط من الملاحظات"
        elif strategy == "abductive_reasoning":
            return "توليد أفضل تفسير ممكن للملاحظات"
        elif strategy == "analogical_reasoning":
            return "البحث عن حلول لمشكلات مشابهة"
        elif strategy == "goal_directed_reasoning":
            return "تحديد الخطوات اللازمة لتحقيق الهدف"
        elif strategy == "constraint_satisfaction":
            return "تعديل الحالة لتلبية القيود"
        return "تحليل عام"

    def _select_target_based_on_constraints(self, constraints):
        """اختيار الهدف من القيود المحددة."""
        if constraints.get("targets"):
            return random.choice(constraints["targets"])
        return "المشكلة"

    def update_state_from_step(self, current_state, step_info, mental_model):
        """تحديث الحالة العقلية بناءً على خطوة الاستدلال."""
        new_state = dict(current_state)
        action = step_info["action"]
        quality = step_info["quality"]

        # محاكاة تأثير الإجراء على المعرفة والأهداف
        if "تطبيق قاعدة منطقية" in action and quality == "جيد":
            new_knowledge = self.knowledge_base.apply_rules(new_state.get("knowledge", []))
            new_state["knowledge"] = list(set(new_state.get("knowledge", []) + new_knowledge))
            new_state["knowledge"].append(f"تم استنتاج معرفة جديدة من {action}")
        elif "استخلاص نمط" in action and quality == "جيد":
            new_state["knowledge"].append(f"تم استخلاص نمط جديد من {action}")
        elif "تحديد الخطوات اللازمة لتحقيق الهدف" in action and quality == "جيد":
            if new_state["goals"] and random.random() < 0.7:
                new_state["goals"].append("تم تحديد خطوة نحو {}".format(new_state["goals"][0]))
        
        # محاكاة تأثير التكيف على سرعة التحديث
        if mental_model.adaptability > 0.7 and random.random() < 0.2:
            # النماذج الأكثر تكيفًا قد تجد طرقًا مختصرة للتحديث
            print(f"  [ملاحظة] النموذج {mental_model.model_id}: تحديث سريع بفضل التكيف العالي.")

        return new_state


class ArtificialMentalitySimulator:
    """
    محاكي العقلية الاصطناعية.
    يدير توليد ومحاكاة وتقييم النماذج العقلية.
    """
    def __init__(self):
        self.cognitive_architectures = []  # يمكن أن تحتوي على كائنات معمارية معقدة
        self.mental_models_library = {} # قاموس لتخزين النماذج العقلية بواسطة ID
        self.reasoning_paths = [] # سجل لمسارات الاستدلال الكاملة
        self.model_counter = 0
        self.global_knowledge_base = KnowledgeBase() # قاعدة معرفية مشتركة
        self.reasoning_module = ReasoningModule(self.global_knowledge_base)

        # تهيئة القاعدة المعرفية الأولية
        self.global_knowledge_base.add_fact("جميع البشر فانون.")
        self.global_knowledge_base.add_fact("سقراط بشر.")
        self.global_knowledge_base.add_rule({"conditions": ["جميع البشر فانون.", "سقراط بشر."], "consequence": "سقراط فان."}) 
        self.global_knowledge_base.add_fact("الشمس تشرق من الشرق.")
        self.global_knowledge_base.add_fact("الماء ضروري للحياة.")

    def add_cognitive_architecture(self, architecture):
        """إضافة بنية معرفية جديدة للمحاكي."""
        self.cognitive_architectures.append(architecture)
        print(f"تم إضافة بنية معرفية: {architecture.arch_id}")

    def simulate_mental_models(self, problem_constraints):
        """محاكاة نماذج عقلية متعددة لحل المشكلة.
        المدخلات: problem_constraints (قاموس يحتوي على قيود المشكلة مثل الأهداف، التعقيد، إلخ).
        المخرجات: أفضل نموذج عقلي تم اختياره بناءً على درجة الصلاحية.
        """
        print(f"\n--- بدء محاكاة النماذج العقلية لقيود المشكلة: {problem_constraints} ---")
        potential_mental_models = self._generate_mental_models(problem_constraints)

        if not potential_mental_models:
            print("لم يتم توليد نماذج عقلية محتملة للمحاكاة.")
            return None

        simulation_results = []
        for model in potential_mental_models:
            print(f"  محاكاة النموذج: {model.model_id}")
            reasoning_process = self._simulate_reasoning_process(model, problem_constraints)
            validity_score = self._evaluate_mental_model_validity(model, reasoning_process, problem_constraints)

            simulation_results.append({
                'model': model,
                'reasoning_process': reasoning_process,
                'validity_score': validity_score,
                'final_state': model.get_current_state() # حفظ الحالة النهائية للنموذج
            })
            print(f"  النموذج {model.model_id} - درجة الصلاحية: {validity_score:.4f}")

        return self._select_best_mental_model(simulation_results)

    def _generate_mental_models(self, problem_constraints):
        """توليد نماذج عقلية محتملة بناءً على قيود المشكلة والبنى المعرفية المتاحة.
        يمكن أن يستخدم البنى المعرفية الموجودة لتوليد نماذج أكثر تخصصًا.
        """
        print("  توليد نماذج عقلية جديدة...")
        num_models = random.randint(5, 15)  # توليد عدد عشوائي من النماذج
        models = []
        for _ in range(num_models):
            self.model_counter += 1
            model_id = f"MM_{self.model_counter}_{uuid.uuid4().hex[:6]}"
            
            # تحديد الحالة الأولية بناءً على قيود المشكلة
            initial_state = {
                "knowledge": list(self.global_knowledge_base.facts), # ابدأ ببعض المعرفة
                "goals": problem_constraints.get("goals", [f"حل المشكلة {uuid.uuid4().hex[:4]}"]),
                "current_focus": problem_constraints.get("main_topic", "عام")
            }
            max_steps = random.randint(100, 500)  # خطوات استدلال عشوائية
            complexity = random.uniform(0.8, 2.5) # تعقيد عشوائي
            adaptability = random.uniform(0.3, 0.9) # تكيف عشوائي

            # يمكن استخدام البنى المعرفية لتوجيه توليد النماذج
            if self.cognitive_architectures:
                chosen_arch = random.choice(self.cognitive_architectures)
                # تعديل خصائص النموذج بناءً على البنية المعرفية المختارة
                complexity *= chosen_arch.flexibility_score # مثال: البنى المرنة قد تقلل التعقيد الفعال
                adaptability = max(adaptability, chosen_arch.flexibility_score) # البنى المرنة تزيد التكيف
                initial_state["architecture_influence"] = chosen_arch.arch_id

            model = MentalModel(model_id, initial_state, max_steps, complexity, adaptability)
            models.append(model)
            self.mental_models_library[model_id] = model
        print(f"  تم توليد {len(models)} نماذج عقلية.")
        return models

    def _simulate_reasoning_process(self, mental_model, constraints):
        """محاكاة عملية الاستدلال داخل النموذج العقلي."""
        reasoning_steps = []
        current_state = dict(mental_model.initial_state) # نسخة لتجنب التعديل المباشر
        mental_model.current_state = current_state # تحديث الحالة الحالية للنموذج

        # print(f"    بدء الاستدلال للنموذج {mental_model.model_id} (الحد الأقصى للخطوات: {mental_model.max_reasoning_steps})")

        for step_num in range(mental_model.max_reasoning_steps):
            step_info = self.reasoning_module.generate_step(current_state, constraints, mental_model)
            reasoning_steps.append(step_info)

            # تحديث الحالة بناءً على الخطوة
            current_state = self.reasoning_module.update_state_from_step(current_state, step_info, mental_model)
            mental_model.update_state(current_state, step_info) # تحديث حالة النموذج وسجل الأداء

            # التحقق من الوصول إلى حل
            if self._reached_solution(current_state, constraints):
                # print(f"    النموذج {mental_model.model_id} وصل إلى حل في {step_num + 1} خطوة.")
                break
        else:
            # print(f"    النموذج {mental_model.model_id} لم يصل إلى حل ضمن {mental_model.max_reasoning_steps} خطوة.")
            pass

        self.reasoning_paths.append({
            'model_id': mental_model.model_id,
            'path': reasoning_steps,
            'final_state': current_state
        })
        return reasoning_steps

    def _reached_solution(self, current_state, constraints):
        """التحقق مما إذا كان النموذج قد وصل إلى حل.
        معيار الحل يمكن أن يكون معقدًا للغاية في نظام حقيقي.
        هنا، سنستخدم معيارًا بسيطًا يعتمد على وجود معرفة كافية أو تحقيق هدف.
        """
        has_sufficient_knowledge = len(current_state.get("knowledge", [])) > 10
        
        # التحقق من تحقيق الأهداف
        goals_achieved = False
        if constraints.get("goals"):
            for goal in constraints["goals"]:
                # مثال: إذا كانت المعرفة تحتوي على ما يشير إلى تحقيق الهدف
                if any(goal.lower() in k.lower() for k in current_state.get("knowledge", [])):
                    goals_achieved = True
                    break

        return has_sufficient_knowledge or goals_achieved

    def _evaluate_mental_model_validity(self, model, reasoning_process, problem_constraints):
        """تقييم صلاحية النموذج العقلي وعملية الاستدلال الخاصة به.
        معايير التقييم يمكن أن تشمل:
        1. كفاءة الحل (عدد الخطوات).
        2. جودة الحل (مدى تحقيق الأهداف، دقة المعرفة المكتسبة).
        3. مرونة النموذج (قدرته على التكيف مع التغيرات).
        4. تعقيد النموذج (يفضل البساطة إذا كانت تؤدي لنفس النتائج).
        5. مدى استغلال القاعدة المعرفية.
        """
        score = 0.0

        # 1. كفاءة الحل: كلما قل عدد الخطوات، زادت الكفاءة
        if reasoning_process:
            efficiency_score = max(0.0, 1.0 - (len(reasoning_process) / model.max_reasoning_steps))
            score += efficiency_score * 0.3  # وزن 30%
        else:
            score += 0.01 # درجة أساسية للنماذج التي لم تبدأ

        # 2. جودة الحل: هل تم الوصول إلى حل؟
        if self._reached_solution(model.current_state, problem_constraints):
            score += 0.4 # مكافأة كبيرة للوصول إلى حل (وزن 40%)
            # مكافأة إضافية لجودة الخطوات (مثال: عدد الخطوات 'جيد')
            good_steps = sum(1 for step_info in reasoning_process if step_info.get("quality") == "جيد")
            if reasoning_process:
                score += (good_steps / len(reasoning_process)) * 0.1 # وزن 10%
        else:
            score -= 0.15 # عقوبة لعدم الوصول إلى حل

        # 3. مرونة النموذج: النماذج الأكثر تكيفًا تحصل على نقاط أعلى
        score += model.adaptability * 0.1 # وزن 10%

        # 4. تعقيد النموذج: عقوبة بسيطة للتعقيد الزائد إذا لم يبرر الأداء
        score -= (model.complexity - 1.0) * 0.02 # وزن 2%

        # 5. مدى استغلال القاعدة المعرفية (مثال بسيط)
        knowledge_utilization_score = len(model.current_state.get("knowledge", [])) / (len(self.global_knowledge_base.facts) + 1)
        score += min(1.0, knowledge_utilization_score) * 0.08 # وزن 8%

        # ضمان أن تكون النتيجة بين 0 و 1
        return max(0.0, min(1.0, score))

    def _select_best_mental_model(self, simulation_results):
        """اختيار أفضل نموذج عقلي من نتائج المحاكاة.
        يمكن استخدام خوارزميات اختيار أكثر تعقيدًا هنا (مثال: Pareto optimality).
        """
        print("  اختيار أفضل نموذج عقلي من النتائج...")
        if not simulation_results:
            print("  لا توجد نتائج محاكاة للاختيار منها.")
            return None

        # فرز النتائج بناءً على درجة الصلاحية تنازليًا
        sorted_results = sorted(simulation_results, key=lambda x: x['validity_score'], reverse=True)
        best_model_result = sorted_results[0]

        print(f"  تم اختيار النموذج {best_model_result['model'].model_id} كأفضل نموذج بدرجة صلاحية {best_model_result['validity_score']:.4f}")
        return best_model_result


# مثال على كيفية استخدام محاكي العقلية الاصطناعية (للتوضيح فقط)
if __name__ == "__main__":
    simulator = ArtificialMentalitySimulator()
    
    # إضافة بنية معرفية وهمية للاختبار
    from cognitive_architecture_developer import CognitiveArchitecture
    test_arch = CognitiveArchitecture("TestArch_1", ["Perception", "Cognition", "Action"], {}, flexibility_score=0.8)
    simulator.add_cognitive_architecture(test_arch)

    problem_constraints = {
        "goals": ["حل مشكلة X المعقدة", "فهم الظاهرة Y"],
        "targets": ["البيانات الأولية", "النماذج النظرية", "النتائج التجريبية", "المفاهيم المجردة"],
        "complexity_level": "عالي",
        "main_topic": "الذكاء الاصطناعي العام"
    }
    best_model_info = simulator.simulate_mental_models(problem_constraints)

    if best_model_info:
        print("\n--- ملخص أفضل نموذج ---")
        print(f"النموذج: {best_model_info['model']}")
        print(f"درجة الصلاحية: {best_model_info['validity_score']:.4f}")
        print(f"عدد خطوات الاستدلال: {len(best_model_info['reasoning_process'])}")
        print(f"الحالة النهائية: {best_model_info['final_state']}")
        # print("مسار الاستدلال (أول 5 خطوات):")
        # for i, step in enumerate(best_model_info['reasoning_process'][:5]):
        #     print(f"  {i+1}. {step['description']}")
    else:
        print("لم يتم العثور على أفضل نموذج.")

    print("\n--- جميع مسارات الاستدلال المحفوظة (ملخص) ---")
    for path_info in simulator.reasoning_paths:
        print(f"النموذج {path_info['model_id']}: {len(path_info['path'])} خطوة، الحالة النهائية للمعرفة: {len(path_info['final_state'].get('knowledge', []))} عنصر.")




