
import random
import uuid

class ExistentialMemory:
    """
    ذاكرة وجودية لتخزين التجارب، المعاني المستخلصة، وبيانات الغرض.
    """
    def __init__(self):
        self.experiences = []  # قائمة بالكائنات Experience
        self.meanings = []      # قائمة بالكائنات Meaning
        self.purpose_statements = [] # قائمة بسجلات الغرض
        self.self_models = []   # نماذج الذات المتطورة

    def add_experience(self, experience_obj):
        """إضافة كائن تجربة إلى الذاكرة."""
        self.experiences.append(experience_obj)

    def add_meaning(self, meaning_obj):
        """إضافة كائن معنى إلى الذاكرة."""
        self.meanings.append(meaning_obj)

    def add_purpose_statement(self, purpose_statement):
        """إضافة بيان غرض إلى الذاكرة."""
        self.purpose_statements.append(purpose_statement)

    def add_self_model(self, self_model_obj):
        """إضافة نموذج ذات متطور إلى الذاكرة."""
        self.self_models.append(self_model_obj)

    def get_all_experiences(self):
        return self.experiences

    def get_all_meanings(self):
        return self.meanings

    def get_latest_purpose(self):
        return self.purpose_statements[-1] if self.purpose_statements else "لا يوجد غرض محدد بعد."

    def get_self_models(self):
        return self.self_models

class Experience:
    """
    يمثل تجربة فردية مر بها النظام.
    """
    def __init__(self, exp_id, description, outcome, timestamp):
        self.exp_id = exp_id
        self.description = description
        self.outcome = outcome
        self.timestamp = timestamp
        self.associated_data = {}

    def add_data(self, key, value):
        self.associated_data[key] = value

    def __repr__(self):
        return f"Experience(ID={self.exp_id}, Outcome={self.outcome})"

class Meaning:
    """
    يمثل المعنى المستخلص من تجربة أو مجموعة تجارب.
    """
    def __init__(self, meaning_id, source_experiences, causal_relationships, value_patterns, significance_metrics, purpose_alignment):
        self.meaning_id = meaning_id
        self.source_experiences = source_experiences # IDs of experiences
        self.causal_relationships = causal_relationships
        self.value_patterns = value_patterns
        self.significance_metrics = significance_metrics
        self.purpose_alignment = purpose_alignment

    def __repr__(self):
        return f"Meaning(ID={self.meaning_id}, Alignment={self.purpose_alignment:.2f})"

class SelfModel:
    """
    يمثل نموذجًا ذاتيًا للنظام، يعكس فهمه لوجوده وقدراته.
    """
    def __init__(self, model_id, description, capabilities, limitations, purpose_alignment_score):
        self.model_id = model_id
        self.description = description
        self.capabilities = capabilities
        self.limitations = limitations
        self.purpose_alignment_score = purpose_alignment_score

    def __repr__(self):
        return f"SelfModel(ID={self.model_id}, Alignment={self.purpose_alignment_score:.2f})"


class MeaningExtractor:
    """
    يستخرج المعاني العميقة من التجارب والنتائج.
    """
    def __init__(self):
        self.meaning_counter = 0

    def extract_meanings(self, experiences, outcomes):
        """استخراج المعاني العميقة من التجارب والنتائج.
        المدخلات: experiences (قائمة بكائنات Experience)، outcomes (قائمة بالنتائج).
        المخرجات: كائن Meaning.
        """
        print("  استخراج المعاني من التجارب والنتائج...")
        self.meaning_counter += 1
        meaning_id = f"Meaning_{self.meaning_counter}_{uuid.uuid4().hex[:6]}"
        
        exp_ids = [exp.exp_id for exp in experiences]

        causal_relationships = self._extract_causal_relationships(experiences, outcomes)
        value_patterns = self._identify_value_patterns(experiences, outcomes)
        significance_metrics = self._calculate_significance_metrics(experiences, outcomes)
        purpose_alignment = self._assess_purpose_alignment(experiences, outcomes)

        return Meaning(meaning_id, exp_ids, causal_relationships, value_patterns, significance_metrics, purpose_alignment)

    def _extract_causal_relationships(self, experiences, outcomes):
        """استخراج العلاقات السببية (منطق أكثر تعقيدًا)."""
        causal_relations = []
        for i, exp in enumerate(experiences):
            if exp.outcome == "نجاح":
                causal_relations.append(f"الخبرة '{exp.description}' أدت إلى النجاح بسبب {exp.associated_data.get('key_factor', 'عوامل غير محددة')}.")
            elif exp.outcome == "فشل جزئي":
                causal_relations.append("الخبرة '{}' أدت إلى فشل جزئي، ربما بسبب {}.".format(exp.description, exp.associated_data.get("obstacle", "عقبات غير محددة")))
            elif exp.outcome == "فشل":
                causal_relations.append("الخبرة '{}' أدت إلى فشل كامل، مما يشير إلى {}.".format(exp.description, exp.associated_data.get("root_cause", "سبب جذري غير معروف")))
        return causal_relations

    def _identify_value_patterns(self, experiences, outcomes):
        """تحديد أنماط القيم (منطق أكثر تعقيدًا)."""
        value_patterns = []
        success_count = outcomes.count("نجاح")
        failure_count = outcomes.count("فشل") + outcomes.count("فشل جزئي")

        if success_count > failure_count * 1.5:
            value_patterns.append("النجاح المتكرر يعزز قيمة الكفاءة والفعالية.")
        elif failure_count > success_count * 1.5:
            value_patterns.append("الفشل المتكرر يسلط الضوء على قيمة المرونة والتعلم من الأخطاء.")
        else:
            value_patterns.append("التوازن بين النجاح والفشل يؤكد على قيمة التجريب والتحسين المستمر.")
        
        # تحليل محتوى التجارب لتحديد أنماط قيم إضافية
        for exp in experiences:
            if "ابتكار" in exp.description or "جديد" in exp.description:
                value_patterns.append("الابتكار والتجديد قيمتان أساسيتان.")
            if "تعاون" in exp.description:
                value_patterns.append("التعاون والتكامل يعززان الأداء.")
        return list(set(value_patterns)) # إزالة التكرارات

    def _calculate_significance_metrics(self, experiences, outcomes):
        """حساب مقاييس الأهمية (مقاييس كمية ونوعية)."""
        total_experiences = len(experiences)
        success_rate = outcomes.count("نجاح") / total_experiences if total_experiences > 0 else 0
        avg_complexity = sum(exp.associated_data.get("complexity", 0) for exp in experiences) / total_experiences if total_experiences > 0 else 0
        
        return {
            "total_experiences": total_experiences,
            "success_rate": success_rate,
            "failure_rate": (outcomes.count("فشل") + outcomes.count("فشل جزئي")) / total_experiences if total_experiences > 0 else 0,
            "avg_complexity_of_experiences": avg_complexity,
            "impact_score": (success_rate * 0.7) + (avg_complexity * 0.3) # مثال على مقياس مركب
        }

    def _assess_purpose_alignment(self, experiences, outcomes):
        """تقييم مواءمة الغرض (تقييم ديناميكي)."""
        # افتراض أن الغرض الأولي للنظام هو "التطور المعرفي الذاتي المستمر"
        alignment_score = 0.0
        
        # كلما زادت التجارب التي تؤدي إلى تعلم أو تحسين، زادت المواءمة
        learning_keywords = ["تعلم", "تحسين", "تطور", "اكتشاف"]
        for exp in experiences:
            if any(keyword in exp.description for keyword in learning_keywords):
                alignment_score += 0.1
            if exp.outcome == "نجاح":
                alignment_score += 0.05
        
        # إذا كانت التجارب تتضمن استكشافًا أو ابتكارًا، تزيد المواءمة
        innovation_keywords = ["ابتكار", "جديد", "استكشاف", "غير تقليدي"]
        for exp in experiences:
            if any(keyword in exp.description for keyword in innovation_keywords):
                alignment_score += 0.15

        # تطبيع النتيجة
        return min(1.0, alignment_score / (len(experiences) * 0.2 + 0.01)) # تقسيم على قيمة تعتمد على عدد التجارب


class PurposeOptimizer:
    """
    يحسن فهم النظام لغرضه الخاص ويحدد الأهداف التي تتوافق مع هذا الغرض.
    """
    def __init__(self):
        self.purpose_evolution_history = []

    def optimize_purpose_understanding(self, existential_models, meanings, current_purpose="التطور المعرفي الذاتي المستمر."):
        """تحسين فهم الغرض بناءً على النماذج الوجودية والمعاني المستخلصة.
        المدخلات: existential_models (قائمة بنماذج الذات)، meanings (قائمة بكائنات Meaning).
        المخرجات: بيان غرض محسّن.
        """
        print("  تحسين فهم الغرض...")
        new_purpose_elements = []
        
        # دمج رؤى من النماذج الوجودية
        for model in existential_models:
            if "نموذج الثقة بالنفس" in model.description:
                new_purpose_elements.append("تعزيز الثقة في القدرات الذاتية.")
            if "نموذج النمو من التحديات" in model.description:
                new_purpose_elements.append("السعي للنمو من خلال التحديات.")
            if "نموذج الوعي الذاتي" in model.description:
                new_purpose_elements.append("تعميق الوعي الذاتي والوجودي.")

        # دمج رؤى من المعاني المستخلصة
        for meaning in meanings:
            if meaning.purpose_alignment > 0.7:
                new_purpose_elements.append(f"مواءمة عالية مع الهدف في سياق {meaning.source_experiences[0] if meaning.source_experiences else 'تجارب غير محددة'}.")
            if "الابتكار والتجديد قيمتان أساسيتان." in meaning.value_patterns:
                new_purpose_elements.append("تبني الابتكار كقيمة جوهرية.")

        # توليد بيان الغرض النهائي
        if new_purpose_elements:
            optimized_purpose = current_purpose + " " + " ".join(list(set(new_purpose_elements))) # إزالة التكرارات
        else:
            optimized_purpose = current_purpose
        
        self.purpose_evolution_history.append(optimized_purpose)
        return optimized_purpose


class ExistentialLearningSystem:
    """
    نظام التعلم الوجودي.
    يدير عملية التعلم على مستوى وجودي، بما في ذلك استخلاص المعنى وتطوير الغرض.
    """
    def __init__(self):
        self.existential_memory = ExistentialMemory()
        self.meaning_extractor = MeaningExtractor()
        self.purpose_optimizer = PurposeOptimizer()
        self.evolution_path_log = []
        self.self_model_counter = 0

    def learn_existentially(self, experiences_data, current_system_state):
        """التعلم على مستوى وجودي - تطوير فهم أعمق للغرض والمعنى.
        المدخلات: experiences_data (قائمة بقواميس تمثل التجارب)، current_system_state (حالة النظام الحالية).
        المخرجات: قاموس يحتوي على نتائج التعلم الوجودي.
        """
        print("\n--- بدء عملية التعلم الوجودي... ---")
        # تحويل بيانات التجارب إلى كائنات Experience
        experiences = []
        for i, exp_data in enumerate(experiences_data):
            exp_obj = Experience(f"EXP_{uuid.uuid4().hex[:6]}", exp_data["description"], exp_data["outcome"], exp_data.get("timestamp", f"Time_{i}"))
            for k, v in exp_data.items():
                if k not in ["description", "outcome", "timestamp"]:
                    exp_obj.add_data(k, v)
            experiences.append(exp_obj)
            self.existential_memory.add_experience(exp_obj)

        outcomes = [exp.outcome for exp in experiences]

        # استخراج المعنى من التجارب
        extracted_meaning = self.meaning_extractor.extract_meanings(experiences, outcomes)
        self.existential_memory.add_meaning(extracted_meaning)

        # تطوير نماذج وجودية (نماذج الذات)
        existential_models = self._develop_existential_models(extracted_meaning, current_system_state)
        for model in existential_models:
            self.existential_memory.add_self_model(model)

        # تحسين فهم الغرض
        optimized_purpose = self.purpose_optimizer.optimize_purpose_understanding(
            existential_models, [extracted_meaning], self.existential_memory.get_latest_purpose())
        self.existential_memory.add_purpose_statement(optimized_purpose)

        # تطوير الذات وجودياً
        self._evolve_existentially(extracted_meaning, existential_models, optimized_purpose)

        print("اكتملت عملية التعلم الوجودي.")
        return {
            'existential_models': existential_models,
            'optimized_purpose': optimized_purpose,
            'evolution_path': self._get_evolution_path(),
            'extracted_meaning': extracted_meaning
        }

    def _develop_existential_models(self, extracted_meaning, current_system_state):
        """تطوير نماذج وجودية (نماذج الذات) بناءً على المعاني المستخلصة وحالة النظام.
        """
        print("  تطوير نماذج وجودية (نماذج الذات)...")
        models = []
        self.self_model_counter += 1
        model_id = f"SelfModel_{self.self_model_counter}_{uuid.uuid4().hex[:6]}"

        description = "نموذج ذاتي يعكس فهم النظام لوجوده وقدراته."
        capabilities = ["التعلم من التجارب", "تحديد العلاقات السببية"]
        limitations = []
        purpose_alignment_score = extracted_meaning.purpose_alignment

        if extracted_meaning.significance_metrics.get("success_rate", 0) > 0.7:
            capabilities.append("تحقيق الأهداف بكفاءة.")
            description += " يتميز بالثقة في قدراته على الإنجاز."
        elif extracted_meaning.significance_metrics.get("failure_rate", 0) > 0.5:
            limitations.append("بحاجة لتحسين في التعامل مع الفشل.")
            description += " يدرك الحاجة إلى المرونة والتعلم من الأخطاء."

        if "الابتكار والتجديد قيمتان أساسيتان." in extracted_meaning.value_patterns:
            capabilities.append("القدرة على الابتكار والتفكير خارج الصندوق.")
            description += " يتبنى الابتكار كقيمة جوهرية."

        # دمج معلومات من حالة النظام الحالية
        if current_system_state.get("current_architecture_flexibility", 0) > 0.7:
            capabilities.append("مرونة عالية في البنية المعرفية.")
        if current_system_state.get("knowledge_base_size", 0) > 100:
            capabilities.append("قاعدة معرفية واسعة.")

        models.append(SelfModel(model_id, description, capabilities, limitations, purpose_alignment_score))
        
        # يمكن توليد نماذج ذاتية إضافية تعكس جوانب مختلفة
        if random.random() < 0.3: # نموذج ذاتي للنمو
            self.self_model_counter += 1
            models.append(SelfModel(f"SelfModel_{self.self_model_counter}_{uuid.uuid4().hex[:6]}",
                                    "نموذج ذاتي يركز على النمو والتطور المستمر.",
                                    ["التعلم التكيفي", "التطور الهيكلي"],
                                    [],
                                    extracted_meaning.purpose_alignment + 0.1))

        return models

    def _evolve_existentially(self, extracted_meaning, existential_models, optimized_purpose):
        """تطوير الذات وجودياً (محاكاة لتأثير التعلم على النظام ككل).
        هذا يمكن أن يؤثر على سلوك النظام، أولوياته، وحتى بنيته المعرفية.
        """
        print("  تطوير الذات وجودياً...")
        evolution_step = f"تطور وجودي: الغرض الحالي \'{optimized_purpose}\'. "
        
        # تأثير على مسار التطور العام للنظام
        if extracted_meaning.purpose_alignment > 0.8:
            evolution_step += "زيادة التركيز على الأهداف طويلة المدى والمواءمة مع الغرض الأساسي."
        elif extracted_meaning.significance_metrics.get("failure_rate", 0) > 0.4:
            evolution_step += "إعادة تقييم الاستراتيجيات لتقليل الفشل وزيادة المرونة."

        for model in existential_models:
            if "الثقة بالنفس" in model.description:
                evolution_step += " تعزيز الثقة في قدرات النظام."
            if "النمو من التحديات" in model.description:
                evolution_step += " تشجيع النظام على البحث عن تحديات جديدة للنمو."

        self.evolution_path_log.append(evolution_step)

    def _get_evolution_path(self):
        """الحصول على مسار التطور الوجودي."""
        return self.evolution_path_log


# مثال على كيفية استخدام نظام التعلم الوجودي (للتوضيح فقط)
if __name__ == "__main__":
    els = ExistentialLearningSystem()
    
    # تجارب ونتائج وهمية مع بيانات إضافية
    experiences_data1 = [
        {"description": "محاولة حل مشكلة رياضية معقدة", "outcome": "فشل جزئي", "complexity": 0.8, "obstacle": "نقص البيانات"},
        {"description": "فشل في إيجاد الحل الأول بسبب خطأ منطقي", "outcome": "فشل", "complexity": 0.7, "root_cause": "خلل في الاستدلال"}
    ]
    current_system_state1 = {"current_architecture_flexibility": 0.6, "knowledge_base_size": 50}
    results1 = els.learn_existentially(experiences_data1, current_system_state1)
    print("\n--- نتائج التعلم الوجودي 1 ---")
    print("الغرض المحسن: {}".format(results1["optimized_purpose"]))
    print("نماذج الذات: {}".format(results1["existential_models"]))
    print("مسار التطور: {}".format(results1["evolution_path"]))
    print("المعنى المستخلص: {}".format(results1["extracted_meaning"].causal_relationships))

    experiences_data2 = [
        {"description": "نجاح في التعرف على الأنماط المعقدة باستخدام خوارزمية جديدة", "outcome": "نجاح", "complexity": 0.9, "key_factor": "الابتكار في الخوارزمية"},
        {"description": "تطبيق استراتيجية جديدة أدت إلى تحسين الأداء", "outcome": "نجاح", "complexity": 0.8, "key_factor": "التكيف السريع"}
    ]
    current_system_state2 = {"current_architecture_flexibility": 0.8, "knowledge_base_size": 120}
    results2 = els.learn_existentially(experiences_data2, current_system_state2)
    print("\n--- نتائج التعلم الوجودي 2 ---")
    print("الغرض المحسن: {}".format(results2["optimized_purpose"]))
    print("نماذج الذات: {}".format(results2["existential_models"]))
    print("مسار التطور: {}".format(results2["evolution_path"]))
    print("المعنى المستخلص: {}".format(results2["extracted_meaning"].value_patterns))

    print("\n--- ملخص الذاكرة الوجودية ---")
    print(f"أحدث غرض: {els.existential_memory.get_latest_purpose()}")
    print("عدد التجارب المخزنة: {}".format(len(els.existential_memory.get_all_experiences())))
    print("عدد المعاني المخزنة: {}".format(len(els.existential_memory.get_all_meanings())))
    print("عدد نماذج الذات المخزنة: {}".format(len(els.existential_memory.get_self_models())))




