
import random
import uuid

class IntuitionBase:
    """
    قاعدة معرفية للحدس، تخزن الحقائق والأنماط والعلاقات الدلالية.
    """
    def __init__(self):
        self.knowledge_graph = {}
        self.pattern_library = []
        self.semantic_relations = {}

    def add_fact(self, fact, category="general"):
        if category not in self.knowledge_graph:
            self.knowledge_graph[category] = set() # استخدام مجموعة لتجنب التكرار
        self.knowledge_graph[category].add(fact)

    def add_pattern(self, pattern_name, elements):
        self.pattern_library.append({"name": pattern_name, "elements": elements})

    def add_semantic_relation(self, entity1, relation, entity2):
        if entity1 not in self.semantic_relations:
            self.semantic_relations[entity1] = []
        self.semantic_relations[entity1].append((relation, entity2))

    def get_related_facts(self, query_term, category=None):
        facts = set()
        if category and category in self.knowledge_graph:
            facts.update([f for f in self.knowledge_graph[category] if query_term.lower() in f.lower()])
        else:
            for cat in self.knowledge_graph:
                facts.update([f for f in self.knowledge_graph[cat] if query_term.lower() in f.lower()])
        return list(facts)

    def get_patterns_matching(self, context_elements):
        matching_patterns = []
        for pattern in self.pattern_library:
            if all(elem.lower() in [ce.lower() for ce in context_elements] for elem in pattern["elements"]):
                matching_patterns.append(pattern["name"])
        return matching_patterns

    def get_semantic_relations(self, entity):
        return self.semantic_relations.get(entity, [])


class Insight:
    """
    يمثل رؤية حدسية تم توليدها.
    """
    def __init__(self, insight_id, description, source_type, confidence=0.5):
        self.insight_id = insight_id
        self.description = description
        self.source_type = source_type # (e.g., 'fact_based', 'pattern_based', 'analogical')
        self.confidence = confidence

    def __repr__(self):
        return f"Insight(ID={self.insight_id}, Source={self.source_type}, Conf={self.confidence:.2f})"


class CreativeLeap:
    """
    يمثل قفزة إبداعية تم إجراؤها.
    """
    def __init__(self, leap_id, description, original_insight_id, validity_score=0.0):
        self.leap_id = leap_id
        self.description = description
        self.original_insight_id = original_insight_id
        self.validity_score = validity_score

    def __repr__(self):
        return f"CreativeLeap(ID={self.leap_id}, Valid={self.validity_score:.2f})"


class InsightGenerator:
    """
    يولد رؤى جديدة بناءً على قاعدة الحدس وسياق المشكلة.
    """
    def __init__(self):
        self.insight_counter = 0

    def generate_insights(self, intuitive_base, problem_context):
        """توليد رؤى جديدة بناءً على قاعدة الحدس وسياق المشكلة.
        المدخلات: intuitive_base (كائن IntuitionBase)، problem_context (قاموس يحتوي على سياق المشكلة).
        المخرجات: قائمة بكائنات Insight.
        """
        print("  توليد رؤى...")
        insights = []

        # 1. رؤى قائمة على الحقائق
        main_topic = problem_context.get("main_topic", "")
        related_facts = intuitive_base.get_related_facts(main_topic)
        if related_facts:
            self.insight_counter += 1
            insight_id = f"Insight_{self.insight_counter}_{uuid.uuid4().hex[:4]}"
            insights.append(Insight(insight_id, f"رؤية من الحقائق ذات الصلة: {random.choice(related_facts)}", "fact_based", confidence=0.7))

        # 2. رؤى قائمة على الأنماط
        keywords = problem_context.get("keywords", [])
        matching_patterns = intuitive_base.get_patterns_matching(keywords)
        if matching_patterns:
            self.insight_counter += 1
            insight_id = f"Insight_{self.insight_counter}_{uuid.uuid4().hex[:4]}"
            insights.append(Insight(insight_id, f"رؤية من الأنماط المطابقة: {random.choice(matching_patterns)}", "pattern_based", confidence=0.65))

        # 3. رؤى قائمة على العلاقات الدلالية
        if main_topic:
            semantic_relations = intuitive_base.get_semantic_relations(main_topic)
            if semantic_relations:
                self.insight_counter += 1
                insight_id = f"Insight_{self.insight_counter}_{uuid.uuid4().hex[:4]}"
                relation, entity = random.choice(semantic_relations)
                insights.append(Insight(insight_id, f"رؤية من العلاقة الدلالية: {main_topic} {relation} {entity}", "semantic_based", confidence=0.75))

        # 4. رؤى توليدية (عشوائية/إبداعية)
        if random.random() < 0.4: # 40% فرصة لتوليد رؤية إبداعية
            self.insight_counter += 1
            insight_id = f"Insight_{self.insight_counter}_{uuid.uuid4().hex[:4]}"
            creative_insight_desc = random.choice([
                "ربما هناك علاقة غير متوقعة بين X و Y.",
                "ماذا لو نظرنا إلى المشكلة من منظور مختلف تمامًا؟",
                "هل يمكن تطبيق حل من مجال آخر على هذه المشكلة؟"
            ])
            insights.append(Insight(insight_id, creative_insight_desc, "generative", confidence=0.5))

        print(f"  تم توليد {len(insights)} رؤى.")
        return insights


class CreativeLeapEngine:
    """
    مسؤول عن إجراء "القفزات المفاهيمية" التي تؤدي إلى فهم حدسي.
    """
    def __init__(self):
        self.leap_counter = 0

    def make_creative_leaps(self, insights, problem_context):
        """إجراء قفزات إبداعية تتجاوز الفهم الحالي.
        المدخلات: insights (قائمة بكائنات Insight)، problem_context (قاموس يحتوي على سياق المشكلة).
        المخرجات: قائمة بكائنات CreativeLeap.
        """
        print("  إجراء قفزات إبداعية...")
        creative_leaps = []

        for insight in insights:
            # استكشاف مسافات مفهومية بعيدة
            conceptual_distances = self._explore_conceptual_distances(insight, problem_context)

            for distance in conceptual_distances:
                # محاولة القفز المفاهيمي
                leap_attempt_desc = self._attempt_conceptual_leap(insight, distance)
                
                # تقييم صلاحية القفزة
                validity_score = self._validate_conceptual_leap(leap_attempt_desc, problem_context)
                
                self.leap_counter += 1
                leap_id = f"Leap_{self.leap_counter}_{uuid.uuid4().hex[:4]}"
                creative_leaps.append(CreativeLeap(leap_id, leap_attempt_desc, insight.insight_id, validity_score))
                
                if validity_score > 0.6:
                    print(f"    تمت قفزة إبداعية ناجحة (صلاحية {validity_score:.2f}): {leap_attempt_desc}")
                else:
                    print(f"    فشلت محاولة القفزة (صلاحية {validity_score:.2f}): {leap_attempt_desc}")

        return creative_leaps

    def _explore_conceptual_distances(self, insight, problem_context):
        """استكشاف مسافات مفهومية بعيدة (منطق أكثر تعقيدًا).
        يمكن أن يشمل البحث عن مجالات بعيدة، أو تطبيق مبادئ مجردة.
        """
        distances = [
            f"ربط \'{insight.description}\' بمجال غير ذي صلة مثل {random.choice(['البيولوجيا', 'الموسيقى', 'الفلسفة القديمة'])}.",
            f"تطبيق مبدأ {random.choice(['التطور الطبيعي', 'التوازن', 'الفوضى المنظمة'])} على \'{insight.description}\'",
            f"عكس الافتراضات الأساسية لـ \'{insight.description}\' والنظر في النقيض.",
            f"توسيع نطاق \'{insight.description}\' ليشمل أبعادًا زمنية أو مكانية مختلفة."
        ]
        return distances

    def _attempt_conceptual_leap(self, insight, distance):
        """محاولة القفزة المفاهيمية (توليد وصف للقفزة)."""
        return f"قفزة مفاهيمية: من \'{insight.description}\' إلى \'{distance}\'"

    def _validate_conceptual_leap(self, leap_attempt_desc, problem_context):
        """التحقق من صحة القفزة المفاهيمية (تقييم أكثر دقة).
        يعتمد على مدى ملاءمة القفزة لسياق المشكلة، ومدى ابتكارها، وجدواها المحتملة.
        """
        validity_score = random.uniform(0.1, 0.9) # درجة صلاحية أولية عشوائية

        # مكافأة على الابتكار (إذا كانت القفزة تبدو غير تقليدية)
        if "غير ذي صلة" in leap_attempt_desc or "عكس الافتراضات" in leap_attempt_desc:
            validity_score += 0.15 # مكافأة على الجرأة
        
        # عقوبة إذا كانت القفزة بعيدة جدًا عن سياق المشكلة
        if not any(keyword.lower() in leap_attempt_desc.lower() for keyword in problem_context.get("keywords", [])):
            validity_score -= 0.2

        # ضمان أن تكون النتيجة بين 0 و 1
        return max(0.0, min(1.0, validity_score))


class IntuitiveUnderstandingGenerator:
    """
    مُولد الفهم الحدسي.
    يدمج قاعدة الحدس، مولد الرؤى، ومحرك القفزة الإبداعية لتكوين فهم شامل.
    """
    def __init__(self):
        self.intuition_base = IntuitionBase()
        self.insight_generator = InsightGenerator()
        self.creative_leap_engine = CreativeLeapEngine()
        self.integrated_understandings = []

        # تهيئة قاعدة الحدس ببعض الحقائق والأنماط والعلاقات الأولية
        self.intuition_base.add_fact("الماء يتجمد عند 0 درجة مئوية.", "فيزياء")
        self.intuition_base.add_fact("الجاذبية تسحب الأجسام نحو الأرض.", "فيزياء")
        self.intuition_base.add_fact("التعلم المستمر يؤدي إلى التطور.", "فلسفة")
        self.intuition_base.add_fact("الخلايا هي الوحدة الأساسية للحياة.", "بيولوجيا")
        self.intuition_base.add_pattern("حل المشكلات", ["مشكلة", "حل", "استراتيجية"])
        self.intuition_base.add_pattern("النمو والتطور", ["نمو", "تطور", "تحسين"])
        self.intuition_base.add_semantic_relation("الذكاء الاصطناعي", "هو فرع من", "علوم الحاسوب")
        self.intuition_base.add_semantic_relation("الابتكار", "يؤدي إلى", "التقدم")
        self.intuition_base.add_semantic_relation("التعلم", "يؤدي إلى", "المعرفة")

    def generate_intuitive_understanding(self, problem_context, known_facts):
        """توليد فهم حدسي للمشكلة يتجاوز التحليل المنطقي.
        المدخلات: problem_context (قاموس يحتوي على سياق المشكلة)، known_facts (قائمة بالحقائق المعروفة).
        المخرجات: وصف نصي للفهم الحدسي المتكامل.
        """
        print("\n--- بدء توليد الفهم الحدسي... ---")
        # تحديث قاعدة الحدس بالحقائق المعروفة الجديدة
        for fact in known_facts:
            self.intuition_base.add_fact(fact, "problem_specific")

        # توليد رؤى
        insights = self.insight_generator.generate_insights(self.intuition_base, problem_context)

        # إجراء قفزات إبداعية
        creative_leaps = self.creative_leap_engine.make_creative_leaps(insights, problem_context)

        # تكوين الفهم الحدسي المتكامل
        integrated_understanding_text = self._integrate_understanding(
            self.intuition_base, insights, creative_leaps)
        
        self.integrated_understandings.append(integrated_understanding_text)
        print("اكتمل توليد الفهم الحدسي.")
        return integrated_understanding_text

    def _integrate_understanding(self, intuitive_base, insights, creative_leaps):
        """تكوين الفهم الحدسي المتكامل من الرؤى والقفزات الإبداعية.
        """
        print("  تكوين الفهم الحدسي المتكامل...")
        integrated_text = "فهم حدسي متكامل للمشكلة:\n"

        if insights:
            integrated_text += "  **رؤى مكتسبة:**\n"
            for insight in insights:
                integrated_text += f"    - {insight.description} (الثقة: {insight.confidence:.2f}, المصدر: {insight.source_type})\n"
        
        if creative_leaps:
            integrated_text += "  **قفزات إبداعية:**\n"
            for leap in creative_leaps:
                integrated_text += f"    - {leap.description} (الصلاحية: {leap.validity_score:.2f})\n"
        
        integrated_text += "\n  يعتمد هذا الفهم على قاعدة حدسية غنية بالحقائق، الأنماط، والعلاقات الدلالية، مما يمكن النظام من تجاوز التحليل المنطقي البحت للوصول إلى حلول مبتكرة وغير تقليدية."
        
        # إضافة ملخص بسيط لقاعدة الحدس
        integrated_text += "\n  **ملخص قاعدة الحدس المستخدمة:**\n"
        integrated_text += f"    - عدد فئات الحقائق: {len(intuitive_base.knowledge_graph)}\n"
        integrated_text += f"    - عدد الأنماط المخزنة: {len(intuitive_base.pattern_library)}\n"
        integrated_text += f"    - عدد العلاقات الدلالية: {len(intuitive_base.semantic_relations)}\n"

        return integrated_text


# مثال على كيفية استخدام مُولد الفهم الحدسي (للتوضيح فقط)
if __name__ == "__main__":
    iug = IntuitiveUnderstandingGenerator()
    
    problem_context1 = {"main_topic": "تطوير ذكاء اصطناعي عام", "keywords": ["ذكاء اصطناعي", "عام", "تطور", "وعي"]}
    known_facts1 = ["التعلم العميق أحدث ثورة في معالجة اللغة.", "الشبكات العصبية تحاكي الدماغ.", "الوعي مفهوم معقد."]
    intuitive_understanding1 = iug.generate_intuitive_understanding(problem_context1, known_facts1)
    print("\n--- الفهم الحدسي للمشكلة 1 ---")
    print(intuitive_understanding1)

    problem_context2 = {"main_topic": "حل مشكلة التغير المناخي", "keywords": ["مناخ", "تغير", "حلول", "طاقة متجددة"]}
    known_facts2 = ["الاحتباس الحراري يسببه انبعاثات الغازات الدفيئة.", "الطاقة الشمسية مصدر نظيف.", "التعاون الدولي ضروري."]
    intuitive_understanding2 = iug.generate_intuitive_understanding(problem_context2, known_facts2)
    print("\n--- الفهم الحدسي للمشكلة 2 ---")
    print(intuitive_understanding2)

    print("\n--- جميع الفهوم الحدسية المتكاملة التي تم توليدها ---")
    for i, understanding in enumerate(iug.integrated_understandings):
        print(f"الفهم {i+1}: {understanding[:100]}...") # طباعة جزء من الفهم




