
import random
import uuid

class CognitiveArchitecture:
    """
    يمثل بنية معرفية فردية للنظام.
    تحدد المكونات، الاتصالات، ودرجة المرونة.
    """
    def __init__(self, arch_id, components, connections, flexibility_score=0.5, innovation_level=0.1):
        self.arch_id = arch_id
        self.components = components  # قائمة بالمكونات (مثال: ["معالجة اللغة", "الذاكرة العاملة"])
        self.connections = connections  # قاموس يمثل الاتصالات بين المكونات
        self.flexibility_score = flexibility_score # مدى مرونة البنية للتكيف
        self.innovation_level = innovation_level # مستوى الابتكار في هذه البنية
        self.performance_history = [] # سجل لأداء هذه البنية عبر الزمن

    def record_performance(self, performance_data):
        """تسجيل بيانات الأداء لهذه البنية."""
        self.performance_history.append(performance_data)

    def get_average_performance(self):
        """حساب متوسط الأداء لهذه البنية."""
        if not self.performance_history:
            return 0.0
        return sum(d.get("validity_score", 0) for d in self.performance_history) / len(self.performance_history)

    def __repr__(self):
        return f"CognitiveArchitecture(ID={self.arch_id}, Comps={len(self.components)}, Flex={self.flexibility_score:.2f}, Innov={self.innovation_level:.2f})"


class ArchitectureComponent:
    """
    يمثل مكونًا فرديًا داخل البنية المعرفية.
    """
    def __init__(self, name, functionality, complexity=1.0):
        self.name = name
        self.functionality = functionality # وصف لوظيفة المكون
        self.complexity = complexity

    def __repr__(self):
        return f"Component({self.name}, Func=\"{self.functionality[:20]}...\")"


class CognitiveArchitectureDeveloper:
    """
    مُطوِّر البنية المعرفية.
    مسؤول عن تحليل الأداء، تحديد نقاط الضعف، وتوليد وتعديل البنى المعرفية.
    """
    def __init__(self):
        self.architecture_templates = self._load_architecture_templates()
        self.innovation_rules = self._initialize_innovation_rules()
        self.developed_architectures = {} # لتخزين البنى المطورة
        self.arch_counter = 0

    def _load_architecture_templates(self):
        """تحميل قوالب البنى المعمارية الأولية.
        هذه القوالب تمثل نقاط بداية لتطوير بنى جديدة.
        """
        print("  تحميل قوالب البنى المعمارية الأولية...")
        templates = []
        # قالب بسيط: إدراك -> معالجة -> فعل
        comp1 = ArchitectureComponent("PerceptionModule", "يستقبل ويحلل المدخلات الحسية.")
        comp2 = ArchitectureComponent("ProcessingUnit", "يعالج المعلومات ويتخذ القرارات.")
        comp3 = ArchitectureComponent("ActionExecutor", "ينفذ الأفعال بناءً على القرارات.")
        templates.append(CognitiveArchitecture("Base_PerceptionAction", 
                                               [comp1, comp2, comp3],
                                               {comp1.name: [comp2.name], comp2.name: [comp3.name]}))

        # قالب أكثر تعقيدًا: مع ذاكرة وتعلم
        comp4 = ArchitectureComponent("SensoryInput", "يجمع البيانات الخام.")
        comp5 = ArchitectureComponent("ShortTermMemory", "يخزن المعلومات مؤقتًا.")
        comp6 = ArchitectureComponent("LongTermMemory", "يخزن المعرفة الدائمة.")
        comp7 = ArchitectureComponent("LearningModule", "يكتسب المعرفة ويعدل السلوك.")
        comp8 = ArchitectureComponent("DecisionMaker", "يتخذ القرارات المعقدة.")
        comp9 = ArchitectureComponent("OutputHandler", "يدير المخرجات.")
        templates.append(CognitiveArchitecture("Advanced_MemoryLearning", 
                                               [comp4, comp5, comp6, comp7, comp8, comp9],
                                               {comp4.name: [comp5.name], comp5.name: [comp6.name, comp8.name],
                                                comp6.name: [comp7.name], comp7.name: [comp8.name],
                                                comp8.name: [comp9.name]}))
        return templates

    def _initialize_innovation_rules(self):
        """تهيئة قواعد الابتكار التي توجه عملية تطوير البنى.
        هذه القواعد يمكن أن تكون استدلالية أو توليدية.
        """
        print("  تهيئة قواعد الابتكار...")
        return [
            "إذا كان الأداء منخفضًا بسبب نقص المعلومات، أضف مكونًا لجمع البيانات.",
            "إذا كان الأداء بطيئًا، حاول تبسيط الاتصالات أو دمج المكونات.",
            "إذا كان النظام غير قادر على التكيف، أضف مكونًا للتعلم التكيفي أو زيادة المرونة.",
            "إذا كانت هناك حاجة لقدرات جديدة، قم بتوليد مكونات بوظائف جديدة.",
            "استكشاف بنى هجينة تجمع بين خصائص القوالب المختلفة."
        ]

    def develop_new_architecture(self, problem_domain, performance_data):
        """تطوير بنية معرفية جديدة متخصصة للمجال بناءً على الأداء الحالي.
        المدخلات: problem_domain (وصف للمجال)، performance_data (بيانات أداء من المحاكي).
        المخرجات: بنية معرفية جديدة أو محسنة.
        """
        print(f"\n--- بدء تطوير بنية معرفية جديدة للمجال: {problem_domain} ---")
        domain_requirements = self._analyze_domain_requirements(problem_domain)
        weaknesses = self._identify_architecture_weaknesses(performance_data)

        # محاولة توليد بنية جديدة
        new_architecture = self._generate_new_architecture(
            domain_requirements, weaknesses, performance_data)

        if not new_architecture:
            print("  فشل في توليد بنية معرفية جديدة بشكل مبدئي.")
            return None

        # اختبار البنية الجديدة
        test_results = self._test_architecture(new_architecture, problem_domain)

        if test_results.get("success", False):
            print(f"  البنية {new_architecture.arch_id} اجتازت الاختبارات الأولية. تحسين...")
            refined_arch = self._refine_architecture(new_architecture, test_results)
            self.developed_architectures[refined_arch.arch_id] = refined_arch
            return refined_arch
        else:
            print(f"  البنية {new_architecture.arch_id} لم تجتز الاختبارات. تكرار التصميم...")
            iterated_arch = self._iterate_architecture_design(new_architecture, test_results)
            if iterated_arch:
                self.developed_architectures[iterated_arch.arch_id] = iterated_arch
            return iterated_arch

    def _analyze_domain_requirements(self, problem_domain):
        """تحليل متطلبات المجال لتحديد الخصائص المطلوبة للبنية.
        """
        print(f"  تحليل متطلبات المجال: {problem_domain}")
        requirements = {"scalability": 0.5, "efficiency": 0.5, "adaptability": 0.5, "novelty": 0.1}
        if "complex" in problem_domain or "advanced" in problem_domain:
            requirements["scalability"] = 0.8
            requirements["adaptability"] = 0.7
            requirements["novelty"] = 0.5 # تشجيع الابتكار للمجالات المعقدة
        if "real-time" in problem_domain:
            requirements["efficiency"] = 0.9
        if "creative" in problem_domain:
            requirements["novelty"] = 0.8
        return requirements

    def _identify_architecture_weaknesses(self, performance_data):
        """تحديد نقاط الضعف في البنى الحالية بناءً على بيانات الأداء.
        """
        print("  تحديد نقاط الضعف في البنى الحالية...")
        weaknesses = []
        if performance_data.get("validity_score", 0) < 0.6:
            weaknesses.append("ضعف في جودة الحلول المنتجة (validity_score منخفض).")
        if performance_data.get("reasoning_process_length", 0) > 300:
            weaknesses.append("عدم كفاءة في عملية الاستدلال (عدد خطوات كبير).")
        if not performance_data.get("model_id") and performance_data.get("attempted_models", 0) > 0:
            weaknesses.append("فشل في توليد نماذج عقلية فعالة أو الوصول إلى حل.")
        if performance_data.get("adaptability_score", 0) < 0.5:
            weaknesses.append("ضعف في قدرة النموذج على التكيف.")
        print(f"  نقاط الضعف المحددة: {weaknesses}")
        return weaknesses

    def _generate_new_architecture(self, requirements, weaknesses, performance_data):
        """توليد بنية معرفية جديدة بناءً على المتطلبات ونقاط الضعف.
        تستخدم قواعد الابتكار وتجمع بين مكونات من القوالب الموجودة أو تولد جديدة.
        """
        print("  توليد بنية معرفية جديدة...")
        self.arch_counter += 1
        arch_id = f"Arch_{self.arch_counter}_{uuid.uuid4().hex[:6]}"

        # البدء بقالب عشوائي أو بناء من الصفر
        base_arch = random.choice(self.architecture_templates) if self.architecture_templates else CognitiveArchitecture("Base", [], {})
        components = [comp for comp in base_arch.components] # نسخ المكونات
        connections = {k: list(v) for k, v in base_arch.connections.items()} # نسخ الاتصالات
        flexibility = base_arch.flexibility_score
        innovation = base_arch.innovation_level

        # تطبيق قواعد الابتكار بناءً على نقاط الضعف والمتطلبات
        for rule in self.innovation_rules:
            if "ضعف في جودة الحلول" in weaknesses and "جمع البيانات" in rule:
                new_comp = ArchitectureComponent("DataAcquisitionUnit", "يجمع ويصفي البيانات.", complexity=1.2)
                if new_comp not in components: components.append(new_comp)
                connections[new_comp.name] = [c.name for c in base_arch.components if "Perception" in c.name or "Input" in c.name]
                flexibility = min(1.0, flexibility + 0.1)
                innovation = min(1.0, innovation + 0.05)
            elif "عدم كفاءة في عملية الاستدلال" in weaknesses and "تبسيط الاتصالات" in rule:
                # مثال: إزالة بعض الاتصالات العشوائية لتبسيط المسار
                if connections:
                    key_to_modify = random.choice(list(connections.keys()))
                    if connections[key_to_modify]:
                        connections[key_to_modify].pop(random.randrange(len(connections[key_to_modify])))
                flexibility = min(1.0, flexibility + 0.05)
                innovation = min(1.0, innovation + 0.03)
            elif "غير قادر على التكيف" in weaknesses and "التعلم التكيفي" in rule:
                new_comp = ArchitectureComponent("AdaptiveLearningCore", "يعدل سلوك النظام بناءً على البيئة.", complexity=1.5)
                if new_comp not in components: components.append(new_comp)
                connections[new_comp.name] = [c.name for c in components if "Processing" in c.name or "Decision" in c.name]
                flexibility = min(1.0, flexibility + 0.2)
                innovation = min(1.0, innovation + 0.1)
            elif "حاجة لقدرات جديدة" in rule and requirements["novelty"] > 0.6:
                # توليد مكون جديد تمامًا
                new_comp_name = f"NoveltyModule_{uuid.uuid4().hex[:4]}"
                new_comp_func = f"يقدم قدرة جديدة في {random.choice(['التحليل', 'التوليد', 'التنبؤ'])}."
                new_comp = ArchitectureComponent(new_comp_name, new_comp_func, complexity=random.uniform(1.0, 2.0))
                if new_comp not in components: components.append(new_comp)
                # ربط عشوائي
                if components and len(components) > 1:
                    src = random.choice([c.name for c in components if c.name != new_comp.name])
                    connections[src] = connections.get(src, []) + [new_comp.name]
                innovation = min(1.0, innovation + 0.15)

        # إضافة مكونات عشوائية أو تعديلات لزيادة التنوع والابتكار
        if random.random() < requirements["novelty"] * 0.5: # احتمالية إضافة مكون عشوائي بناءً على متطلب الابتكار
            new_comp_name = f"RandomComp_{uuid.uuid4().hex[:4]}"
            new_comp_func = f"مكون وظيفي عشوائي {random.choice(['للتصفية', 'للدمج', 'للتوزيع'])}."
            new_comp = ArchitectureComponent(new_comp_name, new_comp_func, complexity=random.uniform(0.5, 1.5))
            if new_comp not in components: components.append(new_comp)
            if components and len(components) > 1:
                src = random.choice([c.name for c in components if c.name != new_comp.name])
                connections[src] = connections.get(src, []) + [new_comp.name]
            innovation = min(1.0, innovation + 0.02)

        # التأكد من أن جميع المكونات لها مدخلات ومخرجات (بشكل مبسط)
        for comp in components:
            if comp.name not in connections and any(comp.name in dests for dests in connections.values()):
                # إذا كان مكونًا مستهدفًا ولكن ليس له مخرجات، أضف مخرجًا عشوائيًا
                if components and len(components) > 1:
                    dest = random.choice([c.name for c in components if c.name != comp.name])
                    connections[comp.name] = [dest]
            elif comp.name not in connections and not any(comp.name in dests for dests in connections.values()):
                # مكون معزول، ربطه بمدخل ومخرج عشوائي
                if components and len(components) > 1:
                    src = random.choice([c.name for c in components if c.name != comp.name])
                    dest = random.choice([c.name for c in components if c.name != comp.name and c.name != src])
                    connections[src] = connections.get(src, []) + [comp.name]
                    connections[comp.name] = [dest]

        return CognitiveArchitecture(arch_id, components, connections, flexibility, innovation)

    def _test_architecture(self, architecture, problem_domain):
        """اختبار البنية الجديدة في بيئة محاكاة.
        تقييم الأداء بناءً على مدى تلبية متطلبات المجال.
        """
        print(f"  اختبار البنية: {architecture.arch_id} في المجال: {problem_domain}")
        # محاكاة أكثر تفصيلاً لنتائج الاختبار
        success_prob = 0.4 + (architecture.flexibility_score * 0.3) + (architecture.innovation_level * 0.2)
        # كلما زاد عدد المكونات، زادت احتمالية التعقيد ولكن أيضًا القدرة
        success_prob += (len(architecture.components) / 10.0) * 0.1 # مكافأة على التعقيد المعقول

        if "complex" in problem_domain:
            success_prob -= 0.15 # المشاكل المعقدة أصعب
        if "real-time" in problem_domain:
            # البنى ذات المكونات الكثيرة قد تكون أبطأ
            success_prob -= (len(architecture.components) / 10.0) * 0.05

        test_success = random.random() < success_prob
        performance_score = random.uniform(0.5, 0.9) if test_success else random.uniform(0.1, 0.4)

        # تقييم مدى تلبية المتطلبات
        requirements_met_score = 0.0
        domain_reqs = self._analyze_domain_requirements(problem_domain)
        if test_success:
            if architecture.flexibility_score >= domain_reqs["adaptability"]:
                requirements_met_score += 0.3
            if performance_score >= domain_reqs["efficiency"]:
                requirements_met_score += 0.3
            if architecture.innovation_level >= domain_reqs["novelty"]:
                requirements_met_score += 0.2
            # يمكن إضافة المزيد من المعايير هنا

        print(f"  نتائج اختبار البنية {architecture.arch_id}: النجاح={test_success}, الأداء={performance_score:.2f}, تلبية المتطلبات={requirements_met_score:.2f}")
        return {"success": test_success, "performance_score": performance_score, 
                "requirements_met_score": requirements_met_score, "details": "تفاصيل الاختبار..."}

    def _refine_architecture(self, architecture, test_results):
        """تحسين البنية المعمارية بناءً على نتائج الاختبار الناجحة.
        يهدف إلى تحسين الكفاءة أو تبسيط البنية دون المساس بالأداء.
        """
        print(f"  تحسين البنية: {architecture.arch_id}")
        # إذا كان الأداء جيدًا جدًا، حاول تبسيط البنية
        if test_results["performance_score"] > 0.85 and len(architecture.components) > 4:
            print("  محاولة تبسيط البنية (إزالة مكونات غير ضرورية)...")
            # إزالة مكون عشوائي إذا كان لا يؤثر سلبًا على الاتصالات الحرجة
            removable_comps = [c for c in architecture.components if len(architecture.connections.get(c.name, [])) == 0] # مكونات ليس لها مخرجات
            if removable_comps:
                comp_to_remove = random.choice(removable_comps)
                architecture.components = [c for c in architecture.components if c != comp_to_remove]
                # تنظيف الاتصالات التي تشير إلى المكون المحذوف
                for k in architecture.connections:
                    architecture.connections[k] = [c for c in architecture.connections[k] if c != comp_to_remove.name]
                print(f"  تم إزالة المكون: {comp_to_remove.name}")

        # زيادة درجة المرونة قليلاً إذا كانت البنية ناجحة
        architecture.flexibility_score = min(1.0, architecture.flexibility_score + 0.05)
        # تسجيل الأداء الحالي للبنية
        architecture.record_performance(test_results)
        print(f"  البنية {architecture.arch_id} بعد التحسين: {architecture}")
        return architecture

    def _iterate_architecture_design(self, architecture, test_results):
        """تكرار تصميم البنية المعمارية إذا فشلت الاختبارات.
        يهدف إلى معالجة نقاط الضعف المحددة وتحسين الأداء.
        """
        print(f"  تكرار تصميم البنية: {architecture.arch_id}")
        # إضافة مكون جديد أو تعديل الاتصالات بشكل جذري لمعالجة الفشل
        if not test_results["success"]:
            print("  إعادة تصميم جذرية لمعالجة الفشل...")
            # إضافة مكون لتعزيز القدرة في المنطقة التي فشلت
            if "ضعف في جودة الحلول" in self._identify_architecture_weaknesses(test_results):
                new_comp = ArchitectureComponent("ErrorCorrectionUnit", "يصحح الأخطاء ويحسن جودة المخرجات.", complexity=1.3)
                if new_comp not in architecture.components: architecture.components.append(new_comp)
                # ربطها بمكونات المعالجة الرئيسية
                processing_comps = [c.name for c in architecture.components if "Processing" in c.name or "Decision" in c.name]
                if processing_comps: architecture.connections[random.choice(processing_comps)] = architecture.connections.get(random.choice(processing_comps), []) + [new_comp.name]
                architecture.flexibility_score = min(1.0, architecture.flexibility_score + 0.1)
                architecture.innovation_level = min(1.0, architecture.innovation_level + 0.05)
            elif "عدم كفاءة في عملية الاستدلال" in self._identify_architecture_weaknesses(test_results):
                new_comp = ArchitectureComponent("EfficiencyOptimizer", "يحسن كفاءة مسارات الاستدلال.", complexity=1.1)
                if new_comp not in architecture.components: architecture.components.append(new_comp)
                # ربطها بمكونات المعالجة الرئيسية
                processing_comps = [c.name for c in architecture.components if "Processing" in c.name or "Decision" in c.name]
                if processing_comps: architecture.connections[random.choice(processing_comps)] = architecture.connections.get(random.choice(processing_comps), []) + [new_comp.name]
                architecture.flexibility_score = min(1.0, architecture.flexibility_score + 0.08)
                architecture.innovation_level = min(1.0, architecture.innovation_level + 0.03)

        # زيادة المرونة بشكل أكبر لتشجيع التكيف في التكرارات اللاحقة
        architecture.flexibility_score = min(1.0, architecture.flexibility_score + 0.05)
        # تسجيل الأداء الحالي للبنية
        architecture.record_performance(test_results)
        print(f"  البنية {architecture.arch_id} بعد التكرار: {architecture}")
        return architecture


# مثال على كيفية استخدام مُطوِّر البنية المعرفية (للتوضيح فقط)
if __name__ == "__main__":
    developer = CognitiveArchitectureDeveloper()
    
    # بيانات أداء وهمية من محاكي العقلية الاصطناعية
    performance_data_good = {"model_id": "MM_1", "validity_score": 0.85, "reasoning_process_length": 80, "adaptability_score": 0.7}
    performance_data_bad = {"model_id": "MM_2", "validity_score": 0.3, "reasoning_process_length": 180, "adaptability_score": 0.4, "attempted_models": 5}

    print("\n--- تطوير بنية لمشكلة عامة بأداء جيد ---")
    new_arch1 = developer.develop_new_architecture("general_problem_solving", performance_data_good)
    if new_arch1:
        print(f"البنية المطورة 1: {new_arch1}")
        print(f"متوسط أداء البنية 1: {new_arch1.get_average_performance():.2f}")

    print("\n--- تطوير بنية لمشكلة معقدة بأداء سيء ---")
    new_arch2 = developer.develop_new_architecture("complex_optimization_problem", performance_data_bad)
    if new_arch2:
        print(f"البنية المطورة 2: {new_arch2}")
        print(f"متوسط أداء البنية 2: {new_arch2.get_average_performance():.2f}")

    print("\n--- تطوير بنية بدون بيانات أداء (سيناريو أولي) ---")
    new_arch3 = developer.develop_new_architecture("initial_setup", {})
    if new_arch3:
        print(f"البنية المطورة 3: {new_arch3}")
        print(f"متوسط أداء البنية 3: {new_arch3.get_average_performance():.2f}")

    print("\n--- جميع البنى المطورة ---")
    for arch_id, arch in developer.developed_architectures.items():
        print(f"- {arch}")




