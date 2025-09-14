
import time
import uuid

class PerformanceMetric:
    """
    يمثل مقياس أداء فردي.
    """
    def __init__(self, name, value, timestamp, unit="", context=None):
        self.name = name
        self.value = value
        self.timestamp = timestamp
        self.unit = unit
        self.context = context if context is not None else {}

    def __repr__(self):
        return "Metric({}={} {}, Time={})".format(self.name, self.value, self.unit, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp)))


class PerformanceEvaluator:
    """
    يقوم بتقييم أداء النظام عبر جمع وتحليل المقاييس من مختلف المكونات.
    """
    def __init__(self):
        self.metrics_history = []
        self.evaluation_reports = []
        self.thresholds = {
            "validity_score": {"min": 0.6, "max": 1.0},
            "reasoning_efficiency": {"min": 0.7, "max": 1.0},
            "adaptability_score": {"min": 0.5, "max": 1.0},
            "knowledge_growth_rate": {"min": 0.01, "max": 0.1},
            "response_time": {"min": 0.0, "max": 2.0} # بالثواني
        }
        print("تم تهيئة مقيم الأداء.")

    def collect_metrics(self, metric_name, value, unit="", context=None):
        """جمع مقياس أداء جديد."""
        timestamp = time.time()
        metric = PerformanceMetric(metric_name, value, timestamp, unit, context)
        self.metrics_history.append(metric)
        print(f"  [Evaluator] تم جمع المقياس: {metric}")
        return metric

    def evaluate_performance(self, evaluation_period_seconds=3600):
        """
        تقييم الأداء بناءً على المقاييس المجمعة خلال فترة زمنية محددة.
        المدخلات: evaluation_period_seconds (الفترة الزمنية للتقييم بالثواني).
        المخرجات: تقرير تقييم الأداء.
        """
        print(f"\n--- بدء تقييم الأداء للفترة الزمنية الأخيرة ({evaluation_period_seconds} ثانية) ---")
        current_time = time.time()
        relevant_metrics = [m for m in self.metrics_history if m.timestamp >= (current_time - evaluation_period_seconds)]

        if not relevant_metrics:
            print("  لا توجد مقاييس ذات صلة للتقييم في الفترة المحددة.")
            return {"status": "no_data", "report_id": str(uuid.uuid4())}

        aggregated_metrics = self._aggregate_metrics(relevant_metrics)
        performance_summary = self._analyze_aggregated_metrics(aggregated_metrics)
        recommendations = self._generate_recommendations(performance_summary)

        report_id = str(uuid.uuid4())
        evaluation_report = {
            "report_id": report_id,
            "timestamp": current_time,
            "period_seconds": evaluation_period_seconds,
            "aggregated_metrics": aggregated_metrics,
            "performance_summary": performance_summary,
            "recommendations": recommendations,
            "status": "completed"
        }
        self.evaluation_reports.append(evaluation_report)
        print(f"  تم إنشاء تقرير تقييم الأداء: {report_id}")
        return evaluation_report

    def _aggregate_metrics(self, metrics):
        """تجميع المقاييس ذات الصلة."""
        aggregated = {}
        for metric in metrics:
            if metric.name not in aggregated:
                aggregated[metric.name] = []
            aggregated[metric.name].append(metric.value)
        
        # حساب المتوسطات للقيم الرقمية
        for name, values in aggregated.items():
            if all(isinstance(v, (int, float)) for v in values):
                aggregated[name] = sum(values) / len(values)
            else:
                # للقيم غير الرقمية، يمكن الاحتفاظ بالقائمة أو اختيار الأكثر تكرارًا
                aggregated[name] = values
        return aggregated

    def _analyze_aggregated_metrics(self, aggregated_metrics):
        """تحليل المقاييس المجمعة وتحديد نقاط القوة والضعف."""
        summary = {"overall_status": "Good", "strengths": [], "weaknesses": [], "areas_for_improvement": []}

        # تقييم Validity Score
        validity_score = aggregated_metrics.get("validity_score", 0.0)
        if validity_score < self.thresholds["validity_score"]["min"]:
            summary["overall_status"] = "Needs Improvement"
            summary["weaknesses"].append(f"درجة الصلاحية منخفضة ({validity_score:.2f}).")
            summary["areas_for_improvement"].append("تحسين جودة الحلول المنتجة.")
        else:
            summary["strengths"].append(f"درجة صلاحية جيدة ({validity_score:.2f}).")

        # تقييم Reasoning Efficiency (مثال: مقلوب عدد الخطوات)
        reasoning_process_length = aggregated_metrics.get("reasoning_process_length", 1)
        # نفترض أن الكفاءة هي 1 / طول العملية، مع تطبيع
        reasoning_efficiency = 1.0 / (reasoning_process_length / 100.0) if reasoning_process_length > 0 else 0.0
        reasoning_efficiency = min(1.0, reasoning_efficiency) # تطبيع للحد الأقصى 1

        if reasoning_efficiency < self.thresholds["reasoning_efficiency"]["min"]:
            summary["overall_status"] = "Needs Improvement"
            summary["weaknesses"].append(f"كفاءة الاستدلال منخفضة ({reasoning_efficiency:.2f}).")
            summary["areas_for_improvement"].append("تبسيط مسارات الاستدلال أو تقليل الخطوات.")
        else:
            summary["strengths"].append(f"كفاءة استدلال جيدة ({reasoning_efficiency:.2f}).")

        # تقييم Adaptability Score
        adaptability_score = aggregated_metrics.get("adaptability_score", 0.0)
        if adaptability_score < self.thresholds["adaptability_score"]["min"]:
            summary["overall_status"] = "Needs Improvement"
            summary["weaknesses"].append(f"قدرة التكيف منخفضة ({adaptability_score:.2f}).")
            summary["areas_for_improvement"].append("زيادة مرونة البنية المعرفية.")
        else:
            summary["strengths"].append(f"قدرة تكيف جيدة ({adaptability_score:.2f}).")

        # تقييم Knowledge Growth Rate (مثال: عدد الحقائق الجديدة)
        knowledge_growth_rate = aggregated_metrics.get("knowledge_growth_rate", 0.0)
        if knowledge_growth_rate < self.thresholds["knowledge_growth_rate"]["min"]:
            summary["areas_for_improvement"].append(f"معدل نمو المعرفة بطيء ({knowledge_growth_rate:.2f}).")
        else:
            summary["strengths"].append(f"معدل نمو معرفة جيد ({knowledge_growth_rate:.2f}).")

        # تقييم Response Time
        response_time = aggregated_metrics.get("response_time", 0.0)
        if response_time > self.thresholds["response_time"]["max"]:
            summary["overall_status"] = "Needs Improvement"
            summary["weaknesses"].append(f"وقت الاستجابة مرتفع ({response_time:.2f} ثانية).")
            summary["areas_for_improvement"].append("تحسين سرعة المعالجة وتقليل زمن الاستجابة.")
        else:
            summary["strengths"].append(f"وقت استجابة جيد ({response_time:.2f} ثانية). ")

        return summary

    def _generate_recommendations(self, performance_summary):
        """توليد توصيات بناءً على ملخص الأداء."""
        recommendations = []
        if "Needs Improvement" in performance_summary["overall_status"]:
            recommendations.append("يجب إجراء دورة تحسين شاملة لمعالجة نقاط الضعف المحددة.")
            recommendations.extend(performance_summary["areas_for_improvement"])
        else:
            recommendations.append("الأداء العام جيد. يمكن التركيز على التحسينات الدقيقة أو استكشاف قدرات جديدة.")
        return recommendations

    def get_latest_report(self):
        return self.evaluation_reports[-1] if self.evaluation_reports else None


# مثال على كيفية استخدام PerformanceEvaluator (للتوضيح فقط)
if __name__ == "__main__":
    evaluator = PerformanceEvaluator()

    # محاكاة جمع المقاييس على مدى فترة
    print("\n--- جمع المقاييس ---")
    evaluator.collect_metrics("validity_score", 0.75, context={"cycle": 1, "model": "MM_001"})
    evaluator.collect_metrics("reasoning_process_length", 120, unit="خطوة", context={"cycle": 1})
    evaluator.collect_metrics("adaptability_score", 0.6, context={"cycle": 1})
    evaluator.collect_metrics("knowledge_growth_rate", 0.05, unit="حقيقة/دورة", context={"cycle": 1})
    evaluator.collect_metrics("response_time", 1.5, unit="ثانية", context={"cycle": 1})

    time.sleep(1) # محاكاة مرور الوقت

    evaluator.collect_metrics("validity_score", 0.55, context={"cycle": 2, "model": "MM_002"})
    evaluator.collect_metrics("reasoning_process_length", 250, unit="خطوة", context={"cycle": 2})
    evaluator.collect_metrics("adaptability_score", 0.4, context={"cycle": 2})
    evaluator.collect_metrics("knowledge_growth_rate", 0.02, unit="حقيقة/دورة", context={"cycle": 2})
    evaluator.collect_metrics("response_time", 3.0, unit="ثانية", context={"cycle": 2})

    # تقييم الأداء
    print("\n--- تقييم الأداء (فترة قصيرة) ---")
    report1 = evaluator.evaluate_performance(evaluation_period_seconds=5) # تقييم آخر 5 ثوانٍ
    print("ملخص التقرير 1:", report1["performance_summary"])
    print("توصيات التقرير 1:", report1["recommendations"])

    print("\n--- تقييم الأداء (فترة أطول) ---")
    report2 = evaluator.evaluate_performance(evaluation_period_seconds=10) # تقييم آخر 10 ثوانٍ
    print("ملخص التقرير 2:", report2["performance_summary"])
    print("توصيات التقرير 2:", report2["recommendations"])

    print("\n--- أحدث تقرير ---")
    print(evaluator.get_latest_report())




