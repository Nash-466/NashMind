# مشروع برهان - نظام الذكاء الاصطناعي لحل مسائل ARC

## نظرة عامة

مشروع برهان هو نظام ذكاء اصطناعي متقدم مصمم لحل مسائل ARC (Abstraction and Reasoning Corpus). يتكون النظام من عدة مكونات متكاملة تعمل معاً لتحليل وحل المسائل المعقدة.

## الميزات الرئيسية

### 🧠 المكونات الأساسية
- **نظام التحليل الشامل**: تحليل الأنماط والهياكل المعقدة
- **محرك الاستراتيجيات المتقدم**: تطبيق استراتيجيات متنوعة لحل المسائل
- **نظام التعلم التكيفي**: تحسين الأداء بناءً على النتائج السابقة
- **محرك المحاكاة السببية**: فهم العلاقات السببية في البيانات

### 🔧 التحسينات الجديدة
- **إدارة التبعيات الذكية**: نظام آمن للتعامل مع المكتبات المفقودة
- **إدارة الأخطاء الموحدة**: نظام شامل لمعالجة الأخطاء والاستثناءات
- **تحسين الأداء والذاكرة**: مراقبة وتحسين استخدام الموارد
- **واجهات موحدة**: تصميم متسق عبر جميع المكونات

## متطلبات النظام

### المتطلبات الأساسية
```
Python >= 3.8
numpy >= 1.21.0
```

### المتطلبات الاختيارية
```
pandas >= 1.3.0          # لمعالجة البيانات المتقدمة
scikit-learn >= 1.0.0    # للتعلم الآلي
scipy >= 1.7.0           # للحوسبة العلمية
torch >= 1.9.0           # للتعلم العميق
scikit-image >= 0.18.0   # لمعالجة الصور
networkx >= 2.6.0        # لمعالجة الرسوم البيانية
optuna >= 2.10.0         # لتحسين المعاملات
```

## التثبيت

### 1. تثبيت المتطلبات الأساسية
```bash
pip install numpy
```

### 2. تثبيت المتطلبات الاختيارية (موصى به)
```bash
pip install -r requirements.txt
```

### 3. التحقق من التثبيت
```bash
python integration_tests.py
```

## الاستخدام

### الاستخدام الأساسي
```bash
python main.py -t arc-agi_training_challenges.json -o submission.json
```

### الخيارات المتقدمة
```bash
# تشغيل في الوضع العميق مع MetaBrain
python main.py -t tasks.json --mode deep --meta --kb-path _kb/meta_kb.json

# تشغيل مع تحسين دوري
python main.py -t tasks.json --optimize-every 10

# تشغيل مع تقييم
python main.py --kaggle-data /path/to/data --kaggle-split eval --evaluate
```

### خيارات سطر الأوامر

| الخيار | الوصف |
|--------|--------|
| `-t, --tasks` | مسار ملف المسائل JSON |
| `-o, --output` | مسار حفظ النتائج (افتراضي: submission.json) |
| `--mode` | وضع التشغيل: fast أو deep |
| `--meta` | تفعيل طبقة MetaBrain |
| `--kaggle-data` | مسار بيانات Kaggle |
| `--evaluate` | تشغيل التقييم |
| `--smoke-import` | اختبار استيراد جميع الوحدات |

## بنية المشروع

```
مشروع برهان/
├── main.py                          # نقطة الدخول الرئيسية
├── requirements.txt                 # متطلبات المشروع
├── README.md                       # هذا الملف
│
├── arc_complete_agent_part1.py     # المحرك الأساسي والحوسبة
├── arc_complete_agent_part2.py     # تحليل الأنماط الشامل
├── arc_complete_agent_part3.py     # إدارة الاستراتيجيات
├── arc_complete_agent_part4.py     # الوكيل الذكي الرئيسي
├── arc_complete_agent_part5.py     # نظام MuZero والتعلم المعزز
├── arc_complete_agent_part6.py     # تحسين المعاملات
├── arc_ultimate_mind_part7.py      # المنسق الرئيسي
│
├── burhan_meta_brain.py            # نظام MetaBrain المتقدم
├── evaluation.py                   # نظام التقييم
├── kaggle_io.py                   # إدارة بيانات Kaggle
│
├── dependency_manager.py           # إدارة التبعيات الذكية
├── error_manager.py               # إدارة الأخطاء الموحدة
├── unified_interfaces.py          # الواجهات الموحدة
├── performance_optimizer.py       # تحسين الأداء
├── integration_tests.py           # اختبارات التكامل
│
└── _kb/                           # قاعدة المعرفة
    └── meta_kb.json
```

## المكونات الرئيسية

### 1. النظام الأساسي (main.py)
- نقطة الدخول الرئيسية
- معالجة الحجج وإعداد النظام
- تنسيق تشغيل جميع المكونات

### 2. محرك التحليل (arc_complete_agent_part2.py)
- تحليل الأنماط الهندسية والمكانية
- استخراج الميزات المتقدمة
- تحليل التماثل والتناسق

### 3. إدارة الاستراتيجيات (arc_complete_agent_part3.py)
- تطبيق التحويلات الأساسية
- تركيب الاستراتيجيات المعقدة
- تقييم فعالية الاستراتيجيات

### 4. نظام MetaBrain (burhan_meta_brain.py)
- سوق الفرضيات الذكي
- تركيب الاستراتيجيات التلقائي
- قاعدة المعرفة التكيفية

### 5. أنظمة الدعم الجديدة
- **dependency_manager.py**: إدارة آمنة للتبعيات
- **error_manager.py**: معالجة شاملة للأخطاء
- **performance_optimizer.py**: مراقبة وتحسين الأداء
- **unified_interfaces.py**: واجهات متسقة

## الاختبار

### تشغيل اختبارات التكامل
```bash
python integration_tests.py
```

### اختبار استيراد الوحدات
```bash
python main.py --smoke-import -t dummy.json
```

### اختبار الأداء
```bash
python -c "from performance_optimizer import performance_monitor; print(performance_monitor.get_system_status())"
```

## استكشاف الأخطاء

### مشاكل شائعة وحلولها

#### 1. خطأ في استيراد المكتبات
```
ModuleNotFoundError: No module named 'pandas'
```
**الحل**: تثبيت المتطلبات الاختيارية
```bash
pip install pandas
```

#### 2. مشاكل الذاكرة
```
MemoryError
```
**الحل**: تفعيل تحسين الذاكرة
```python
from performance_optimizer import performance_monitor
performance_monitor.memory_manager.cleanup()
```

#### 3. أخطاء في معالجة الملفات
```
FileNotFoundError
```
**الحل**: التأكد من وجود ملفات البيانات
```bash
ls arc-agi_training_challenges.json
```

### سجلات النظام
يمكن تفعيل السجلات التفصيلية:
```bash
python main.py -t tasks.json --log-level DEBUG
```

## المساهمة

### إضافة مكونات جديدة
1. وراثة من الفئات الأساسية في `unified_interfaces.py`
2. استخدام نظام إدارة الأخطاء
3. تطبيق تحسينات الأداء
4. إضافة اختبارات التكامل

### مثال على مكون جديد
```python
from unified_interfaces import BaseAnalyzer, AnalysisResult
from error_manager import safe_execute, ErrorSeverity

class MyAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__("MyAnalyzer")
    
    @safe_execute("MyAnalyzer", "analyze", ErrorSeverity.MEDIUM)
    def analyze(self, task_data):
        # تنفيذ التحليل
        return AnalysisResult(...)
```

## الترخيص

هذا المشروع مفتوح المصدر ومتاح للاستخدام والتطوير.

## الدعم

للحصول على الدعم أو الإبلاغ عن مشاكل، يرجى:
1. تشغيل اختبارات التكامل أولاً
2. فحص سجلات النظام
3. التأكد من تثبيت المتطلبات بشكل صحيح

## إحصائيات المشروع

- **إجمالي الملفات**: 20+ ملف Python
- **إجمالي الأسطر**: 15,000+ سطر من الكود
- **المكونات الرئيسية**: 8 مكونات أساسية
- **أنظمة الدعم**: 5 أنظمة دعم متقدمة
- **الاختبارات**: نظام اختبارات تكامل شامل

## التحديثات الأخيرة

### الإصدار الحالي
- ✅ إصلاح مشاكل BOM في جميع الملفات
- ✅ نظام إدارة التبعيات الذكي
- ✅ نظام إدارة الأخطاء الموحد
- ✅ واجهات برمجة موحدة
- ✅ تحسين الأداء والذاكرة
- ✅ اختبارات التكامل الشاملة
- ✅ توثيق كامل للنظام
\n## Benchmark Results (100 tasks)
\n### Training (Top 5)
\n### Evaluation (Top 5)
\n## Benchmark Results (100 tasks)
\n### Training (Top 5)
- perfect_arc_system_v2: exact=1/100 sim=64.4%
- enhanced_efficient_zero: exact=0/100 sim=27.5%
- deep_learning_arc_system: exact=0/100 sim=27.5%
- arc_learning_solver: exact=0/100 sim=27.5%
- neural_pattern_learner: exact=0/100 sim=27.5%
\n### Evaluation (Top 5)
- perfect_arc_system_v2: exact=0/100 sim=52.2%
- ultimate_generalized_arc_system: exact=0/100 sim=9.1%
- neural_pattern_learner: exact=0/100 sim=7.6%
- arc_learning_solver: exact=0/100 sim=7.6%
- deep_learning_arc_system: exact=0/100 sim=7.6%
