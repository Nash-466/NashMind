
import time
import uuid

class SecurityPolicy:
    """
    يمثل سياسة أمان فردية.
    """
    def __init__(self, policy_id, name, description, rules, enforcement_action, severity):
        self.policy_id = policy_id
        self.name = name
        self.description = description
        self.rules = rules  # قائمة بالقواعد (مثال: {"condition": "user_role == admin", "action": "allow"})
        self.enforcement_action = enforcement_action # (e.g., 'block', 'alert', 'quarantine')
        self.severity = severity # (e.g., 'low', 'medium', 'high')
        self.is_active = True

    def activate(self):
        self.is_active = True

    def deactivate(self):
        self.is_active = False

    def __repr__(self):
        return f"Policy(ID={self.policy_id}, Name={self.name}, Active={self.is_active}, Action={self.enforcement_action})"


class SecurityPolicyEnforcer:
    """
    منفذ سياسة الأمان.
    يطبق سياسات الأمان على أنشطة النظام ويضمن الامتثال.
    """
    def __init__(self):
        self.policies = self._load_default_policies()
        self.enforcement_log = []
        print("تم تهيئة منفذ سياسة الأمان.")

    def _load_default_policies(self):
        """
        تحميل سياسات الأمان الافتراضية.
        """
        print("  تحميل سياسات الأمان الافتراضية...")
        policies = []
        policies.append(SecurityPolicy(
            "POL_001", "Access Control for Critical Data", 
            "يمنع الوصول غير المصرح به إلى البيانات الحساسة.",
            [{"condition": "resource_type == critical_data and user_role != authorized_admin", "action": "deny"}],
            "block", "critical"
        ))
        policies.append(SecurityPolicy(
            "POL_002", "Rate Limiting for API Calls", 
            "يحد من عدد استدعاءات API لمنع إساءة الاستخدام.",
            [{"condition": "api_calls_per_minute > 100", "action": "alert"}],
            "alert", "medium"
        ))
        policies.append(SecurityPolicy(
            "POL_003", "Software Integrity Check", 
            "يتحقق من سلامة ملفات النظام الأساسية.",
            [{"condition": "file_checksum_mismatch", "action": "quarantine"}],
            "quarantine", "high"
        ))
        return policies

    def add_policy(self, policy):
        """إضافة سياسة أمان جديدة."""
        self.policies.append(policy)
        print(f"  تم إضافة سياسة جديدة: {policy.name}")

    def remove_policy(self, policy_id):
        """إزالة سياسة أمان موجودة."""
        self.policies = [p for p in self.policies if p.policy_id != policy_id]
        print(f"  تم إزالة السياسة: {policy_id}")

    def enforce_activity(self, activity_data):
        """
        تطبيق سياسات الأمان على نشاط معين.
        المدخلات: activity_data (قاموس يمثل النشاط المراد تطبيقه).
        المخرجات: قائمة بالإجراءات المتخذة.
        """
        print("\n--- تطبيق السياسات على النشاط: {} ---".format(activity_data.get("description", "نشاط غير محدد")))
        enforcement_actions_taken = []

        for policy in self.policies:
            if not policy.is_active:
                continue

            for rule in policy.rules:
                condition_met = self._evaluate_condition(rule["condition"], activity_data)
                if condition_met:
                    action = policy.enforcement_action
                    log_entry = {
                        "timestamp": time.time(),
                        "policy_id": policy.policy_id,
                        "policy_name": policy.name,
                        "activity_id": activity_data.get("id", "N/A"),
                        "action_taken": action,
                        "description": "تم تطبيق إجراء {} بسبب انتهاك السياسة {}.".format(action, policy.name)

                    }
                    self.enforcement_log.append(log_entry)
                    enforcement_actions_taken.append(log_entry)
                    print("  [PolicyEnforced] {}".format(log_entry["description"]))
                    # في نظام حقيقي، سيتم تنفيذ الإجراء الفعلي هنا (مثل حظر الوصول)
                    if action == "block":
                        print("    [BLOCK] تم حظر النشاط: {}".format(activity_data.get("description")))
                    elif action == "alert":
                        print("    [ALERT] تم إرسال تنبيه بخصوص النشاط: {}".format(activity_data.get("description")))

                    elif action == "quarantine":
                        print("    [QUARANTINE] تم عزل المورد المرتبط بالنشاط: {}".format(activity_data.get("description")))

                    break # تطبيق إجراء واحد لكل سياسة
        return enforcement_actions_taken

    def _evaluate_condition(self, condition_str, activity_data):
        """
        تقييم شرط السياسة مقابل بيانات النشاط.
        هذه دالة بسيطة لتقييم الشروط.
        """
        # مثال بسيط: تحويل الشرط إلى تعبير بايثون وتقييمه
        # يجب توخي الحذر الشديد عند استخدام eval في بيئة إنتاجية
        try:
            # استبدال المتغيرات في الشرط بقيمها من activity_data
            # يجب أن تكون أسماء المتغيرات في الشرط مطابقة للمفاتيح في activity_data
            # مثال: 



            # condition: "resource_type == 'critical_data' and user_role != 'authorized_admin'"
            # activity_data: {"resource_type": "critical_data", "user_role": "guest"}
            # eval("'critical_data' == 'critical_data' and 'guest' != 'authorized_admin'") -> True
            
            # هذا التنفيذ هو مجرد محاكاة وغير آمن للاستخدام الإنتاجي
            # في نظام حقيقي، يجب استخدام محلل تعبيرات آمن
            if "resource_type == critical_data" in condition_str and activity_data.get("resource_type") == "critical_data" and activity_data.get("user_role") != "authorized_admin":
                return True
            if "api_calls_per_minute > 100" in condition_str and activity_data.get("api_calls_per_minute", 0) > 100:
                return True
            if "file_checksum_mismatch" in condition_str and activity_data.get("file_checksum_mismatch", False):
                return True
        except Exception as e:
            print(f"  [PolicyError] خطأ في تقييم الشرط \"{condition_str}\": {e}")
        return False

    def get_enforcement_log(self):
        return self.enforcement_log


# مثال على كيفية استخدام SecurityPolicyEnforcer (للتوضيح فقط)
if __name__ == "__main__":
    spe = SecurityPolicyEnforcer()

    # محاكاة أنشطة النظام
    print("\n--- محاكاة أنشطة النظام لتطبيق السياسات ---")
    activity1 = {"id": "ACT_001", "user_role": "guest", "resource_type": "critical_data", "description": "محاولة وصول ضيف إلى بيانات حساسة."}
    spe.enforce_activity(activity1)

    activity2 = {"id": "ACT_002", "user_role": "authorized_admin", "resource_type": "critical_data", "description": "وصول مسؤول مصرح به إلى بيانات حساسة."}
    spe.enforce_activity(activity2)

    activity3 = {"id": "ACT_003", "api_calls_per_minute": 150, "description": "معدل استدعاءات API مرتفع."}
    spe.enforce_activity(activity3)

    activity4 = {"id": "ACT_004", "file_checksum_mismatch": True, "description": "تم اكتشاف عدم تطابق في المجموع الاختباري لملف النظام."}
    spe.enforce_activity(activity4)

    print("\n--- سجل تطبيق السياسات ---")
    for log_entry in spe.get_enforcement_log():
        print(log_entry)

    print("\n--- تعطيل سياسة وتكرار النشاط ---")
    spe.policies[0].deactivate() # تعطيل سياسة الوصول إلى البيانات الحساسة
    print(f"تم تعطيل السياسة: {spe.policies[0].name}")
    spe.enforce_activity(activity1) # تكرار النشاط الأول




