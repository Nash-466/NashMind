



from .threat_detection_module import ThreatDetectionModule
from .security_policy_enforcer import SecurityPolicyEnforcer
import time

class SecurityManager:
    """
    مدير الأمان.
    يدمج جميع مكونات الأمان ويوفر واجهة موحدة لإدارة أمان النظام.
    """
    def __init__(self, communication_manager):
        self.threat_detector = ThreatDetectionModule()
        self.policy_enforcer = SecurityPolicyEnforcer()
        self.communication_manager = communication_manager
        self._setup_internal_subscriptions()
        print("تم تهيئة مدير الأمان.")

    def _setup_internal_subscriptions(self):
        """إعداد الاشتراكات الداخلية لمعالجة الرسائل من المكونات الأخرى."""
        self.communication_manager.subscribe_internal_messages(
            "SecurityManager_MonitorActivity", self._handle_monitor_activity_request)
        self.communication_manager.subscribe_internal_messages(
            "SecurityManager_EnforcePolicy", self._handle_enforce_policy_request)
        print("  مدير الأمان مشترك في ناقل الرسائل الداخلي.")

    def _handle_monitor_activity_request(self, message):
        """معالجة طلبات مراقبة النشاط الواردة داخليًا."""
        activity_log = message["payload"].get("activity_log")
        if activity_log:
            print("  [SecurityManager] تلقى طلب مراقبة نشاط: {}".format(activity_log.get("description")))
            detected_threats = self.threat_detector.monitor_system_activity(activity_log)
            if detected_threats:
                for threat in detected_threats:
                    self.communication_manager.publish_internal_message(
                        "threat_detected", {"threat": threat.__dict__}, "SecurityManager")

    def _handle_enforce_policy_request(self, message):
        """معالجة طلبات تطبيق السياسات الواردة داخليًا."""
        activity_data = message["payload"].get("activity_data")
        if activity_data:
            print("  [SecurityManager] تلقى طلب تطبيق سياسة على نشاط: {}".format(activity_data.get("description")))

            self.policy_enforcer.enforce_activity(activity_data)

    def run_security_scan(self):
        """
        تشغيل فحص أمني شامل للنظام.
        """
        print("\n==================================================")
        print("بدء فحص أمني شامل...")
        print("==================================================")
        
        # مثال: فحص سلامة الملفات (محاكاة)
        print("  فحص سلامة الملفات...")
        # في نظام حقيقي، سيتم حساب المجموع الاختباري للملفات ومقارنته بالقيم المخزنة
        # هنا، سنقوم بمحاكاة اكتشاف عدم تطابق
        if time.time() % 10 < 2: # 20% فرصة لاكتشاف عدم تطابق
            activity_data = {"id": f"SCAN_{time.time()}", "file_checksum_mismatch": True, "description": "تم اكتشاف عدم تطابق في المجموع الاختباري لملف النظام أثناء الفحص."}
            self.policy_enforcer.enforce_activity(activity_data)
        else:
            print("  لم يتم العثور على مشاكل في سلامة الملفات.")

        # مثال: فحص التهديدات المكتشفة التي لم يتم التعامل معها
        print("\n  فحص التهديدات المكتشفة...")
        unhandled_threats = self.threat_detector.get_detected_threats(status="detected")
        if unhandled_threats:
            print("  تم العثور على {} تهديدًا لم يتم التعامل معه:".format(len(unhandled_threats)))
            for threat in unhandled_threats:
                print("    - {}".format(threat))
                # يمكن اتخاذ إجراءات تلقائية هنا، مثل رفع مستوى الخطورة أو إرسال تنبيه
                self.communication_manager.publish_internal_message(
                    "unhandled_threat_alert", {"threat": threat.__dict__}, "SecurityManager")
        else:
            print("  لا توجد تهديدات مكتشفة لم يتم التعامل معها.")

        print("\n==================================================")
        print("اكتمل الفحص الأمني الشامل.")
        print("==================================================")

    def get_security_status(self):
        return {
            "total_threats_detected": len(self.threat_detector.detected_threats),
            "active_policies": len([p for p in self.policy_enforcer.policies if p.is_active]),
            "enforcement_log_size": len(self.policy_enforcer.enforcement_log)
        }


# مثال على كيفية استخدام SecurityManager (للتوضيح فقط)
if __name__ == "__main__":
    from aces_system.interfaces.communication_manager import CommunicationManager
    comm_manager = CommunicationManager()
    security_manager = SecurityManager(comm_manager)

    # محاكاة نشر رسائل داخلية لمراقبة الأنشطة
    print("\n--- محاكاة نشر رسائل لمراقبة الأنشطة ---")
    activity1 = {"event_type": "unusual_login_attempts", "user": "unknown", "action": "login", "source": "external_ip", "description": "100 محاولة تسجيل دخول فاشلة من IP غريب."}
    comm_manager.publish_internal_message("SecurityManager_MonitorActivity", {"activity_log": activity1}, "ACES_Main")

    activity2 = {"id": "ACT_001", "user_role": "guest", "resource_type": "critical_data", "description": "محاولة وصول ضيف إلى بيانات حساسة."}
    comm_manager.publish_internal_message("SecurityManager_EnforcePolicy", {"activity_data": activity2}, "ACES_Main")

    # معالجة الرسائل الداخلية
    print("\n--- معالجة الرسائل الداخلية ---")
    comm_manager.process_messages()

    # تشغيل فحص أمني
    security_manager.run_security_scan()

    print("\n--- حالة الأمان ---")
    print(security_manager.get_security_status())




