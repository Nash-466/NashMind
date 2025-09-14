
import time
import uuid
import random

class Threat:
    """
    يمثل تهديدًا محتملاً تم اكتشافه.
    """
    def __init__(self, threat_id, threat_type, severity, description, source, timestamp):
        self.threat_id = threat_id
        self.threat_type = threat_type  # (e.g., 'malware', 'intrusion_attempt', 'data_exfiltration')
        self.severity = severity      # (e.g., 'low', 'medium', 'high', 'critical')
        self.description = description
        self.source = source          # (e.g., 'network_log', 'system_activity', 'external_intelligence')
        self.timestamp = timestamp
        self.status = "detected"      # (e.g., 'detected', 'analyzed', 'mitigated', 'false_positive')

    def __repr__(self):
        return f"Threat(ID={self.threat_id}, Type={self.threat_type}, Severity={self.severity}, Status={self.status})"


class ThreatDetectionModule:
    """
    وحدة كشف التهديدات.
    تراقب أنشطة النظام وتحدد التهديدات المحتملة.
    """
    def __init__(self):
        self.threat_signatures = self._load_threat_signatures()
        self.detected_threats = []
        self.anomaly_threshold = 0.8 # عتبة الكشف عن الشذوذ
        print("تم تهيئة وحدة كشف التهديدات.")

    def _load_threat_signatures(self):
        """
        تحميل توقيعات التهديدات المعروفة.
        في نظام حقيقي، سيتم تحميلها من قاعدة بيانات أو خدمة استخبارات التهديدات.
        """
        print("  تحميل توقيعات التهديدات...")
        return [
            {"pattern": "unusual_login_attempts", "type": "intrusion_attempt", "severity": "high"},
            {"pattern": "high_data_transfer_outbound", "type": "data_exfiltration", "severity": "critical"},
            {"pattern": "unauthorized_file_access", "type": "privilege_escalation", "severity": "high"},
            {"pattern": "suspicious_process_creation", "type": "malware", "severity": "medium"},
            {"pattern": "unexpected_api_calls", "type": "api_abuse", "severity": "medium"}
        ]

    def monitor_system_activity(self, activity_log):
        """
        مراقبة سجلات نشاط النظام للكشف عن التهديدات.
        المدخلات: activity_log (قاموس يمثل سجل النشاط).
        المخرجات: قائمة بالتهديدات المكتشفة.
        """
        print("\n--- مراقبة نشاط النظام: {} ---".format(activity_log.get("description", "نشاط غير محدد")))
        newly_detected_threats = []
        
        # الكشف القائم على التوقيع
        for signature in self.threat_signatures:
            if signature["pattern"] in activity_log.get("event_type", "") or \
               signature["pattern"] in activity_log.get("description", ""):
                threat_id = str(uuid.uuid4())
                threat = Threat(threat_id, signature["type"], signature["severity"], 
                                "تم الكشف عن تهديد مطابق للتوقيع: {}".format(signature["pattern"]), 
                                activity_log.get("source", "system_log"), time.time())
                newly_detected_threats.append(threat)
                self.detected_threats.append(threat)
                print(f"  [ThreatDetected] {threat}")

        # الكشف القائم على الشذوذ (مثال مبسط)
        anomaly_score = self._calculate_anomaly_score(activity_log)
        if anomaly_score > self.anomaly_threshold:
            threat_id = str(uuid.uuid4())
            threat = Threat(threat_id, "anomaly_detection", "medium", 
                            f"تم الكشف عن نشاط شاذ بدرجة {anomaly_score:.2f}.", 
                            activity_log.get("source", "system_log"), time.time())
            newly_detected_threats.append(threat)
            self.detected_threats.append(threat)
            print(f"  [AnomalyDetected] {threat}")

        return newly_detected_threats

    def _calculate_anomaly_score(self, activity_log):
        """
        حساب درجة الشذوذ لنشاط معين.
        في نظام حقيقي، سيتضمن ذلك نماذج تعلم آلة معقدة.
        """
        score = 0.0
        if activity_log.get("user") == "unknown":
            score += 0.3
        if activity_log.get("action") == "delete_critical_file":
            score += 0.5
        if activity_log.get("data_volume", 0) > 1000000: # أكثر من 1 ميجابايت
            score += 0.4
        
        # إضافة بعض العشوائية لمحاكاة التعقيد
        score += random.uniform(0, 0.2)
        return min(1.0, score) # ضمان أن تكون النتيجة بين 0 و 1

    def get_detected_threats(self, status=None):
        if status:
            return [t for t in self.detected_threats if t.status == status]
        return self.detected_threats

    def update_threat_status(self, threat_id, new_status):
        for threat in self.detected_threats:
            if threat.threat_id == threat_id:
                threat.status = new_status
                print(f"  [ThreatUpdate] تم تحديث حالة التهديد {threat_id} إلى {new_status}.")
                return True
        return False


# مثال على كيفية استخدام ThreatDetectionModule (للتوضيح فقط)
if __name__ == "__main__":
    tdm = ThreatDetectionModule()

    # محاكاة أنشطة النظام
    print("\n--- محاكاة أنشطة النظام ---")
    activity1 = {"event_type": "user_login", "user": "admin", "action": "login", "source": "web_app", "description": "محاولة تسجيل دخول ناجحة."}
    tdm.monitor_system_activity(activity1)

    activity2 = {"event_type": "unusual_login_attempts", "user": "unknown", "action": "login", "source": "external_ip", "description": "100 محاولة تسجيل دخول فاشلة من IP غريب."}
    tdm.monitor_system_activity(activity2)

    activity3 = {"event_type": "file_access", "user": "guest", "action": "read_file", "file": "critical_data.txt", "source": "local_system", "description": "وصول مستخدم ضيف إلى ملف حساس."}
    tdm.monitor_system_activity(activity3)

    activity4 = {"event_type": "data_transfer", "user": "system", "action": "upload", "data_volume": 2000000, "source": "internal_server", "description": "نقل كمية كبيرة من البيانات إلى خارج الشبكة."}
    tdm.monitor_system_activity(activity4)

    activity5 = {"event_type": "normal_operation", "user": "dev", "action": "compile_code", "source": "dev_env", "description": "عملية تطوير روتينية."}
    tdm.monitor_system_activity(activity5)

    print("\n--- التهديدات المكتشفة ---")
    for threat in tdm.get_detected_threats():
        print(threat)

    # تحديث حالة تهديد
    if tdm.get_detected_threats():
        first_threat_id = tdm.get_detected_threats()[0].threat_id
        tdm.update_threat_status(first_threat_id, "analyzed")
        print("\nالتهديدات بعد التحديث:")
        for threat in tdm.get_detected_threats():
            print(threat)

    print("\nالتهديدات ذات الحالة \"detected\":")
    for threat in tdm.get_detected_threats(status="detected"):
        print(threat)




