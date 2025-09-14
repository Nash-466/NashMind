
import unittest
import sys
import os
import time

# إضافة المسار الجذر للمشروع إلى sys.path للسماح بالاستيراد
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aces_system.aces_system import ACES
from aces_system.core.artificial_mentality_simulator import ArtificialMentalitySimulator, MentalModel
from aces_system.core.cognitive_architecture_developer import CognitiveArchitectureDeveloper, CognitiveArchitecture
from aces_system.core.existential_learning_system import ExistentialLearningSystem, Experience
from aces_system.core.intuitive_understanding_generator import IntuitiveUnderstandingGenerator, Insight
from aces_system.interfaces.communication_manager import CommunicationManager
from aces_system.interfaces.user_interface_manager import UserInterfaceManager
from aces_system.self_improvement.performance_evaluator import PerformanceEvaluator
from aces_system.self_improvement.self_optimization_engine import SelfOptimizationEngine
from aces_system.security.threat_detection_module import ThreatDetectionModule
from aces_system.security.security_policy_enforcer import SecurityPolicyEnforcer
from aces_system.security.security_manager import SecurityManager

class TestACESSystem(unittest.TestCase):

    def setUp(self):
        """إعداد بيئة الاختبار قبل كل اختبار."""
        self.aces = ACES()
        # إعادة تعيين بعض الحالات لضمان اختبارات نظيفة
        self.aces.mentality_simulator.mental_models_library = {}
        self.aces.mentality_simulator.reasoning_paths = []
        self.aces.architecture_developer.developed_architectures = []
        self.aces.existential_learner.existential_memory.experiences = []
        self.aces.existential_learner.existential_memory.self_models = []
        self.aces.existential_learner.existential_memory.purposes = []
        self.aces.intuitive_generator.integrated_understandings = []
        self.aces.performance_evaluator.metrics_history = []
        self.aces.performance_evaluator.evaluation_reports = []
        self.aces.self_optimization_engine.optimization_history = []
        self.aces.security_manager.threat_detector.detected_threats = []
        self.aces.security_manager.policy_enforcer.enforcement_log = []
        # مسح قائمة انتظار الرسائل لـ CommunicationManager
        self.aces.communication_manager.internal_bus.message_queue = []
        self.aces.communication_manager.http_protocol.inbox = []
        self.aces.communication_manager.http_protocol.outbox = []
        self.aces.communication_manager.websocket_protocol.message_queue = []
        self.aces.communication_manager.websocket_protocol.connected_clients = {}


    def test_initialization(self):
        """اختبار تهيئة نظام ACES."""
        self.assertIsInstance(self.aces.mentality_simulator, ArtificialMentalitySimulator)
        self.assertIsInstance(self.aces.architecture_developer, CognitiveArchitectureDeveloper)
        self.assertIsInstance(self.aces.existential_learner, ExistentialLearningSystem)
        self.assertIsInstance(self.aces.intuitive_generator, IntuitiveUnderstandingGenerator)
        self.assertIsInstance(self.aces.communication_manager, CommunicationManager)
        self.assertIsInstance(self.aces.user_interface_manager, UserInterfaceManager)
        self.assertIsInstance(self.aces.performance_evaluator, PerformanceEvaluator)
        self.assertIsInstance(self.aces.self_optimization_engine, SelfOptimizationEngine)
        self.assertIsInstance(self.aces.security_manager, SecurityManager)
        self.assertIn("architecture_flexibility", self.aces.current_system_state)

    def test_communication_manager(self):
        """اختبار وظائف مدير الاتصالات."""
        comm_manager = self.aces.communication_manager
        
        # اختبار إرسال واستقبال رسائل HTTP
        comm_manager.http_protocol.simulate_incoming_message("TestSender", "Hello HTTP")
        received_messages = comm_manager.receive_external_messages()
        self.assertEqual(len(received_messages), 1)
        self.assertEqual(received_messages[0]["payload"], "Hello HTTP")

        send_result = comm_manager.send_external_message("http", "TestRecipient", "Reply HTTP")
        self.assertEqual(send_result["status"], "sent")
        self.assertEqual(len(comm_manager.http_protocol.outbox), 1)

        # اختبار ناقل الرسائل الداخلي
        test_callback_triggered = False
        def test_callback(message):
            nonlocal test_callback_triggered
            test_callback_triggered = True
            self.assertEqual(message["payload"], "Internal Test Message")
        
        comm_manager.subscribe_internal_messages("TestComponent", test_callback)
        comm_manager.publish_internal_message("test_topic", "Internal Test Message", "TestSender")
        comm_manager.process_internal_messages()
        self.assertTrue(test_callback_triggered)

    def test_user_interface_manager(self):
        """اختبار وظائف مدير واجهة المستخدم."""
        ui_manager = self.aces.user_interface_manager
        comm_manager = self.aces.communication_manager

        # اختبار معالجة المدخلات النصية
        raw_input = {"type": "text", "content": "Test command"}
        response = ui_manager.process_user_interaction(raw_input)
        self.assertIn("Test command", response["content"])
        
        # التأكد من أن الأمر تم نشره داخليًا
        comm_manager.process_internal_messages() # لمعالجة user_command
        self.assertGreater(len(comm_manager.internal_bus.message_queue), 0)
        # التأكد من أن استجابة النظام تم نشرها للعرض
        comm_manager.process_internal_messages() # لمعالجة UserInterfaceManager_Output
        self.assertEqual(len(comm_manager.internal_bus.message_queue), 0) # يجب أن تكون فارغة بعد المعالجة

        # اختبار المدخلات غير المدعومة
        raw_input_unsupported = {"type": "unsupported", "content": "Invalid"}
        response_unsupported = ui_manager.process_user_interaction(raw_input_unsupported)
        self.assertIn("غير مدعوم حاليًا", response_unsupported["content"])

    def test_performance_evaluator(self):
        """اختبار وظائف مقيم الأداء."""
        evaluator = self.aces.performance_evaluator

        evaluator.collect_metrics("validity_score", 0.8)
        evaluator.collect_metrics("reasoning_process_length", 100)
        evaluator.collect_metrics("adaptability_score", 0.7)
        evaluator.collect_metrics("knowledge_growth_rate", 0.05)
        evaluator.collect_metrics("response_time", 1.2)

        report = evaluator.evaluate_performance(evaluation_period_seconds=5)
        self.assertEqual(report["status"], "completed")
        self.assertIn("Good", report["performance_summary"]["overall_status"])
        self.assertGreater(len(evaluator.metrics_history), 0)
        self.assertGreater(len(evaluator.evaluation_reports), 0)

    def test_self_optimization_engine(self):
        """اختبار وظائف محرك التحسين الذاتي."""
        optimizer = self.aces.self_optimization_engine
        evaluator = self.aces.performance_evaluator

        # محاكاة تقرير أداء سيء
        evaluator.collect_metrics("validity_score", 0.4)
        evaluator.collect_metrics("reasoning_process_length", 300)
        evaluator.collect_metrics("adaptability_score", 0.3)
        evaluator.collect_metrics("knowledge_growth_rate", 0.005)
        evaluator.collect_metrics("response_time", 4.0)
        bad_report = evaluator.evaluate_performance(evaluation_period_seconds=5)

        optimization_results = optimizer.analyze_and_optimize(bad_report, self.aces)
        self.assertEqual(optimization_results["status"], "optimized")
        self.assertGreater(len(optimizer.optimization_history), 0)
        self.assertIn("parameter_tuning", str(optimizer.optimization_history[0]["actions"]))

    def test_threat_detection_module(self):
        """اختبار وظائف وحدة كشف التهديدات."""
        tdm = self.aces.security_manager.threat_detector

        activity1 = {"event_type": "user_login", "user": "admin", "description": "Normal login."}
        detected = tdm.monitor_system_activity(activity1)
        self.assertEqual(len(detected), 0) # لا يوجد تهديد

        activity2 = {"event_type": "unusual_login_attempts", "user": "unknown", "description": "Brute force."}
        detected = tdm.monitor_system_activity(activity2)
        self.assertGreater(len(detected), 0) # يجب اكتشاف تهديد
        self.assertEqual(tdm.get_detected_threats()[0].threat_type, "intrusion_attempt")

    def test_security_policy_enforcer(self):
        """اختبار وظائف منفذ سياسة الأمان."""
        spe = self.aces.security_manager.policy_enforcer

        activity1 = {"id": "ACT_001", "user_role": "guest", "resource_type": "critical_data", "description": "Access critical data."}
        actions = spe.enforce_activity(activity1)
        self.assertGreater(len(actions), 0)
        self.assertEqual(actions[0]["action_taken"], "block")

        activity2 = {"id": "ACT_002", "user_role": "authorized_admin", "resource_type": "critical_data", "description": "Admin access critical data."}
        actions = spe.enforce_activity(activity2)
        self.assertEqual(len(actions), 0) # لا يوجد انتهاك

    def test_security_manager(self):
        """اختبار وظائف مدير الأمان."""
        sec_manager = self.aces.security_manager
        comm_manager = self.aces.communication_manager

        # محاكاة طلب مراقبة نشاط
        activity_log = {"event_type": "unusual_login_attempts", "user": "hacker", "description": "Suspicious login."}
        comm_manager.publish_internal_message("SecurityManager_MonitorActivity", {"activity_log": activity_log}, "Test")
        comm_manager.process_internal_messages()
        self.assertGreater(len(sec_manager.threat_detector.get_detected_threats()), 0)

        # محاكاة طلب فحص أمني
        comm_manager.publish_internal_message("security_scan_request", {}, "Test")
        comm_manager.process_internal_messages()
        # يمكن التحقق من سجلات الفحص أو التهديدات المكتشفة هنا
        self.assertIn("اكتمل الفحص الأمني.", comm_manager.internal_bus.message_queue[0]["payload"]["data"]["response"])

    def test_evolution_cycle_integration(self):
        """اختبار دورة التطور المتكاملة."""
        initial_problem = {
            "goals": ["حل مشكلة X"],
            "targets": ["هدف 1"],
            "complexity_level": "متوسط",
            "main_topic": "اختبار التكامل",
            "keywords": ["تكامل", "اختبار"]
        }
        results = self.aces.run_evolution_cycle(initial_problem)
        self.assertIsNotNone(results)
        self.assertIn("best_mental_model_info", results)
        self.assertIn("new_architecture", results)
        self.assertIn("existential_learning_results", results)
        self.assertIn("intuitive_understanding", results)
        self.assertIn("performance_report", results)
        self.assertIn("optimization_results", results)

        # التحقق من تحديث حالة النظام
        self.assertGreater(self.aces.current_system_state["overall_performance_score"], 0.0)
        self.assertGreater(self.aces.current_system_state["knowledge_base_size"], 0)

    def test_user_input_to_evolution_cycle(self):
        """اختبار بدء دورة تطور من أمر المستخدم."""
        initial_knowledge_size = len(self.aces.system_knowledge_base.facts)
        initial_performance_score = self.aces.current_system_state["overall_performance_score"]

        user_input = {"type": "text", "content": "تشغيل دورة تطور"}
        response = self.aces.process_user_input(user_input)
        self.assertIn("بدء دورة تطور جديدة", response["content"])

        # معالجة الرسائل الداخلية التي تطلق دورة التطور
        self.aces.communication_manager.process_internal_messages()
        self.aces.communication_manager.process_internal_messages()
        self.aces.communication_manager.process_internal_messages()
        self.aces.communication_manager.process_internal_messages()
        self.aces.communication_manager.process_internal_messages()
        self.aces.communication_manager.process_internal_messages()

        # التحقق من أن دورة التطور قد أثرت على حالة النظام
        self.assertGreater(len(self.aces.system_knowledge_base.facts), initial_knowledge_size)
        self.assertGreater(self.aces.current_system_state["overall_performance_score"], initial_performance_score)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)



