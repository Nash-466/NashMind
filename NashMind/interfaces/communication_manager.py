
import json
import time
import uuid

class CommunicationProtocol:
    """
    فئة أساسية لبروتوكولات الاتصال المختلفة.
    """
    def __init__(self, protocol_name, version):
        self.protocol_name = protocol_name
        self.version = version

    def send_message(self, recipient, message_content):
        raise NotImplementedError("يجب على الفئات الفرعية تنفيذ هذه الوظيفة.")

    def receive_message(self):
        raise NotImplementedError("يجب على الفئات الفرعية تنفيذ هذه الوظيفة.")

    def __repr__(self):
        return f"Protocol({self.protocol_name} v{self.version})"


class HTTPProtocol(CommunicationProtocol):
    """
    بروتوكول HTTP للاتصال (محاكاة).
    """
    def __init__(self):
        super().__init__("HTTP", "1.1")
        self.inbox = []
        self.outbox = []

    def send_message(self, recipient, message_content):
        message = {
            "id": str(uuid.uuid4()),
            "sender": "ACES_System",
            "recipient": recipient,
            "type": "HTTP_REQUEST",
            "payload": message_content,
            "timestamp": time.time()
        }
        self.outbox.append(message)
        print(f"  [HTTP] إرسال رسالة إلى {recipient}: {message_content[:50]}...")
        return {"status": "sent", "message_id": message["id"]}

    def receive_message(self):
        if self.inbox:
            message = self.inbox.pop(0)
            print("  [HTTP] استلام رسالة: {}...".format(message["payload"][:50]))
            return message
        return None

    def simulate_incoming_message(self, sender, payload):
        message = {
            "id": str(uuid.uuid4()),
            "sender": sender,
            "recipient": "ACES_System",
            "type": "HTTP_RESPONSE",
            "payload": payload,
            "timestamp": time.time()
        }
        self.inbox.append(message)


class WebSocketProtocol(CommunicationProtocol):
    """
    بروتوكول WebSocket للاتصال (محاكاة).
    """
    def __init__(self):
        super().__init__("WebSocket", "1.0")
        self.connected_clients = {}
        self.message_queue = []

    def connect_client(self, client_id):
        if client_id not in self.connected_clients:
            self.connected_clients[client_id] = {"status": "connected", "last_ping": time.time()}
            print(f"  [WS] تم توصيل العميل: {client_id}")
            return True
        return False

    def disconnect_client(self, client_id):
        if client_id in self.connected_clients:
            del self.connected_clients[client_id]
            print(f"  [WS] تم قطع اتصال العميل: {client_id}")
            return True
        return False

    def send_message(self, recipient, message_content):
        if recipient in self.connected_clients:
            message = {
                "id": str(uuid.uuid4()),
                "sender": "ACES_System",
                "recipient": recipient,
                "type": "WS_MESSAGE",
                "payload": message_content,
                "timestamp": time.time()
            }
            self.message_queue.append(message) # في الواقع، سيتم إرسالها فورًا
            print(f"  [WS] إرسال رسالة إلى {recipient}: {message_content[:50]}...")
            return {"status": "sent", "message_id": message["id"]}
        print(f"  [WS] العميل {recipient} غير متصل.")
        return {"status": "failed", "reason": "client_not_connected"}

    def receive_message(self):
        if self.message_queue:
            message = self.message_queue.pop(0) # محاكاة الاستلام من قائمة الانتظار
            print("  [WS] استلام رسالة: {}...".format(message["payload"][:50]))
            return message
        return None

    def simulate_incoming_message(self, sender, payload):
        if sender in self.connected_clients:
            message = {
                "id": str(uuid.uuid4()),
                "sender": sender,
                "recipient": "ACES_System",
                "type": "WS_MESSAGE",
                "payload": payload,
                "timestamp": time.time()
            }
            self.message_queue.append(message)
            return True
        return False


class InternalMessageBus:
    """
    نظام ناقل رسائل داخلي للتواصل بين مكونات ACES.
    """
    def __init__(self):
        self.subscribers = {}
        self.message_queue = []

    def subscribe(self, component_id, callback_function):
        if component_id not in self.subscribers:
            self.subscribers[component_id] = []
        self.subscribers[component_id].append(callback_function)
        print(f"  [BUS] {component_id} مشترك في ناقل الرسائل الداخلي.")

    def publish(self, topic, message_content, sender_id="System"):
        message = {
            "id": str(uuid.uuid4()),
            "sender": sender_id,
            "topic": topic,
            "payload": message_content,
            "timestamp": time.time()
        }
        self.message_queue.append(message)
        print(f"  [BUS] نشر رسالة على الموضوع \'{topic}\' من {sender_id}: {str(message_content)[:50]}...")

    def process_messages(self):
        processed_count = 0
        new_queue = []
        for message in self.message_queue:
            topic = message["topic"]
            handled = False
            for component_id, callbacks in self.subscribers.items():
                # في نظام حقيقي، قد يكون هناك منطق توجيه أكثر تعقيدًا بناءً على الموضوع
                # هنا، سنفترض أن جميع المشتركين يعالجون جميع الرسائل (لأغراض المحاكاة)
                for callback in callbacks:
                    try:
                        callback(message) # تمرير الرسالة كاملة إلى وظيفة رد الاتصال
                        handled = True
                    except Exception as e:
                        print(f"  [BUS ERROR] خطأ في معالجة الرسالة بواسطة {component_id}: {e}")
            if handled:
                processed_count += 1
            else:
                new_queue.append(message) # إعادة الرسائل غير المعالجة إلى قائمة الانتظار
        self.message_queue = new_queue
        return processed_count


class CommunicationManager:
    """
    يدير جميع قنوات الاتصال الخارجية والداخلية للنظام.
    """
    def __init__(self):
        self.http_protocol = HTTPProtocol()
        self.websocket_protocol = WebSocketProtocol()
        self.internal_bus = InternalMessageBus()
        self.active_channels = {
            "http": self.http_protocol,
            "websocket": self.websocket_protocol
        }
        print("تم تهيئة مدير الاتصالات.")

    def send_external_message(self, channel_type, recipient, message_content):
        """إرسال رسالة عبر قناة اتصال خارجية محددة."""
        if channel_type in self.active_channels:
            return self.active_channels[channel_type].send_message(recipient, message_content)
        print(f"  قناة الاتصال \'{channel_type}\' غير مدعومة أو غير نشطة.")
        return {"status": "failed", "reason": "unsupported_channel"}

    def receive_external_messages(self):
        """استلام الرسائل من جميع القنوات الخارجية النشطة."""
        all_messages = []
        for channel_type, protocol in self.active_channels.items():
            message = protocol.receive_message()
            while message:
                all_messages.append(message)
                message = protocol.receive_message()
        return all_messages

    def publish_internal_message(self, topic, message_content, sender_id):
        """نشر رسالة على ناقل الرسائل الداخلي."""
        self.internal_bus.publish(topic, message_content, sender_id)

    def subscribe_internal_messages(self, component_id, callback_function):
        """الاشتراك في رسائل ناقل الرسائل الداخلي."""
        self.internal_bus.subscribe(component_id, callback_function)

    def process_internal_messages(self):
        """معالجة جميع الرسائل المعلقة في ناقل الرسائل الداخلي."""
        return self.internal_bus.process_messages()

    def get_status(self):
        return {
            "http_outbox_size": len(self.http_protocol.outbox),
            "http_inbox_size": len(self.http_protocol.inbox),
            "websocket_connected_clients": len(self.websocket_protocol.connected_clients),
            "internal_bus_queue_size": len(self.internal_bus.message_queue)
        }


# مثال على كيفية استخدام CommunicationManager (للتوضيح فقط)
if __name__ == "__main__":
    comm_manager = CommunicationManager()

    # محاكاة رسائل واردة
    comm_manager.http_protocol.simulate_incoming_message("UserApp", "طلب تحليل بيانات X.")
    comm_manager.websocket_protocol.connect_client("Client_A")
    comm_manager.websocket_protocol.simulate_incoming_message("Client_A", "تحديث حالة النظام.")

    # استلام ومعالجة الرسائل الخارجية
    print("\n--- استلام الرسائل الخارجية ---")
    incoming_external_messages = comm_manager.receive_external_messages()
    for msg in incoming_external_messages:
        print("رسالة خارجية: من {}، النوع {}: {}".format(msg["sender"], msg["type"], msg["payload"]))

    # إرسال رسائل خارجية
    print("\n--- إرسال الرسائل الخارجية ---")
    comm_manager.send_external_message("http", "ExternalService", "طلب معلومات من قاعدة بيانات.")
    comm_manager.send_external_message("websocket", "Client_A", "تم معالجة طلبك.")
    comm_manager.send_external_message("websocket", "Client_B", "رسالة لعميل غير متصل.")

    # مثال على استخدام ناقل الرسائل الداخلي
    print("\n--- ناقل الرسائل الداخلي ---")

    def system_core_callback(message):
        print("  [Core] تلقى رسالة داخلية: {}".format(message["payload"]))
        # يمكن أن يقوم Core بمعالجة الرسالة هنا
        if "تحليل بيانات" in message["payload"]:
            comm_manager.publish_internal_message("data_analysis_results", "تم تحليل البيانات بنجاح.", "SystemCore")

    def ui_component_callback(message):
        print("  [UI] تلقى رسالة داخلية: {}".format(message["payload"]))

    comm_manager.subscribe_internal_messages("SystemCore", system_core_callback)
    comm_manager.subscribe_internal_messages("UIComponent", ui_component_callback)

    comm_manager.publish_internal_message("user_request", "يرجى تحليل البيانات.", "InteractionEngine")
    comm_manager.publish_internal_message("system_event", "حدث نظام هام.", "ACES_Main")

    print("  معالجة الرسائل الداخلية...")
    processed_count = comm_manager.process_internal_messages()
    print(f"  تم معالجة {processed_count} رسالة داخلية.")

    print("  معالجة الرسائل الداخلية مرة أخرى (لرسائل الرد)...")
    processed_count = comm_manager.process_internal_messages()
    print(f"  تم معالجة {processed_count} رسالة داخلية.")

    print("\n--- حالة مدير الاتصالات ---")
    print(comm_manager.get_status())




