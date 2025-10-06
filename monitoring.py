import time
import threading
from typing import Dict, Any
from database import db
from config import PROCESSING_CONFIG


class ProcessMonitor:
    def __init__(self):
        self.active_processes = {}
        self.monitoring = False

    def start_monitoring(self):
        """بدء مراقبة العمليات"""
        self.monitoring = True
        monitor_thread = threading.Thread(target=self._monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()

    def _monitor_loop(self):
        """حلقة المراقبة الرئيسية"""
        while self.monitoring:
            try:
                self._check_processes()
                time.sleep(5)  # التحقق كل 5 ثواني
            except Exception as e:
                print(f"❌ خطأ في المراقبة: {e}")

    def _check_processes(self):
        """التحقق من حالة العمليات"""
        # يمكن إضافة منطق لمراقبة استخدام الذاكرة والوقت
        pass

    def add_process(self, process_id: str):
        """إضافة عملية للمراقبة"""
        self.active_processes[process_id] = {
            "start_time": time.time(),
            "last_update": time.time(),
            "error_count": 0
        }

    def remove_process(self, process_id: str):
        """إزالة عملية من المراقبة"""
        if process_id in self.active_processes:
            del self.active_processes[process_id]

    def report_error(self, process_id: str):
        """الإبلاغ عن خطأ في العملية"""
        if process_id in self.active_processes:
            self.active_processes[process_id]["error_count"] += 1
            self.active_processes[process_id]["last_update"] = time.time()

            # إذا تجاوزت الأخطاء الحد المسموح
            if self.active_processes[process_id]["error_count"] >= PROCESSING_CONFIG["max_404_retries"]:
                db.update_process_status(process_id, "error", 0, "تم إيقاف التحليل بسبب أخطاء متكررة")
                self.remove_process(process_id)


# إنشاء monitor عالمي
process_monitor = ProcessMonitor()
