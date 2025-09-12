import cv2
import numpy as np
from typing import List, Dict, Any
from config import PROCESSING_CONFIG
from model_loader import model_loader
import logging
logger = logging.getLogger(__name__)


class PersonDetector:
    def __init__(self):
        self.model = model_loader.load_yolo_model("yolov8n.pt")
        self.threshold = PROCESSING_CONFIG["person_detection_threshold"]

    def detect_persons(self, frame: np.ndarray, frame_number: int) -> List[Dict[str, Any]]:
        """كشف الأشخاص في إطار الفيديو"""
        if self.model is None:
            print("❌ نموذج كشف الأشخاص غير متاح")
            return []

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.model(frame_rgb, conf=self.threshold, verbose=False) #, classes=[self.class_id]

            persons = []

            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for i, box in enumerate(result.boxes):
                        confidence = float(box.conf[0])
                        if confidence >= self.threshold:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                            # التأكد من أن المربع ضمن حدود الصورة
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(frame.shape[1], x2)
                            y2 = min(frame.shape[0], y2)

                            # حساب عرض وارتفاع الشخص
                            width = x2 - x1
                            height = y2 - y1

                            # تجاهل الأشخاص الصغيرة جداً
                            if width > 30 and height > 30:
                                persons.append({
                                    "frame_number": frame_number,
                                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                    "confidence": confidence,
                                    "person_id": i
                                })

            return persons

        except Exception as e:
            print(f"❌ خطأ في كشف الأشخاص: {e}")
            return []

    def cleanup(self):
        self.model = None
        logger.info("🧹 تم تنظيف موارد PersonDetector")