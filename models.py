
import torch
import cv2
import numpy as np
import easyocr
from PIL import Image
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging
from collections import Counter
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
import types  # تم إضافة هذا الاستيراد
from config import MODELS_DIR, PROCESSING_CONFIG, MODEL_CONFIG
from model_loader import model_loader
from activity_recognizer import ActivityRecognizer
from ultralytics import YOLO
from PIL import Image, ImageEnhance
from scrfd import SCRFD, Threshold
import supervision as sv

# إعداد التسجيل
logger = logging.getLogger(__name__)

# تحديد الجهاز
device = model_loader.device


class FaceDetector:
    def __init__(self):
        self.scrfd_detector = SCRFD.from_path("scrfd.onnx", providers=["CUDAExecutionProvider"])

        if self.scrfd_detector is None:
            raise Exception("فشل تحميل نموذج SCRFD لكشف الوجوه.")
        self.threshold = Threshold(probability=PROCESSING_CONFIG["face_detection_threshold"])
        self.frame_counter = 0
        self.sampling_interval = PROCESSING_CONFIG["frame_sampling_interval"]

        # إعدادات NMS (بدون نافذة منزلقة)
        self.nms_threshold = 0.4  # عتبة NMS لإزالة الاكتشافات المتداخلة

    def detect_faces(self, frame: np.ndarray, frame_number: int) -> List[Dict[str, Any]]:
        """كشف الوجوه في إطار الفيديو مباشرة (بدون نافذة منزلقة)"""
        self.frame_counter += 1
        # أخذ عينة من الإطارات إذا لزم الأمر
        if self.frame_counter % self.sampling_interval != 0:
            return []

        height, width, _ = frame.shape
        all_raw_detections = []  # لتخزين الاكتشافات قبل NMS

        try:
            # تحويل الإطار إلى PIL Image للكشف المباشر
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # الكشف عن الوجوه مباشرة على الإطار بأكمله
            faces_in_frame = self.scrfd_detector.detect(pil_frame, threshold=self.threshold)

            # جمع الاكتشافات
            for face in faces_in_frame:
                x1 = int(face.bbox.upper_left.x)
                y1 = int(face.bbox.upper_left.y)
                x2 = int(face.bbox.lower_right.x)
                y2 = int(face.bbox.lower_right.y)

                # التأكد من أن الإحداثيات ضمن حدود الإطار
                x1 = max(0, min(x1, width))
                y1 = max(0, min(y1, height))
                x2 = max(x1, min(x2, width))
                y2 = max(y1, min(y2, height))

                all_raw_detections.append({
                    "bbox": [x1, y1, x2, y2],  # تنسيق [x1, y1, x2, y2]
                    "confidence": face.probability,
                    "keypoints": face.keypoints  # SCRFD يعيد keypoints مباشرة
                })

            # تطبيق NMS لإزالة الاكتشافات المتداخلة
            final_detections = self._apply_nms(all_raw_detections)

            faces = []
            for i, det in enumerate(final_detections):
                x1, y1, x2, y2 = det["bbox"]
                width_face = x2 - x1
                height_face = y2 - y1

                faces.append({
                    "frame_number": frame_number,
                    "bbox": [x1, y1, width_face, height_face],  # تنسيق [x, y, w, h]
                    "confidence": det["confidence"],
                    "face_id": i,  # فهرس فريد لكل وجه بعد NMS
                    "keypoints": {  # تحويل نقاط الوجه إلى قاموس لسهولة التخزين
                        "left_eye": {"x": det["keypoints"].left_eye.x, "y": det["keypoints"].left_eye.y},
                        "right_eye": {"x": det["keypoints"].right_eye.x, "y": det["keypoints"].right_eye.y},
                        "nose": {"x": det["keypoints"].nose.x, "y": det["keypoints"].nose.y},
                        "left_mouth": {"x": det["keypoints"].left_mouth.x, "y": det["keypoints"].left_mouth.y},
                        "right_mouth": {"x": det["keypoints"].right_mouth.x, "y": det["keypoints"].right_mouth.y}
                    }
                })

            if faces:
                logger.info(f"👥 تم اكتشاف {len(faces)} وجوه في الإطار {frame_number} بعد NMS.")

            return faces

        except Exception as e:
            logger.error(f"❌ خطأ في كشف الوجوه: {e}")
            return []

    def _apply_nms(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """تطبيق Non-Maximum Suppression على الاكتشافات"""
        if not detections:
            return []

        boxes = np.array([d["bbox"] for d in detections])  # [x1, y1, x2, y2]
        scores = np.array([d["confidence"] for d in detections])

        # cv2.dnn.NMSBoxes تتوقع [x, y, w, h]، لذا نحتاج للتحويل
        boxes_xywh = np.array([[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes])

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(),
            scores.tolist(),
            score_threshold=float(self.threshold.probability),
            nms_threshold=self.nms_threshold
        )

        if isinstance(indices, np.ndarray):
            indices = indices.flatten().tolist()
        elif isinstance(indices, list) and len(indices) > 0 and isinstance(indices[0], list):
            indices = [i[0] for i in indices]
        else:
            indices = []  # في حالة عدم وجود اكتشافات بعد NMS

        return [detections[i] for i in indices]

    def cleanup(self):
        """تنظيف الموارد"""
        if hasattr(self, 'scrfd_detector'):
            self.scrfd_detector = None
            logger.info("🧹 تم تنظيف موارد FaceDetector")


class FrameEnhancer:
    def __init__(self, brightness=1.0, contrast=1.0, saturation=1.0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def enhance_frame(self, frame):
        if frame is None or frame.size == 0:
            logger.warning("⚠️ تم تمرير إطار فارغ إلى FrameEnhancer.enhance_frame.")
            return None
        # تحويل BGR إلى RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # تحويل numpy array إلى PIL Image
        img_pil = Image.fromarray(img_rgb)
        # تطبيق تحسينات السطوع والتباين والتشبع
        if self.brightness != 1.0:
            img_pil = ImageEnhance.Brightness(img_pil).enhance(self.brightness)
        if self.contrast != 1.0:
            img_pil = ImageEnhance.Contrast(img_pil).enhance(self.contrast)
        if self.saturation != 1.0:
            img_pil = ImageEnhance.Color(img_pil).enhance(self.saturation)
        return img_pil


class TextDetector:
    def __init__(self):
        self.detection_threshold = PROCESSING_CONFIG["text_detection_threshold"]
        self.enabled = PROCESSING_CONFIG["text_detection_enabled"]
        self.min_text_confidence = PROCESSING_CONFIG["min_text_confidence"]
        self.languages = MODEL_CONFIG["easyocr_languages"]
        self.reader = None
        self.frame_counter = 0
        self.sampling_interval = PROCESSING_CONFIG["frame_sampling_interval"]

        if self.enabled:
            self._setup_easyocr()

    def _setup_easyocr(self):
        """إعداد EasyOCR مع الإصلاح"""
        try:
            from config import EASYOCR_CONFIG
            logger.info("📥 جاري تحميل EasyOCR...")

            gpu = EASYOCR_CONFIG["gpu_enabled"] and torch.cuda.is_available()
            model_dir = Path(EASYOCR_CONFIG["model_storage_directory"])
            model_dir.mkdir(parents=True, exist_ok=True)

            # ✅ التصحيح: استخدام المعلمات الصحيحة
            self.reader = easyocr.Reader(
                lang_list=self.languages,
                gpu=gpu,
                model_storage_directory=str(model_dir),
                download_enabled=EASYOCR_CONFIG["download_enabled"],
                detector=EASYOCR_CONFIG["detector"],  # ✅ تصحيح اسم المعلمة
                recognizer=EASYOCR_CONFIG["recognizer"]  # ✅ تصحيح اسم المعلمة
            )

            logger.info(f"✅ تم تحميل EasyOCR بنجاح على {'GPU' if gpu else 'CPU'}")

        except Exception as e:
            logger.error(f"❌ فشل في تحميل EasyOCR: {e}")
            self.reader = None
            self.enabled = False

    def detect_text(self, frame: np.ndarray, frame_number: int) -> List[Dict[str, Any]]:
        """كشف النص في إطار الفيديو"""
        if self.reader is None or not self.enabled:
            return []

        # أخذ عينة من الإطارات فقط
        self.frame_counter += 1
        if self.frame_counter % self.sampling_interval != 0:
            return []

        try:
            # تحسين الصورة لتحسين دقة التعرف على النص
            enhanced_frame = self._enhance_image_for_text(frame)
            results = self.reader.readtext(enhanced_frame, paragraph=False)

            text_data = []
            for (bbox, text, confidence) in results:
                if confidence >= self.min_text_confidence:
                    points = np.array(bbox).astype(int)
                    x1, y1 = np.min(points[:, 0]), np.min(points[:, 1])
                    x2, y2 = np.max(points[:, 0]), np.max(points[:, 1])
                    width, height = x2 - x1, y2 - y1

                    # تجاهل النصوص الصغيرة جداً
                    if width < 10 or height < 10:
                        continue

                    language = self._detect_language(text)
                    text_data.append({
                        "frame_number": int(frame_number),
                        "bbox": [int(x1), int(y1), int(width), int(height)],
                        "text": text.strip(),
                        "confidence": float(confidence),
                        "language": language
                    })

            if text_data:
                logger.info(f"📝 تم اكتشاف {len(text_data)} نصوص في الإطار {frame_number}")

            return text_data

        except Exception as e:
            logger.error(f"❌ خطأ في كشف النص: {e}")
            return []

    def _enhance_image_for_text(self, frame: np.ndarray) -> np.ndarray:
        """تحسين الصورة لتحسين التعرف على النص"""
        if frame is None or frame.size == 0:
            logger.warning("⚠️ تم تمرير إطار فارغ إلى FrameEnhancer.enhance_frame.")
            return None
        try:
            # تحويل إلى تدرجات الرمادي
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # تطبيق تصحيح التباين
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # تحويل مرة أخرى إلى BGR
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        except Exception:
            return frame  # العودة إلى الصورة الأصلية في حالة الخطأ

    def _detect_language(self, text: str) -> str:
        """كشف اللغة تلقائياً بناء على النص"""
        arabic_chars = "ابتثجحخدذرزسشصضطظعغفقكلمنهويءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوي"

        if any(char in arabic_chars for char in text):
            return "ar"
        elif any(char.isalpha() for char in text):
            return "en"
        return "unknown"

    def cleanup(self):
        """تنظيف ذاكرة EasyOCR"""
        if self.reader:
            try:
                # تنظيف Models من الذاكرة
                self.reader = None
                logger.info("🧹 تم تنظيف ذاكرة EasyOCR")
            except Exception as e:
                logger.error(f"⚠️ خطأ في تنظيف EasyOCR: {e}")


class SpeechRecognizer:
    def __init__(self):
        self.model_name = MODEL_CONFIG["speech_recognition_model"]
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        """تحميل نموذج التعرف على الكلام"""
        try:
            logger.info(f"📥 تحميل نموذج التعرف على الكلام ({self.model_name}) على {self.device}...")
            self.model = model_loader.load_whisper_model(self.model_name)

            if self.model is not None:
                logger.info(f"✅ تم تحميل نموذج التعرف على الكلام على {self.device}")
            else:
                logger.error("❌ فشل في تحميل نموذج التعرف على الكلام")

        except Exception as e:
            logger.error(f"❌ خطأ في تحميل نموذج التعرف على الكلام: {e}")
            self.model = None

    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """تحويل الصوت إلى نص"""
        if self.model is None:
            return {"text": "", "language": "unknown", "confidence": 0.0}

        try:
            # التحقق من وجود ملف الصوت
            if not Path(audio_path).exists():
                logger.error(f"❌ ملف الصوت غير موجود: {audio_path}")
                return {"text": "", "language": "unknown", "confidence": 0.0}

            # تنظيف ذاكرة GPU إذا كان متاحًا
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("🎵 جاري تحويل الصوت إلى نص...")
            result = self.model.transcribe(audio_path, fp16=torch.cuda.is_available())

            # حفظ النص في ملف
            try:
                text_output_path = Path(audio_path).parent / "transcription.txt"
                with open(text_output_path, 'w', encoding='utf-8') as f:
                    f.write(result["text"])
                logger.info(f"✅ تم حفظ النص الصوتي في: {text_output_path}")
            except Exception as e:
                logger.error(f"⚠️ خطأ في حفظ النص الصوتي: {e}")

            return {
                "text": result["text"],
                "language": result.get("language", "unknown"),
                "confidence": 0.9,  # قيمة تقريبية
                "segments": result.get("segments", [])
            }

        except Exception as e:
            logger.error(f"❌ خطأ في تحويل الصوت إلى نص: {e}")
            return {"text": "", "language": "unknown", "confidence": 0.0}

    def cleanup(self):
        """تنظيف الموارد"""
        self.model = None
        logger.info("🧹 تم تنظيف موارد SpeechRecognizer")


class ObjectTracker:
    def __init__(self):
        # 1. تعريف self.device أولاً (قبل أي تحميل)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 2. تهيئة المتغيرات الأساسية (قبل التحميل، لتجنب AttributeError إذا فشل التحميل)
        self.model = None
        self.image_processor = None
        self.id2label = None
        self.frame_counter = 0  # <--- تصحيح: تعريف frame_counter هنا مبكرًا (لكن سنعيد تعيينه إذا نجح التحميل)
        self.tracker = None
        self.box_annotator = None
        self.label_annotator = None
        self.trace_annotator = None
        self.sampling_interval = PROCESSING_CONFIG["frame_sampling_interval"]
        self.min_track_length = PROCESSING_CONFIG.get("min_track_length", 5)
        self.detection_threshold = PROCESSING_CONFIG.get("object_detection_threshold", 0.3)
        self.confidence_threshold = PROCESSING_CONFIG.get("object_confidence_threshold", 0.5)

        # 3. محاولة تحميل النموذج
        try:
            logger.info(f"📥 تحميل نموذج RT-DETR v2 r18vd على {self.device}...")
            self.image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
            self.model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd").to(self.device)
            self.model.eval()

            # 4. تعيين self.id2label بعد نجاح التحميل
            self.id2label = self.model.config.id2label

            # التحقق من أن id2label يحتوي على "person" (ID 0 في COCO)
            if 0 not in self.id2label or self.id2label[0] != "person":
                logger.warning("⚠️ فئة 'person' (ID 0) غير موجودة في id2label. قد يؤثر على التتبع.")

            logger.info(f"✅ تم تحميل RT-DETR v2 بنجاح. عدد الفئات: {len(self.id2label)}")

        except Exception as e:
            logger.error(f"❌ خطأ في تحميل نموذج RT-DETR v2: {e}")
            # 5. رفع استثناء لإيقاف التهيئة إذا فشل التحميل (يمنع إنشاء كائن غير صالح)
            raise Exception(f"فشل تحميل نموذج RT-DETR v2 r18vd للتتبع: {e}")

        # 6. تهيئة مكونات supervision فقط إذا نجح التحميل (هنا يتم إكمال التهيئة)
        try:
            self.tracker = sv.ByteTrack()
            self.box_annotator = sv.BoxAnnotator()
            self.label_annotator = sv.LabelAnnotator()
            self.trace_annotator = sv.TraceAnnotator()
            self.frame_counter = 0  # <--- إعادة تعيين frame_counter هنا للتأكيد (بعد النجاح)

            logger.info(f"✅ تم تهيئة ObjectTracker (ByteTrack + RT-DETR v2) بنجاح")
        except Exception as e:
            logger.error(f"❌ خطأ في تهيئة مكونات supervision: {e}")
            raise Exception(f"فشل في تهيئة ObjectTracker: {e}")

    def track_objects(self, frame: np.ndarray, frame_number: int) -> Tuple[sv.Detections, List[Dict[str, Any]]]:
        """
        يكشف ويتتبع جميع الكائنات (بما في ذلك الأشخاص) باستخدام RT-DETR v2 r18vd و ByteTrack.
        يعيد كائن sv.Detections وقائمة ببيانات التتبع للأشخاص.
        """
        all_detections = []
        person_tracks = []

        # 7. تحقق إضافي لـ frame_counter (للمتانة، في حالة مشكلة في التهيئة)
        if not hasattr(self, 'frame_counter'):
            logger.error("❌ ObjectTracker غير مُهيأ بشكل صحيح: frame_counter غير موجود")
            return sv.Detections.empty(), []

        self.frame_counter += 1

        # أخذ عينة من الإطارات إذا لزم الأمر
        if self.frame_counter % self.sampling_interval != 0:
            return sv.Detections.empty(), []

        try:
            # التحقق من أن النموذج مُحمل
            if self.model is None or self.image_processor is None:
                logger.error("❌ النموذج غير مُحمل. تأكد من نجاح التهيئة.")
                return sv.Detections.empty(), []

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = self.image_processor(images=rgb_frame, return_tensors="pt").to(self.device)

            with torch.no_grad():
                with torch.autocast(device_type="cuda" if self.device.type == "cuda" else "cpu",
                                    dtype=torch.float16 if self.device.type == "cuda" else torch.float32):
                    outputs = self.model(**inputs)

            # الحصول على النتائج من post_process_object_detection
            processed_results = self.image_processor.post_process_object_detection(
                outputs,
                target_sizes=torch.tensor([(h, w)], device=self.device),
                threshold=self.detection_threshold
            )[0]

            # التحقق من وجود اكتشافات (boxes, scores, labels)
            if 'boxes' not in processed_results or len(processed_results['boxes']) == 0:
                return sv.Detections.empty(), []

            # إنشاء sv.Detections من نتائج transformers
            detections = sv.Detections.from_transformers(
                transformers_results=processed_results,
                id2label=self.id2label
            )

            # فلترة بناءً على الثقة
            if detections.confidence is not None:
                mask = detections.confidence > self.confidence_threshold
                detections = detections[mask]

            if len(detections) == 0:
                return sv.Detections.empty(), []

            # تحديث التتبع باستخدام ByteTrack
            detections = self.tracker.update_with_detections(detections)

            # استخراج البيانات المطلوبة
            for i in range(len(detections)):
                if detections.xyxy is None or i >= len(detections.xyxy):
                    continue  # تخطي إذا كانت البيانات غير صالحة

                bbox = detections.xyxy[i].tolist()  # [x1, y1, x2, y2]
                if len(bbox) != 4:
                    logger.warning(f"⚠️ bbox غير صالح في الإطار {frame_number}: {bbox}. تخطي.")
                    continue

                conf = detections.confidence[i] if detections.confidence is not None else 0.0
                cls_id = detections.class_id[i] if detections.class_id is not None else 0
                x1, y1, x2, y2 = map(int, bbox)
                class_name = self.id2label.get(int(cls_id), f"unknown_{cls_id}")
                track_id = detections.tracker_id[i] if detections.tracker_id is not None else None

                all_detections.append({
                    "bbox": bbox,
                    "confidence": conf,
                    "class_name": class_name,
                    "frame_number": frame_number,
                    "track_id": track_id if class_name == "person" else None
                })

                if class_name == "person" and track_id is not None:
                    person_tracks.append({
                        "track_id": track_id,
                        "frame_number": frame_number,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "class_name": class_name,
                    })

            if len(detections) > 0:
                logger.info(
                    f"🔄 تم كشف وتتبع {len(detections)} كائن ({len(person_tracks)} شخص) في الإطار {frame_number}")

            return detections, person_tracks

        except Exception as e:
            logger.error(f"❌ خطأ في كشف وتتبع الكائنات (ByteTrack): {e}")
            import traceback
            logger.error(f"تفاصيل الخطأ: {traceback.format_exc()}")  # للتشخيص
            return sv.Detections.empty(), []

    def draw_tracks(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """رسم الكائنات والمسارات على الإطار"""
        if len(detections) == 0 or detections.xyxy is None:
            return frame

        # بناء labels مع تحقق
        labels = []
        for cls_id, tid, conf in zip(detections.class_id, detections.tracker_id, detections.confidence):
            if tid is not None and cls_id is not None and conf is not None:
                class_name = self.id2label.get(int(cls_id), 'unknown')
                labels.append(f"#{int(tid)} {class_name} {conf:.2f}")
            else:
                labels.append("unknown")

        # رسم الصناديق
        annotated_frame = self.box_annotator.annotate(frame.copy(), detections=detections)
        # رسم التسميات
        annotated_frame = self.label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
        # رسم المسارات
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections=detections)

        return annotated_frame

    def cleanup(self):
        """تنظيف الموارد"""
        try:
            self.model = None
            self.image_processor = None
            self.tracker = None
            self.box_annotator = None
            self.label_annotator = None
            self.trace_annotator = None
            self.id2label = None
            logger.info("🧹 تم تنظيف موارد ObjectTracker (ByteTrack + RT-DETR)")
        except Exception as e:
            logger.error(f"⚠️ خطأ في تنظيف ObjectTracker: {e}")



