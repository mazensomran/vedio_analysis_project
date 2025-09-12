import torch
import cv2
import numpy as np
import easyocr

from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging

from config import MODELS_DIR, PROCESSING_CONFIG, MODEL_CONFIG
from model_loader import model_loader
from person_detector import PersonDetector
from ultralytics import YOLO
from PIL import Image, ImageEnhance
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

logger = logging.getLogger(__name__)
device = model_loader.device


class FaceDetector:
    def __init__(self):
        self.model = model_loader.load_yolo_model(MODEL_CONFIG["face_detection_model"], "face")
        self.threshold = PROCESSING_CONFIG["face_detection_threshold"]
        self.min_face_size = PROCESSING_CONFIG["min_face_size"]
        self.max_face_size = PROCESSING_CONFIG["max_face_size"]
        self.aspect_ratio_min = PROCESSING_CONFIG["face_aspect_ratio_min"]
        self.aspect_ratio_max = PROCESSING_CONFIG["face_aspect_ratio_max"]
        self.frame_counter = 0
        self.sampling_interval = PROCESSING_CONFIG["frame_sampling_interval"]
        self.class_names = ['face']

    def detect_faces(self, frame: np.ndarray, frame_number: int) -> List[Dict[str, Any]]:
        """كشف الوجوه في إطار الفيديو مع أخذ العينات والتصفية"""
        if self.model is None:
            return []

        # أخذ عينة من الإطارات فقط
        self.frame_counter += 1
        if self.frame_counter % self.sampling_interval != 0:
            return []

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(frame_rgb, conf=self.threshold, verbose=False)

            faces = []
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

                            width = x2 - x1
                            height = y2 - y1

                            # تصفية الوجوه بناء على الحجم ونسبة العرض إلى الارتفاع
                            if self._is_valid_face(width, height):
                                faces.append({
                                    "frame_number": frame_number,
                                    "bbox": [x1, y1, width, height],
                                    "confidence": confidence,
                                    "face_id": i
                                })

            if faces:
                logger.info(f"👥 تم اكتشاف {len(faces)} وجوه في الإطار {frame_number}")

            return faces

        except Exception as e:
            logger.error(f"❌ خطأ في كشف الوجوه: {e}")
            return []

    def _is_valid_face(self, width: int, height: int) -> bool:
        """التأكد من أن الكائن المكتشف هو وجه حقيقي"""
        # التحقق من الحجم
        if width < self.min_face_size or height < self.min_face_size:
            return False

        if width > self.max_face_size or height > self.max_face_size:
            return False

        if height > 0:
            aspect_ratio = width / height
            if aspect_ratio < self.aspect_ratio_min or aspect_ratio > self.aspect_ratio_max:
                return False

        return True

    def cleanup(self):
        """تنظيف الموارد"""
        if hasattr(self, 'model'):
            self.model = None
            logger.info("🧹 تم تنظيف موارد FaceDetector")

class FrameEnhancer:
    def __init__(self, model_path='RealESRGAN_x4plus.pth', device=None, brightness=1.0, contrast=1.0,
                         saturation=1.0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.half = self.device.type == 'cuda'
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        self.upsampler = RealESRGANer(
                    scale=4,
                    model_path=model_path,
                    model=model,
                    tile=512,
                    tile_pad=128,
                    pre_pad=0,
                    half=self.half,
                    device=self.device
                )
    def enhance_frame(self, frame):
        # تحويل BGR إلى RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        '''with torch.no_grad():
            output_rgb, _ = self.upsampler.enhance(img_rgb, outscale=4)'''
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
                detector=EASYOCR_CONFIG["detector"],
                recognizer=EASYOCR_CONFIG["recognizer"]
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
        self.person_detector = PersonDetector()
        self.tracking_threshold = PROCESSING_CONFIG["tracking_threshold"]
        self.max_tracking_distance = PROCESSING_CONFIG["max_tracking_distance"]
        self.min_track_length = PROCESSING_CONFIG["min_track_length"]
        self.tracks = defaultdict(list)
        self.next_track_id = 1
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]

    def track_objects(self, frame: np.ndarray, persons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """تتبع الأشخاص بين الإطارات"""
        tracking_data = []
        for person in persons:
            bbox_list = person["bbox"]
            center_x = bbox_list[0] + bbox_list[2] / 2
            center_y = bbox_list[1] + bbox_list[3] / 2
            min_distance = float('inf')
            best_track_id = None
            for track_id, track in self.tracks.items():
                if track:
                    last_pos = track[-1]
                    distance = ((center_x - last_pos[0]) ** 2 + (center_y - last_pos[1]) ** 2) ** 0.5
                    if distance < min_distance and distance < self.max_tracking_distance:
                        min_distance = distance
                        best_track_id = track_id

            if best_track_id is not None:
                track_id = best_track_id
                self.tracks[track_id].append((center_x, center_y))
            else:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[track_id] = [(center_x, center_y)]
            tracking_data.append({
                "track_id": track_id,
                "frame_number": person["frame_number"],
                "bbox": bbox_list.tolist(),
                "confidence": person["confidence"],
                "type": "person"
            })
        if tracking_data:
                logger.info(f"🔄 تم تتبع {len(tracking_data)} أشخاص في الإطار {frame.shape[0]}") # استخدم frame.shape[0] أو أي مؤشر للإطار

        return tracking_data

    def draw_tracks(self, frame: np.ndarray, tracking_data: List[Dict[str, Any]]) -> np.ndarray:
        """رسم المسارات على الإطار"""
        for track in tracking_data:
            track_id = track["track_id"]
            bbox = track["bbox"]
            color = self.colors[track_id % len(self.colors)]

            x1, y1, x2, y2 = map(int, bbox)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if track_id in self.tracks and len(self.tracks[track_id]) >= self.min_track_length:
                path = self.tracks[track_id]
                for i in range(1, len(path)):
                    cv2.line(frame,
                             (int(path[i - 1][0]), int(path[i - 1][1])),
                             (int(path[i][0]), int(path[i][1])),
                             color, 2)

                cv2.putText(frame, f"Person {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    def reset(self):
        """إعادة تعيين المتعقب"""
        self.tracks.clear()
        self.next_track_id = 1
        logger.info("🔄 تم إعادة تعيين المتعقب")

    def cleanup(self):
        """تنظيف الموارد"""
        self.tracks.clear()
        if hasattr(self, 'person_detector'):
            self.person_detector.cleanup()
            self.person_detector = None
        logger.info("🧹 تم تنظيف موارد ObjectTracker")

class GeneralObjectDetector:
    def __init__(self, model_name="yolov8n.pt", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_name)
        self.model.to(self.device)
        self.frame_counter = 0
        self.model.fuse()  # تحسين الأداء
        self.threshold = PROCESSING_CONFIG["person_detection_threshold"]
        self.sampling_interval = PROCESSING_CONFIG["frame_sampling_interval"]
    def detect_objects(self, frame, frame_number):

        self.frame_counter += 1
        if self.frame_counter % self.sampling_interval != 0:
            return []
        results = self.model(frame)
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            class_names = [self.model.names[cid] for cid in class_ids]
            for box, confidence, cname in zip(boxes, confidences, class_names):
                detections.append({
                    "bbox": box,
                    "confidence": confidence,
                    "class_name": cname,
                    "frame_number" : frame_number
                })
        return detections


