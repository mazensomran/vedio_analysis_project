import torch
from ultralytics import YOLO
import whisper
from pathlib import Path
import requests
from typing import Optional, List, Dict, Any, Tuple
import logging
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoFeatureExtractor, VideoMAEForVideoClassification
from config import MODELS_DIR, MODEL_CONFIG, PROCESSING_CONFIG, GPU_AVAILABLE

logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self):
        self.device = torch.device(MODEL_CONFIG["device"])
        self.models_dir = MODELS_DIR
        self.models_dir.mkdir(exist_ok=True)
        self.loaded_models = {}
        self.model_cache = {}

        logger.info(f"🎯 جهاز المعالجة: {self.device}")
        if GPU_AVAILABLE:
            print(f"🎯 ذاكرة GPU الإجمالية: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    def _download_model(self, model_url: str, model_path: Path) -> bool:
        """تحميل النموذج من URL"""
        try:
            logger.info(f"📥 جاري تحميل النموذج من: {model_url}")

            response = requests.get(model_url, stream=True, timeout=MODEL_CONFIG["model_download_timeout"])
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            logger.info(f"📊 التقدم: {progress:.1f}%", end='\r')

            logger.info(f"\n✅ تم تحميل النموذج بنجاح: {model_path.name}")
            return True

        except Exception as e:
            logger.info(f"❌ فشل في تحميل النموذج: {e}")
            if model_path.exists():
                model_path.unlink()
            return False

    def _setup_model_cache(self):
        """إعداد ذاكرة التخزين المؤقت للنماذج"""
        if MODEL_CONFIG["optimize_for_inference"]:
            torch.backends.cudnn.benchmark = True
            if GPU_AVAILABLE:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    def load_yolo_model(self, model_name: str = None, model_type: str = "face") -> Optional[YOLO]:
        if model_name is None:
            model_name = MODEL_CONFIG["face_detection_model"] if model_type == "face" else MODEL_CONFIG[
                "person_detection_model"]

        cache_key = f"yolo_{model_name}_{model_type}"
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]

        try:
            model_path = self.models_dir / model_name
            model_urls = {
                "yolov8n-face.pt": "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt",
                "yolov8s-face.pt": "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8s-face.pt",
                "yolov8m-face.pt": "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8m-face.pt",
                "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
                "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
                "yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
            }

            if not model_path.exists():
                if model_name in model_urls:
                    if not self._download_model(model_urls[model_name], model_path):
                        logger.info(f"⚠️ استخدام النموذج المضمن في YOLO: {model_name}")
                        model = YOLO(model_name)
                else:
                    logger.info(f"⚠️ النموذج {model_name} غير معروف، استخدام النموذج المضمن")
                    model = YOLO(model_name)
            else:
                logger.info(f"📂 تحميل النموذج من التخزين المحلي: {model_name}")
                model = YOLO(str(model_path))

            model.to(self.device)
            model.model.float()
            if hasattr(model, 'model'):
                model.model.float()

            logger.info(f"✅ تم تحميل وتفعيل نموذج YOLO ({model_name}) بنجاح على {self.device}")
            self.model_cache[cache_key] = model
            return model

        except Exception as e:
            logger.info(f"❌ خطأ في تحميل YOLO {model_name}: {e}")
            return None

    def load_whisper_model(self, model_name: str = None) -> Optional[Any]:
        """تحميل نموذج Whisper"""
        if model_name is None:
            model_name = MODEL_CONFIG["speech_recognition_model"]

        cache_key = f"whisper_{model_name}"
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]

        try:
            logger.info(f"📥 تحميل نموذج Whisper: {model_name}")

            available_models = MODEL_CONFIG["available_whisper_models"]
            if model_name not in available_models:
                logger.info(f"⚠️ النموذج {model_name} غير متاح، استخدام 'base' بدلاً منه")
                model_name = "base"

            model = whisper.load_model(
                model_name,
                download_root=str(self.models_dir / "whisper"),
                device=self.device
            )

            model = model.float()

            self.model_cache[cache_key] = model
            print(f"✅ تم تحميل Whisper ({model_name}) بنجاح على {self.device}")
            return model

        except Exception as e:
            logger.info(f"❌ خطأ في تحميل Whisper: {e}")
            return None

    def load_blip_model(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        """تحميل نموذج BLIP لتوليد الوصف"""
        try:
            if model_name not in self.model_cache:
                processor = BlipProcessor.from_pretrained(model_name)
                # Force float32 for better compatibility
                model = BlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32) # <--- CHANGE THIS LINE
                model.to(self.device)
                self.model_cache[model_name] = (processor, model)
                logger.info(f"✅ تم تحميل نموذج BLIP: {model_name}")
            return self.model_cache[model_name]
        except Exception as e:
            logger.info(f"❌ فشل تحميل نموذج BLIP: {e}")
            return None, None

    def load_videomae_model(self):
        """تحميل نموذج VideoMAE لتصنيف النشاط في الفيديو."""
        model_key = "videomae"
        if model_key in self.model_cache:
            logger.info("✅ تم العثور على نموذج VideoMAE في الذاكرة المؤقتة.")
            return self.model_cache[model_key]
        try:
            model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
            logger.info(f"📥 جاري تحميل نموذج VideoMAE: {model_name}")
            processor = AutoFeatureExtractor.from_pretrained(model_name)
            # Force float32 for better compatibility
            model = VideoMAEForVideoClassification.from_pretrained(model_name, torch_dtype=torch.float32).to(self.device) # <--- CHANGE THIS LINE
            # Note: .to(self.device) should be after from_pretrained if you specify dtype
            self.model_cache[model_key] = (processor, model)
            logger.info("✅ تم تحميل نموذج VideoMAE بنجاح.")
            return processor, model
        except Exception as e:
            logger.error(f"❌ فشل تحميل نموذج VideoMAE: {e}")
            return None, None


    def load_person_detector(self) -> Optional[Any]:
        """تحميل نموذج كشف الأشخاص"""
        return self.load_yolo_model(MODEL_CONFIG["person_detection_model"], "person")

    def load_face_detector(self) -> Optional[Any]:
        """تحميل نموذج كشف الوجوه"""
        return self.load_yolo_model(MODEL_CONFIG["face_detection_model"], "face")

    def clear_model_cache(self, model_key: str = None):
        """مسح ذاكرة التخزين المؤقت للنماذج"""
        if model_key:
            if model_key in self.model_cache:
                # تفريغ النموذج من الذاكرة
                model_instance = self.model_cache[model_key]
                if isinstance(model_instance, tuple): # إذا كان tuple (مثل BLIP: processor, model)
                    for item in model_instance:
                        del item
                else:
                    del model_instance
                del self.model_cache[model_key]
                logger.info(f"🧹 تم مسح النموذج من الذاكرة: {model_key}")
        else:
            for key in list(self.model_cache.keys()):
                self.clear_model_cache(key)
            if GPU_AVAILABLE:
                torch.cuda.empty_cache()
            logger.info("🧹 تم مسح جميع النماذج من الذاكرة")

    def get_model_memory_usage(self) -> Dict[str, float]:
        """الحصول على استخدام الذاكرة للنماذج"""
        memory_usage = {}

        for model_name, model in self.model_cache.items():
            try:
                if hasattr(model, 'parameters'):
                    # حساب حجم النموذج
                    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
                    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
                    total_size = (param_size + buffer_size) / 1024 ** 2  # MB

                    memory_usage[model_name] = total_size
            except:
                pass

        return memory_usage

    def load_scene_recognition_model(self):
        videomae_processor, videomae_model = self.load_videomae_model()
        blip_processor, blip_model = self.load_blip_model()
        return videomae_processor, videomae_model, blip_processor, blip_model

    def cleanup(self):
        """تنظيف جميع الموارد"""
        self.model_cache.clear()
        if GPU_AVAILABLE:
            torch.cuda.empty_cache()
        logger.info("🧹 تم تنظيف جميع موارد ModelLoader")


# إنشاء loader عالمي
model_loader = ModelLoader()

def cleanup_models():
    """تنظيف جميع النماذج من الذاكرة"""
    model_loader.cleanup()

