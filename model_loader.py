
import torch
from transformers import AutoImageProcessor
import whisper
import os
from pathlib import Path
import requests
import time
from typing import Optional, List, Dict, Any, Tuple
import logging
from config import MODELS_DIR, MODEL_CONFIG, PROCESSING_CONFIG, GPU_AVAILABLE
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from scrfd import SCRFD, Threshold
import torch
# إعداد التسجيل
logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self):
        self.device = torch.device(MODEL_CONFIG["device"])
        self.models_dir = MODELS_DIR
        self.models_dir.mkdir(exist_ok=True)
        self.loaded_models = {}
        self.model_cache = {}

        print(f"🎯 جهاز المعالجة: {self.device}")
        if GPU_AVAILABLE:
            print(f"🎯 ذاكرة GPU الإجمالية: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    def _download_model(self, model_url: str, model_path: Path) -> bool:
        """تحميل النموذج من URL"""
        try:
            print(f"📥 جاري تحميل النموذج من: {model_url}")

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
                            print(f"📊 التقدم: {progress:.1f}%", end='\r')

            print(f"\n✅ تم تحميل النموذج بنجاح: {model_path.name}")
            return True

        except Exception as e:
            print(f"❌ فشل في تحميل النموذج: {e}")
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


    def load_scrfd_model(self, model_path: Optional[Path] = None) -> Optional[SCRFD]:
        # إذا لم يُمرر مسار، استخدم المسار الافتراضي من الإعدادات
        if model_path is None:
            model_path = MODEL_CONFIG["scrfd_model_path"]
        # تحويل model_path إلى كائن Path إذا كان str
        if isinstance(model_path, str):
            model_path = Path(model_path)
        # (اختياري) استخدام التخزين المؤقت
        cache_key = f"scrfd_{model_path.name}"
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]
        try:
            if not model_path.exists():
                print(f"⚠️ نموذج SCRFD غير موجود في المسار: {model_path}. يرجى تنزيله يدوياً.")
                return None
            print(f"📂 تحميل نموذج SCRFD من التخزين المحلي: {model_path.name}")
            model = SCRFD.from_path(str(model_path))
            self.model_cache[cache_key] = model
            print(f"✅ تم تحميل نموذج SCRFD بنجاح.")
            return model
        except Exception as e:
            print(f"❌ خطأ في تحميل SCRFD: {e}")
            return None

    def load_whisper_model(self, model_name: str = None) -> Optional[Any]:
        """تحميل نموذج Whisper"""
        if model_name is None:
            model_name = MODEL_CONFIG["speech_recognition_model"]

        cache_key = f"whisper_{model_name}"
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]

        try:
            print(f"📥 تحميل نموذج Whisper: {model_name}")

            # التحقق من أن النموذج متاح
            available_models = MODEL_CONFIG["available_whisper_models"]
            if model_name not in available_models:
                print(f"⚠️ النموذج {model_name} غير متاح، استخدام 'base' بدلاً منه")
                model_name = "base"

            # تحميل النموذج مع استخدام float32
            model = whisper.load_model(
                model_name,
                download_root=str(self.models_dir / "whisper"),
                device=self.device
            )

            # استخدام float32 بدلاً من half لتجنب المشاكل
            model = model.float()

            self.model_cache[cache_key] = model
            print(f"✅ تم تحميل Whisper ({model_name}) بنجاح على {self.device}")
            return model

        except Exception as e:
            print(f"❌ خطأ في تحميل Whisper: {e}")
            return None

    def load_qwen2_vl_model(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
        """تحميل نموذج Qwen2-VL لتوليد الوصف"""
        cache_key = f"qwen2_vl_{model_name}"
        if cache_key in self.model_cache:
            logger.info(f"✅ تم العثور على نموذج Qwen2-VL ({model_name}) في الذاكرة المؤقتة.")
            return self.model_cache[cache_key]
        try:
            logger.info(f"📥 جاري تحميل نموذج Qwen2-VL: {model_name}")
            # استخدام AutoProcessor و AutoModelForVision2Seq
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto")
            model.to( self.device )
            # default processer
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
            self.model_cache[cache_key] = (processor, model)
            logger.info(f"✅ تم تحميل نموذج Qwen2-VL: {model_name} بنجاح على {self.device}")
            return processor, model
        except Exception as e:
            logger.error(f"❌ فشل تحميل نموذج Qwen2-VL: {e}")
            return None, None


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
            # مسح جميع النماذج
            for key in list(self.model_cache.keys()):
                self.clear_model_cache(key) # استخدام الدالة نفسها لتفريغ كل نموذج على حدة
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
        """تحميل نموذج Qwen2-VL فقط لتحليل المشهد/النشاط."""
        qwen2_vl_processor, qwen2_vl_model = self.load_qwen2_vl_model()
        # نُرجع قيمتين فقط الآن
        return qwen2_vl_processor, qwen2_vl_model

    def cleanup(self):
        """تنظيف جميع الموارد"""
        self.model_cache.clear()
        if GPU_AVAILABLE:
            torch.cuda.empty_cache()
        print("🧹 تم تنظيف جميع موارد ModelLoader")


def load_text_recognition_model():
    """تحميل نموذج التعرف على النص"""
    # سيتم التعامل مع EasyOCR في مكان آخر
    return None


def load_speech_recognition_model():
    """تحميل نموذج التعرف على الكلام"""
    return model_loader.load_whisper_model()


# إنشاء loader عالمي
model_loader = ModelLoader()

def cleanup_models():
    """تنظيف جميع النماذج من الذاكرة"""
    model_loader.cleanup()

