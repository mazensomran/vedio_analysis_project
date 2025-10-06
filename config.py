import os
import torch
from pathlib import Path
from typing import List, Dict, Any

# المسارات الأساسية
BASE_DIR = Path(__file__).resolve().parent

WORKING_DIR = Path("/kaggle/working/") # تعريف مجلد العمل الجديد
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models" # يمكن أن يكون هذا داخل WORKING_DIR أو يبقى في BASE_DIR إذا كانت النماذج محملة مسبقًا
DATABASE_DIR = BASE_DIR / "database"
LOGS_DIR = BASE_DIR / "logs"

# إنشاء المجلدات إذا لم تكن موجودة
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
DATABASE_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# التحقق من توفر GPU
GPU_AVAILABLE = torch.cuda.is_available()
GPU_COUNT = torch.cuda.device_count() if GPU_AVAILABLE else 0
GPU_NAME = torch.cuda.get_device_name(0) if GPU_AVAILABLE else "None"

print(f"🎯 حالة GPU: {'متاح' if GPU_AVAILABLE else 'غير متاح'}")
if GPU_AVAILABLE:
    print(f"🎯 عدد أجهزة GPU: {GPU_COUNT}")
    print(f"🎯 اسم GPU: {GPU_NAME}")

# إعدادات التطبيق
APP_CONFIG = {
    "host": "127.0.0.1",
    "port": 8000,
    "debug": True,
    "max_file_size": 100 * 1024 * 1024,  # 100MB
    "allowed_extensions": [".mp4", ".avi", ".mov", ".mkv", ".webm"],
    "max_video_length": 1800,  # 30 دقيقة كحد أقصى للفيديو
    "max_concurrent_processes": 2,  # الحد الأقصى للعمليات المتزامنة
    "temp_file_cleanup_hours": 24,  # تنظيف الملفات المؤقتة بعد 24 ساعة
}

# إعدادات المعالجة
PROCESSING_CONFIG = {
    # إعدادات عامة
    "target_width": 640,
    "target_height": 360,
    "target_fps": 15,
    "enable_fast_processing": True,
    "max_workers": 2 if GPU_AVAILABLE else 1,
    "frame_sampling_interval": 2,  # أخذ عينة كل 10 إطارات
    "batch_size": 4 if GPU_AVAILABLE else 2,  # حجم الدفعة للمعالجة

    # إعدادات كشف الوجوه
    "face_detection_threshold": 0.3,
    "min_face_size": 20,  # الحد الأدنى لحجم الوجه (بكسل)
    "max_face_size": 300,  # الحد الأقصى لحجم الوجه (بكسل)
    "face_aspect_ratio_min": 0.7,  # نسبة العرض إلى الارتفاع الدنيا للوجه
    "face_aspect_ratio_max": 1.4,  # نسبة العرض إلى الارتفاع القصوى للوجه
    "target_face_width": 128,  # العرض المستهدف للوجه المستخرج
    "target_face_height": 128, # الارتفاع المستهدف للوجه المستخرج
    "face_padding_ratio": 0.2, # نسبة الهامش حول الوجه قبل القص
    "enable_face_enhancement": False,  # تمكين تحسين دقة الوجوه
    "min_face_size_for_enhancement": 50,  # الحد الأدنى لحجم الوجه للتحسين

    # إعدادات كشف الأشخاص
    "person_detection_threshold": 0.5,
    "min_person_size": 50,  # الحد الأدنى لحجم الشخص (بكسل)

    # إعدادات كشف النص
    "text_detection_threshold": 0.5,
    "text_detection_enabled": True,
    "text_recognition_enabled": True,
    "min_text_confidence": 0.3,  # الحد الأدنى لثقة التعرف على النص

    # إعدادات التتبع
    "tracking_threshold": 0.5,
    "max_tracking_distance": 100,  # المسافة القصوى للتتبع بين الإطارات (بكسل)
    "min_track_length": 5,  # الحد الأدنى لطول المسار لاعتباره صالحاً

    # إعدادات التعرف على النشاط
    "activity_recognition_interval": 20,  # تحليل النشاط كل 30 إطار
    "activity_history_size": 100,  # عدد التحليلات السابقة لتخزينها

    # إعدادات إدارة الأخطاء
    "max_404_retries": 20,  # الحد الأقصى لمحاولات 404 قبل الإيقاف
    "max_processing_time": 3600,  # الحد الأقصى لزمن المعالجة (ثانية)
    "retry_delay_seconds": 2,  # وقت الانتظار بين المحاولات الفاشلة

    # إعدادات الذاكرة والأداء
    "gpu_enabled": GPU_AVAILABLE,
    "gpu_memory_limit": 0.8 if GPU_AVAILABLE else 0,  # استخدام 80% من ذاكرة GPU
    "cpu_usage_limit": 0.7,  # الحد الأقصى لاستخدام CPU
    "memory_cleanup_interval": 50,  # تنظيف الذاكرة كل 50 إطار

    # إعدادات الجودة
    "video_quality": 23,  # جودة الفيديو المحول (كلما قل الرقم زادت الجودة)
    "image_quality": 85,  # جودة الصور المحفوظة
}

# إعدادات النماذج
MODEL_CONFIG = {
    # نماذج كشف الوجوه
    "face_detection_model": "yolov8m-face.pt",
    "scrfd_model_path": "sscrfd.onnx",
    "available_face_models": [
        "yolov8n-face.pt",
        "yolov8s-face.pt",
        "yolov8m-face.pt",
        "yolov8l-face.pt"
    ],

    # نماذج كشف الأشخاص
    "person_detection_model": "yolov8s.pt",
    "available_person_models": [
        "yolov8n.pt",
        "yolov8s.pt",
        "yolov8m.pt",
        "yolov8l.pt",
        "yolov8x.pt"
    ],

    # نماذج التعرف على النص
    "text_detection_model": "easyocr",
    "text_recognition_model": "easyocr",
    "easyocr_languages": ["ar", "en"],

    # نماذج التعرف على الكلام
    "speech_recognition_model": "base",
    "available_whisper_models": [
        "tiny", "tiny.en",
        "base", "base.en",
        "small", "small.en",
        "medium", "medium.en",
        "large", "large-v1", "large-v2", "large-v3"
    ],

    # نماذج التعرف على المشهد والنشاط
    "scene_recognition_model": "Salesforce/blip-image-captioning-base",
    "activity_recognition_model": "Salesforce/blip-image-captioning-base",
    "available_scene_models": [
        "Salesforce/blip-image-captioning-base",
        "google/vit-base-patch16-224",
        "facebook/convnext-tiny-224"
    ],

    # نماذج التتبع
    "tracking_model": "bytetrack",
    "available_tracking_models": ["bytetrack", "deepsort", "sort"],

    "face_enhancement_model": "EDSR_x2.pt",

    # إعدادات الجهاز
    "device": "cuda" if GPU_AVAILABLE else "cpu",
    "precision": "fp32", # <--- CHANGE THIS LINE FROM "fp16" to "fp32"
    "optimize_for_inference": True,

    # إعدادات التنزيل والتخزين
    "model_download_timeout": 300,  # 5 دقائق
    "model_retry_attempts": 3,
    "model_cache_size": 1024 * 1024 * 1024,  # 1GB

# نماذج التعرف على النشاط (Video Swin Transformer)
    "video_activity_model": "microsoft/video-swin-tiny-patch4-window7-224", # نموذج صغير وسريع للبدء
    "available_video_activity_models": [
        "microsoft/video-swin-tiny-patch4-window7-224",
        "microsoft/video-swin-base-patch4-window7-224"]
}

# إعدادات EasyOCR
EASYOCR_CONFIG = {
    "gpu_enabled": GPU_AVAILABLE,
    "model_storage_directory": str(MODELS_DIR / "easyocr"),
    "download_enabled": True,
    "recog_network": "standard",
    "detector": True,  # ✅ استخدام detector بدلاً من detector_enabled
    "recognizer": True, # ✅ استخدام recognizer بدلاً من recognizer_enabled
    "batch_size": 10,
    "model_precision": "fp16", # if GPU_AVAILABLE else "fp32",  # ✅ استخدام fp32 بدلاً من fp16
    "detector_threshold": 0.3,
    "recognizer_threshold": 0.3,
    "text_min_size": 10,
    "text_max_size": 500,
    "paragraph": False,
    "detail": 1,
}

# إعدادات قاعدة البيانات
DATABASE_CONFIG = {
    "db_path": DATABASE_DIR / "video_analysis.db",
    "backup_path": DATABASE_DIR / "backups",
    "max_backups": 10,
    "backup_interval_hours": 24,
    "tables": {
        "processes": "processes",
        "faces": "faces",
        "texts": "texts",
        "transcriptions": "transcriptions",
        "tracking": "tracking",
        "scenes": "scenes",
        "activities": "activities",
        "errors": "errors"
    },
    "connection_timeout": 30,
    "journal_mode": "WAL",
    "cache_size": -2000,  # 2MB
}

# إعدادات التسجيل (Logging)
LOGGING_CONFIG = {
    "log_level": "INFO",
    "log_file": LOGS_DIR / "video_analysis.log",
    "max_log_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5,
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
}

# إعدادات الأمان
SECURITY_CONFIG = {
    "cors_origins": ["*"],
    "rate_limiting_enabled": True,
    "max_requests_per_minute": 60,
    "api_key_required": False,
    "allowed_file_types": ["video/mp4", "video/avi", "video/quicktime", "video/x-msvideo"],
    "max_filename_length": 255,
    "sanitize_filenames": True,
}

# إعدادات الأداء والتحسين
PERFORMANCE_CONFIG = {
    "thread_pool_size": 4,
    "io_timeout_seconds": 30,
    "max_retry_attempts": 3,
    "cache_enabled": True,
    "cache_size": 100,  # عدد العناصر المخزنة
    "cache_ttl_seconds": 300,  # 5 دقائق
    "compression_enabled": True,
    "preload_models": False,  # تحميل النماذج مسبقاً عند التشغيل
}

# إعدادات معالجة الصوت
AUDIO_CONFIG = {
    "sample_rate": 16000,
    "channels": 1,
    "audio_format": "mp3",
    "audio_bitrate": "64k",
    "silence_threshold": 0.01,
    "min_audio_length": 1.0,  # ثانية
    "max_audio_length": 300.0,  # ثانية
}

# إعدادات معالجة الفيديو
VIDEO_CONFIG = {
    "codec": "libx264",
    "preset": "medium",
    "crf": 23,
    "pix_fmt": "yuv420p",
    "keyframe_interval": 30,
    "buffer_size": 10*1024 * 1024,  # 1MB
    "threads": 2,
}

# إعدادات الترجمة واللغات
LANGUAGE_CONFIG = {
    "supported_languages": ["ar", "en"],
    "default_language": "ar",
    "auto_detect_language": True,
    "translation_enabled": False,
    "language_detection_confidence": 0.7,
}


# إنشاء المجلدات الإضافية
def setup_directories():
    """إعداد جميع المجلدات المطلوبة"""
    directories = [
        EASYOCR_CONFIG["model_storage_directory"],
        DATABASE_CONFIG["backup_path"],
        OUTPUTS_DIR / "temp",
        OUTPUTS_DIR / "cache",
        LOGS_DIR / "archived",
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print("✅ تم إعداد جميع المجلدات بنجاح")


# وظائف مساعدة
def get_app_url() -> str:
    """الحصول على رابط التطبيق"""
    return f"http://{APP_CONFIG['host']}:{APP_CONFIG['port']}"


def get_model_path(model_name: str) -> Path:
    """الحصول على المسار الكامل للنموذج"""
    return MODELS_DIR / model_name


def get_output_path(process_id: str, filename: str = "") -> Path:
    """الحصول على مسار المخرجات للعملية"""
    process_dir = OUTPUTS_DIR / process_id
    if filename:
        return process_dir / filename
    return process_dir


def check_processing_config() -> bool:
    """التحقق من صحة إعدادات المعالجة"""
    try:
        # التحقق من وجود النماذج المطلوبة
        required_models = [
            MODEL_CONFIG["face_detection_model"],
            MODEL_CONFIG["person_detection_model"],
        ]

        print("🔍 التحقق من إعدادات المعالجة...")
        print(f"📁 مجلد النماذج: {MODELS_DIR}")
        print(f"🎯 GPU متاح: {GPU_AVAILABLE}")
        print(f"⚡ وضع المعالجة: {'سريع' if PROCESSING_CONFIG['enable_fast_processing'] else 'عادي'}")

        return True

    except Exception as e:
        print(f"❌ خطأ في التحقق من الإعدادات: {e}")
        return False


def get_available_memory() -> Dict[str, float]:
    """الحصول على معلومات الذاكرة المتاحة"""
    memory_info = {}

    if GPU_AVAILABLE:
        memory_info["gpu_total"] = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        memory_info["gpu_allocated"] = torch.cuda.memory_allocated() / 1024 ** 3
        memory_info["gpu_reserved"] = torch.cuda.memory_reserved() / 1024 ** 3
        memory_info["gpu_available"] = memory_info["gpu_total"] - memory_info["gpu_allocated"]

    return memory_info


def optimize_for_hardware():
    """تحسين الإعدادات بناءً على الهاردوير المتاح"""
    if not GPU_AVAILABLE:
        # تقليل الإعدادات عند استخدام CPU فقط
        PROCESSING_CONFIG["batch_size"] = max(1, PROCESSING_CONFIG["batch_size"] // 2)
        PROCESSING_CONFIG["max_workers"] = 1
        PROCESSING_CONFIG["gpu_memory_limit"] = 0
        MODEL_CONFIG["precision"] = "fp32"

        print("⚠️  تم تقليل إعدادات الأداء لاستخدام CPU فقط")


# تنفيذ الإعدادات
setup_directories()
optimize_for_hardware()

# طباعة معلومات التهيئة
print("🎯 إعدادات نظام تحليل الفيديو:")
print(f"🌐 عنوان التطبيق: {get_app_url()}")
print(f"📁 مجلد التحميلات: {UPLOAD_DIR}")
print(f"📁 مجلد المخرجات: {OUTPUTS_DIR}")
print(
    f"📊 ذاكرة GPU المتاحة: {get_available_memory().get('gpu_available', 0):.2f} GB" if GPU_AVAILABLE else "📊 لا يوجد GPU")

# التحقق من الإعدادات
if not check_processing_config():
    print("⚠️  هناك مشاكل في إعدادات المعالجة، قد يؤثر على الأداء")

print("✅ تم تحميل إعدادات التطبيق بنجاح")
