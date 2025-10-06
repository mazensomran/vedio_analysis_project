import torch
import gc


def setup_gpu():
    """إعداد GPU وتحسين الأداء"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🎯 GPU متاح: {torch.cuda.get_device_name(0)}")
        print(f"🎯 ذاكرة GPU الإجمالية: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

        # تحسين إعدادات الذاكرة
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        return device
    else:
        print("⚠️  GPU غير متاح، سيتم استخدام CPU")
        return torch.device("cpu")


def clear_gpu_memory():
    """تنظيف ذاكرة GPU"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def get_gpu_memory_usage():
    """الحصول على استخدام ذاكرة GPU"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        return allocated, reserved
    return 0, 0


def print_gpu_memory_usage():
    """طباعة استخدام ذاكرة GPU"""
    if torch.cuda.is_available():
        allocated, reserved = get_gpu_memory_usage()
        print(f"💾 ذاكرة GPU المستخدمة: {allocated:.2f} GB / {reserved:.2f} GB محجوزة")
        
