import cv2
import numpy as np
import torch
from typing import Dict, Any, List
from collections import Counter
import logging
from model_loader import ModelLoader
from translation_utils import MarianTranslator
from models import VideoEnhancer
from PIL import Image
from qwen_vl_utils import process_vision_info
import traceback

logger = logging.getLogger(__name__)

model_loader = ModelLoader()
translator = MarianTranslator()  # تهيئة المترجم هنا
video_enhancer = VideoEnhancer()

def move_to_device(obj, device):
    """نقل كل tensors داخل dict/list/tensor إلى نفس الجهاز"""
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    else:
        return obj


class ActivityRecognizer:

    def __init__(self):
        try:
            # تحميل نموذج Qwen2-VL فقط
            self.qwen2_vl_proc, self.qwen2_vl_model = model_loader.load_qwen2_vl_model()
            if self.qwen2_vl_model is None:
                raise Exception("فشل تحميل نموذج Qwen2-VL.")

            # device=torch.device("cuda:0")
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # self.device =device  #next(self.qwen2_vl_model.parameters()).device
            self.qwen2_vl_model = self.qwen2_vl_model.to(self.device)
            # self.qwen2_vl_model.parameters().to(device)
            self.qwen2_vl_model.eval()
            print(f"✅ تم تهيئة ActivityRecognizer بنجاح بنموذج Qwen2-VL على {self.device}.")
        except Exception as e:
            self.qwen2_vl_model = None
            self.device = None
            print(f"❌ فشل تهيئة ActivityRecognizer. فشل تحميل النماذج: {e}   traceback  {traceback.format_exc()}")

    def recognize_activity(self, prompt: str, video_path: str, fsp: float, pixels_size: int,
                           max_new_tokens: int = 600, temperature: float = 0.3,
                           top_p: float = 0.9, top_k: int = 50, do_sample: bool = True,
                           enable_enhancement: bool = False, enhancement_strength: int = 2):

        # معالجة القيم None
        max_new_tokens = max_new_tokens if max_new_tokens is not None else 600
        temperature = temperature if temperature is not None else 0.3
        top_p = top_p if top_p is not None else 0.9
        top_k = top_k if top_k is not None else 50
        do_sample = do_sample if do_sample is not None else True

        print("✅ معاملات النموذج بعد المعالجة:")
        print(
            f"max_new_tokens={max_new_tokens}, temperature={temperature}, top_p={top_p}, top_k={top_k}, do_sample={do_sample}")
        print(f"enable_enhancement={enable_enhancement}, enhancement_strength={enhancement_strength}")

        # تحسين الفيديو إذا كان مطلوباً
        final_video_path = video_path
        if enable_enhancement:
            print("🎨 تفعيل تحسين جودة الفيديو...")
            final_video_path = video_enhancer.enhance_video(video_path, enhancement_strength)
            print(f"📹 مسار الفيديو النهائي: {final_video_path}")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,  # مسار الفيديو الكامل
                        "max_pixels": pixels_size * pixels_size,  # تقليل الدقة لتوفير الذاكرة
                        "fps": fsp,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # تفريغ ذاكرة قبل الاستدلال
        torch.cuda.empty_cache()
        # Preparation for inference
        text = self.qwen2_vl_proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.qwen2_vl_proc(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # inputs = move_to_device(inputs, self.device)
        # device = next(model.parameters()).device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # inputs = inputs.to(self.device)
        # Inference مع no_grad لتوفير الذاكرة
        with torch.no_grad():
            generated_ids = self.qwen2_vl_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
            )


        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        # تفريغ فوري بعد الاستدلال
        del inputs  # حذف المدخلات
        torch.cuda.empty_cache()
        # Decode the generated text
        output_text = self.qwen2_vl_proc.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def cleanup(self):
        """تنظيف الموارد"""
        self.qwen2_vl_proc = None
        self.qwen2_vl_model = None
        self.device = None
        logger.info("🧹 تم تنظيف موارد ActivityRecognizer")
