
import cv2
import numpy as np
import torch
from typing import Dict, Any, List
from collections import Counter
import logging
from model_loader import ModelLoader
from translation_utils import MarianTranslator
#from qwen2_VL import Qwen2_VL # تأكد من وجود هذا الاستيراد
from PIL import Image
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)

model_loader = ModelLoader()
translator = MarianTranslator() # تهيئة المترجم هنا

class ActivityRecognizer:


    def __init__(self):
        try:
            # تحميل نموذج Qwen2-VL فقط
            self.qwen2_vl_proc, self.qwen2_vl_model = model_loader.load_qwen2_vl_model()

            if self.qwen2_vl_model is None:
                raise Exception("فشل تحميل نموذج Qwen2-VL.")

            self.device = next(self.qwen2_vl_model.parameters()).device
            self.qwen2_vl_model.eval()
            print(f"✅ تم تهيئة ActivityRecognizer بنجاح بنموذج Qwen2-VL.")
        except Exception as e:
            self.qwen2_vl_model = None
            self.device = None
            print(f"❌ فشل تهيئة ActivityRecognizer. فشل تحميل النماذج: {e}")


    def recognize_activity(self,prompt:str, video_path: str, fsp: float, pixels_size: int):
        """
        تقوم هذه الدالة بتحليل النشاط والبيئة باستخدام Qwen2-VL.
        يمكنها العمل على إطار واحد أو تجميع الإطارات في دفعات.
        """
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
        inputs = inputs.to(self.device) 
        # Inference مع no_grad لتوفير الذاكرة
        with torch.no_grad():
            generated_ids = self.qwen2_vl_model.generate( 
                **inputs,
                max_new_tokens=150
            )
        # Trim the generated output to remove the input prompt
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
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
