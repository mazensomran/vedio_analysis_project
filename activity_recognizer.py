import cv2
import numpy as np
import torch
from typing import Dict, Any, List
from collections import Counter
import logging
from model_loader import ModelLoader
from translation_utils import MarianTranslator # تأكد من وجود هذا الاستيراد
from PIL import Image

logger = logging.getLogger(__name__)

model_loader = ModelLoader()
translator = MarianTranslator() # تهيئة المترجم هنا


class ActivityRecognizer:
    # عدد الإطارات المطلوب لتشغيل نموذج VideoMAE (حجم النافذة)
    FRAME_SEQUENCE_LENGTH = 16
    # خطوة التقدم للنافذة المنزلقة (عدد الإطارات التي تتحركها النافذة في كل مرة)
    # قيمة 4 تعني تداخل 12 إطارًا (16 - 4)
    SLIDING_WINDOW_STRIDE = 8

    def __init__(self):
        try:
            # تحميل كلا النموذجين من model_loader
            self.videomae_proc, self.videomae_model, self.blip_proc, self.blip_model = model_loader.load_scene_recognition_model()

            if self.videomae_model is None or self.blip_model is None:
                raise Exception("فشل تحميل أحد النماذج.")

            self.device = next(self.videomae_model.parameters()).device
            self.videomae_model.eval()
            self.blip_model.eval()

            # المخزن المؤقت للإطارات لتكوين تسلسل
            self.frame_buffer = []
            # عداد للإطارات التي تم إضافتها إلى المخزن المؤقت منذ آخر تحليل
            self.frames_since_last_analysis = 0

            print("✅ تم تهيئة ActivityRecognizer بنجاح بنموذجي VideoMAE و BLIP")
        except Exception as e:
            self.videomae_model = self.blip_model = None
            self.device = None
            print(f"❌ فشل تهيئة ActivityRecognizer. فشل تحميل النماذج: {e}")

        # تاريخ المشاهد للتحليل التراكمي
        self.scene_history = []
        self.max_history = 50

        # إحصائيات النشاط
        self.activity_stats = {
            "total_frames_processed": 0, # إجمالي الإطارات التي مرت على recognize_activity
            "successful_detections": 0,  # عدد مرات التحليل الناجح (نافذة كاملة)
            "failed_detections": 0       # عدد مرات فشل التحليل (نافذة كاملة)
        }

    def _analyze_blip(self, frame_pil: Image.Image) -> Dict[str, Any]:
        """تحليل إطار واحد بواسطة نموذج BLIP لتوليد وصف نصي."""
        if self.blip_model is None:
            return {"description": "النموذج غير متاح", "confidence": 0.0, "description_ar": "النموذج غير متاح"}

        try:
            inputs = self.blip_proc(images=frame_pil, text="A video of", return_tensors="pt").to(self.device)
            out = self.blip_model.generate(**inputs, max_length=50)
            description_en = self.blip_proc.decode(out[0], skip_special_tokens=True).strip()
            confidence = 1.0  # BLIP لا يعطي قيمة ثقة مباشرة
            description_ar = translator.translate(description_en) if translator else description_en

            return {"description": description_en, "confidence": confidence, "description_ar": description_ar}
        except Exception as e:
            logger.error(f"❌ خطأ في تحليل BLIP: {e}")
            return {"description": "فشل توليد الوصف", "confidence": 0.0, "description_ar": "فشل توليد الوصف"}

    def _analyze_videomae(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """تحليل تسلسل من الإطارات بواسطة VideoMAE لتصنيف النشاط."""
        if self.videomae_model is None:
            return {"activity": "unknown", "confidence": 0.0, "activity_ar": "غير معروف"}

        try:
            # تأكد أن عدد الإطارات في التسلسل يطابق FRAME_SEQUENCE_LENGTH
            if len(frames) != self.FRAME_SEQUENCE_LENGTH:
                logger.warning(f"⚠️ عدد الإطارات لتسلسل VideoMAE غير متطابق. المتوقع: {self.FRAME_SEQUENCE_LENGTH}, الفعلي: {len(frames)}")

            frames_pil = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
            inputs = self.videomae_proc(frames_pil, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.videomae_model(**inputs)

            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            confidence = torch.nn.functional.softmax(logits, dim=-1)[0, predicted_class_idx].item()
            activity_label_en = self.videomae_model.config.id2label[predicted_class_idx]
            activity_label_ar = translator.translate(activity_label_en) if translator else activity_label_en

            return {"activity": activity_label_en, "confidence": round(confidence, 4), "activity_ar": activity_label_ar}
        except Exception as e:
            logger.error(f"❌ خطأ في تحليل VideoMAE: {e}")
            return {"activity": "فشل التحليل", "confidence": 0.0, "activity_ar": "فشل التحليل"}

    def recognize_activity(self, frame: np.ndarray, frame_number: int) -> Dict[str, Any]:
        """
        تقوم هذه الدالة بدمج الإطارات في مخزن مؤقت (buffer) وتطبيق النافذة المنزلقة.
        عندما يكتمل المخزن، تقوم بتحليل النشاط والبيئة وتُرجع النتائج.
        """
        self.activity_stats["total_frames_processed"] += 1

        if self.videomae_model is None or self.blip_model is None:
            # لا نزيد failed_detections هنا لأن هذا ليس فشل تحليل نافذة كاملة
            return {
                "activity": "unknown",
                "activity_ar": "غير معروف",
                "description": "النماذج غير متاحة",
                "description_ar": "النماذج غير متاحة",
                "confidence": 0.0,
                "frame_number": frame_number,
                "status": "error"
            }

        # إضافة الإطار الحالي إلى المخزن المؤقت
        self.frame_buffer.append(frame)
        self.frames_since_last_analysis += 1

        # التحليل فقط عندما يكون المخزن المؤقت كبيرًا بما يكفي (حجم النافذة)
        # وعندما نكون قد تقدمنا بما يكفي (خطوة التقدم)
        if len(self.frame_buffer) >= self.FRAME_SEQUENCE_LENGTH and \
           self.frames_since_last_analysis >= self.SLIDING_WINDOW_STRIDE:

            # 1. تحليل النشاط بواسطة VideoMAE على النافذة الحالية
            # نأخذ آخر FRAME_SEQUENCE_LENGTH إطارًا من المخزن المؤقت
            current_window_frames = self.frame_buffer[-self.FRAME_SEQUENCE_LENGTH:]
            videomae_result = self._analyze_videomae(current_window_frames)

            # 2. تحليل البيئة بواسطة BLIP (على الإطار الأوسط أو الأخير من النافذة)
            blip_frame = current_window_frames[self.FRAME_SEQUENCE_LENGTH // 2] # الإطار الاوسط في النافذة
            blip_result = self._analyze_blip(Image.fromarray(cv2.cvtColor(blip_frame, cv2.COLOR_BGR2RGB)))

            # دمج النتائج
            combined_result = {
                "activity": videomae_result.get("activity", "unknown"),
                "activity_ar": videomae_result.get("activity_ar", "غير معروف"),
                "confidence": videomae_result.get("confidence", 0.0),
                "description": blip_result.get("description", "لا يوجد وصف")  ,
                "description_ar": blip_result.get("description_ar", "لا يوجد وصف"),
                "frame_number": frame_number, # رقم الإطار الحالي (نهاية النافذة)
                "status": "success"
            }

            # تحديث السجل التاريخي
            self.scene_history.append(combined_result)
            if len(self.scene_history) > self.max_history:
                self.scene_history.pop(0)

            self.activity_stats["successful_detections"] += 1

            # تحريك النافذة: إزالة الإطارات القديمة من المخزن المؤقت
            # نحتفظ فقط بالإطارات اللازمة للنافذة التالية
            # إذا كان المخزن المؤقت أكبر من FRAME_SEQUENCE_LENGTH، نقوم بقصه
            # هذا يضمن أن المخزن المؤقت لا ينمو بلا حدود
            if len(self.frame_buffer) > self.FRAME_SEQUENCE_LENGTH:
                # نحذف الإطارات الزائدة من البداية
                self.frame_buffer = self.frame_buffer[len(self.frame_buffer) - self.FRAME_SEQUENCE_LENGTH:]

            # إعادة تعيين عداد الإطارات منذ آخر تحليل
            self.frames_since_last_analysis = 0

            return combined_result

        else:
            # إذا لم يكتمل التسلسل بعد أو لم نصل لخطوة التقدم، ارجع نتيجة "pending"
            # يمكننا هنا إرجاع حالة المخزن المؤقت
            return {
                "activity": "pending",
                "activity_ar": "جاري",
                "confidence": 0.0,
                "description": f"جاري جمع الإطارات ({len(self.frame_buffer)}/{self.FRAME_SEQUENCE_LENGTH})",
                "description_ar": f"جاري جمع الإطارات ({len(self.frame_buffer)}/{self.FRAME_SEQUENCE_LENGTH})",
                "frame_number": frame_number,
                "status": "pending"
            }

    def get_dominant_activity(self) -> Dict[str, Any]:
        """تحديد النشاط السائد والوصف السائد من السجل التاريخي."""
        if not self.scene_history:
            return {
                "dominant_activity": "لا يوجد تحليل",
                "dominant_activity_ar": "لا يوجد تحليل",
                "dominant_description": "لا يوجد وصف",
                "dominant_description_ar": "لا يوجد وصف",
                "top_activities": [],
                "top_activities_ar": [],
                "top_descriptions": [],
                "top_descriptions_ar": [],
                "total_samples": 0
            }

        all_activities_en = [item["activity"] for item in self.scene_history if "activity" in item and item["activity"] != "pending"]
        all_activities_ar = [item.get("activity_ar", item["activity"]) for item in self.scene_history if "activity" in item and item["activity"] != "pending"]
        all_descriptions_en = [item["description"] for item in self.scene_history if "description" in item and item["description"] != "لا يوجد وصف"]
        all_descriptions_ar = [item.get("description_ar", item["description"]) for item in self.scene_history if "description" in item and item["description"] != "لا يوجد وصف"]

        activity_counts_en = Counter(all_activities_en)
        activity_counts_ar = Counter(all_activities_ar)
        description_counts_en = Counter(all_descriptions_en)
        description_counts_ar = Counter(all_descriptions_ar)

        most_common_activities_en = activity_counts_en.most_common(5)
        most_common_activities_ar = activity_counts_ar.most_common(5)
        most_common_descriptions_en = description_counts_en.most_common(5)
        most_common_descriptions_ar = description_counts_ar.most_common(5)

        dominant_activity_en = most_common_activities_en[0][0] if most_common_activities_en else "unknown"
        dominant_activity_ar = most_common_activities_ar[0][0] if most_common_activities_ar else "غير محدد"
        dominant_description_en = most_common_descriptions_en[0][0] if most_common_descriptions_en else "no description"
        dominant_description_ar = most_common_descriptions_ar[0][0] if most_common_descriptions_ar else "لا يوجد وصف"

        return {
            "dominant_activity": dominant_activity_en,
            "dominant_activity_ar": dominant_activity_ar,
            "dominant_description": dominant_description_en,
            "dominant_description_ar": dominant_description_ar,
            "top_activities": most_common_activities_en,
            "top_activities_ar": most_common_activities_ar,
            "top_descriptions": most_common_descriptions_en,
            "top_descriptions_ar": most_common_descriptions_ar,
            "total_samples": len(self.scene_history)
        }

    def get_activity_statistics(self) -> Dict[str, Any]:
        """الحصول على إحصائيات النشاط"""
        success_rate = (self.activity_stats["successful_detections"] /
                        self.activity_stats["total_frames_processed"]) * 100 if self.activity_stats[
                                                                                    "total_frames_processed"] > 0 else 0

        return {
            "total_frames_processed": self.activity_stats["total_frames_processed"],
            "successful_detections": self.activity_stats["successful_detections"],
            "failed_detections": self.activity_stats["failed_detections"],
            "success_rate_percentage": round(success_rate, 2),
            "current_history_size": len(self.scene_history),
            "max_history_size": self.max_history
        }

    def get_current_activity_analysis(self, frame: np.ndarray, frame_number: int) -> Dict[str, Any]:
        """الحصول على التحليل الحالي للنشاط"""
        # هذه الدالة ستستدعي recognize_activity التي تقوم بالتحليل الفعلي
        activity_analysis_per_frame = self.recognize_activity(frame, frame_number)
        stats = self.get_activity_statistics()
        dominant_analysis = self.get_dominant_activity()
        return {
            "activity_analysis_per_frame": activity_analysis_per_frame,
            "dominant_activity_en": dominant_analysis['dominant_activity'],
            "dominant_activity_ar": dominant_analysis['dominant_activity_ar'],
            "dominant_description_en": dominant_analysis['dominant_description'],
            "dominant_description_ar": dominant_analysis['dominant_description_ar'],
            "top_activities_en": dominant_analysis["top_activities"],
            "top_activities_ar": dominant_analysis["top_activities_ar"],
            "top_descriptions_en": dominant_analysis["top_descriptions"],
            "top_descriptions_ar": dominant_analysis["top_descriptions_ar"],
            "statistics": stats,
            "recent_scenes": self.scene_history[-10:] if self.scene_history else []
        }

    def reset_history(self):
        """إعادة تعيين السجل التاريخي"""
        self.scene_history.clear()
        self.frame_buffer.clear() # مسح المخزن المؤقت أيضاً
        self.frames_since_last_analysis = 0
        self.activity_stats = { # إعادة تعيين الإحصائيات
            "total_frames_processed": 0,
            "successful_detections": 0,
            "failed_detections": 0
        }
        logger.info("🔄 تم إعادة تعيين سجل ActivityRecognizer")

    def cleanup(self):
        """تنظيف الموارد"""
        self.scene_history.clear()
        self.frame_buffer.clear()
        self.frames_since_last_analysis = 0
        self.videomae_proc = None
        self.videomae_model = None
        self.blip_proc = None
        self.blip_model = None
        self.device = None
        logger.info("🧹 تم تنظيف موارد ActivityRecognizer")

