import cv2
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import subprocess
import numpy as np
from datetime import datetime
import torch
import time
import threading
from translation_utils import MarianTranslator
from config import OUTPUTS_DIR, PROCESSING_CONFIG, MODEL_CONFIG
from database import db
from models import FaceDetector, TextDetector, SpeechRecognizer, GeneralObjectDetector, ObjectTracker, FrameEnhancer
from activity_recognizer import ActivityRecognizer
from model_loader import model_loader
from monitoring import ProcessMonitor
import logging

logger = logging.getLogger(__name__)
monitor = ProcessMonitor()

stop_processing = False
current_process_id = None
translator = MarianTranslator()

TARGET_FACE_WIDTH = PROCESSING_CONFIG["target_face_width"]
TARGET_FACE_HEIGHT = PROCESSING_CONFIG["target_face_height"]
FACE_PADDING_RATIO = PROCESSING_CONFIG["face_padding_ratio"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

enhancer = FrameEnhancer(model_path='RealESRGAN_x4plus.pth')

def set_stop_processing(value: bool, process_id: Optional[str] = None):
    """تعيين حالة إيقاف المعالجة"""
    global stop_processing, current_process_id
    stop_processing = value
    if process_id:
        current_process_id = process_id


def get_stop_processing():
    """الحصول على حالة إيقاف المعالجة"""
    return stop_processing


def setup_output_dirs(process_id: str) -> Tuple[Path, Path, Path]:
    """إعداد المجلدات اللازمة للمعالجة"""
    process_dir = OUTPUTS_DIR / process_id
    faces_dir = process_dir / "faces"
    video_dir = process_dir / "video"
    audio_dir = process_dir / "audio"

    for directory in [process_dir, faces_dir, video_dir, audio_dir]:
        directory.mkdir(exist_ok=True, parents=True)

    return process_dir, faces_dir, video_dir

def extract_audio(video_path: str, output_audio_path: str) -> bool:
    """استخراج الصوت من الفيديو"""
    try:
        command = [
            'ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a',
            output_audio_path, '-y', '-loglevel', 'error'
        ]
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info(f"✅ تم استخراج الصوت بنجاح: {output_audio_path}")
            return True
        else:
            logger.info(f"❌ فشل في استخراج الصوت: {result.stderr}")
            return False

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"❌ خطأ في استخراج الصوت: {e}")
        return False


def find_track_id_for_bbox(bbox, tracks, iou_threshold=0.3):
    """
    تبحث عن track_id في قائمة tracks الذي يتطابق مع bbox بناءً على مقياس IoU.

    Returns:
        track_id المناسب إذا وجد، أو None إذا لم يتم العثور على تطابق.
    """

    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        unionArea = boxAArea + boxBArea - interArea
        if unionArea == 0:
            return 0
        return interArea / unionArea

    best_iou = 0
    best_track_id = None
    for track in tracks:
        track_bbox = track['bbox']
        current_iou = iou(bbox, track_bbox)
        if current_iou > best_iou and current_iou >= iou_threshold:
            best_iou = current_iou
            best_track_id = track['track_id']
    return best_track_id

def process_single_frame(frame: np.ndarray, frame_number: int,
                        face_detector: FaceDetector, text_detector: TextDetector,
                        object_detector: GeneralObjectDetector, object_tracker: ObjectTracker,
                        activity_recognizer: ActivityRecognizer,
                        options: Dict[str, Any], process_id: str, faces_dir: Path) -> Tuple[np.ndarray, List, List, List, List, Dict, List]:
    """معالجة إطار واحد مع كشف جميع الكائنات وتتبع الأشخاص"""

    processed_frame = frame.copy()
    all_objects = []
    all_faces = []
    all_texts = []
    all_tracks = []
    activity_data = {}
    persons_data = []

    # تحسين الإطار
    enhanced_img_pil = enhancer.enhance_frame(frame)
    enhanced_frame = cv2.cvtColor(np.array(enhanced_img_pil), cv2.COLOR_RGB2BGR)

    # كشف الوجوه
    if options.get("enable_face_detection", True) and face_detector:
        faces = face_detector.detect_faces(enhanced_frame, frame_number)
        all_faces.extend(faces)
        for j, face in enumerate(faces):
            bbox = face["bbox"]
            x, y, w, h = bbox

            # اضافة حشو للوجوه المكتشفة للحصول على نتائج افضل
            pad_x = int(w * FACE_PADDING_RATIO / 2)
            pad_y = int(h * FACE_PADDING_RATIO / 2)

            x1_padded = max(0, x - pad_x)
            y1_padded = max(0, y - pad_y)
            x2_padded = min(frame.shape[1], x + w + pad_x)
            y2_padded = min(frame.shape[0], y + h + pad_y)

            face_img = frame[y1_padded:y2_padded, x1_padded:x2_padded]

            if face_img.shape[0] > 0 and face_img.shape[1] > 0:
                current_height, current_width = face_img.shape[:2]
                if current_width < TARGET_FACE_WIDTH or current_height < TARGET_FACE_HEIGHT:
                    face_img = cv2.resize(face_img, (
                    max(current_width * 2, TARGET_FACE_WIDTH), max(current_height * 2, TARGET_FACE_HEIGHT)),
                                          interpolation=cv2.INTER_CUBIC)
                face_img_resized = cv2.resize(face_img, (TARGET_FACE_WIDTH, TARGET_FACE_HEIGHT),
                                              interpolation=cv2.INTER_AREA) #توحيد قياس صور الوجوه المكتشفة

                face_filename = f"face_{frame_number}_{j}.jpg"
                face_path = faces_dir / face_filename
                try:
                    cv2.imwrite(str(face_path), face_img_resized)
                    face["image_path"] = str(face_path.relative_to(OUTPUTS_DIR / process_id / "faces"))
                except Exception as e:
                    logger.error(f"❌ خطأ في حفظ صورة الوجه: {e}")
                    face["image_path"] = ""
            else:
                logger.info(f"⚠️ تم اكتشاف وجه فارغ في الإطار {frame_number}، لن يتم حفظه.")
                face["image_path"] = ""

    # كشف النصوص
    if options.get("enable_text_detection", True) and text_detector:
        texts = text_detector.detect_text(enhanced_frame, frame_number)
        all_texts.extend(texts)

    # كشف جميع الكائنات
    all_detections = []
    if object_detector:
        all_detections = object_detector.detect_objects(enhanced_frame, frame_number)

    # فصل الأشخاص وتتبعهم
    person_detections = [d for d in all_detections if d["class_name"] == "person"]

    tracks = []
    if options.get("enable_tracking", True) and object_tracker:
        tracks = object_tracker.track_objects(frame, person_detections)
        all_tracks.extend(tracks)

    # رسم جميع الكائنات
    for det in all_detections:
        bbox = det["bbox"]
        class_name = det["class_name"]
        confidence = det["confidence"]
        color = (0, 255, 0) if class_name != "person" else (255, 0, 0)
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(processed_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # رسم مسارات الأشخاص
    if all_tracks:
        processed_frame = object_tracker.draw_tracks(processed_frame, all_tracks)

    # تحليل النشاط (ActivityRecognizer يدير النافذة المنزلقة داخليًا)
    if options.get("enable_activity_recognition", True) and activity_recognizer:
        activity_data = activity_recognizer.recognize_activity(enhanced_frame, frame_number)

    # حفظ الكائنات مع ربط track_id للأشخاص
    for det in all_detections:
        track_id = None
        if det["class_name"] == "person":
            track_id = find_track_id_for_bbox(det["bbox"], all_tracks)
            bbox_to_save = det["bbox"].tolist()
            x1, y1, x2, y2 = bbox_to_save
            persons_data.append({
                "process_id": process_id,
                "frame_number": frame_number,
                "class_name": det["class_name"],
                "bbox_x1": float(x1),
                "bbox_y1": float(y1),
                "bbox_x2": float(x2),
                "bbox_y2": float(y2),
                "confidence": float(det["confidence"]),
                "track_id": track_id
            })
        else: # Other objects
            bbox_to_save = det["bbox"].tolist()
            x1, y1, x2, y2 = bbox_to_save
            all_objects.append({
                "process_id": process_id,
                "frame_number": frame_number,
                "class_name": det["class_name"],
                "bbox_x1": float(x1),
                "bbox_y1": float(y1),
                "bbox_x2": float(x2),
                "bbox_y2": float(y2),
                "confidence": float(det["confidence"]),
                "track_id": None # لا يوجد track_id للكائنات غير الأشخاص
            })

    return processed_frame, all_objects, all_faces, all_texts, all_tracks, activity_data, persons_data


def monitor_processing(process_id: str, total_frames: int, cap: cv2.VideoCapture):
    """مراقبة عملية المعالجة والتحقق من الأخطاء"""
    error_count = 0
    max_errors = PROCESSING_CONFIG.get("max_404_retries", 20)

    while not get_stop_processing():
        try:
            # التحقق من حالة العملية في قاعدة البيانات
            process_info = db.get_process_status(process_id)
            if not process_info or process_info["status"] in ["completed", "error", "stopped"]:
                break

            # التحقق من أن الفيديو لا يزال مفتوحاً
            if not cap.isOpened():
                error_count += 1
                print(f"⚠️ الفيديو مغلق - الخطأ رقم {error_count}")

            # إذا تجاوزت الأخطاء الحد المسموح
            if error_count >= max_errors:
                print(f"❌ تم إيقاف التحليل بسبب {max_errors} أخطاء متتالية")
                set_stop_processing(True, process_id)
                db.update_process_status(process_id, "error", 0,
                                         f"تم إيقاف التحليل بسبب {max_errors} أخطاء متتالية")
                break

            time.sleep(2)  # الانتظار لمدة ثانيتين بين كل فحص

        except Exception as e:
            print(f"❌ خطأ في المراقبة: {e}")
            error_count += 1
            time.sleep(2)

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj


def process_video(input_path: str, process_id: str, options: Dict[str, Any]):
    """معالجة الفيديو الرئيسية مع تحسينات GPU واكتشاف جميع الكائنات مع تتبع الأشخاص وتحسين دقة الوجوه"""
    start_time = time.time()

    global stop_processing

    # تهيئة المتغيرات
    text_detector = None
    face_detector = None
    speech_recognizer = None
    activity_recognizer = None
    object_tracker = None
    object_detector = None
    # face_enhancer = None # لم يعد يستخدم هنا بشكل مباشر

    cap = None
    out = None

    try:
        set_stop_processing(False, process_id)

        process_dir, faces_dir, video_dir = setup_output_dirs(process_id)

        db.update_process_status(process_id, "processing", 5, "جاري فتح الفيديو")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            db.update_process_status(process_id, "error", 0, "تعذر فتح ملف الفيديو")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        print(f"📊 معلومات الفيديو: {total_frames} إطار, {fps:.2f} FPS, {width}x{height}")
        print(f"⏱️ المدة: {duration:.2f} ثانية")

        monitor_thread = threading.Thread(
            target=monitor_processing,
            args=(process_id, total_frames, cap),
            daemon=True
        )
        monitor_thread.start()

        output_video_path = str(video_dir / "analyzed_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        db.update_process_status(process_id, "processing", 10, "جاري تحميل النماذج")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()

        face_detector = FaceDetector() if options.get("enable_face_detection", True) else None
        text_detector = TextDetector() if options.get("enable_text_detection", True) else None
        object_tracker = ObjectTracker() if options.get("enable_tracking", True) else None
        activity_recognizer = ActivityRecognizer() if options.get("enable_activity_recognition", True) else None
        speech_recognizer = SpeechRecognizer() if options.get("enable_audio_transcription", True) else None

        object_detector = GeneralObjectDetector(model_name="yolov8s.pt", device=MODEL_CONFIG["device"])

        db.update_process_status(process_id, "processing", 15, "تم تحميل النماذج بنجاح")

        transcription_result = None
        if options.get("enable_audio_transcription", True) and speech_recognizer:
            db.update_process_status(process_id, "processing", 20, "جاري استخراج الصوت")
            audio_path = str(process_dir / "audio" / "extracted_audio.mp3")

            if extract_audio(input_path, audio_path):
                db.update_process_status(process_id, "processing", 25, "جاري تحويل الصوت إلى نص")
                transcription_result = speech_recognizer.transcribe_audio(audio_path)

                if transcription_result and transcription_result["text"]:
                    db.add_transcription(process_id, transcription_result)
                    print(f"✅ تم تحويل الصوت إلى نص: {len(transcription_result['text'])} حرف")
                else:
                    print("⚠️ لم يتم العثور على نص في الصوت")
            else:
                print("⚠️ فشل في استخراج الصوت")

            speech_recognizer.cleanup()
            model_loader.clear_model_cache(f"whisper_{MODEL_CONFIG['speech_recognition_model']}")
            del speech_recognizer
            speech_recognizer = None

        frame_number = 0
        all_faces = []
        all_texts = []
        all_tracking_data = []
        all_activities = [] # لتخزين نتائج تحليل النشاط من recognize_activity
        all_objects = []
        all_people = []

        db.update_process_status(process_id, "processing", 30, "جاري معالجة إطارات الفيديو")

        while True:
            if get_stop_processing():
                print("⏹️ تم إيقاف المعالجة يدوياً")
                db.update_process_status(process_id, "stopped",
                                         int((frame_number / total_frames) * 65) + 30,
                                         "تم إيقاف التحليل يدوياً")
                break

            monitor.start_monitoring()
            ret, frame = cap.read()
            if not ret:
                break

            # معالجة إطار واحد في كل مرة
            processed_frame, current_objects, current_faces, current_texts, current_tracks, current_activity_data, current_persons = \
                process_single_frame(frame, frame_number, face_detector, text_detector,
                                     object_detector, object_tracker, activity_recognizer,
                                     options, process_id, faces_dir)

            out.write(processed_frame)

            all_objects.extend(current_objects)
            all_faces.extend(current_faces)
            all_texts.extend(current_texts)
            all_tracking_data.extend(current_tracks)
            all_people.extend(current_persons)

            # إضافة نتائج النشاط فقط إذا كانت ليست "pending"
            if current_activity_data and current_activity_data.get("status") == "success":
                all_activities.append(current_activity_data)

            frame_number += 1

            if frame_number % 10 == 0:
                progress = 30 + int((frame_number / total_frames) * 65)
                progress = min(progress, 95)
                db.update_process_status(process_id, "processing", progress,
                                         f"جاري معالجة الإطار {frame_number}/{total_frames}")

        # بعد انتهاء قراءة الإطارات، قد يكون هناك إطارات متبقية في المخزن المؤقت لـ ActivityRecognizer
        # ولكنها لا تشكل نافذة كاملة. لا نحتاج لمعالجتها هنا لأن ActivityRecognizer يتعامل معها.

        if get_stop_processing():
            print("💾 جاري حفظ النتائج الحالية بعد الإيقاف")
            final_progress = 30 + int((frame_number / total_frames) * 65)
            db.update_process_status(process_id, "stopped", final_progress,
                                     "تم إيقاف التحليل وحفظ النتائج الحالية")
        else:
            db.update_process_status(process_id, "processing", 95, "جاري حفظ النتائج النهائية")

        # إحصاءات الوجوه المحسنة (هذا الجزء لم يتغير)
        enhanced_faces = [face for face in all_faces if face.get("enhanced", False)]
        if enhanced_faces:
            print(f"✨ تم تحسين دقة {len(enhanced_faces)} وجه من أصل {len(all_faces)}")

        if all_faces:
            faces_output_path = process_dir / "faces_data.json"
            with open(faces_output_path, 'w', encoding='utf-8') as f:
                json.dump(all_faces, f, ensure_ascii=False, indent=2)
            print(f"✅ تم حفظ {len(all_faces)} وجه في ملف JSON")

        if all_texts:
            texts_output_path = process_dir / "texts_data.json"
            with open(texts_output_path, 'w', encoding='utf-8') as f:
                json.dump(all_texts, f, ensure_ascii=False, indent=2)
            print(f"✅ تم حفظ {len(all_texts)} نص في ملف JSON")

        if all_tracking_data:
            tracking_output_path = process_dir / "tracking_data.json"
            with open(tracking_output_path, 'w', encoding='utf-8') as f:
                json.dump(convert_numpy_types(all_tracking_data), f, ensure_ascii=False, indent=2)
            print(f"✅ تم حفظ {len(all_tracking_data)} عملية تتبع في ملف JSON")

        if all_objects:
            objects_output_path = process_dir / "objects_data.json"
            with open(objects_output_path, 'w', encoding='utf-8') as f:
                json.dump(convert_numpy_types(all_objects), f, ensure_ascii=False, indent=2)
            print(f"✅ تم حفظ {len(all_objects)} كائن في ملف JSON")

        objects = list(set(obj["class_name"] for obj in all_objects))
        objects_ar = [translator.translate(obj) for obj in objects]
        # الحصول على التحليل النهائي للنشاط من ActivityRecognizer
        # لا نحتاج لتمرير إطار وهمي هنا، فقط نطلب التحليل التراكمي
        final_activity_analysis = activity_recognizer.get_dominant_activity()
        activity_stats = activity_recognizer.get_activity_statistics()

        activity_output = {
            "dominant_activity_en": final_activity_analysis.get("dominant_activity"),
            "dominant_activity_ar": final_activity_analysis.get("dominant_activity_ar"),
            "dominant_description_en": final_activity_analysis.get("dominant_description"),
            "dominant_description_ar": final_activity_analysis.get("dominant_description_ar"),
            "top_activities_en": final_activity_analysis.get("top_activities"),
            "top_activities_ar": final_activity_analysis.get("top_activities_ar"),
            "top_descriptions_en": final_activity_analysis.get("top_descriptions"),
            "top_descriptions_ar": final_activity_analysis.get("top_descriptions_ar"),
            "statistics": activity_stats, # استخدام الإحصائيات المباشرة
            "recent_scenes": activity_recognizer.scene_history[-10:] if activity_recognizer.scene_history else [],
            "per_frame_results": all_activities # هذه هي النتائج التي تم الحصول عليها من كل نافذة مكتملة
        }

        unique_activities = list({item["activity"] for item in all_activities if "activity" in item})
        unique_descriptions = list({item["description"] for item in all_activities if "description" in item})

        # ترجمة الأنشطة الفريدة
        unique_activities_ar = [translator.translate(act) for act in unique_activities]
        # ترجمة التوصيفات الفريدة
        unique_descriptions_ar = [translator.translate(desc) for desc in unique_descriptions]

        activity_file = process_dir / "activity_analysis.json"
        with open(activity_file, "w", encoding="utf-8") as f:
            json.dump(activity_output, f, ensure_ascii=False, indent=2)
        print(f"✅ تم حفظ تحليل النشاط في {activity_file}")

        # حفظ النشاط السائد والوصف في قاعدة البيانات
        if activity_output.get("dominant_activity_en") and activity_output.get("dominant_description_en"):
            db.add_scene_analysis(
                process_id=process_id,
                scene_info={
                    "activity": activity_output["dominant_activity_en"],
                    "activity_ar": activity_output["dominant_activity_ar"],
                    "description": activity_output["dominant_description_en"],
                    "description_ar": activity_output["dominant_description_ar"],
                    "confidence": 1.0, # الثقة هنا هي للتحليل السائد، وليست ثقة النموذج الفردية
                    "frame_number": frame_number # هذا هو الإطار الأخير الذي تم معالجته
                }
            )

        end_time = time.time()
        processing_duration = end_time - start_time

        results = {
            "process_id": process_id,
            "total_frames": total_frames,
            "frames_processed": frame_number,
            "fps": fps,
            "resolution": f"{width}x{height}",
            "duration_seconds": duration,
            "faces_detected": len(all_faces),
            "faces_enhanced": len(enhanced_faces),
            "poeple_detected": len(all_people),
            "texts_detected": len(all_texts),
            "tracks_detected": len(set(track["track_id"] for track in all_tracking_data if "track_id" in track and track["track_id"] is not None)),
            "objects_detected": (len(objects), objects),
            "objects_ar":objects_ar,
            "transcription": transcription_result,
            "activity_analysis": {
            "dominant_activity_en": activity_output.get("dominant_activity_en") if activity_recognizer else None,
            "dominant_activity_ar": activity_output.get("dominant_activity_ar") if activity_recognizer else None,
            "dominant_description_en": activity_output.get("dominant_description_en") if activity_recognizer else None,
            "dominant_description_ar": activity_output.get("dominant_description_ar") if activity_recognizer else None,
            "top_activities": activity_output.get("top_activities_en") if activity_recognizer else [],
            "top_descriptions": activity_output.get("top_descriptions_en") if activity_recognizer else [],
            "top_activities_ar": activity_output.get("top_activities_ar") if activity_recognizer else [],
            "top_descriptions_ar": activity_output.get("top_descriptions_ar") if activity_recognizer else [],
            "statistics": activity_output.get("statistics") if activity_recognizer else {},
            "unique_activities": unique_activities,
            "unique_activities_ar": unique_activities_ar,
            "unique_descriptions": unique_descriptions,
            "unique_descriptions_ar": unique_descriptions_ar,
        },
            "processing_date": datetime.now().isoformat(),
            "processing_options": options,
            "face_enhancement_enabled": options.get("enable_face_enhancement", False),
            "processing_status": "stopped" if get_stop_processing() else "completed",
            "processing_duration_seconds": processing_duration

        }

        results_file = process_dir / "final_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✅ تم حفظ النتائج النهائية في {results_file}")

        if cap:
            cap.release()
        if out:
            out.release()

        model_loader.cleanup()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not get_stop_processing():
            db.update_process_status(process_id, "completed", 100, "تمت المعالجة بنجاح")
            monitor.remove_process(process_id)
            print("🎉 تمت معالجة الفيديو بنجاح!")

    except Exception as e:
        error_msg = f"خطأ في المعالجة: {str(e)}"
        monitor.report_error(process_id)
        print(f"❌ {error_msg}")

        try:
            current_progress = 30 + int((frame_number / total_frames) * 65) if 'frame_number' in locals() else 0
            db.update_process_status(process_id, "error", current_progress, error_msg)
        except:
            print("⚠️ لا يمكن تحديث حالة العملية في قاعدة البيانات")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model_loader.cleanup()

    finally:
        try:
            if cap and cap.isOpened():
                cap.release()
            if out:
                out.release()
            try:
                model_loader.cleanup()
            except Exception as e:
                print(f"⚠️ خطأ أثناء تنظيف الموارد: {e}")

            if text_detector and hasattr(text_detector, 'cleanup'):
                text_detector.cleanup()
            if face_detector and hasattr(face_detector, 'cleanup'):
                face_detector.cleanup()
            if object_tracker and hasattr(object_tracker, 'cleanup'):
                object_tracker.cleanup()
            if activity_recognizer and hasattr(activity_recognizer, 'cleanup'):
                activity_recognizer.cleanup()
            if speech_recognizer and hasattr(speech_recognizer, 'cleanup'):
                speech_recognizer.cleanup()

        except Exception as e:
            print(f"⚠️ خطأ في التنظيف: {e}")

        set_stop_processing(False, None)


def get_processing_status(process_id: str) -> Tuple[Dict[str, Any], str, str]:
    """الحصول على حالة المعالجة"""
    try:
        process_info = db.get_process_status(process_id)

        if not process_info:
            return {}, "not_found", "لم يتم العثور على العملية"

        # تحميل النتائج من ملف final_results.json
        process_dir = OUTPUTS_DIR / process_id
        results_file = process_dir / "final_results.json"

        results = {}
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)

        # تحديث الحالة والرسالة من قاعدة البيانات
        results["status"] = process_info["status"]
        results["message"] = process_info["message"]
        results["progress"] = process_info["progress"]

        # إضافة معلومات إضافية إذا كانت العملية مكتملة أو متوقفة
        if process_info["status"] in ["completed", "stopped"]:
            # تحميل بيانات الوجوه إذا كانت موجودة
            faces_file = process_dir / "faces_data.json"
            if faces_file.exists():
                with open(faces_file, 'r', encoding='utf-8') as f:
                    results["faces_data"] = json.load(f)
            else:
                results["faces_data"] = []

            # تحميل بيانات النصوص إذا كانت موجودة
            texts_file = process_dir / "texts_data.json"
            if texts_file.exists():
                with open(texts_file, 'r', encoding='utf-8') as f:
                    results["extracted_texts"] = json.load(f)
            else:
                results["extracted_texts"] = []

            # تحميل بيانات التتبع إذا كانت موجودة
            tracking_file = process_dir / "tracking_data.json"
            if tracking_file.exists():
                with open(tracking_file, 'r', encoding='utf-8') as f:
                    results["tracking_data"] = json.load(f)
            else:
                results["tracking_data"] = []

            # تحميل بيانات النشاط إذا كانت موجودة
            activity_file = process_dir / "activity_analysis.json"
            if activity_file.exists():
                with open(activity_file, 'r', encoding='utf-8') as f:
                    activity_data = json.load(f)
                    results["activity_analysis"] = activity_data # تم تغيير هذا ليعكس الهيكل الجديد
            else:
                results["activity_analysis"] = {}

            # التأكد من وجود حقول المدة وعدد الإطارات
            if "duration_seconds" not in results:
                results["duration_seconds"] = 0
            if "total_frames" not in results:
                results["total_frames"] = 0
            if "fps" not in results:
                results["fps"] = 0

        return results, process_info["status"], process_info["message"]

    except Exception as e:
        print(f"❌ خطأ في الحصول على حالة المعالجة: {e}")
        return {}, "error", f"خطأ في الحصول على الحالة: {str(e)}"


def stop_video_processing(process_id: str):
    """إيقاف معالجة الفيديو"""
    global stop_processing, current_process_id

    if current_process_id == process_id:
        set_stop_processing(True, process_id)
        return True
    else:
        print(f"⚠️ لا توجد عملية نشطة بالمعرف {process_id}")
        return False


def cleanup_processing():
    """تنظيف عمليات المعالجة"""
    global stop_processing, current_process_id
    set_stop_processing(True, None)
    current_process_id = None

    # تنظيف ذاكرة GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

