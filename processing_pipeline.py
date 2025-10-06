
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

from cv2 import Mat
from numpy import ndarray, dtype

from translation_utils import MarianTranslator
from config import UPLOAD_DIR, OUTPUTS_DIR, PROCESSING_CONFIG, MODEL_CONFIG
from database import db
# استيراد الفئات المعدلة
from models import FaceDetector, TextDetector, SpeechRecognizer, ObjectTracker, FrameEnhancer
from activity_recognizer import ActivityRecognizer
from model_loader import model_loader
import gc

stop_processing = False
current_process_id = None
translator = MarianTranslator()

TARGET_FACE_WIDTH = PROCESSING_CONFIG["target_face_width"]
TARGET_FACE_HEIGHT = PROCESSING_CONFIG["target_face_height"]
FACE_PADDING_RATIO = PROCESSING_CONFIG["face_padding_ratio"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

enhancer = FrameEnhancer()


def set_stop_processing(value: bool, process_id: Optional[str] = None):
    global stop_processing, current_process_id
    stop_processing = value
    if process_id:
        current_process_id = process_id


def get_stop_processing():
    return stop_processing


def setup_output_dirs(process_id: str) -> Tuple[Path, Path, Path]:
    process_dir = OUTPUTS_DIR / process_id
    faces_dir = process_dir / "faces"
    video_dir = process_dir / "video"
    audio_dir = process_dir / "audio"

    for directory in [process_dir, faces_dir, video_dir, audio_dir]:
        directory.mkdir(exist_ok=True, parents=True)

    return process_dir, faces_dir, video_dir


def extract_audio(video_path: str, output_audio_path: str) -> bool:
    try:
        command = [
            'ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a',
            output_audio_path, '-y', '-loglevel', 'error'
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ تم استخراج الصوت بنجاح: {output_audio_path}")
            return True
        else:
            print(f"❌ فشل في استخراج الصوت: {result.stderr}")
            return False
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"❌ خطأ في استخراج الصوت: {e}")
        return False


def process_single_frame(frame: np.ndarray, frame_number: int, detection_step,
                         face_detector: FaceDetector, text_detector: TextDetector,
                         object_tracker: ObjectTracker,
                         options: Dict[str, Any], process_id: str, faces_dir: Path) -> tuple[
    ndarray | Any, list[dict[str, str | float | None | int | Any]], list[dict[str, Any]], list[dict[str, Any]], list[
        dict[str, Any]], list[dict[str, str | float | None | int | Any]]]:
    """معالجة إطار واحد مع كشف جميع الكائنات وتتبع الأشخاص"""

    all_objects = []  # الكائنات المكتشفة في هذا الإطار (بما في ذلك الأشخاص)
    all_faces = []
    all_texts = []
    all_tracks = []  # بيانات التتبع للأشخاص في هذا الإطار
    activity_data = {}
    persons_data = []

    # تحسين الإطار
    enhanced_img_pil = enhancer.enhance_frame(frame)
    processed_frame = cv2.cvtColor(np.array(enhanced_img_pil), cv2.COLOR_RGB2BGR)

    if (options.get("enable_text_detection", True) and text_detector) and frame_number % detection_step == 0:
        texts = text_detector.detect_text(processed_frame, frame_number)
        all_texts.extend(texts)

    if (options.get("enable_text_detection", True) and text_detector) and frame_number % detection_step == 0:
        texts = text_detector.detect_text(processed_frame, frame_number)
        all_texts.extend(texts)
        # رسم مربعات الإحاطة للنصوص
        for text_det in texts:
            x, y, w, h = text_det["bbox"]
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # أصفر للنصوص
            cv2.putText(processed_frame, text_det["text"], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    if (options.get("enable_tracking", True) and object_tracker) and frame_number % detection_step == 0:

        sv_detections, current_person_tracks = object_tracker.track_objects(processed_frame, frame_number)
        all_tracks.extend(current_person_tracks)

        # تحويل sv.Detections إلى التنسيق القديم لـ all_objects و persons_data
        for i in range(len(sv_detections)):
            bbox = sv_detections.xyxy[i].tolist()
            conf = sv_detections.confidence[i]
            cls_id = sv_detections.class_id[i]
            class_name = object_tracker.id2label.get(int(cls_id), f"unknown_{cls_id}")
            track_id = sv_detections.tracker_id[i] if sv_detections.tracker_id is not None else None

            # إضافة الكائن إلى all_objects
            all_objects.append({
                "process_id": process_id,
                "frame_number": frame_number,
                "class_name": class_name,
                "bbox_x1": float(bbox[0]),
                "bbox_y1": float(bbox[1]),
                "bbox_x2": float(bbox[2]),
                "bbox_y2": float(bbox[3]),
                "confidence": float(conf),
                "track_id": track_id if class_name == "person" else None  # فقط الأشخاص لهم track_id
            })

            # إضافة الأشخاص إلى persons_data (إذا كان class_name هو "person")
            if class_name == "person" and track_id is not None:
                persons_data.append(all_objects[-1])  # يمكن إعادة استخدام الكائن الأخير المضاف

        # رسم الكائنات والمسارات على الإطار باستخدام sv.Detections
        processed_frame = object_tracker.draw_tracks(processed_frame, sv_detections)

    if (options.get("enable_face_detection", True) and face_detector) and frame_number % detection_step == 0:
        faces = face_detector.detect_faces(processed_frame, frame_number)
        all_faces.extend(faces)
        for j, face in enumerate(faces):
            bbox = face["bbox"]  # [x, y, w, h]
            x, y, w, h = bbox

            # رسم مربع الإحاطة للوجه
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # أخضر للوجوه
            cv2.putText(processed_frame, f"Face {face['face_id']}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

            # رسم نقاط الوجه (keypoints)
            if "keypoints" in face and face["keypoints"]:
                keypoints = face["keypoints"]
                for kp_name in ["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"]:
                    if kp_name in keypoints:
                        point = keypoints[kp_name]
                        cv2.circle(processed_frame, (int(point["x"]), int(point["y"])), 2, (255, 0, 0),
                                   -1)  # أزرق لنقاط الوجه

            # حفظ صورة الوجه
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
                                              interpolation=cv2.INTER_AREA)

                face_filename = f"face_{frame_number}_{j}.jpg"
                face_path = faces_dir / face_filename
                try:
                    cv2.imwrite(str(face_path), face_img_resized)
                    face["image_path"] = str(face_path.relative_to(OUTPUTS_DIR / process_id / "faces"))
                except Exception as e:
                    print(f"❌ خطأ في حفظ صورة الوجه: {e}")
                    face["image_path"] = ""
            else:
                print(f"⚠️ تم اكتشاف وجه فارغ في الإطار {frame_number}، لن يتم حفظه.")
                face["image_path"] = ""

    # إرجاع all_objects_in_frame بدلاً من all_objects
    return processed_frame, all_objects, all_faces, all_texts, all_tracks, persons_data


def monitor_processing(process_id: str, total_frames: int, cap: cv2.VideoCapture):
    error_count = 0
    max_errors = PROCESSING_CONFIG.get("max_404_retries", 20)

    while not get_stop_processing():
        try:
            process_info = db.get_process_status(process_id)
            if not process_info or process_info["status"] in ["completed", "error", "stopped"]:
                break

            if not cap.isOpened():
                error_count += 1
                print(f"⚠️ الفيديو مغلق - الخطأ رقم {error_count}")

            if error_count >= max_errors:
                print(f"❌ تم إيقاف التحليل بسبب {max_errors} أخطاء متتالية")
                set_stop_processing(True, process_id)
                db.update_process_status(process_id, "error", 0,
                                         f"تم إيقاف التحليل بسبب {max_errors} أخطاء متتالية")
                break

            time.sleep(2)

        except Exception as e:
            print(f"❌ خطأ في المراقبة: {e}")
            error_count += 1
            time.sleep(2)


def convert_serializable_types(obj: Any) -> Any:
    """
    تحويل أنواع البيانات غير القياسية (NumPy, PyTorch) إلى أنواع JSON-serializable.
    تدعم القواميس والقوائم المتداخلة بشكل recursive.
    """
    if obj is None:
        return None
    
    if isinstance(obj, dict):
        return {key: convert_serializable_types(value) for key, value in obj.items()}
    
    if isinstance(obj, list):
        return [convert_serializable_types(item) for item in obj]
    
    # التعامل مع PyTorch Tensors (السبب الرئيسي في التتبع)
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:  # scalar tensor (مثل confidence)
            return float(obj.item())
        else:  # array tensor (مثل bbox)
            # نقل إلى CPU إذا كان على GPU، ثم تحويل إلى list
            return obj.cpu().tolist()
    
    # التعامل مع NumPy types
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # أنواع أخرى قياسية (لا نحتاج تحويل)
    return obj


def process_video(input_path: str, process_id: str, options: Dict[str, Any]):
    start_time = time.time()
    global stop_processing

    text_detector = None
    face_detector = None
    speech_recognizer = None
    activity_recognizer = None
    object_tracker = None

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
        detection_step = 1
        activity_analysis_fps = options.get("activity_fps", 1.0)
        if activity_analysis_fps <= 0:
            activity_analysis_fps = 1.0

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
        object_tracker = ObjectTracker() if options.get(
            "enable_tracking", True) else None
        activity_recognizer = ActivityRecognizer() if options.get("enable_activity_recognition", True) else None
        speech_recognizer = SpeechRecognizer() if options.get("enable_audio_transcription", True) else None

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

        frame_number = 0
        all_faces = []
        all_texts = []
        all_tracking_data = []
        all_activities = []
        all_objects_overall = []
        all_people_overall = []

        db.update_process_status(process_id, "processing", 30, "جاري معالجة إطارات الفيديو")

        while True:
            if get_stop_processing():
                print("⏹️ تم إيقاف المعالجة يدوياً")
                db.update_process_status(process_id, "stopped",
                                         int((frame_number / total_frames) * 65) + 30,
                                         "تم إيقاف التحليل يدوياً")
                break

            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, current_objects_in_frame, current_faces, current_texts, current_tracks_in_frame, current_persons_data = \
                process_single_frame(frame, frame_number, detection_step, face_detector, text_detector,
                                     object_tracker, options, process_id, faces_dir)

            out.write(processed_frame)

            all_objects_overall.extend(current_objects_in_frame)
            all_faces.extend(current_faces)
            all_texts.extend(current_texts)
            all_tracking_data.extend(current_tracks_in_frame)
            all_people_overall.extend(current_persons_data)

            frame_number += 1

            if frame_number % 10 == 0:
                progress = 30 + int((frame_number / total_frames) * 65)
                progress = min(progress, 95)
                db.update_process_status(process_id, "processing", progress,
                                         f"جاري معالجة الإطار {frame_number}/{total_frames}")

        if get_stop_processing():
            print("💾 جاري حفظ النتائج الحالية بعد الإيقاف")
            final_progress = 30 + int((frame_number / total_frames) * 65)
            db.update_process_status(process_id, "stopped", final_progress,
                                     "تم إيقاف التحليل وحفظ النتائج الحالية")
        else:
            db.update_process_status(process_id, "processing", 95, "جاري حفظ النتائج النهائية")

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
                json.dump(convert_serializable_types(all_tracking_data), f, ensure_ascii=False, indent=2)
            print(f"✅ تم حفظ {len(all_tracking_data)} عملية تتبع في ملف JSON")

        if all_objects_overall:
            objects_output_path = process_dir / "objects_data.json"
            with open(objects_output_path, 'w', encoding='utf-8') as f:
                json.dump(convert_serializable_types(all_objects_overall), f, ensure_ascii=False, indent=2)
            print(f"✅ تم حفظ {len(all_objects_overall)} كائن في ملف JSON")

        objects_unique = list(set(obj["class_name"] for obj in all_objects_overall))
        objects_ar = [translator.translate(obj) for obj in objects_unique]

        if activity_recognizer and options.get("enable_activity_recognition", True):
            db.update_process_status(process_id, "processing", 96, "جاري تحليل النشاط والبيئة")
            activity_prompt_text = options.get("activity_prompt",
                                               "Describe the main activities and environment in the video.")

            activity_analysis_en = activity_recognizer.recognize_activity(
                prompt=activity_prompt_text,
                video_path=input_path,
                fsp=activity_analysis_fps,
                pixels_size=336
            )
            activity_analysis_ar = translator.translate(activity_analysis_en)
            activity_output = {
                "activity_analysis_en": activity_analysis_en,
                "activity_analysis_ar": activity_analysis_ar
            }
            activity_file = process_dir / "activity_analysis.json"
            with open(activity_file, "w", encoding="utf-8") as f:
                json.dump(activity_output, f, ensure_ascii=False, indent=2)
            print(f"✅ تم حفظ تحليل النشاط في {activity_file}")

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
            "poeple_detected": len(all_people_overall),
            "texts_detected": len(all_texts),
            "tracks_detected": len(set(track["track_id"] for track in all_tracking_data if
                                       "track_id" in track and track["track_id"] is not None)),
            "objects_detected": (len(objects_unique), objects_unique),
            "objects_ar": objects_ar,
            "transcription": transcription_result,
            "activity_analysis": {
                "activity_analysis_en": activity_output.get("activity_analysis_en") if activity_recognizer else None,
                "activity_analysis_ar": activity_output.get("activity_analysis_ar") if activity_recognizer else None
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
            print("🎉 تمت معالجة الفيديو بنجاح!")

    except Exception as e:
        error_msg = f"خطأ في المعالجة: {str(e)}"
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

            model_loader.cleanup()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            print(f"⚠️ خطأ في التنظيف: {e}")

        set_stop_processing(False, None)


def get_processing_status(process_id: str) -> Tuple[Dict[str, Any], str, str]:
    try:
        process_info = db.get_process_status(process_id)

        if not process_info:
            return {}, "not_found", "لم يتم العثور على العملية"

        process_dir = OUTPUTS_DIR / process_id
        results_file = process_dir / "final_results.json"

        results = {}
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)

        results["status"] = process_info["status"]
        results["message"] = process_info["message"]
        results["progress"] = process_info["progress"]

        if process_info["status"] in ["completed", "stopped"]:
            faces_file = process_dir / "faces_data.json"
            if faces_file.exists():
                with open(faces_file, 'r', encoding='utf-8') as f:
                    results["faces_data"] = json.load(f)
            else:
                results["faces_data"] = []

            texts_file = process_dir / "texts_data.json"
            if texts_file.exists():
                with open(texts_file, 'r', encoding='utf-8') as f:
                    results["extracted_texts"] = json.load(f)
            else:
                results["extracted_texts"] = []

            tracking_file = process_dir / "tracking_data.json"
            if tracking_file.exists():
                with open(tracking_file, 'r', encoding='utf-8') as f:
                    results["tracking_data"] = json.load(f)
            else:
                results["tracking_data"] = []

            activity_file = process_dir / "activity_analysis.json"
            if activity_file.exists():
                with open(activity_file, 'r', encoding='utf-8') as f:
                    activity_data = json.load(f)
                    results["activity_analysis"] = activity_data
            else:
                results["activity_analysis"] = {}

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
    global stop_processing, current_process_id

    if current_process_id == process_id:
        set_stop_processing(True, process_id)
        return True
    else:
        print(f"⚠️ لا توجد عملية نشطة بالمعرف {process_id}")
        return False


def cleanup_processing():
    global stop_processing, current_process_id
    set_stop_processing(True, None)
    current_process_id = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

