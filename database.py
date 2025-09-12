import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from config import DATABASE_CONFIG


class VideoAnalysisDB:
    def __init__(self):
        self.db_path = DATABASE_CONFIG["db_path"]
        self.tables = DATABASE_CONFIG["tables"]
        self._init_db()

    def _init_db(self):
        """تهيئة قاعدة البيانات وإنشاء الجداول إذا لم تكن موجودة"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.tables['processes']} (
                process_id TEXT PRIMARY KEY,
                filename TEXT,
                original_path TEXT,
                status TEXT,
                progress INTEGER DEFAULT 0,
                message TEXT,
                options TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.tables['faces']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                process_id TEXT,
                frame_number INTEGER,
                face_id INTEGER,
                bbox TEXT,
                confidence REAL,
                image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (process_id) REFERENCES {self.tables['processes']} (process_id)
            )
        """)

        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.tables['texts']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                process_id TEXT,
                frame_number INTEGER,
                bbox TEXT,
                text TEXT,
                confidence REAL,
                language TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (process_id) REFERENCES {self.tables['processes']} (process_id)
            )
        """)

        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.tables['transcriptions']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                process_id TEXT,
                text TEXT,
                language TEXT,
                confidence REAL,
                segments TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (process_id) REFERENCES {self.tables['processes']} (process_id)
            )
        """)

        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.tables['tracking']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                process_id TEXT,
                track_id INTEGER,
                frame_number INTEGER,
                bbox TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (process_id) REFERENCES {self.tables['processes']} (process_id)
            )
        """)

        cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.tables['scenes']} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        process_id TEXT,
                        frame_number INTEGER,
                        activity_label_en TEXT,  -- النشاط من VideoMAE (الإنجليزية)
                        activity_label_ar TEXT,  -- النشاط من VideoMAE (العربية)
                        activity_confidence REAL,
                        description_en TEXT,     -- الوصف من BLIP (الإنجليزية)
                        description_ar TEXT,     -- الوصف من BLIP (العربية)
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (process_id) REFERENCES {self.tables['processes']} (process_id)
                    )
                """)

        conn.commit()
        conn.close()

    def add_process(self, process_id: str, filename: str, original_path: str, options: Dict[str, Any]):
        """إضافة عملية جديدة إلى قاعدة البيانات"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(f"""
                INSERT INTO {self.tables['processes']} (process_id, filename, original_path, status, options)
                VALUES (?, ?, ?, ?, ?)
            """, (process_id, filename, original_path, "processing", json.dumps(options)))

            conn.commit()
            conn.close()
            print(f"✅ تم إضافة العملية {process_id} إلى قاعدة البيانات")

        except Exception as e:
            print(f"❌ خطأ في إضافة العملية إلى قاعدة البيانات: {e}")

    def update_process_status(self, process_id: str, status: str, progress: int = 0, message: str = ""):
        """تحديث حالة العملية"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(f"""
            UPDATE {self.tables['processes']} 
            SET status = ?, progress = ?, message = ?, updated_at = CURRENT_TIMESTAMP
            WHERE process_id = ?
        """, (status, progress, message, process_id))

        conn.commit()
        conn.close()

    def get_process_status(self, process_id: str) -> Optional[Dict[str, Any]]:
        """الحصول على حالة العملية"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT process_id, filename, status, progress, message, options, created_at, updated_at
            FROM {self.tables['processes']} 
            WHERE process_id = ?
        """, (process_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "process_id": row[0],
                "filename": row[1],
                "status": row[2],
                "progress": row[3],
                "message": row[4],
                "options": json.loads(row[5]) if row[5] else {},
                "created_at": row[6],
                "updated_at": row[7]
            }
        return None

    def add_faces(self, process_id: str, faces_data: List[Dict[str, Any]]):
        """إضافة وجوه مكتشفة إلى قاعدة البيانات"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for face in faces_data:
            cursor.execute(f"""
                INSERT INTO {self.tables['faces']} (process_id, frame_number, face_id, bbox, confidence, image_path)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                process_id,
                face["frame_number"],
                face.get("face_id", 0),
                json.dumps(face["bbox"]),
                face["confidence"],
                face["image_path"]
            ))

        conn.commit()
        conn.close()

    def add_texts(self, process_id: str, texts_data: List[Dict[str, Any]]):
        """إضافة نصوص مستخرجة إلى قاعدة البيانات"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for text in texts_data:
            cursor.execute(f"""
                INSERT INTO {self.tables['texts']} (process_id, frame_number, bbox, text, confidence, language)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                process_id,
                text["frame_number"],
                json.dumps(text["bbox"]),
                text["text"],
                text["confidence"],
                text.get("language", "unknown")
            ))

        conn.commit()
        conn.close()

    def add_transcription(self, process_id: str, transcription_data: Dict[str, Any]):
        """إضافة نص صوتي إلى قاعدة البيانات"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(f"""
            INSERT INTO {self.tables['transcriptions']} (process_id, text, language, confidence, segments)
            VALUES (?, ?, ?, ?, ?)
        """, (
            process_id,
            transcription_data["text"],
            transcription_data.get("language", "unknown"),
            transcription_data.get("confidence", 0.0),
            json.dumps(transcription_data.get("segments", []))
        ))

        conn.commit()
        conn.close()

    def add_tracking_data(self, process_id: str, tracking_data: List[Dict[str, Any]]):
        """إضافة بيانات التتبع إلى قاعدة البيانات"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for track in tracking_data:
            cursor.execute(f"""
                INSERT INTO {self.tables['tracking']} (process_id, track_id, frame_number, bbox, confidence)
                VALUES (?, ?, ?, ?, ?)
            """, (
                process_id,
                track["track_id"],
                track["frame_number"],
                json.dumps(track["bbox"]),
                track["confidence"]
            ))

        conn.commit()
        conn.close()

    def add_scene_analysis(self, process_id: str, scene_info: Dict[str, Any]):
        """إضافة تحليل المشهد/النشاط إلى قاعدة البيانات"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        activity_label_en = scene_info.get("activity", "unknown")
        activity_label_ar = scene_info.get("activity_ar", "غير معروف")
        activity_confidence = scene_info.get("confidence", 0.0)
        description_en = scene_info.get("description", "لا يوجد وصف")
        description_ar = scene_info.get("description_ar", "لا يوجد وصف")
        frame_number = scene_info.get("frame_number")
        cursor.execute(
            f"INSERT INTO {self.tables['scenes']} (process_id, frame_number, activity_label_en, activity_label_ar, activity_confidence, description_en, description_ar) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (process_id, frame_number, activity_label_en, activity_label_ar, activity_confidence, description_en, description_ar)
        )
        conn.commit()
        conn.close()

    def get_process_results(self, process_id: str) -> Dict[str, Any]:
        """الحصول على نتائج العملية"""
        process_info = self.get_process_status(process_id)
        if not process_info:
            return {}

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # الحصول على الوجوه
        cursor.execute(f"SELECT COUNT(*) FROM {self.tables['faces']} WHERE process_id = ?", (process_id,))
        faces_count = cursor.fetchone()[0]

        # الحصول على النصوص
        cursor.execute(f"SELECT COUNT(*) FROM {self.tables['texts']} WHERE process_id = ?", (process_id,))
        texts_count = cursor.fetchone()[0]

        # الحصول على الترجمة الصوتية
        cursor.execute(f"SELECT text, language FROM {self.tables['transcriptions']} WHERE process_id = ?",
                       (process_id,))
        transcription_row = cursor.fetchone()
        transcription = {"text": transcription_row[0], "language": transcription_row[1]} if transcription_row else None

        # الحصول على بيانات التتبع
        cursor.execute(f"SELECT COUNT(DISTINCT track_id) FROM {self.tables['tracking']} WHERE process_id = ?",
                       (process_id,))
        tracks_count = cursor.fetchone()[0]

        # الحصول على تحليل المشهد
        cursor.execute(f"""
                    SELECT activity_label_en, activity_label_ar, activity_confidence, description_en, description_ar
                    FROM {self.tables['scenes']}
                    WHERE process_id = ?
                    ORDER BY frame_number DESC LIMIT 1
                """, (process_id,))
        scene_row = cursor.fetchone()
        scene_analysis = {
            "activity_en": scene_row[0],
            "activity_ar": scene_row[1],
            "confidence": scene_row[2],
            "description_en": scene_row[3],
            "description_ar": scene_row[4]
        } if scene_row else None
        conn.close()

        return {
            "process_id": process_id,
            "status": process_info["status"],
            "progress": process_info["progress"],
            "message": process_info["message"],
            "faces_detected": faces_count,
            "extracted_texts": texts_count,
            "transcription": transcription,
            "tracks_count": tracks_count,
            "scene_analysis": scene_analysis,
            "created_at": process_info["created_at"],
            "updated_at": process_info["updated_at"]
        }


db = VideoAnalysisDB()