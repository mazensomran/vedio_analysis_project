import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import shutil
import warnings
import uuid
import torch
import cv2
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime
import aiofiles
import subprocess
from typing import Dict, Any, List, Optional

from shared.config import UPLOAD_DIR, OUTPUTS_DIR, APP_CONFIG, EASYOCR_CONFIG
from shared.gpu_utils import setup_gpu, print_gpu_memory_usage
from shared.database import db
from shared.monitoring import ProcessMonitor
import logging

logger = logging.getLogger(__name__)

# إنشاء مجلد EasyOCR إذا لم يكن موجوداً
easyocr_dir = Path(EASYOCR_CONFIG["model_storage_directory"])
easyocr_dir.mkdir(parents=True, exist_ok=True)
print(f"📁 مجلد EasyOCR: {easyocr_dir}")

# تقليل تحذيرات HuggingFace
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

monitor = ProcessMonitor()

# قاموس لتخزين العمليات النشطة
active_processes = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    from shared.config import get_app_url

    # إعداد GPU
    device = setup_gpu()
    print_gpu_memory_usage()

    logger.info("🎉 تم تشغيل نظام تحليل الفيديو بنجاح!")
    logger.info(f"🌐 يمكنك الوصول إلى التطبيق على: {get_app_url()}")

    monitor.start_monitoring()
    logger.info("✅  بدأت مراقبة العمليات")

    yield

    # Shutdown code
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # تنظيف جميع العمليات النشطة
    for process_id in list(active_processes.keys()):
        stop_video_processing(process_id)

    cleanup_processing()
    logger.info("👋 إيقاف نظام تحليل الفيديو...")


app = FastAPI(
    title="نظام تحليل الفيديو للأدلة الجنائية",
    description="نظام متكامل لتحليل الفيديو والصوت باستخدام الذكاء الاصطناعي",
    version="1.0.0",
    docs_url="/api-docs",
    redoc_url="/api-redoc",
    lifespan=lifespan
)

# تمكين CORS للوصول من أي domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# خدمة الملفات الثابتة
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# إنشاء مجلد التحميلات إذا لم يكن موجوداً
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# HTML Template for the Web Interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>نظام تحليل الفيديو للأدلة الجنائية</title>
    <style>
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --accent-color: #e74c3c;
            --light-color: #ecf0f1;
            --dark-color: #2c3e50;
            --success-color: #27ae60;
            --warning-color: #f39c12;
            --border-radius: 10px;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Segoe UI', 'Tahoma', 'Geneva', 'Verdana', sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: var(--border-radius);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            overflow: hidden;
        }

        header {
            background: var(--primary-color);
            color: white;
            padding: 2rem;
            text-align: center;
        }

        header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }

        header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }

        .main-content {
            padding: 2rem;
        }

        .nav-tabs {
            display: flex;
            background: #f8f9fa;
            border-bottom: 2px solid #dee2e6;
            margin-bottom: 2rem;
        }

        .nav-tab {
            padding: 1rem 2rem;
            cursor: pointer;
            border: none;
            background: none;
            font-size: 1rem;
            transition: all 0.3s ease;
        }

        .nav-tab.active {
            background: white;
            border-bottom: 3px solid var(--secondary-color);
            font-weight: bold;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        .upload-section {
            text-align: center;
            margin-bottom: 2rem;
        }

        .file-drop-area {
            border: 3px dashed var(--secondary-color);
            border-radius: var(--border-radius);
            padding: 3rem;
            margin: 1rem 0;
            cursor: pointer;
            transition: all 0.3s ease;
            background: #f8f9fa;
        }

        .file-drop-area:hover {
            background-color: var(--light-color);
            border-color: var(--primary-color);
        }

        .file-drop-area.active {
            background-color: #d6eaf8;
            border-color: var(--success-color);
        }

        .file-input {
            display: none;
        }

        .btn {
            background: var(--secondary-color);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: var(--border-radius);
            cursor: pointer;
            font-size: 1rem;
            transition: all 0.3s ease;
            margin: 0.5rem;
            text-decoration: none;
            display: inline-block;
        }

        .btn:hover {
            background: var(--primary-color);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }

        .btn:disabled {
            background: #95a5a6;
            cursor: not-allowed;
            transform: none;
        }

        .btn-primary { background: var(--secondary-color); }
        .btn-success { background: var(--success-color); }
        .btn-warning { background: var(--warning-color); }
        .btn-danger { background: var(--accent-color); }
        .btn-info { background: #17a2b8; }


        .options-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }

        .option-item {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: var(--border-radius);
            border-left: 4px solid var(--secondary-color);
        }

        .progress-container {
            margin: 2rem 0;
            background: white;
            padding: 1.5rem;
            border-radius: var(--border-radius);
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .progress-bar {
            width: 100%;
            height: 20px;
            background: #f0f0f0;
            border-radius: 10px;
            overflow: hidden;
            margin: 1rem 0;
        }

        .progress {
            height: 100%;
            background: linear-gradient(90deg, var(--secondary-color), var(--success-color));
            width: 0%;
            transition: width 0.3s ease;
        }

        .results-section {
            margin-top: 2rem;
        }

        .result-card {
            background: var(--light-color);
            padding: 1.5rem;
            margin: 1rem 0;
            border-radius: var(--border-radius);
            border-left: 4px solid var(--secondary-color);
        }

        .face-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }

        .face-item {
            text-align: center;
            background: white;
            padding: 1rem;
            border-radius: var(--border-radius);
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }

        .face-image {
            width: 100%;
            height: 120px;
            object-fit: cover;
            border-radius: var(--border-radius);
            border: 2px solid var(--secondary-color);
        }

        .status-message {
            padding: 1rem;
            margin: 1rem 0;
            border-radius: var(--border-radius);
            text-align: center;
        }

        .status-success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }

        .status-error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }

        .status-info {
            background-color: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }

        .status-warning {
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }

        .video-preview {
            width: 100%;
            max-width: 600px;
            margin: 1rem auto;
            display: block;
            border-radius: var(--border-radius);
        }

        footer {
            text-align: center;
            padding: 2rem;
            background: var(--primary-color);
            color: white;
            margin-top: 2rem;
        }

        .api-links {
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin: 1rem 0;
        }

        .processing-actions {
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin: 1rem 0;
        }

        @media (max-width: 768px) {
            .container {
                margin: 10px;
                padding: 10px;
            }

            header h1 {
                font-size: 2rem;
            }

            .file-drop-area {
                padding: 2rem;
            }

            .nav-tab {
                padding: 0.8rem 1rem;
                font-size: 0.9rem;
            }

            .btn {
                padding: 10px 20px;
                font-size: 0.9rem;
            }

            .options-grid {
                grid-template-columns: 1fr;
            }

            .processing-actions {
                flex-direction: column;
                align-items: center;
            }
        }

        code {
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
        }

        .hidden {
            display: none;
        }

        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }

        .result-item {
            background: white;
            padding: 1rem;
            border-radius: var(--border-radius);
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }

        .person-track {
            border: 2px solid #3498db;
            padding: 0.5rem;
            margin: 0.5rem 0;
            border-radius: var(--border-radius);
        }

        .activity-badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 15px;
            background: #3498db;
            color: white;
            font-size: 0.8rem;
            margin: 0.2rem;
        }

        /* Modal Styles */
        .modal {
            display: none; /* Hidden by default */
            position: fixed; /* Stay in place */
            z-index: 1; /* Sit on top */
            left: 0;
            top: 0;
            width: 100%; /* Full width */
            height: 100%; /* Full height */
            overflow: auto; /* Enable scroll if needed */
            background-color: rgba(0,0,0,0.4); /* Black w/ opacity */
        }

        .modal-content {
            background-color: #fefefe;
            margin: 5% auto; /* 15% from the top and centered */
            padding: 20px;
            border: 1px solid #888;
            width: 80%; /* Could be more or less, depending on screen size */
            border-radius: var(--border-radius);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            position: relative;
        }

        .close-button {
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
        }

        .close-button:hover,
        .close-button:focus {
            color: black;
            text-decoration: none;
            cursor: pointer;
        }

        .file-browser ul {
            list-style-type: none;
            padding: 0;
        }

        .file-browser li {
            padding: 8px 0;
            border-bottom: 1px solid #eee;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .file-browser li:last-child {
            border-bottom: none;
        }

        .file-browser li a {
            text-decoration: none;
            color: var(--primary-color);
            font-weight: bold;
        }

        .file-browser li a:hover {
            color: var(--secondary-color);
        }

        .file-browser .file-icon {
            font-size: 1.2em;
            color: var(--secondary-color);
        }

        .file-browser .folder-icon {
            color: var(--warning-color);
        }

        .file-viewer {
            margin-top: 20px;
            text-align: center;
        }

        .file-viewer img, .file-viewer video {
            max-width: 100%;
            height: auto;
            border-radius: var(--border-radius);
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
                .results-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }

        .results-table th, .results-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: right; /* ليتناسب مع اللغة العربية */
        }

        .results-table th {
            background-color: #f2f2f2;
            font-weight: bold;
        }

        .results-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }

        .results-table ul {
            list-style-type: none;
            padding: 0;
            margin: 0;
        }

        .results-table li {
            padding: 2px 0;
        }

                /* شبكة الإحصائيات */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }

        .stat-card {
            background: white;
            padding: 1.5rem;
            border-radius: var(--border-radius);
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 4px solid var(--secondary-color);
        }

        .stat-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }

        .stat-number {
            font-size: 1.8rem;
            font-weight: bold;
            color: var(--primary-color);
        }

        .stat-label {
            color: #666;
            font-size: 0.9rem;
        }

        /* عناصر تحكم الفيديو */
        .video-container {
            position: relative;
            margin-bottom: 1rem;
        }

        .video-controls {
            display: flex;
            justify-content: center;
            gap: 0.5rem;
            margin-top: 1rem;
            flex-wrap: wrap;
        }

        .btn-small {
            padding: 8px 16px;
            font-size: 0.8rem;
        }

        .video-info {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: var(--border-radius);
            margin-top: 1rem;
        }

        .video-info p {
            margin: 0.3rem 0;
            display: flex;
            justify-content: space-between;
        }

        /* عناصر التحكم بالجدول */
        .table-controls {
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }

        .search-input {
            flex: 1;
            padding: 0.5rem;
            border: 1px solid #ddd;
            border-radius: var(--border-radius);
            min-width: 200px;
        }

        .filter-select {
            padding: 0.5rem;
            border: 1px solid #ddd;
            border-radius: var(--border-radius);
            background: white;
        }

        .results-table-container {
            overflow-x: auto;
            max-height: 400px;
            overflow-y: auto;
        }

        .slider {
            width: 100%;
            margin: 10px 0;
        }
        .slider + span {
            font-weight: bold;
            color: var(--secondary-color);
            display: block;
        }
        textarea.form-control {
            width: 100%;
            min-height: 80px;
            resize: vertical; /* يسمح بالتوسع العمودي فقط */
            font-family: inherit;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: var(--border-radius);
        }
        textarea.form-control:focus {
            outline: none;
            border-color: var(--secondary-color);
            box-shadow: 0 0 5px rgba(52, 152, 219, 0.3);
        }
        .slider + span {
            font-weight: bold;
            color: var(--secondary-color);
            display: block;
        }
        small {
            color: #666;
            font-size: 0.8em;
            display: block;
            margin-top: 5px;
        }

        /* تحسينات لعرض الوجوه */
        .face-controls {
            margin-bottom: 1rem;
            padding: 0.5rem;
            background: #f8f9fa;
            border-radius: var(--border-radius);
            text-align: center;
        }

        .face-item {
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            text-align: center;
            padding: 0.5rem;
        }

        .face-item:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }

        .face-item .btn-small {
            margin-top: 0.5rem;
            padding: 0.25rem 0.5rem;
            font-size: 0.8rem;
            width: 100%;
        }

        .face-info {
            margin-top: 0.5rem;
            font-size: 0.9rem;
        }

        .more-faces {
            text-align: center;
            font-style: italic;
            color: #666;
            margin-top: 1rem;
            padding: 1rem;
        }

        /* نافذة عرض الوجه المحسنة */
        .face-modal-container {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            max-height: 80vh;
            overflow-y: auto;
        }

        .face-preview-section {
            display: flex;
            gap: 1.5rem;
            align-items: flex-start;
        }

        .face-image-container {
            flex: 2;
            text-align: center;
            background: #f8f9fa;
            padding: 1rem;
            border-radius: var(--border-radius);
            border: 2px solid #e9ecef;
        }

        .face-info-panel {
            flex: 1;
            background: white;
            padding: 1rem;
            border-radius: var(--border-radius);
            border-left: 4px solid var(--secondary-color);
            min-width: 200px;
        }

        .face-controls-section {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: var(--border-radius);
        }

        .enhancement-controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 0.8rem;
            margin-bottom: 1rem;
        }

        .enhancement-controls .btn {
            padding: 0.8rem 0.5rem;
            font-size: 0.9rem;
            white-space: nowrap;
        }

        .enhancement-sliders {
            background: white;
            padding: 1rem;
            border-radius: var(--border-radius);
            border: 1px solid #dee2e6;
        }

        .slider-control {
            margin-bottom: 1rem;
        }

        .slider-control label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: bold;
            color: #495057;
        }

        .slider-control .slider {
            width: 100%;
            margin: 0.5rem 0;
        }

        .hidden {
            display: none !important;
        }

        /* تحسينات للعرض على الجوال */
        @media (max-width: 768px) {
            .face-preview-section {
                flex-direction: column;
            }

            .face-info-panel {
                min-width: auto;
            }

            .enhancement-controls {
                grid-template-columns: 1fr;
            }

            .face-modal-container {
                max-height: 90vh;
            }
        }

        /* تحسينات التقليب */
        .faces-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding: 1rem;
            background: #f8f9fa;
            border-radius: var(--border-radius);
            flex-wrap: wrap;
            gap: 1rem;
        }

        .faces-pagination {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .faces-pagination .btn-small {
            padding: 0.5rem 1rem;
            font-size: 0.9rem;
        }

        .page-info {
            font-weight: bold;
            color: var(--primary-color);
        }

        .faces-per-page {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .faces-per-page select {
            padding: 0.5rem;
            border: 1px solid #ddd;
            border-radius: var(--border-radius);
            background: white;
        }

        .btn-disabled {
            background: #95a5a6 !important;
            cursor: not-allowed;
            opacity: 0.6;
        }

        .no-faces {
            text-align: center;
            padding: 2rem;
            color: #666;
            font-style: italic;
        }

        /* تحسينات لعرض الإجراءات */
        .enhancement-stack {
            background: white;
            padding: 1rem;
            border-radius: var(--border-radius);
            margin-top: 1rem;
            border-left: 4px solid var(--info-color);
        }

        .enhancement-stack h5 {
            margin-bottom: 0.5rem;
            color: var(--primary-color);
        }

        .enhancement-item {
            display: flex;
            justify-content: space-between;
            padding: 0.5rem;
            border-bottom: 1px solid #eee;
        }

        .enhancement-item:last-child {
            border-bottom: none;
        }

        /* تحسينات للعرض على الجوال */
        @media (max-width: 768px) {
            .faces-header {
                flex-direction: column;
                align-items: stretch;
            }

            .faces-pagination {
                justify-content: center;
            }

            .faces-per-page {
                justify-content: center;
            }

            .face-grid {
                grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            }

            .enhancement-controls {
                grid-template-columns: 1fr;
            }
        }

                /* تنسيق قسم الضبط المتقدم */
        .advanced-settings {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: var(--border-radius);
            border-left: 4px solid var(--warning-color);
            margin-top: 1rem;
        }

        .advanced-settings h4 {
            margin-bottom: 1rem;
            color: var(--primary-color);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .slider-value {
            font-weight: bold;
            color: var(--secondary-color);
            background: #f8f9fa;
            padding: 2px 8px;
            border-radius: 4px;
            border: 1px solid #dee2e6;
            min-width: 40px;
            display: inline-block;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎥 نظام تحليل الفيديو للأدلة الجنائية</h1>
            <p>نموذج أولي لتحليل الفيديو والصوت باستخدام الذكاء الاصطناعي</p>
        </header>

        <div class="nav-tabs">
            <button class="nav-tab active" onclick="showTab('upload')">رفع فيديو</button>
            <button class="nav-tab" onclick="showTab('results')">النتائج</button>
            <button class="nav-tab" onclick="showTab('api')">واجهة API</button>
            <button class="nav-tab" onclick="showTab('help')">المساعدة</button>
        </div>

        <div class="main-content">
            <!-- Upload Tab -->
            <div class="tab-content active" id="tab-upload">
                <div class="upload-section">
                    <h2>📤 رفع فيديو للتحليل</h2>
                    <p>اختر ملف فيديو (MP4, AVI, MOV) بحد أقصى 100MB</p>

                    <div class="file-drop-area" id="fileDropArea" onclick="document.getElementById('fileInput').click()">
                        <p>📁 اسحب وأسقط ملف الفيديو هنا أو</p>
                        <input type="file" id="fileInput" class="file-input" accept="video/*">
                        <button class="btn btn-primary">اختر ملف</button>
                    </div>

                    <video id="uploadedVideoPreview" class="video-preview hidden" controls></video>
                    <div id="videoInfo" class="result-card hidden">
                        <h3>معلومات الفيديو:</h3>
                        <p><strong>الاسم:</strong> <span id="videoName"></span></p>
                        <p><strong>الحجم:</strong> <span id="videoSize"></span></p>
                        <p><strong>النوع:</strong> <span id="videoType"></span></p>
                        <p><strong>المدة:</strong> <span id="videoDuration"></span></p>
                        <p><strong>عدد الإطارات:</strong> <span id="videoFrames"></span></p>
                        <p><strong>معدل الإطارات (FPS):</strong> <span id="videoFPS"></span></p>
                    </div>


                    <div class="options-grid">
                        <div class="option-item" id="detectionStepContainer">
                            <label for="detectionStep">خطوة الكشف (للتسريع، 1=كل إطار):</label>
                            <input type="range" id="detectionStep" min="1" max="10" step="1" value="1" class="slider">
                            <span id="detectionStepValue">1</span>
                            <small>زيادة القيمة تسرع المعالجة لكن تقلل الدقة</small>
                        </div>
                        <div class="option-item">
                            <input type="checkbox" id="enableAudio" >
                            <label for="enableAudio">🎵 تحويل الصوت إلى نص</label>
                        </div>
                        <div class="option-item">
                            <input type="checkbox" id="enableFaces" >
                            <label for="enableFaces">👥 اكتشاف الوجوه</label>
                        </div>
                        <div class="option-item hidden" id="faceThresholdContainer";">
                            <label for="faceThreshold">عتبة كشف الوجوه (0.1 - 1.0):</label>
                            <input type="range" id="faceThreshold" min="0.1" max="1.0" step="0.01" value="0.3" class="slider">
                            <span id="faceThresholdValue">0.3</span>
                        </div>
                        <div class="option-item">
                            <input type="checkbox" id="enableText" >
                            <label for="enableText">📝 استخراج النصوص</label>
                        </div>
                        <div class="option-item hidden" id="textThresholdContainer";">
                            <label for="textThreshold">عتبة كشف النصوص (0.1 - 1.0):</label>
                            <input type="range" id="textThreshold" min="0.1" max="1.0" step="0.01" value="0.3" class="slider">
                            <span id="textThresholdValue">0.3</span>
                        </div>
                        <div class="option-item">
                            <input type="checkbox" id="enableTracking" >
                            <label for="enableTracking">🔄 تتبع حركة الأشخاص</label>
                        </div>
                        <div class="option-item hidden" id="objectThresholdContainer";">
                            <label for="objectThreshold">عتبة كشف الكائنات (0.1 - 1.0):</label>
                            <input type="range" id="objectThreshold" min="0.1" max="01.0" step="0.01" value="0.5" class="slider">
                            <span id="objectThresholdValue">0.5</span>
                        </div>
                        <div class="option-item">
                            <input type="checkbox" id="enableActivity" >
                            <label for="enableActivity">🎯 تحليل النشاط والبيئة</label>
                        </div>
                        <div class="option-item hidden" id="activityPromptContainer">
                            <label for="activityPromptPreset">نوع التحليل:</label>
                            <select id="activityPromptPreset" class="form-control" onchange="loadPromptPreset(this.value)">
                                <option value="forensic">🔍 التحليل الشامل للأدلة الجنائية</option>
                                <option value="threats">⚠️ كشف التهديدات والأسلحة</option>
                                <option value="theft">💰 تحليل السرقة والاعتداء على الممتلكات</option>
                                <option value="behavior">🚶 تحليل الحركات والسلوكيات المشبوهة</option>
                                <option value="temporal">⏰ التحليل الزمني والتسلسلي للأحداث</option>
                                <option value="custom">✏️ تخصيص يدوي (اكتب Prompt خاص)</option>
                            </select>

                            <div id="customPromptContainer" class="hidden" style="margin-top: 10px;">
                                <label for="activityPrompt">أدخل الـ Prompt المخصص:</label>
                                <textarea id="activityPrompt" rows="3" class="form-control" placeholder="اكتب مطالبة مفصلة هنا..."></textarea>
                            </div>

                            <div id="presetDescription" class="status-message status-info" style="margin-top: 10px; font-size: 0.9em;">
                                <strong>التحليل الشامل للأدلة الجنائية:</strong> تحليل كامل للفيديو يشمل البيئة، الأشخاص، الأنشطة المشبوهة، وجمع الأدلة
                            </div>
                        </div>
                        <div class="option-item hidden" id="activityFpsContainer">
                            <label for="activityFps">دقة التحليل (FPS):</label>
                            <input type="number" id="activityFps" class="form-control" value="1" min="1" step="1">
                        </div>
                        <div class="option-item hidden" id="advancedSettingsContainer">
                            <h4>⚙️ ضبط متقدم</h4>
                            <div class="options-grid">

                                <div class="option-item">
                                    <input type="checkbox" id="enable_video_enhancement">
                                    <label for="enable_video_enhancement">🎨 تحسين جودة الفيديو (للفيديوهات منخفضة الدقة)</label>
                                    <small>يحسن وضوح الفيديو قبل التحليل - يستهلك ذاكرة إضافية</small>
                                </div>

                                <div class="option-item hidden" id="enhancementStrengthContainer">
                                    <label for="enhancementStrength">قوة التحسين (1-5):</label>
                                    <input type="range" id="enhancementStrength" min="1" max="5" step="1" value="2" class="slider">
                                    <span id="enhancementStrengthValue">2</span>
                                    <small>1 = تحسين خفيف (أسرع), 5 = تحسين قوي (أبطأ)</small>
                                </div>
                                <div class="option-item">
                                    <label for="maxNewTokens">Max Tokens (1-1500):</label>
                                    <input type="range" id="maxNewTokens" min="1" max="1500" step="10" value="600" class="slider">
                                    <span id="maxNewTokensValue" class="slider-value">600</span>
                                </div>
                                <div class="option-item">
                                    <input type="checkbox" id="doSample">
                                    <label for="doSample">Do Sample</label>
                                </div>

                                <div class="option-item">
                                    <label for="temperature">Temperature (0-1):</label>
                                    <input type="range" id="temperature" min="0" max="1" step="0.1" value="0.2" class="slider">
                                    <span id="temperatureValue">0.2</span>
                                </div>
                                <div class="option-item">
                                    <label for="topP">Top P (0-1):</label>
                                    <input type="range" id="topP" min="0" max="1" step="0.01" value="0.85" class="slider">
                                    <span id="topPValue">0.85</span>
                                </div>
                                <div class="option-item">
                                    <label for="topK">Top K (1-100):</label>
                                    <input type="range" id="topK" min="1" max="100" step="1" value="50" class="slider">
                                    <span id="topKValue">50</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="processing-actions">
                        <button class="btn btn-success" id="analyzeBtn" onclick="analyzeVideo()" disabled>
                            🚀 بدء التحليل
                        </button>
                        <button class="btn btn-danger" id="stopBtn" onclick="stopAnalysis()" disabled>
                            ⏹️ إيقاف التحليل
                        </button>
                    </div>
                </div>

                <div class="progress-container hidden" id="progressContainer">
                    <h3>⏳ جاري المعالجة...</h3>
                    <div class="progress-bar">
                        <div class="progress" id="progressBar"></div>
                    </div>
                    <p id="progressText">0%</p>
                    <p id="processingDetails">جاري الإعداد...</p>
                    <div id="statusMessage" class="status-message status-info">
                        جاري إعداد النظام...
                    </div>
                </div>
            </div>

            <!-- Results Tab -->
            <div class="tab-content" id="tab-results">
                <h2>📊 نتائج التحليل</h2>
                <div id="resultsContent">
                    <p class="status-message status-info">لم يتم تحليل أي فيديو بعد</p>
                </div>
                <!-- حاوية جديدة لجدول النتائج النهائية -->
                <div id="finalResultsTableContainer" class="result-card hidden">
                    <h3>📋 ملخص النتائج النهائية</h3>
                    <table id="finalResultsTable" class="results-table">
                        <!-- سيتم ملء هذا الجدول بواسطة JavaScript -->
                    </table>
                </div>
            </div>

            <!-- API Tab -->
            <div class="tab-content" id="tab-api">
                <h2>🔌 واجهة برمجة التطبيقات (API)</h2>
                <div class="result-card">
                    <h3>📋 Endpoints المتاحة:</h3>
                    <ul>
                        <li><code>POST /analyze-video</code> - رفع فيديو للتحليل</li>
                        <li><code>GET /results/&#123;process_id&#125;</code> - الحصول على النتائج</li>
                        <li><code>POST /stop-analysis/&#123;process_id&#125;</code> - إيقاف التحليل</li>
                        <li><code>GET /outputs/&#123;process_id&#125;/&#123;filename&#125;</code> - تحميل الملفات</li>
                        <li><code>GET /outputs_list/&#123;process_id&#125;</code> - قائمة ملفات المخرجات</li>
                        <li><code>GET /health</code> - حالة الخادم</li>
                    </ul>

                    <h3>📝 مثال استخدام:</h3>
                    <div class="result-card">
                        <pre><code># رفع فيديو
curl -X POST "{{base_url}}/analyze-video" \\
  -F "file=@video.mp4" \\
  -F "enable_audio_transcription=true" \\
  -F "enable_face_detection=true" \\
  -F "enable_text_detection=true" \\
  -F "enable_tracking=true" \\
  -F "enable_activity_recognition=true" \\
  -F "activity_batch_processing=true"

# الحصول على النتائج
curl "{{base_url}}/results/process_id"

# إيقاف التحليل
curl -X POST "{{base_url}}/stop-analysis/process_id"</code></pre>
                    </div>

                    <div class="api-links">
                        <a href="/api-docs" class="btn btn-primary" target="_blank">📖 واجهة API التفاعلية</a>
                        <a href="/api-redoc" class="btn btn-success" target="_blank">📚 الوثائق الكاملة</a>
                    </div>
                </div>
            </div>

            <!-- Help Tab -->
            <div class="tab-content" id="tab-help">
                <h2>❓ المساعدة</h2>
                <div class="result-card">
                    <h3>🎯 تعليمات الاستخدام:</h3>
                    <ol>
                        <li>اختر ملف فيديو من جهازك</li>
                        <li>اضبط خيارات التحليل المطلوبة</li>
                        <li>اضغط على "بدء التحليل"</li>
                        <li>يمكنك إيقاف التحليل في أي وقت باستخدام زر "إيقاف التحليل"</li>
                        <li>انتظر حتى اكتمال المعالجة</li>
                        <li>شاهد النتائج في قسم "النتائج"</li>
                    </ol>

                    <h3>⏱️ أوقات المعالجة المتوقعة:</h3>
                    <ul>
                        <li>فيديو 1 دقيقة: 2-3 دقائق</li>
                        <li>فيديو 5 دقائق: 10-15 دقيقة</li>
                        <li>فيديو 10 دقائق: 20-30 دقيقة</li>
                    </ul>

                    <h3>📋 متطلبات النظام:</h3>
                    <ul>
                        <li>متصفح حديث (Chrome, Firefox, Safari, Edge)</li>
                        <li>اتصال بالإنترنت</li>
                        <li>لا حاجة لتثبيت أي برامج</li>
                    </ul>

                    <h3>⚠️ استكشاف الأخطاء وإصلاحها:</h3>
                    <ul>
                        <li>إذا توقف التحليل، اضغط على زر الإيقاف لحفظ النتائج الحالية</li>
                        <li>تأكد من أن حجم الفيديو لا يتجاوز 100MB</li>
                        <li>إذا استمرت المشاكل، جرب إعادة تحميل الصفحة</li>
                    </ul>
                </div>
            </div>
        </div>

        <footer>
            <p>© 2025 نظام تحليل الفيديو للأدلة الجنائية - الإصدار 1.0.0</p>
            <p>⏰ حالة الخادم: <span id="serverStatus">جاري التحقق...</span></p>
        </footer>
    </div>

    <!-- Output Files Modal -->
    <div id="outputModal" class="modal">
        <div class="modal-content">
            <span class="close-button" onclick="closeModal()">&times;</span>
            <h2>📂 ملفات المخرجات</h2>
            <div id="outputFileList" class="file-browser">
                <!-- File list will be loaded here -->
            </div>
            <div id="fileViewer" class="file-viewer">
                <!-- File content (image/video) will be displayed here -->
            </div>
        </div>
    </div>

    <!-- Enhanced Face Modal -->
    <div id="faceModal" class="modal">
        <div class="modal-content" style="max-width: 95%; max-height: 95%;">
            <span class="close-button" onclick="closeFaceModal()">&times;</span>
            <h2>👤 معاينة وتحسين الوجه</h2>

            <div class="face-modal-container">
                <div class="face-preview-section">
                    <div class="face-image-container">
                        <img id="modalFaceImage" src="" alt="Face Preview" 
                             style="max-width: 100%; max-height: 60vh; border: 2px solid #ddd; border-radius: 10px;">
                    </div>
                    <div class="face-info-panel">
                        <h4>معلومات الوجه</h4>
                        <p><strong>الإطار:</strong> <span id="modalFaceFrame">-</span></p>
                        <p><strong>الثقة:</strong> <span id="modalFaceConfidence">-</span>%</p>
                        <p><strong>الحالة:</strong> <span id="modalFaceStatus">الأصلية</span></p>
                        <p><strong>الإجراءات:</strong> <span id="modalFaceActions">لا يوجد</span></p>
                    </div>
                </div>

                <div class="face-controls-section">
                    <h4>🛠️ أدوات التحسين</h4>
                    <div class="enhancement-controls">
                        <button class="btn btn-primary" onclick="applyEnhancement('super_resolution')">
                            🔍 تحسين الدقة
                        </button>
                        <button class="btn btn-info" onclick="applyEnhancement('sharpen')">
                            ⚡ زيادة الحدة
                        </button>
                        <button class="btn btn-warning" onclick="applyEnhancement('contrast')">
                            🌈 زيادة التباين
                        </button>
                        <button class="btn btn-success" onclick="applyEnhancement('smooth')">
                            💫 تنعيم الصورة
                        </button>
                        <button class="btn btn-secondary" onclick="undoEnhancement()">
                            ↩️ تراجع
                        </button>
                        <button class="btn btn-danger" onclick="saveEnhancedFace()">
                            💾 حفظ الصورة المحسنة
                        </button>
                    </div>

                    <div class="enhancement-sliders hidden" id="enhancementSliders">
                        <div class="slider-control">
                            <label for="sharpenAmount">قوة الحدة:</label>
                            <input type="range" id="sharpenAmount" min="1" max="5" step="0.5" value="2" class="slider">
                            <span id="sharpenValue">2</span>
                        </div>
                        <div class="slider-control">
                            <label for="contrastAmount">قوة التباين:</label>
                            <input type="range" id="contrastAmount" min="1" max="3" step="0.1" value="1.5" class="slider">
                            <span id="contrastValue">1.5</span>
                        </div>
                        <div class="slider-control">
                            <label for="smoothAmount">قوة التنعيم:</label>
                            <input type="range" id="smoothAmount" min="1" max="10" step="1" value="3" class="slider">
                            <span id="smoothValue">3</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>


    <script>
    // Global variables
    let currentProcessId = null;
    let checkInterval = null;
    const baseUrl = window.location.origin;

    // متغيرات التقليب العالمية
    let currentFacesPage = 1;
    let facesPerPage = 4;
    let currentResults = null;
    let totalFaces = 0;

    // Page initialization
    document.addEventListener('DOMContentLoaded', function() {
        setupDragAndDrop();
        checkServerStatus();
        updateApiExamples();
        checkActiveProcesses();
        setupEnhancementSliders();
        setupAdvancedSettingsSliders();
        loadPromptPreset('forensic');

        initializeOptionVisibility();
        // تحكم في عتبة الوجوه
        document.getElementById('enableFaces').addEventListener('change', function() {
            const container = document.getElementById('faceThresholdContainer');
            const slider = document.getElementById('faceThreshold');
            if (this.checked) {
                container.style.display = 'block';
                slider.disabled = false;
            } else {
                container.style.display = 'none';
                slider.disabled = true;
            }
        });

        // التحكم في خيار تحسين الفيديو
        const enableVideoEnhancement = document.getElementById('enable_video_enhancement');
        const enhancementContainer = document.getElementById('enhancementStrengthContainer');

        if (enableVideoEnhancement && enhancementContainer) {
            // تعيين الحالة الافتراضية
            if (!enableVideoEnhancement.checked) {
                enhancementContainer.classList.add('hidden');
            }

            // إضافة event listener
            enableVideoEnhancement.addEventListener('change', function() {
                if (this.checked) {
                    enhancementContainer.classList.remove('hidden');
                } else {
                    enhancementContainer.classList.add('hidden');
                }
            });
        }

        // تحكم في عتبة النصوص
        document.getElementById('enableText').addEventListener('change', function() {
            const container = document.getElementById('textThresholdContainer');
            const slider = document.getElementById('textThreshold');
            if (this.checked) {
                container.style.display = 'block';
                slider.disabled = false;
            } else {
                container.style.display = 'none';
                slider.disabled = true;
            }
        });
        // تحكم في عتبة الكائنات
        document.getElementById('enableTracking').addEventListener('change', function() {
            const container = document.getElementById('objectThresholdContainer');
            const slider = document.getElementById('objectThreshold');
            if (this.checked) {
                container.style.display = 'block';
                slider.disabled = false;
            } else {
                container.style.display = 'none';
                slider.disabled = true;
            }
        });

        // دالة مساعدة للتحكم في container بناءً على checkbox
        /*function toggleContainer(checkboxId, containerId, sliderId) {
            const checkbox = document.getElementById(checkboxId);
            const container = document.getElementById(containerId);
            const slider = document.getElementById(sliderId);

            // التحقق الافتراضي عند التحميل (يظهر إذا كان checkbox محدد)
            if (checkbox.checked) {
                container.style.display = 'block';
                slider.disabled = false;
            } else {
                container.style.display = 'none';
                slider.disabled = true;
            }

            // Event listener للتغييرات اللاحقة
            checkbox.addEventListener('change', function() {
                if (this.checked) {
                    container.style.display = 'block';
                    slider.disabled = false;
                } else {
                    container.style.display = 'none';
                    slider.disabled = true;
                }
            });
        }

        // تطبيق الدالة على كل checkbox (للظهور الافتراضي والتحكم)
        toggleContainer('enableFaces', 'faceThresholdContainer', 'faceThreshold');
        toggleContainer('enableText', 'textThresholdContainer', 'textThreshold');
        toggleContainer('enableTracking', 'objectThresholdContainer', 'objectThreshold');

                // إضافة جديدة: تحكم في عناصر تحليل النشاط والبيئة (prompt و FPS والضبط المتقدم)
        function toggleActivityContainers(checkboxId, promptContainerId, fpsContainerId, advancedContainerId) {
            const checkbox = document.getElementById(checkboxId);
            const promptContainer = document.getElementById(promptContainerId);
            const fpsContainer = document.getElementById(fpsContainerId);
            const advancedContainer = document.getElementById(advancedContainerId);
            const promptTextarea = document.getElementById('activityPrompt');
            const fpsInput = document.getElementById('activityFps');

            // التحقق الافتراضي عند التحميل (يظهر إذا كان checkbox محدد)
            if (checkbox.checked) {
                promptContainer.style.display = 'block';
                fpsContainer.style.display = 'block';
                advancedContainer.style.display = 'block';
                if (promptTextarea) promptTextarea.disabled = false;
                if (fpsInput) fpsInput.disabled = false;
                loadPromptPreset('forensic');
            } else {
                promptContainer.style.display = 'none';
                fpsContainer.style.display = 'none';
                advancedContainer.style.display = 'none';
                if (promptTextarea) promptTextarea.disabled = true;
                if (fpsInput) fpsInput.disabled = true;
            }

            // Event listener للتغييرات اللاحقة
            checkbox.addEventListener('change', function() {
                if (this.checked) {
                    promptContainer.style.display = 'block';
                    fpsContainer.style.display = 'block';
                    advancedContainer.style.display = 'block';
                    if (promptTextarea) promptTextarea.disabled = false;
                    if (fpsInput) fpsInput.disabled = false;
                } else {
                    promptContainer.style.display = 'none';
                    fpsContainer.style.display = 'none';
                    advancedContainer.style.display = 'none';
                    if (promptTextarea) promptTextarea.disabled = true;
                    if (fpsInput) fpsInput.disabled = true;
                }
            });
        }

        toggleActivityContainers('enableActivity', 'activityPromptContainer', 'activityFpsContainer', 'advancedSettingsContainer');*/

        // تحديث قيم السلايدرز (لعرض القيمة الحالية)

        function updateSliderValue(sliderId, valueId) {
            const slider = document.getElementById(sliderId);
            const valueSpan = document.getElementById(valueId);

            if (slider && valueSpan) {
                // تعيين القيمة الأولية
                valueSpan.textContent = slider.value;

                // إضافة event listener للتحديث عند التغيير
                slider.addEventListener('input', function() {
                    valueSpan.textContent = this.value;
                });

                // أيضًا تحديث عند تحرير السلايدر
                slider.addEventListener('change', function() {
                    valueSpan.textContent = this.value;
                });
            }
        }
        updateSliderValue('detectionStep', 'detectionStepValue');
        updateSliderValue('faceThreshold', 'faceThresholdValue');
        updateSliderValue('textThreshold', 'textThresholdValue');
        updateSliderValue('objectThreshold', 'objectThresholdValue');
        // ✅ تحديث قيم سلايدرز الضبط المتقدم
        setupAdvancedSettingsSliders();

        // ✅ تحميل الـ Prompt الافتراضي
        loadPromptPreset('forensic');

        // توسيع textarea تلقائياً (اختياري لتحسين UX)
        const textarea = document.getElementById('activityPrompt');
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
        // ✅ إضافة event listeners إضافية للتأكد من عمل السلايدرز
        setTimeout(() => {
            console.log("🔧 Checking sliders initialization...");
            setupAdvancedSettingsSliders();
        }, 500);
    });

    // ✅ دالة جديدة لتهيئة حالة الظهور الافتراضية
    function initializeOptionVisibility() {
        // التحكم في عتبة الوجوه
        const enableFacesCheckbox = document.getElementById('enableFaces');
        const faceThresholdContainer = document.getElementById('faceThresholdContainer');
        const faceThresholdSlider = document.getElementById('faceThreshold');

        if (enableFacesCheckbox && faceThresholdContainer) {
            // تعيين الحالة الافتراضية
            if (!enableFacesCheckbox.checked) {
                faceThresholdContainer.classList.add('hidden');
                if (faceThresholdSlider) faceThresholdSlider.disabled = true;
            }

            // إضافة event listener
            enableFacesCheckbox.addEventListener('change', function() {
                if (this.checked) {
                    faceThresholdContainer.classList.remove('hidden');
                    if (faceThresholdSlider) faceThresholdSlider.disabled = false;
                } else {
                    faceThresholdContainer.classList.add('hidden');
                    if (faceThresholdSlider) faceThresholdSlider.disabled = true;
                }
            });
        }

        // التحكم في عتبة النصوص
        const enableTextCheckbox = document.getElementById('enableText');
        const textThresholdContainer = document.getElementById('textThresholdContainer');
        const textThresholdSlider = document.getElementById('textThreshold');

        if (enableTextCheckbox && textThresholdContainer) {
            if (!enableTextCheckbox.checked) {
                textThresholdContainer.classList.add('hidden');
                if (textThresholdSlider) textThresholdSlider.disabled = true;
            }

            enableTextCheckbox.addEventListener('change', function() {
                if (this.checked) {
                    textThresholdContainer.classList.remove('hidden');
                    if (textThresholdSlider) textThresholdSlider.disabled = false;
                } else {
                    textThresholdContainer.classList.add('hidden');
                    if (textThresholdSlider) textThresholdSlider.disabled = true;
                }
            });
        }

        // التحكم في عتبة الكائنات
        const enableTrackingCheckbox = document.getElementById('enableTracking');
        const objectThresholdContainer = document.getElementById('objectThresholdContainer');
        const objectThresholdSlider = document.getElementById('objectThreshold');

        if (enableTrackingCheckbox && objectThresholdContainer) {
            if (!enableTrackingCheckbox.checked) {
                objectThresholdContainer.classList.add('hidden');
                if (objectThresholdSlider) objectThresholdSlider.disabled = true;
            }

            enableTrackingCheckbox.addEventListener('change', function() {
                if (this.checked) {
                    objectThresholdContainer.classList.remove('hidden');
                    if (objectThresholdSlider) objectThresholdSlider.disabled = false;
                } else {
                    objectThresholdContainer.classList.add('hidden');
                    if (objectThresholdSlider) objectThresholdSlider.disabled = true;
                }
            });
        }

        // التحكم في عناصر تحليل النشاط
        const enableActivityCheckbox = document.getElementById('enableActivity');
        const activityPromptContainer = document.getElementById('activityPromptContainer');
        const activityFpsContainer = document.getElementById('activityFpsContainer');
        const advancedSettingsContainer = document.getElementById('advancedSettingsContainer');

        if (enableActivityCheckbox && activityPromptContainer) {
            if (!enableActivityCheckbox.checked) {
                activityPromptContainer.classList.add('hidden');
                activityFpsContainer.classList.add('hidden');
                advancedSettingsContainer.classList.add('hidden');
            }

            enableActivityCheckbox.addEventListener('change', function() {
                if (this.checked) {
                    activityPromptContainer.classList.remove('hidden');
                    activityFpsContainer.classList.remove('hidden');
                    advancedSettingsContainer.classList.remove('hidden');
                } else {
                    activityPromptContainer.classList.add('hidden');
                    activityFpsContainer.classList.add('hidden');
                    advancedSettingsContainer.classList.add('hidden');
                }
            });
        }
    }

    // Setup drag and drop for files
    function setupDragAndDrop() {
        const dropArea = document.getElementById('fileDropArea');
        const fileInput = document.getElementById('fileInput');
        const uploadedVideoPreview = document.getElementById('uploadedVideoPreview');
        const videoInfoDiv = document.getElementById('videoInfo');

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, unhighlight, false);
        });

        function highlight() {
            dropArea.classList.add('active');
        }

        function unhighlight() {
            dropArea.classList.remove('active');
        }

        dropArea.addEventListener('drop', handleDrop, false);
        fileInput.addEventListener('change', handleFileSelect, false);

        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            handleFiles(files);
        }

        function handleFileSelect(e) {
            const files = e.target.files;
            handleFiles(files);
        }

        function handleFiles(files) {
            if (files.length > 0) {
                const file = files[0];
                if (file.type.startsWith('video/')) {
                    displaySelectedFile(file);
                    const fileURL = URL.createObjectURL(file);
                    uploadedVideoPreview.src = fileURL;
                    uploadedVideoPreview.classList.remove('hidden');

                    uploadedVideoPreview.onloadedmetadata = function() {
                        document.getElementById('videoDuration').textContent = formatDuration(this.duration);
                        document.getElementById('videoFPS').textContent = 'جاري التحديد...';
                        document.getElementById('videoFrames').textContent = 'جاري التحديد...';
                        videoInfoDiv.classList.remove('hidden');
                    };
                } else {
                    showStatus('يرجى اختيار ملف فيديو صالح', 'error');
                    uploadedVideoPreview.classList.add('hidden');
                    videoInfoDiv.classList.add('hidden');
                }
            }
        }
    }

    // تعريف الـ Prompts المحددة مسبقاً
    const promptPresets = {
        'forensic': `You are a forensic video analysis expert. Analyze this surveillance footage systematically with focus on detecting crimes and illegal activities:

        **🔍 Environmental & Context Analysis:**
        - Describe location, time, lighting conditions, and weather
        - Identify venue type (store, street, building, etc.)
        - Note visible landmarks, signs, or distinctive features

        **👥 Suspect Person Analysis:**
        - Count and describe all individuals (approximate age, gender, clothing, distinctive features)
        - Identify masked individuals, people wearing unusual clothing, or attempting to conceal identity
        - Track movements and interactions between people

        **🚨 Criminal Activities - Priority Detection:**
        🔴 **Critical Events:** Weapons presence, assaults, fights, shootings, kidnappings, armed robberies
        🟡 **Suspicious Behaviors:** Unauthorized entry, property damage, theft, hiding objects, rapid movements
        🟢 **Unusual Patterns:** Loitering, frequent coming/going, abandoned objects, vehicle circling

        **⚖️ Specific Criminal Indicators:**
        - Carrying bladed weapons or firearms
        - Breaking locks or doors
        - Shoplifting or property theft
        - Physical assault on persons
        - Exchange of suspicious materials (drugs)
        - Use of force or threats

        **⏱️ Temporal Analysis:**
        - Record exact timestamps of significant events
        - Document sequence of critical incidents
        - Identify timing patterns in activities

        **📸 Evidence Collection:**
        - License plates, vehicle descriptions
        - Clear faces (quality assessment for identification)
        - Objects carried or exchanged
        - Digital evidence (phones, cameras in use)

        Provide detailed description and confidence levels for each observation. Highlight the three most serious incidents requiring immediate investigation.`,

            'threats': `As a security threat detection specialist, focus specifically on:

        **🔫 Weapons & Dangerous Objects:**
        - Firearms (handguns, rifles, shotguns)
        - Knives, sharp objects, hazardous materials
        - Explosives, suspicious packages
        - Tools used for breaking/entering (crowbars, hammers)

        **🚩 Threat Indicators:**
        - Aggressive body language, fighting stances
        - Concealed hands, clothing bulges suggesting hidden objects
        - Protective gear (gloves, masks, helmets)
        - Coordinated group movements suggesting planned action

        **⚠️ Imminent Danger Signals:**
        - Hostage situations, physical restraints
        - Panic reactions from bystanders
        - Rapid evacuation or hiding behaviors
        - Sounds of gunshots, screams, breaking glass

        **👮 Response Assessment:**
        - Police/security presence and response time
        - Civilian reactions and escape patterns
        - Medical emergency responses

        Provide detailed description with confidence levels for each observation. Prioritize immediate threats and provide practical recommendations for law enforcement responses.`,

            'theft': `Focus on property crimes and theft detection:

        **🛍️ Theft Behaviors:**
        - Shoplifting: concealing merchandise, avoiding cameras
        - Bag/package tampering
        - Unauthorized access to restricted areas
        - Breaking into vehicles or buildings

        **💥 Property Assault:**
        - Vandalism: graffiti, broken windows, damaged property
        - Forced entry: broken locks, pried doors
        - Arson attempts, fire-related activities

        **👥 Accomplice Patterns:**
        - Lookouts/distractions working with perpetrators
        - Getaway vehicles and drivers
        - Signal systems between individuals

        **📹 Evidence Collection:**
        - Clear facial captures of perpetrators
        - Vehicle make/model/color/license plates
        - Stolen items description and handling
        - Escape routes and directions

        Provide detailed description, specifying confidence levels for each observation. Document the complete crime timeline from preparation to escape.`,

            'behavior': `Analyze behavioral patterns and suspicious movements:

        **🤔 Suspicious Behavioral Cues:**
        - Nervousness: frequent looking around, checking watches
        - Attempted disguise: hats, sunglasses, masks in inappropriate contexts
        - Unnatural loitering without clear purpose
        - Testing security measures (checking doors, cameras)

        **🚶 Movement Analysis:**
        - Erratic or evasive walking patterns
        - Rapid direction changes to avoid detection
        - Crouching, hiding, or moving in shadows
        - Unusual gathering/dispersal patterns

        **🔍 Pre-incident Indicators:**
        - Surveillance of locations (casing)
        - Equipment preparation (putting on gloves, masks)
        - Communication signals (phone calls, hand signals)
        - Positioning for ambush or attack

        **🎭 Contextual Abnormalities:**
        - Inappropriate clothing for weather/occasion
        - Carrying unusual objects for the location
        - Mismatched group behavior (some watching while others act)

        Provide detailed description with confidence levels for each observation, and suggest follow-up monitoring actions.`,

            'temporal': `Conduct detailed temporal analysis of events:

        **⏰ Chronological Event Mapping:**
        - Create minute-by-minute timeline of significant activities
        - Document exact sequence of critical incidents
        - Note duration of suspicious activities

        **🔄 Pattern Recognition:**
        - Repetitive behaviors or regular visits
        - Timing correlations between different individuals
        - Peak activity periods and lulls

        **🔗 Cause-and-Effect Analysis:**
        - Trigger events initiating suspicious activities
        - Chain reactions between different parties
        - Response patterns to external stimuli

        **⏱️ Timing Anomalies:**
        - Activities occurring at unusual hours
        - Synchronized actions between distant individuals
        - Precise timing suggesting planning/rehearsal

        **📊 Evidence Timeline:**
        - First/last appearance of key individuals
        - Time windows for critical evidentiary moments
        - Duration of observable criminal acts

        Provide detailed description and confidence levels for each observation, presenting results in timeline format consistent with event sequence and video frames.`,

        'emergency': `🔴 Emergency Analysis - For Immediate Response:

        **🚨 Immediate Danger Assessment:**
        - Is there immediate danger to lives?
        - Are there injuries or need for medical assistance?
        - Is the crime still ongoing?

        **📞 Urgent Contact Information:**
        - Exact location of incident
        - Number of suspects and movement direction
        - Type of weapons used (if any)
        - Number and condition of victims

        **🎯 Response Priorities:**
        - Secure area and protect civilians
        - Track and contain suspects
        - Provide urgent medical assistance
        - Preserve crime scene

        Provide brief emergency report with most critical information for immediate response.`};

    // أوصاف الـ Prompts
    const promptDescriptions = {
        'forensic': 'التحليل الجنائي الشامل: فحص كامل للفيديو مع التركيز على الأدلة الجنائية والأنشطة الإجرامية',
        'threats': 'كشف التهديدات الفورية: يركز على الأسلحة والأنشطة الخطرة والاستجابة للطوارئ',
        'theft': 'جرائم الممتلكات: مخصص للسرقة والتخريب والاعتداء على الممتلكات',
        'behavior': 'السلوكيات المشبوهة: يرصد الحركات والأنماط غير الطبيعية',
        'temporal': 'التحليل الزمني: يركز على التسلسل الزمني والأنماط الزمنية للأحداث',
        'emergency': 'الطوارئ والاستجابة السريعة: لتقييم المواقف الخطرة فورياً',
        'custom': 'التخصيص اليدوي: اكتب الـ Prompt الذي تريد استخدامه بشكل مخصص'
    };

    // تحميل الـ Prompt المحدد
    function loadPromptPreset(presetValue) {
        const customContainer = document.getElementById('customPromptContainer');
        const descriptionDiv = document.getElementById('presetDescription');
        const promptTextarea = document.getElementById('activityPrompt');

        if (presetValue === 'custom') {
            // إظهار حقل الإدخال المخصص
            customContainer.classList.remove('hidden');
            promptTextarea.value = ''; // مسح النص الحالي
            promptTextarea.placeholder = 'اكتب الـ Prompt المخصص هنا...';
            descriptionDiv.innerHTML = `<strong>التخصيص اليدوي:</strong> اكتب الـ Prompt الذي تريد استخدامه بشكل مخصص`;
        } else {
            // إخفاء حقل الإدخال المخصص وتعيين الـ Prompt المحدد
            customContainer.classList.add('hidden');
            promptTextarea.value = promptPresets[presetValue];
            descriptionDiv.innerHTML = `<strong>${getPresetDisplayName(presetValue)}:</strong> ${promptDescriptions[presetValue]}`;
        }
    }

    // الحصول على الاسم المعروض للـ Prompt
    function getPresetDisplayName(presetValue) {
        const presetNames = {
            'forensic': '🔍 التحليل الشامل للأدلة الجنائية',
            'threats': '⚠️ كشف التهديدات والأسلحة',
            'theft': '💰 تحليل السرقة والاعتداء على الممتلكات', 
            'behavior': '🚶 تحليل الحركات والسلوكيات المشبوهة',
            'temporal': '⏰ التحليل الزمني والتسلسلي للأحداث',
            'custom': '✏️ تخصيص يدوي'
        };
        return presetNames[presetValue] || presetValue;
    }

    // Check for active processes on page load
    async function checkActiveProcesses() {
        try {
            const response = await fetch('/active-processes');
            if (response.ok) {
                const processes = await response.json();
                if (processes.length > 0) {
                    currentProcessId = processes[0].process_id;
                    showStatus('تم اكتشاف عملية تحليل نشطة', 'info');
                    document.getElementById('stopBtn').disabled = false;
                    startProgressTracking();
                }
            }
        } catch (error) {
            console.log('لا توجد عمليات نشطة');
        }
    }

    // Display selected file info
    function displaySelectedFile(file) {
        const dropArea = document.getElementById('fileDropArea');
        dropArea.innerHTML = `
            <p><strong>📁 ${file.name}</strong></p>
            <p>📏 الحجم: ${formatFileSize(file.size)}</p>
            <p>🎬 النوع: ${file.type}</p>
        `;
        document.getElementById('analyzeBtn').disabled = false;
        window.selectedFile = file;
        document.getElementById('videoName').textContent = file.name;
        document.getElementById('videoSize').textContent = formatFileSize(file.size);
        document.getElementById('videoType').textContent = file.type;
    }

    // Check server health
    async function checkServerStatus() {
        try {
            const response = await fetch('/health');
            if (response.ok) {
                document.getElementById('serverStatus').textContent = '✅ يعمل بشكل صحيح';
                showStatus('الخادم يعمل بشكل صحيح', 'success');
            } else {
                document.getElementById('serverStatus').textContent = '❌ هناك مشكلة';
                showStatus('الخادم لا يستجيب بشكل صحيح', 'error');
            }
        } catch (error) {
            document.getElementById('serverStatus').textContent = '❌ غير متصل';
            showStatus('لا يمكن الاتصال بالخادم', 'error');
        }
    }

    // Update API examples with base URL
    function updateApiExamples() {
        const codeElements = document.querySelectorAll('code');
        codeElements.forEach(el => {
            el.textContent = el.textContent.replace('{{base_url}}', baseUrl);
        });
    }

        // Start video analysis
    async function analyzeVideo() {
        if (!window.selectedFile) {
            showStatus('يرجى اختيار ملف فيديو أولاً', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('file', window.selectedFile);
        formData.append('enable_audio_transcription', document.getElementById('enableAudio').checked);
        formData.append('enable_face_detection', document.getElementById('enableFaces').checked);
        formData.append('enable_text_detection', document.getElementById('enableText').checked);
        formData.append('enable_tracking', document.getElementById('enableTracking').checked);
        formData.append('enable_activity_recognition', document.getElementById('enableActivity').checked);
        formData.append('face_threshold', document.getElementById('faceThreshold').value);
        formData.append('text_threshold', document.getElementById('textThreshold').value);
        formData.append('object_threshold', document.getElementById('objectThreshold').value);
        formData.append('detection_step', document.getElementById('detectionStep').value || 1);

        if (document.getElementById('enableActivity').checked) {
            formData.append('activity_prompt', document.getElementById('activityPrompt').value);
            formData.append('activity_fps', document.getElementById('activityFps').value);

            // إضافة معاملات الضبط المتقدم
            formData.append('max_new_tokens', document.getElementById('maxNewTokens').value);
            formData.append('temperature', document.getElementById('temperature').value);
            formData.append('top_p', document.getElementById('topP').value);
            formData.append('top_k', document.getElementById('topK').value);
            formData.append('do_sample', document.getElementById('doSample').checked);
        } else {
            formData.append('activity_prompt', ''); // أو قيمة افتراضية أخرى
            formData.append('activity_fps', '1'); // أو قيمة افتراضية أخرى
        }


        try {
            showStatus('جاري رفع الفيديو وتحليله...', 'info');
            document.getElementById('progressContainer').classList.remove('hidden');
            document.getElementById('analyzeBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;

            const response = await fetch('/analyze-video', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const result = await response.json();
                currentProcessId = result.process_id;
                showStatus('بدأت المعالجة. جاري تتبع التقدم...', 'success');
                startProgressTracking();
            } else {
                const error = await response.json();
                throw new Error(error.detail || 'فشل في بدء المعالجة');
            }
        } catch (error) {
            showStatus(`خطأ: ${error.message}`, 'error');
            document.getElementById('analyzeBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
        }
    }

    // Stop ongoing analysis
    async function stopAnalysis() {
        if (!currentProcessId) {
            showStatus('لا توجد عملية تحليل جارية', 'warning');
            return;
        }

        try {
            showStatus('جاري إيقاف التحليل وحفظ النتائج...', 'info');
            document.getElementById('stopBtn').disabled = true;

            const response = await fetch(`/stop-analysis/${currentProcessId}`, {
                method: 'POST'
            });

            if (response.ok) {
                const result = await response.json();
                showStatus('تم إيقاف التحليل بنجاح', 'success');
                clearInterval(checkInterval);
                await fetchResults(currentProcessId);
            } else {
                throw new Error('فشل في إيقاف التحليل');
            }
        } catch (error) {
            showStatus(`خطأ في إيقاف التحليل: ${error.message}`, 'error');
            document.getElementById('stopBtn').disabled = false;
        }
    }

    // Track processing progress
    function startProgressTracking() {
        clearInterval(checkInterval);
        checkInterval = setInterval(async () => {
            try {
                const response = await fetch(`/results/${currentProcessId}`);
                if (response.ok) {
                    const result = await response.json();
                    updateProgress(result);

                    if (result.status === 'completed' || result.status === 'stopped') {
                        clearInterval(checkInterval);
                        showFinalResults(result);
                        showTab('results');
                        document.getElementById('stopBtn').disabled = true;
                    } else if (result.status === 'error') {
                        clearInterval(checkInterval);
                        showStatus('حدث خطأ أثناء المعالجة', 'error');
                        document.getElementById('stopBtn').disabled = true;
                    }
                }
            } catch (error) {
                console.error('Error checking progress:', error);
            }
        }, 2000);
    }

    // Update progress bar and text
    function updateProgress(data) {
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');
        const processingDetails = document.getElementById('processingDetails');

        if (data.progress !== undefined) {
            const progress = data.progress;
            progressBar.style.width = `${progress}%`;
            progressText.textContent = `${progress}%`;
            if (data.message) {
                showStatus(data.message, 'info');
            }
            if (data.results && data.results.frames_processed && data.results.total_frames) {
                processingDetails.textContent =
                    `تم معالجة ${data.results.frames_processed} من ${data.results.total_frames} إطار`;
            } else {
                processingDetails.textContent = data.message;
            }
        }
    }

    // Fetch and display results
    async function fetchResults(processId) {
        try {
            const response = await fetch(`/results/${processId}`);
            if (response.ok) {
                const results = await response.json();
                showFinalResults(results);
            }
        } catch (error) {
            showStatus('خطأ في جلب النتائج', 'error');
            console.error('Error fetching results:', error);
        }
    }

    // Display final results
    function showFinalResults(results) {
        document.getElementById('progressContainer').classList.add('hidden');
        const resultsContent = document.getElementById('resultsContent');
        currentResults = results;

        resultsContent.innerHTML = generateResultsHTML(results.results);
        showStatus('تم عرض النتائج بنجاح!', 'success');
    }

    // Generate HTML for the results
    function generateResultsHTML(results) {
        let html = '';

    // إضافة الإحصائيات السريعة
    html += createQuickStats(results);

    // إضافة معاينة الفيديو المحسنة
    html += createVideoPlayer(results);

    // إضافة الجدول التفاعلي
    html += createInteractiveTable(results);

    totalFaces = results.faces_data ? results.faces_data.length : 0;

    // إصلاح مسار الفيديو
    const analyzedVideoPath = results.analyzed_video_path 
        ? `/outputs/${currentProcessId}/${results.analyzed_video_path}`
        : `/outputs/${currentProcessId}/video/analyzed_video_web.mp4`;
    
        // إضافة timestamp لمنع التخزين المؤقت
    const videoWithTimestamp = `${analyzedVideoPath}?t=${Date.now()}`;
    
    const facesFolderPath = `/outputs/${currentProcessId}/faces/`;
    const outputFolderPath = `/outputs/${currentProcessId}/`;

    const facesPerPageSelect = document.getElementById('facesPerPageSelect');
    facesPerPage = facesPerPageSelect ? parseInt(facesPerPageSelect.value) : 4;

    const maxFacesToShow = document.getElementById('maxFacesDisplay') ? 
    parseInt(document.getElementById('maxFacesDisplay').value) || 4 : 4;
    // عرض الوجوه مع التقليب
    const totalPages = Math.ceil(totalFaces / facesPerPage);

    // التأكد من أن الصفحة الحالية ضمن النطاق الصحيح
    if (currentFacesPage > totalPages && totalPages > 0) {
        currentFacesPage = totalPages;
    } else if (currentFacesPage < 1 && totalFaces > 0) {
        currentFacesPage = 1;
    }

    const startIndex = (currentFacesPage - 1) * facesPerPage;
    const endIndex = Math.min(startIndex + facesPerPage, totalFaces);
    const currentPageFaces = results.faces_data ? results.faces_data.slice(startIndex, endIndex) : [];
    totalFaces = results.faces_data ? results.faces_data.length : 0;

    html += `
        <div class="result-card">
            <h3>👥 الوجوه المكتشفة</h3>
            <div class="faces-header">
                <div class="faces-info">
                    <p>عرض ${totalFaces > 0 ? startIndex + 1 : 0}-${endIndex} من ${totalFaces} وجه</p>
                </div>
                <div class="faces-pagination">
                    <button class="btn btn-small ${currentFacesPage === 1 ? 'btn-disabled' : 'btn-primary'}" 
                            onclick="changeFacesPage(-1)" ${currentFacesPage === 1 ? 'disabled' : ''}>
                        ⬅️ السابق
                    </button>
                    <span class="page-info">الصفحة ${currentFacesPage} من ${totalPages}</span>
                    <button class="btn btn-small ${currentFacesPage === totalPages ? 'btn-disabled' : 'btn-primary'}" 
                            onclick="changeFacesPage(1)" ${currentFacesPage === totalPages ? 'disabled' : ''}>
                        التالي ➡️
                    </button>
                </div>
                <div class="faces-per-page">
                    <label for="facesPerPageSelect">عدد الوجوه في الصفحة:</label>
                    <select id="facesPerPageSelect" onchange="changeFacesPerPage(this.value)">
                        <option value="4" ${facesPerPage === 4 ? 'selected' : ''}>4</option>
                        <option value="8" ${facesPerPage === 8 ? 'selected' : ''}>8</option>
                        <option value="12" ${facesPerPage === 12 ? 'selected' : ''}>12</option>
                        <option value="16" ${facesPerPage === 16 ? 'selected' : ''}>16</option>
                    </select>
                </div>
            </div>

            ${currentPageFaces.length > 0 ? `
                <div class="face-grid">
                    ${currentPageFaces.map(face => `
                        <div class="face-item">
                            <img src="/outputs/${currentProcessId}/faces/${face.image_path?.split('/').pop() || 'default.jpg'}"
                                 alt="Face" class="face-image"
                                 onclick="openFaceModal('${face.image_path}', ${face.confidence}, ${face.frame_number})"
                                 onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTUwIiBoZWlnaHQ9IjE1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTUwIiBoZWlnaHQ9IjE1MCIgZmlsbD0iI2VlZSIvPjx0ZXh0IHg9Ijc1IiB5PSI3NSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iIGZpbGw9IiM5OTkiPvCfpqjwn5CSPC90ZXh0Pjwvc3ZnPg=='">
                            <div class="face-info">
                                <p>الإطار: ${face.frame_number}</p>
                                <p>الثقة: ${Math.round(face.confidence * 100)}%</p>
                            </div>
                            <button class="btn btn-small btn-primary" 
                                    onclick="quickEnhanceFace('${face.image_path}', ${face.frame_number}, this)">
                                ✨ تحسين سريع
                            </button>
                        </div>
                    `).join('')}
                </div>
            ` : '<p class="no-faces">' + (totalFaces > 0 ? 'لا توجد وجوه في هذه الصفحة' : 'لم يتم اكتشاف أي وجوه') + '</p>'}
        </div>
    `;


    html += `
        <div class="result-card">
            <h3>📊 نظرة عامة على التحليل</h3>
            <p>حالة المعالجة: <strong>${results.status === 'completed' ? 'مكتمل' : 'متوقف'}</strong></p>
            <p>عدد الإطارات: <strong>${results.total_frames || 0}</strong></p>
            <p>الإطارات المعالجة: <strong>${results.frames_processed || 0}</strong></p>
            <p>مدة الفيديو: <strong>${results.duration_seconds ? formatDuration(results.duration_seconds) : 'غير معروف'}</strong></p>
            <p>معدل الإطارات (FPS): <strong>${results.fps ? results.fps.toFixed(2) : 'غير معروف'}</strong></p>
        </div>

        <div class="results-grid">

            <div class="result-item">
                <h3>📝 النصوص المستخرجة</h3>
                <p>الإجمالي: <strong>${results.texts_detected || 0}</strong></p>
                ${results.extracted_texts && results.extracted_texts.length > 0 ? `
                    ${results.extracted_texts.slice(0, 5).map(text => `
                        <div class="result-card">
                            <p><strong>${text.text}</strong></p>
                            <p>الإطار: ${text.frame_number} | الثقة: ${Math.round(text.confidence * 100)}%</p>
                            <p>اللغة: ${text.language === 'ar' ? 'عربي' : 'إنجليزي'}</p>
                        </div>
                    `).join('')}
                ` : '<p>لم يتم استخراج أي نصوص</p>'}
            </div>
        </div>

        <div class="results-grid">
            <div class="result-item">
                <h3>🔄 تتبع الأشخاص</h3>
                <p>عدد المسارات: <strong>${results.tracks_detected || 0}</strong></p>
                ${results.tracking_data && results.tracking_data.length > 0 ? `
                    ${Array.from(new Set(results.tracking_data.map(t => t.track_id))).slice(0, 3).map(trackId => `
                        <div class="person-track">
                            <p><strong>الشخص ${trackId}</strong></p>
                            <p>تم تتبعه في ${results.tracking_data.filter(t => t.track_id === trackId).length} إطار</p>
                        </div>
                    `).join('')}
                ` : '<p>لم يتم تتبع أي أشخاص</p>'}
            </div>

            <div class="result-item">
                <h3>🎯 تحليل النشاط والبيئة</h3>
                <p><strong>تحليل النشاط والبيئة (إنجليزي):</strong> ${results.activity_analysis && results.activity_analysis.activity_analysis_en ? results.activity_analysis.activity_analysis_en : 'غير معروف'}</p>
                <p><strong>تحليل النشاط والبيئة (عربي):</strong> ${results.activity_analysis && results.activity_analysis.activity_analysis_ar ? results.activity_analysis.activity_analysis_ar : 'غير معروف'}</p>
            </div>
        </div>
        `;

    if (results.transcription && results.transcription.text) {
        html += `
            <div class="result-card">
                <h3>🎵 تحليل الصوت</h3>
                <p>اللغة: <strong>${results.transcription.language || 'غير معروفة'}</strong></p>
                <p>النص الكامل:</p>
                <div class="result-card">
                    <p style="white-space: pre-wrap;">${results.transcription.text}</p>
                </div>
            </div>
        `;
    }

        // Add the final results table here
        html += `
            <div id="finalResultsTableContainer" class="result-card">
                <h3>📋 ملخص النتائج النهائية</h3>
                <table id="finalResultsTable" class="results-table">
                    ${generateFinalResultsTableHTML(results)}
                </table>
            </div>
        `;

        // Add the download and action buttons section
        html += `
            <div class="result-card">
                <h3>📥 تحميل النتائج</h3>
                <div class="processing-actions">
                    <a href="/outputs/${currentProcessId}/final_results.json" class="btn btn-primary" download>
                        📄 التقرير الكامل (JSON)
                    </a>
                    <a href="/outputs/${currentProcessId}/video/analyzed_video.mp4" class="btn btn-success" download>
                        🎥 الفيديو المحلل
                    </a>
                    ${results.transcription ? `
                    <a href="/outputs/${currentProcessId}/audio/transcription.txt" class="btn btn-warning" download>
                        🔊 النص الصوتي
                    </a>
                    ` : ''}
                    <button class="btn btn-info" onclick="openOutputBrowser('${currentProcessId}')">
                        📂 تصفح ملفات المخرجات
                    </button>
                    <button class="btn btn-danger" onclick="analyzeNewVideo()">
                        🎥 تحليل فيديو جديد
                    </button>
                </div>
            </div>
        `;

        return html;
    }

    // إضافة دالة تغيير الصفحة
    function changeFacesPage(direction) {
        const newPage = currentFacesPage + direction;
        const totalPages = Math.ceil(totalFaces / facesPerPage);

        if (newPage >= 1 && newPage <= totalPages) {
            currentFacesPage = newPage;
            // إعادة تحميل النتائج مع الحفاظ على حالة الصفحة
            fetchResults(currentProcessId);
        }
    }

    // إضافة دالة تغيير عدد الوجوه في الصفحة
    function changeFacesPerPage(newValue) {
        facesPerPage = parseInt(newValue);
        currentFacesPage = 1; // العودة للصفحة الأولى
        fetchResults(currentProcessId);
    }
    // Generate HTML for the final results table
    function generateFinalResultsTableHTML(results) {
        let tableHtml = `
            <thead>
                <tr>
                    <th>الميزة</th>
                    <th>القيمة</th>
                </tr>
            </thead>
            <tbody>
        `;

        const addRow = (label, value) => {
            tableHtml += `
                <tr>
                    <td><strong>${label}</strong></td>
                    <td>${value}</td>
                </tr>
            `;
        };

        // Add basic data
        addRow('حالة المعالجة', results.processing_status === 'completed' ? 'مكتمل' : 'متوقف');
        addRow('إجمالي الإطارات', results.total_frames || 0);
        addRow('الإطارات المعالجة', results.frames_processed || 0);
        addRow('مدة الفيديو', results.duration_seconds ? formatDuration(results.duration_seconds) : 'غير معروف');
        addRow('معدل الإطارات (FPS)', results.fps ? results.fps.toFixed(2) : 'غير معروف');
        addRow('الوجوه المكتشفة', results.faces_detected || 0);
        addRow('الوجوه المحسنة', results.faces_enhanced || 0);
        addRow('النصوص المكتشفة', results.texts_detected || 0);
        addRow('مسارات التتبع', results.tracks_detected || 0);
        addRow('الكائنات المكتشفة', results.objects_detected && results.objects_detected[1] ? results.objects_detected[1].join(', ') : 'لا يوجد');

        // Add activity analysis data
        if (results.activity_analysis) {
                addRow('تحليل النشاط والبيئة (انجليزي)', results.activity_analysis.activity_analysis_en || 'غير معروف');

                }
        }

        // Add transcription info
        if (results.transcription && results.transcription.text) {
            addRow('لغة النسخ الصوتي', results.transcription.language || 'غير معروفة');
            addRow('النص الصوتي (أول 200 حرف)', results.transcription.text.substring(0, 200) + '...');
        } else {
            addRow('النسخ الصوتي', 'غير متاح');
        }
        if (results.processing_duration_seconds) {
            addRow('زمن المعالجة', formatDuration(results.processing_duration_seconds));
        addRow('تاريخ المعالجة', results.processing_date || 'غير معروف');

        tableHtml += `
            </tbody>
        `;
        return tableHtml;
    }

    // Switch between tabs
    function showTab(tabName) {
        document.querySelectorAll('.nav-tab').forEach(tab => tab.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
        document.querySelector(`.nav-tab[onclick="showTab('${tabName}')"]`).classList.add('active');
        document.getElementById(`tab-${tabName}`).classList.add('active');
    }

    // Show status message
    function showStatus(message, type) {
        const statusDiv = document.getElementById('statusMessage');
        statusDiv.textContent = message;
        statusDiv.className = `status-message status-${type}`;
    }

    // Format file size
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Format duration
    function formatDuration(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        let result = '';
        if (hours > 0) {
            result += `${hours} ساعة `;
        }
        if (minutes > 0) {
            result += `${minutes} دقيقة `;
        }
        result += `${remainingSeconds} ثانية`;
        return result.trim();
    }

    // Modal functions for browsing outputs
    async function openOutputBrowser(processId) {
        const modal = document.getElementById('outputModal');
        const fileListDiv = document.getElementById('outputFileList');
        const fileViewerDiv = document.getElementById('fileViewer');
        fileListDiv.innerHTML = '<ul><li>جاري تحميل الملفات...</li></li></ul>';
        fileViewerDiv.innerHTML = '';
        modal.style.display = 'block';

        try {
            const response = await fetch(`/outputs_list/${processId}`);
            if (response.ok) {
                const files = await response.json();
                displayOutputFiles(files, processId, fileListDiv);
            } else {
                fileListDiv.innerHTML = '<ul><li>خطأ في تحميل قائمة الملفات.</li></ul>';
            }
        } catch (error) {
            console.error('Error fetching output files:', error);
            fileListDiv.innerHTML = '<ul><li>خطأ في الاتصال بالخادم.</li></ul>';
        }
    }

    function displayOutputFiles(files, processId, fileListDiv, currentPath = '') {
        let html = '<ul>';
        if (currentPath !== '') {
            const parentPath = currentPath.substring(0, currentPath.lastIndexOf('/'));
            html += `<li><a href="#" onclick="navigateOutputFolder('${processId}', '${parentPath}')">⬆️ .. (العودة)</a></li>`;
        }

        files.forEach(item => {
            const fullPath = currentPath === '' ? item.name : `${currentPath}/${item.name}`;
            if (item.type === 'directory') {
                html += `<li><a href="#" onclick="navigateOutputFolder('${processId}', '${fullPath}')">📁 ${item.name}</a></li>`;
            } else {
                const fileUrl = `/outputs/${processId}/${fullPath}`;
                const fileExtension = item.name.split('.').pop().toLowerCase();
                let icon = '📄';
                if (['mp4', 'avi', 'mov', 'mkv', 'webm'].includes(fileExtension)) {
                    icon = '🎥';
                    html += `<li><a href="#" onclick="viewFile('${fileUrl}', 'video')">${icon} ${item.name}</a></li>`;
                } else if (['jpg', 'jpeg', 'png', 'gif', 'bmp'].includes(fileExtension)) {
                    icon = '🖼️';
                    html += `<li><a href="#" onclick="viewFile('${fileUrl}', 'image')">${icon} ${item.name}</a></li>`;
                } else if (['txt', 'json'].includes(fileExtension)) {
                    icon = '📝';
                    html += `<li><a href="#" onclick="viewFile('${fileUrl}', 'text')">${icon} ${item.name}</a></li>`;
                } else {
                    html += `<li><a href="${fileUrl}" download>${icon} ${item.name}</a></li>`;
                }
            }
        });
        html += '</ul>';
        fileListDiv.innerHTML = html;
    }

    async function navigateOutputFolder(processId, path) {
        const fileListDiv = document.getElementById('outputFileList');
        fileListDiv.innerHTML = '<ul><li>جاري تحميل الملفات...</li></ul>';
        document.getElementById('fileViewer').innerHTML = '';
        try {
            const response = await fetch(`/outputs_list/${processId}?path=${path}`);
            if (response.ok) {
                const files = await response.json();
                displayOutputFiles(files, processId, fileListDiv, path);
            } else {
                fileListDiv.innerHTML = '<ul><li>خطأ في تحميل قائمة الملفات.</li></ul>';
            }
        } catch (error) {
            console.error('Error navigating output folder:', error);
            fileListDiv.innerHTML = '<ul><li>خطأ في الاتصال بالخادم.</li></ul>';
        }
    }

    async function viewFile(fileUrl, type) {
        const fileViewerDiv = document.getElementById('fileViewer');
        fileViewerDiv.innerHTML = '';
        if (type === 'image') {
            fileViewerDiv.innerHTML = `<img src="${fileUrl}" alt="عرض الصورة">`;
        } else if (type === 'video') {
            fileViewerDiv.innerHTML = `<video controls src="${fileUrl}"></video>`;
        } else if (type === 'text') {
            try {
                const response = await fetch(fileUrl);
                if (response.ok) {
                    const textContent = await response.text();
                    fileViewerDiv.innerHTML = `<pre>${textContent}</pre>`;
                } else {
                    fileViewerDiv.innerHTML = `<p>خطأ في تحميل الملف النصي.</p>`;
                }
            } catch (error) {
                console.error('Error fetching text file:', error);
                fileViewerDiv.innerHTML = `<p>خطأ في عرض الملف النصي.</p>`;
            }
        }
    }

    function closeModal() {
        document.getElementById('outputModal').style.display = 'none';
        document.getElementById('fileViewer').innerHTML = '';
    }

    window.onclick = function(event) {
        const modal = document.getElementById('outputModal');
        if (event.target == modal) {
            modal.style.display = "none";
            document.getElementById('fileViewer').innerHTML = '';
        }
    }

    // Start a new video analysis
    function analyzeNewVideo() {
        currentProcessId = null;
        clearInterval(checkInterval);
        currentFacesPage = 1; 
        facesPerPage = 4;
        document.getElementById('fileDropArea').innerHTML = `
            <p>📁 اسحب وأسقط ملف الفيديو هنا أو</p>
            <input type="file" id="fileInput" class="file-input" accept="video/*">
            <button class="btn btn-primary">اختر ملف</button>
        `;
        document.getElementById('analyzeBtn').disabled = true;
        document.getElementById('stopBtn').disabled = true;
        document.getElementById('resultsContent').innerHTML = '<p class="status-message status-info">لم يتم تحليل أي فيديو بعد</p>';
        document.getElementById('finalResultsTableContainer').classList.add('hidden');
        document.getElementById('progressContainer').classList.add('hidden');
        document.getElementById('uploadedVideoPreview').classList.add('hidden');
        document.getElementById('uploadedVideoPreview').src = '';
        document.getElementById('videoInfo').classList.add('hidden');
        setupDragAndDrop();
        showTab('upload');
        currentFacesPage = 1;
        showStatus('جاهز لتحليل فيديو جديد', 'info');
    }

        // إنشاء الإحصائيات السريعة
    function createQuickStats(results) {
        return `
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-icon">👥</div>
                <div class="stat-info">
                    <div class="stat-number">${results.faces_detected || 0}</div>
                    <div class="stat-label">الوجوه</div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">📝</div>
                <div class="stat-info">
                    <div class="stat-number">${results.texts_detected || 0}</div>
                    <div class="stat-label">النصوص</div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">🔄</div>
                <div class="stat-info">
                    <div class="stat-number">${results.tracks_detected || 0}</div>
                    <div class="stat-label">المسارات</div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">⏱️</div>
                <div class="stat-info">
                    <div class="stat-number">${results.processing_duration_seconds ? Math.round(results.processing_duration_seconds) : 0}</div>
                    <div class="stat-label">ثانية</div>
                </div>
            </div>
        </div>
        `;
    }

    // إنشاء مشغل الفيديو مع عناصر التحكم
    function createVideoPlayer(results) {
        // استخدام المسار الصحيح للفيديو المحول
        const analyzedVideoPath = results.analyzed_video_path 
            ? `/outputs/${currentProcessId}/${results.analyzed_video_path}`
            : `/outputs/${currentProcessId}/video/analyzed_video_web.mp4`;
    
        // إضافة timestamp لمنع التخزين المؤقت
        const videoWithTimestamp = `${analyzedVideoPath}?t=${Date.now()}`;
    
        return `
        <div class="result-card">
            <h3>🎥 الفيديو المحلل</h3>
            <div class="video-container">
                <video id="mainVideo" class="video-preview" controls preload="metadata" playsinline>
                    <source src="${videoWithTimestamp}" type="video/mp4">
                    متصفحك لا يدعم تشغيل الفيديو.
                </video>
                <div class="video-controls">
                    <button onclick="togglePlayPause()" class="btn btn-small">⏯️ تشغيل/إيقاف</button>
                    <button onclick="skipBackward()" class="btn btn-small">⏪ 5 ثوانٍ</button>
                    <button onclick="skipForward()" class="btn btn-small">⏩ 5 ثوانٍ</button>
                    <button onclick="toggleFullscreen()" class="btn btn-small">📺 ملء الشاشة</button>
                    <button onclick="downloadVideo('${analyzedVideoPath}')" class="btn btn-small">📥 تحميل</button>
                </div>
            </div>
            <div class="video-info">
                <p><strong>الدقة:</strong> ${results.resolution || 'غير معروف'}</p>
                <p><strong>المدة:</strong> ${results.duration_seconds ? formatDuration(results.duration_seconds) : 'غير معروف'}</p>
                <p><strong>معدل الإطارات:</strong> ${results.fps ? results.fps.toFixed(2) + ' fps' : 'غير معروف'}</p>
            </div>
        </div>
        `;
    }

    // دوال التحكم بالفيديو
    function togglePlayPause() {
        const video = document.getElementById('mainVideo');
        if (video.paused) {
            video.play();
        } else {
            video.pause();
        }
    }

    function skipBackward() {
        const video = document.getElementById('mainVideo');
        video.currentTime = Math.max(0, video.currentTime - 5);
    }

    function skipForward() {
        const video = document.getElementById('mainVideo');
        video.currentTime = Math.min(video.duration, video.currentTime + 5);
    }

    function toggleFullscreen() {
        const video = document.getElementById('mainVideo');
        if (!document.fullscreenElement) {
            video.requestFullscreen().catch(err => {
                console.log(`Error attempting to enable fullscreen: ${err.message}`);
            });
        } else {
            document.exitFullscreen();
        }
    }

    function downloadVideo(videoPath) {
        const link = document.createElement('a');
        link.href = videoPath;
        link.download = 'analyzed_video.mp4';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    // إنشاء جدول تفاعلي للنتائج
    function createInteractiveTable(results) {
        return `
        <div class="result-card">
            <h3>📋 النتائج التفصيلية</h3>
            <div class="table-controls">
                <input type="text" id="searchTable" placeholder="🔍 ابحث في النتائج..." class="search-input" onkeyup="filterTable()">
                <select id="filterCategory" class="filter-select" onchange="filterTable()">
                    <option value="all">جميع الفئات</option>
                    <option value="faces">الوجوه</option>
                    <option value="texts">النصوص</option>
                    <option value="objects">الكائنات</option>
                </select>
            </div>
            <div class="results-table-container">
                <table class="results-table" id="detailedResults">
                    <thead>
                        <tr>
                            <th>النوع</th>
                            <th>العدد</th>
                            <th>التفاصيل</th>
                            <th>الثقة</th>
                            <th>الإطار</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${generateTableRows(results)}
                    </tbody>
                </table>
            </div>
        </div>
        `;
    }

    function generateTableRows(results) {
        let rows = '';

        // إضافة الوجوه
        if (results.faces_data && Array.isArray(results.faces_data)) {
            results.faces_data.forEach(face => {
                rows += `
                <tr data-category="faces">
                    <td>👥 وجه</td>
                    <td>1</td>
                    <td>معرف ${face.face_id || 'غير معروف'}</td>
                    <td>${Math.round(face.confidence * 100)}%</td>
                    <td>${face.frame_number}</td>
                </tr>
                `;
            });
        }

        // إضافة النصوص
        if (results.extracted_texts && Array.isArray(results.extracted_texts)) {
            results.extracted_texts.forEach(text => {
                rows += `
                <tr data-category="texts">
                    <td>📝 نص</td>
                    <td>1</td>
                    <td>${text.text ? text.text.substring(0, 30) : ''}${text.text && text.text.length > 30 ? '...' : ''}</td>
                    <td>${Math.round(text.confidence * 100)}%</td>
                    <td>${text.frame_number}</td>
                </tr>
                `;
            });
        }

        // إضافة الكائنات (إذا كانت متاحة)
        if (results.objects_detected && Array.isArray(results.objects_detected[1])) {
            results.objects_detected[1].forEach((obj, index) => {
                rows += `
                <tr data-category="objects">
                    <td>📦 كائن</td>
                    <td>1</td>
                    <td>${obj}</td>
                    <td>100%</td>
                    <td>غير محدد</td>
                </tr>
                `;
            });
        }

        return rows;
    }

    // تصفية الجدول
    function filterTable() {
        const input = document.getElementById('searchTable');
        const filter = input.value.toUpperCase();
        const category = document.getElementById('filterCategory').value;
        const table = document.getElementById('detailedResults');
        const tr = table.getElementsByTagName('tr');

        for (let i = 1; i < tr.length; i++) {
            const td = tr[i].getElementsByTagName('td');
            let show = false;
            for (let j = 0; j < td.length; j++) {
                if (td[j]) {
                    const txtValue = td[j].textContent || td[j].innerText;
                    if (txtValue.toUpperCase().indexOf(filter) > -1) {
                        show = true;
                        break;
                    }
                }
            }

            const rowCategory = tr[i].getAttribute('data-category');
            const categoryMatch = category === 'all' || rowCategory === category;

            tr[i].style.display = show && categoryMatch ? '' : 'none';
        }
    }

    // متغيرات عالمية لإدارة حالة الوجه
    let currentFaceData = {
        originalPath: '',
        currentState: null,
        frameNumber: 0,
        confidence: 0,
        enhancementStack: [], // سجل التحسينات المطبقة
        currentImage: '',
        enhancementHistory: [] // تاريخ الصور للتراجع
    };

    // فتح نافذة الوجه
    function openFaceModal(imagePath, confidence, frameNumber) {
        const originalImageUrl = `/outputs/${currentProcessId}/faces/${imagePath.split('/').pop()}`;

        currentFaceData = {
            originalPath: imagePath,
            currentState: 'original',
            frameNumber: frameNumber,
            confidence: confidence,
            enhancementStack: [],
            currentImage: originalImageUrl,
            enhancementHistory: [{
                image: originalImageUrl,
                state: 'original',
                action: 'الصورة الأصلية'
            }]
        };

        document.getElementById('modalFaceImage').src = currentFaceData.currentImage;
        document.getElementById('modalFaceFrame').textContent = frameNumber;
        document.getElementById('modalFaceConfidence').textContent = Math.round(confidence * 100);
        document.getElementById('modalFaceStatus').textContent = 'الأصلية';
        updateActionsDisplay();

        document.getElementById('faceModal').style.display = 'block';
    }
    // تحديث عرض الإجراءات
    function updateActionsDisplay() {
        const actionsElement = document.getElementById('modalFaceActions');
        if (currentFaceData.enhancementStack.length === 0) {
            actionsElement.textContent = 'لا يوجد';
        } else {
            actionsElement.textContent = currentFaceData.enhancementStack
                .map(enh => getEnhancementName(enh.type))
                .join(' → ');
        }
    }

    // تطبيق التحسينات بشكل تسلسلي
    async function applyEnhancement(type) {
        const imageElement = document.getElementById('modalFaceImage');
        const statusElement = document.getElementById('modalFaceStatus');

        try {
            statusElement.textContent = `جاري تطبيق ${getEnhancementName(type)}...`;

            // إضافة التحسين إلى السجل
            currentFaceData.enhancementStack.push({
                type: type,
                parameters: getEnhancementParameters(type),
                timestamp: new Date().toISOString()
            });

            const response = await fetch('/enhance-face-sequence', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    process_id: currentProcessId,
                    original_path: currentFaceData.originalPath,
                    enhancement_stack: currentFaceData.enhancementStack
                })
            });

            if (response.ok) {
                const result = await response.json();

                // تحديث الحالة الحالية
                currentFaceData.currentImage = `/outputs/${currentProcessId}/faces/${result.enhanced_filename}?t=${Date.now()}`;
                currentFaceData.currentState = 'enhanced';

                // إضافة إلى تاريخ التراجع
                currentFaceData.enhancementHistory.push({
                    image: currentFaceData.currentImage,
                    state: 'enhanced',
                    action: getEnhancementName(type)
                });

                imageElement.src = currentFaceData.currentImage;
                statusElement.textContent = `محسنة - ${getEnhancementName(type)}`;
                updateActionsDisplay();

                toggleEnhancementSliders(type);
            } else {
                // إزالة التحسين من السجل إذا فشل
                currentFaceData.enhancementStack.pop();
                throw new Error('فشل في تطبيق التحسين');
            }
        } catch (error) {
            statusElement.textContent = 'خطأ في التحسين';
            console.error('Error enhancing face:', error);
        }
    }

    // التراجع عن آخر عملية
    function undoEnhancement() {
        if (currentFaceData.enhancementHistory.length > 1) {
            // إزالة آخر تحسين من السجل
            currentFaceData.enhancementStack.pop();
            currentFaceData.enhancementHistory.pop();

            // العودة إلى الحالة السابقة
            const previousState = currentFaceData.enhancementHistory[currentFaceData.enhancementHistory.length - 1];
            currentFaceData.currentImage = previousState.image;
            currentFaceData.currentState = previousState.state;

            document.getElementById('modalFaceImage').src = currentFaceData.currentImage;
            document.getElementById('modalFaceStatus').textContent = 
                previousState.state === 'original' ? 'الأصلية' : 'محسنة';
            updateActionsDisplay();

            // إذا كان هناك تحسينات متبقية، نعيد تطبيقها
            if (currentFaceData.enhancementStack.length > 0) {
                reapplyEnhancements();
            }
        }
    }

    // إعادة تطبيق التحسينات (للعرض الفوري)
    async function reapplyEnhancements() {
        if (currentFaceData.enhancementStack.length === 0) return;

        try {
            const response = await fetch('/enhance-face-sequence', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    process_id: currentProcessId,
                    original_path: currentFaceData.originalPath,
                    enhancement_stack: currentFaceData.enhancementStack
                })
            });

            if (response.ok) {
                const result = await response.json();
                currentFaceData.currentImage = `/outputs/${currentProcessId}/faces/${result.enhanced_filename}?t=${Date.now()}`;
                document.getElementById('modalFaceImage').src = currentFaceData.currentImage;
            }
        } catch (error) {
            console.error('Error reapplying enhancements:', error);
        }
    }
    // إغلاق نافذة الوجه
    function closeFaceModal() {
        document.getElementById('faceModal').style.display = 'none';
        resetFaceImage();
    }

    // الحصول على اسم التحسين
    function getEnhancementName(type) {
        const names = {
            'super_resolution': 'تحسين الدقة',
            'sharpen': 'زيادة الحدة',
            'contrast': 'زيادة التباين',
            'smooth': 'تنعيم الصورة'
        };
        return names[type] || type;
    }

    // الحصول على معاملات التحسين
    function getEnhancementParameters(type) {
        const params = {};
        switch(type) {
            case 'sharpen':
                params.strength = parseFloat(document.getElementById('sharpenAmount').value);
                break;
            case 'contrast':
                params.strength = parseFloat(document.getElementById('contrastAmount').value);
                break;
            case 'smooth':
                params.strength = parseFloat(document.getElementById('smoothAmount').value);
                break;
        }
        return params;
    }

    // إظهار/إخفاء عناصر التحكم المنزلقة
    function toggleEnhancementSliders(type) {
        const sliders = document.getElementById('enhancementSliders');
        if (['sharpen', 'contrast', 'smooth'].includes(type)) {
            sliders.classList.remove('hidden');
        } else {
            sliders.classList.add('hidden');
        }
    }


    // حفظ الصورة المحسنة
    async function saveEnhancedFace() {
        if (currentFaceData.enhancementStack.length === 0) {
            alert('⚠️ يرجى تطبيق تحسين على الصورة أولاً');
            return;
        }

        try {
            const response = await fetch('/save-enhanced-face', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    process_id: currentProcessId,
                    original_path: currentFaceData.originalPath,
                    enhanced_path: currentFaceData.currentImage.split('/').pop().split('?')[0],
                    enhancement_stack: currentFaceData.enhancementStack
                })
            });

            if (response.ok) {
                const result = await response.json();
                alert('✅ تم حفظ الصورة المحسنة بنجاح');
                document.getElementById('modalFaceStatus').textContent += ' - محفوظة';
            } else {
                throw new Error('فشل في حفظ الصورة');
            }
        } catch (error) {
            alert('❌ خطأ في حفظ الصورة');
            console.error('Error saving enhanced face:', error);
        }
    }

    // تحسين سريع للوجه في الشبكة
    async function quickEnhanceFace(imagePath, frameNumber, buttonElement) {
        try {
            buttonElement.disabled = true;
            buttonElement.textContent = 'جاري التحسين...';

            const response = await fetch('/enhance-face', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    process_id: currentProcessId,
                    image_path: imagePath,
                    enhancement_type: 'super_resolution'
                })
            });

            if (response.ok) {
                const result = await response.json();
                // تحديث الصورة في الشبكة
                const faceElement = buttonElement.closest('.face-item');
                const imgElement = faceElement.querySelector('.face-image');
                imgElement.src = `/outputs/${currentProcessId}/faces/${result.enhanced_filename}?t=${Date.now()}`;
                buttonElement.textContent = '✓ تم التحسين';
                buttonElement.classList.remove('btn-primary');
                buttonElement.classList.add('btn-success');
            } else {
                throw new Error('فشل في تحسين الصورة');
            }
        } catch (error) {
            buttonElement.textContent = '❌ خطأ';
            console.error('Error in quick enhance:', error);
        }
    }

    // إعادة تعيين الصورة
    function resetFaceImage() {
        if (currentFaceData.originalPath) {
            currentFaceData.enhancementStack = [];
            currentFaceData.enhancementHistory = [{
                image: `/outputs/${currentProcessId}/faces/${currentFaceData.originalPath.split('/').pop()}`,
                state: 'original',
                action: 'الصورة الأصلية'
            }];
            currentFaceData.currentImage = currentFaceData.enhancementHistory[0].image;

            document.getElementById('modalFaceImage').src = currentFaceData.currentImage;
            document.getElementById('modalFaceStatus').textContent = 'الأصلية';
            updateActionsDisplay();
        }
    }

    // تحديث قيم السلايدرز
    function setupEnhancementSliders() {
        const sliders = [
            { id: 'sharpenAmount', valueId: 'sharpenValue' },
            { id: 'contrastAmount', valueId: 'contrastValue' },
            { id: 'smoothAmount', valueId: 'smoothValue' }
        ];

        sliders.forEach(slider => {
            const element = document.getElementById(slider.id);
            const valueElement = document.getElementById(slider.valueId);
            if (element && valueElement) {
                element.addEventListener('input', function() {
                    valueElement.textContent = this.value;
                });
                valueElement.textContent = element.value;
            }
        });
    }

    // استدعاء الإعداد عند تحميل الصفحة
    document.addEventListener('DOMContentLoaded', function() {
        setupEnhancementSliders();
        setupAdvancedSettingsSliders();
        loadPromptPreset('forensic');
    });
        // التحكم في إظهار قسم الضبط المتقدم
    function toggleAdvancedSettings(show) {
        const advancedContainer = document.getElementById('advancedSettingsContainer');
        if (show) {
            advancedContainer.classList.remove('hidden');
        } else {
            advancedContainer.classList.add('hidden');
        }
    }

    // تحديث قيم السلايدرز للضبط المتقدم
    function setupAdvancedSettingsSliders() {
        // تحديث جميع سلايدرز الضبط المتقدم
        const advancedSliders = [
            { id: 'maxNewTokens', valueId: 'maxNewTokensValue' },
            { id: 'temperature', valueId: 'temperatureValue' },
            { id: 'topP', valueId: 'topPValue' },
            { id: 'topK', valueId: 'topKValue' },
            { id: 'enhancementStrength', valueId: 'enhancementStrengthValue' }
        ];

        advancedSliders.forEach(slider => {
            updateAdvancedSliderValue(slider.id, slider.valueId);
        });
    }

    // دالة مخصصة لتحديث قيم الضبط المتقدم
    function updateAdvancedSliderValue(sliderId, valueId) {
        const slider = document.getElementById(sliderId);
        const valueSpan = document.getElementById(valueId);

        if (slider && valueSpan) {
            // تحديث القيمة فوراً عند التحميل
            valueSpan.textContent = slider.value;

            // إضافة مستمع event للتحديث عند التغيير
            slider.addEventListener('input', function() {
                valueSpan.textContent = this.value;
                console.log(`✅ ${sliderId} updated to: ${this.value}`); // للتdebug
            });

            // إضافة مستمع event للتحديث عند تغيير الزر أيضاً
            slider.addEventListener('change', function() {
                valueSpan.textContent = this.value;
            });
        } else {
            console.error(`❌ Element not found: ${sliderId} or ${valueId}`);
        }
    }

</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def read_root():
    return HTMLResponse(content=HTML_TEMPLATE)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/active-processes")
async def get_active_processes():
    """الحصول على قائمة بالعمليات النشطة"""
    try:
        # البحث عن العمليات التي لا تزال قيد المعالجة
        active = []
        for process_id in list(active_processes.keys()):
            process_info = db.get_process_status(process_id)
            if process_info and process_info["status"] in ["processing", "starting"]:
                active.append({
                    "process_id": process_id,
                    "filename": process_info.get("filename", ""),
                    "status": process_info["status"],
                    "progress": process_info["progress"]
                })
        return active
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في الحصول على العمليات النشطة: {str(e)}")


SERVICE_CONFIG = {
    "audio": {"url": "http://audio_transcription_service:8001", "timeout": 300},
    "faces": {"url": "http://face_detection_service:8002", "timeout": 60},
    "texts": {"url": "http://text_detection_service:8003", "timeout": 60},
    "objects": {"url": "http://objects_detection_and_tracking_service:8004", "timeout": 60},
    "activity": {"url": "http://activity_analysis_service:8005", "timeout": 600}
}

import asyncio
import aiohttp

# دالة انتظار جاهزية الخدمة
async def wait_for_service(service_name: str, max_retries: int = 10, delay: int = 5) -> bool:
    """
    تنتظر حتى تكون الخدمة جاهزة من خلال التحقق من نقطة نهاية /health.

    :param service_name: اسم الخدمة (مثل 'activity')
    :param max_retries: عدد المحاولات القصوى (افتراضي: 10)
    :param delay: التأخير بين المحاولات بالثواني (افتراضي: 5)
    :return: True إذا كانت الخدمة جاهزة، False إلا
    """
    service_config = SERVICE_CONFIG.get(service_name)
    if not service_config:
        logger.error(f"❌ Service {service_name} not found in SERVICE_CONFIG")
        return False

    health_url = f"{service_config['url']}/health"

    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("status") == "healthy":
                            logger.info(f"✅ Service {service_name} is ready (attempt {attempt + 1})")
                            return True
                        else:
                            logger.warning(f"⚠️ Service {service_name} health check failed: {data}")
                    else:
                        logger.warning(f"⚠️ Service {service_name} returned status {response.status}")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"⚠️ Attempt {attempt + 1} failed for {service_name}: {e}")

        # انتظر قبل المحاولة التالية
        await asyncio.sleep(delay)

    logger.error(f"❌ Service {service_name} is not ready after {max_retries} attempts")
    return False


async def call_service(service_name: str, endpoint: str, files=None, data=None):
    """دالة مساعدة للاتصال بالخدمات"""

    if not await wait_for_service(service_name, max_retries=10, delay=5):
        logger.error(f"❌ Service {service_name} is not ready, skipping call")
        return None

    service_config = SERVICE_CONFIG[service_name]
    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                form_data = aiohttp.FormData()

                # إضافة الملفات إذا كانت موجودة
                if files:
                    for key, file_info in files.items():
                        form_data.add_field(
                            key,
                            file_info['content'],
                            filename=file_info['filename'],
                            content_type=file_info.get('content_type', 'application/octet-stream')
                        )

                # إضافة البيانات إذا كانت موجودة
                if data:
                    for key, value in data.items():
                        if value is not None:
                            form_data.add_field(key, str(value))

                # زيادة المهلة لخدمة تحليل النشاط
                timeout = aiohttp.ClientTimeout(total=service_config['timeout'])

                async with session.post(
                        f"{service_config['url']}/{endpoint}",
                        data=form_data,
                        timeout=timeout
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Service {service_name} returned status {response.status}: {error_text}")

                        # إعادة المحاولة للأخطاء 5xx
                        if response.status >= 500 and attempt < max_retries - 1:
                            logger.info(f"🔄 إعادة المحاولة {attempt + 1} لخدمة {service_name}...")
                            await asyncio.sleep(retry_delay)
                            continue
                        return None

        except asyncio.TimeoutError:
            logger.error(f"❌ Timeout calling {service_name} service (المحاولة {attempt + 1})")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                continue
            return None
        except Exception as e:
            logger.error(f"❌ Error calling {service_name} service (المحاولة {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                continue
            return None

    return None

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


def convert_video_to_web_format(input_path: str, output_path: str) -> bool:
    """تحويل الفيديو إلى تنسيق متوافق مع الويب"""
    try:
        if not Path(input_path).exists():
            print(f"❌ ملف الفيديو الأصلي غير موجود: {input_path}")
            return False

        # استخدام إعدادات متوافقة مع جميع المتصفحات
        command = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-profile:v', 'baseline',  # لضمان التوافق
            '-level', '3.0',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',  # للتحميل التدريجي
            '-y',
            '-loglevel', 'error',
            output_path
        ]

        result = subprocess.run(command, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print(f"✅ تم تحويل الفيديو بنجاح: {output_path}")
            return True
        else:
            print(f"❌ فشل تحويل الفيديو: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"❌ انتهت مهلة تحويل الفيديو")
        return False
    except Exception as e:
        print(f"❌ خطأ في تحويل الفيديو: {e}")
        return False

async def process_video(input_path: str, process_id: str, options: dict):
    """الدالة الرئيسية لمعالجة الفيديو - محدثة للخدمات"""
    try:
        # إنشاء مجلدات العملية
        process_dir = OUTPUTS_DIR / process_id
        faces_dir = process_dir / "faces"
        video_dir = process_dir / "video"
        audio_dir = process_dir / "audio"

        for directory in [process_dir, faces_dir, video_dir, audio_dir]:
            directory.mkdir(exist_ok=True, parents=True)
        # في دالة process_video، أضف هذه السجلات
        logger.info(f"🎬 بدء معالجة الفيديو: {process_id}")
        logger.info(f"⚙️ خيارات المعالجة: {options}")

        # في نهاية process_video

        db.update_process_status(process_id, "processing", 5, "جاري فتح الفيديو")

        await asyncio.sleep(0.1)

        start_time = time.time()
        # الحصول على معلومات الفيديو الأساسية
        video_info = await get_video_info(input_path)
        if not video_info:
            db.update_process_status(process_id, "error", 0, "تعذر فتح ملف الفيديو")
            return

        all_results = {
            "process_id": process_id,
            **video_info,
            "processing_date": datetime.now().isoformat(),
            "processing_options": options
        }

        # 1. معالجة الصوت (إذا مُفعل)
        if options.get("enable_audio_transcription", True):
            await process_audio_service(input_path, process_id, options, all_results)

        # 2. معالجة الفيديو (الوجوه، النصوص، الكائنات)
        if any([options.get("enable_face_detection", True),
                options.get("enable_text_detection", True),
                options.get("enable_tracking", True)]):
            await process_video_services(input_path, process_id, options, all_results)

        # 3. تحليل النشاط (إذا مُفعل)
        if options.get("enable_activity_recognition", True):
            await process_activity_service(input_path, process_id, options, all_results)

        # 4. حفظ النتائج النهائية
        final_video_path = await ensure_video_output(process_id, input_path)
        if final_video_path:
            all_results["analyzed_video_path"] = str(Path(final_video_path).relative_to(OUTPUTS_DIR / process_id))

        # حفظ النتائج النهائية
        await save_final_results(process_id, all_results)
        await save_final_results(process_id, all_results)

        db.update_process_status(process_id, "completed", 100, "تمت المعالجة بنجاح")
        logger.info(f"🎉 تمت معالجة الفيديو بنجاح! العملية: {process_id}")
        logger.info(f"✅ انتهت معالجة الفيديو: {process_id}")
        logger.info(f"📊 النتائج المجمعة: {all_results.keys()}")
        end_time = time.time()
        processing_duration = end_time - start_time
        all_results["processing_duration_seconds"] = str(processing_duration)


    except Exception as e:
        error_msg = f"خطأ في المعالجة: {str(e)}"
        logger.error(f"❌ {error_msg}")
        db.update_process_status(process_id, "error", 0, error_msg)


async def ensure_video_output(process_id: str, input_path: str):
    """التأكد من وجود فيديو مخرج، وإنشاء نسخة من الفيديو الأصلي إذا لزم الأمر"""
    try:
        video_dir = OUTPUTS_DIR / process_id / "video"
        web_video_path = video_dir / "analyzed_video_web.mp4"

        # إذا كان الفيديو المحول موجوداً، استخدمه
        if web_video_path.exists():
            return str(web_video_path)

        # إذا لم يكن موجوداً، أنشئ نسخة من الفيديو الأصلي
        if convert_video_to_web_format(input_path, str(web_video_path)):
            return str(web_video_path)
        else:
            # إذا فشل التحويل، انسخ الفيديو الأصلي
            original_copy = video_dir / "original_video.mp4"
            import shutil
            shutil.copy2(input_path, original_copy)
            return str(original_copy)

    except Exception as e:
        logger.error(f"❌ خطأ في إنشاء فيديو المخرجات: {e}")
        return None

async def process_audio_service(input_path: str, process_id: str, options: dict, all_results: dict):
    """معالجة الصوت عبر الخدمة المخصصة"""
    try:
        db.update_process_status(process_id, "processing", 10, "جاري معالجة الصوت")

        with open(input_path, 'rb') as video_file:
            files = {
                'file': {
                    'content': video_file.read(),
                    'filename': Path(input_path).name
                }
            }
            response = await call_service("audio", "transcribe", files=files)
            print(response)

            if response:
                all_results["transcription"] = response

                # حفظ النتيجة في ملف
                transcription_file = OUTPUTS_DIR / process_id / "transcription.json"
                with open(transcription_file, 'w', encoding='utf-8') as f:
                    json.dump(response, f, ensure_ascii=False, indent=2)

                logger.info(f"✅ تم تحويل الصوت إلى نص: {len(response.get('text', ''))} حرف")
            else:
                logger.error("❌ فشل في خدمة تحويل الصوت إلى نص")
                all_results["transcription"] = {"error": "فشل في معالجة الصوت"}

    except Exception as e:
        logger.error(f"❌ خطأ في معالجة الصوت: {e}")
        all_results["transcription"] = {"error": str(e)}


async def process_video_services(input_path: str, process_id: str, options: dict, all_results: dict):
    """معالجة إطارات الفيديو عبر الخدمات المخصصة"""
    video_dir = OUTPUTS_DIR / process_id / "video"
    try:
        db.update_process_status(process_id, "processing", 20, "جاري معالجة إطارات الفيديو")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return

        # إعداد فيديو الإخراج
        output_video_path = OUTPUTS_DIR / process_id / "video" / "analyzed_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, all_results["fps"],
                              (all_results["width"], all_results["height"]))

        frame_number = 0
        all_faces = []
        all_texts = []
        all_objects = []
        all_tracks = []

        # إضافة متغير لتتبع المسارات
        trajectory_history = {}

        detection_step = options.get("detection_step", 1)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = frame.copy()

            # معالجة الإطار كل detection_step إطارات
            if frame_number % detection_step == 0:
                # معالجة متوازية للخدمات
                tasks = []

                if options.get("enable_face_detection", True):
                    tasks.append(process_frame_faces(frame, frame_number, process_id, options))

                if options.get("enable_text_detection", True):
                    tasks.append(process_frame_texts(frame, frame_number, options))

                if options.get("enable_tracking", True):
                    tasks.append(process_frame_objects(frame, frame_number, options))

                # تنفيذ جميع المهام بشكل متوازي
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # معالجة النتائج
                for result in results:
                    if isinstance(result, Exception):
                        continue
                    if result and isinstance(result, dict):
                        if "faces" in result:
                            all_faces.extend(result["faces"])
                            processed_frame = draw_faces_on_frame(processed_frame, result["faces"])
                        if "texts" in result:
                            all_texts.extend(result["texts"])
                            processed_frame = draw_texts_on_frame(processed_frame, result["texts"])
                        if "objects" in result:
                            all_objects.extend(result["objects"])
                            all_tracks.extend(result.get("tracks", []))
                            processed_frame = draw_objects_on_frame(processed_frame, result)

                            # رسم المسارات
                            processed_frame, trajectory_history = draw_trajectories_on_frame(
                                processed_frame, result.get("tracks", []), trajectory_history
                            )

            out.write(processed_frame)
            frame_number += 1

            # تحديث التقدم
            if frame_number % 10 == 0:
                progress = 20 + int((frame_number / all_results["total_frames"]) * 60)
                progress = min(progress, 80)
                db.update_process_status(process_id, "processing", progress,
                                         f"جاري معالجة الإطار {frame_number}/{all_results['total_frames']}")

        out.release()
        cap.release()

        # استخدام الفيديو المحول فقط
        web_video_path = str(video_dir / "analyzed_video_web.mp4")
        if convert_video_to_web_format(output_video_path, web_video_path):
            # حذف الفيديو الأصلي والاحتفاظ بالمحول فقط
            if os.path.exists(output_video_path):
                os.remove(output_video_path)
            final_video_path = web_video_path
            print("✅ تم تحويل الفيديو إلى تنسيق ويب متوافق")
        else:
            final_video_path = output_video_path
            print("⚠️ فشل تحويل الفيديو، سيتم استخدام الفيديو الأصلي")

        # حفظ نتائج الكشف
        await save_detection_results(process_id, all_faces, all_texts, all_objects, all_tracks)

        all_results.update({
            "frames_processed": frame_number,
            "faces_detected": len(all_faces),
            "texts_detected": len(all_texts),
            "objects_detected": len(all_objects),
            "tracks_detected": len(set(track["track_id"] for track in all_tracks if track.get("track_id"))),
            "video_filename": f"{process_id}/video/analyzed_video_web.mp4"
        })

    except Exception as e:
        logger.error(f"❌ خطأ في معالجة إطارات الفيديو: {e}")


async def process_frame_faces(frame: np.ndarray, frame_number: int, process_id: str, options: dict):
    """معالجة وجه واحدة عبر خدمة الوجوه"""
    try:
        # حفظ الإطار مؤقتاً
        _, buffer = cv2.imencode('.jpg', frame)
        image_data = buffer.tobytes()

        files = {
            'file': {
                'content': image_data,
                'filename': f'frame_{frame_number}.jpg'
            }
        }
        data = {'threshold': options.get('face_threshold', 0.3)}

        response = await call_service("faces", "detect", files=files, data=data)

        if response and 'faces' in response:
            faces = response.get('faces', [])
            logger.info(f"✅ تم كشف {len(faces)} وجوه في الإطار {frame_number}")

            processed_faces = []
            for face in faces:
                if isinstance(face, dict) and 'bbox' in face:
                    face['frame_number'] = frame_number
                    # حفظ صورة الوجه
                    await save_face_image(frame, face, frame_number, process_id)
                    processed_faces.append(face)
                else:
                    logger.warning(f"⚠️ تنسيق بيانات الوجه غير صالح: {face}")

            return {"faces": processed_faces}
        else:
            logger.warning(f"⚠️ لم يتم العثور على وجوه في الإطار {frame_number} أو استجابة غير صالحة")
            return {"faces": []}

    except Exception as e:
        logger.error(f"❌ خطأ في كشف الوجوه للإطار {frame_number}: {e}")
        return {"faces": []}


async def process_frame_texts(frame: np.ndarray, frame_number: int, options: dict):
    """معالجة النصوص في إطار واحد عبر خدمة النصوص"""
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        image_data = buffer.tobytes()

        files = {
            'file': {
                'content': image_data,
                'filename': f'frame_{frame_number}.jpg'
            }
        }
        data = {'threshold': options.get('text_threshold', 0.3)}

        response = await call_service("texts", "detect", files=files, data=data)

        if response:
            texts = response.get('texts', [])

            for text in texts:
                text['frame_number'] = frame_number

            return {"texts": texts}

    except Exception as e:
        logger.error(f"❌ خطأ في كشف النصوص للإطار {frame_number}: {e}")

    return {"texts": []}


async def process_frame_objects(frame: np.ndarray, frame_number: int, options: dict):
    """معالجة الكائنات في إطار واحد عبر خدمة الكائنات"""
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        image_data = buffer.tobytes()

        files = {
            'file': {
                'content': image_data,
                'filename': f'frame_{frame_number}.jpg'
            }
        }
        data = {'threshold': options.get('object_threshold', 0.5)}

        response = await call_service("objects", "detect", files=files, data=data)

        if response:
            objects_data = response.get('objects', [])
            tracks_data = response.get('tracks', [])

            # التأكد من أن البيانات في التنسيق الصحيح
            processed_objects = []
            for obj in objects_data:
                if isinstance(obj, dict):
                    obj['frame_number'] = frame_number
                    # تحويل numpy types إلى Python types
                    if 'confidence' in obj and hasattr(obj['confidence'], 'item'):
                        obj['confidence'] = obj['confidence'].item()
                    processed_objects.append(obj)

            processed_tracks = []
            for track in tracks_data:
                if isinstance(track, dict):
                    track['frame_number'] = frame_number
                    processed_tracks.append(track)

            return {
                "objects": processed_objects,
                "tracks": processed_tracks
            }
        else:
            return {"objects": [], "tracks": []}

    except Exception as e:
        logger.error(f"❌ خطأ في كشف الكائنات للإطار {frame_number}: {e}")
        return {"objects": [], "tracks": []}


async def process_activity_service(input_path: str, process_id: str, options: dict, all_results: dict):
    """معالجة تحليل النشاط عبر الخدمة المخصصة"""
    try:
        db.update_process_status(process_id, "processing", 85, "جاري تحليل النشاط والبيئة")

        with open(input_path, 'rb') as video_file:
            files = {
                'file': {
                    'content': video_file.read(),
                    'filename': Path(input_path).name
                }
            }
            data = {
                'prompt': options.get('activity_prompt', 'Describe the activities in this video'),
                'max_new_tokens': options.get('max_new_tokens', 600),
                'temperature': options.get('temperature', 0.3),
                'top_p': options.get('top_p', 0.9),
                'top_k': options.get('top_k', 50),
                'do_sample': options.get('do_sample', True),
                'enable_enhancement': options.get('enable_video_enhancement', False),
                'enhancement_strength': options.get('enhancement_strength', 2),
                'fps': options.get('activity_fps', 1.0),
                'pixels_size': 512
            }

            response = await call_service("activity", "analyze", files=files, data=data)

            if response:
                all_results["activity_analysis"] = response

                # حفظ النتيجة
                activity_file = OUTPUTS_DIR / process_id / "activity_analysis.json"
                with open(activity_file, 'w', encoding='utf-8') as f:
                    json.dump(response, f, ensure_ascii=False, indent=2)

                logger.info("✅ تم تحليل النشاط والبيئة بنجاح")
            else:
                logger.error("❌ فشل في تحليل النشاط والبيئة")
                all_results["activity_analysis"] = {"error": "فشل في تحليل النشاط"}

    except Exception as e:
        logger.error(f"❌ خطأ في تحليل النشاط: {e}")
        all_results["activity_analysis"] = {"error": str(e)}


async def save_face_image(frame: np.ndarray, face: dict, frame_number: int, process_id: str):
    """حفظ صورة الوجه المقتطع"""
    try:
        bbox = face.get('bbox', [])
        if not bbox or len(bbox) < 4:
            logger.error(f"❌ إحداثيات الوجه غير صالحة: {bbox}")
            return

        FACE_PADDING_RATIO = 0.2

        if len(bbox) == 4:

            x1 = max(int(bbox[0]), 0)
            y1 = max(int(bbox[1]), 0)
            width_face = max(int(bbox[2]), 0)
            height_face = max(int(bbox[3]), 0)

            x2 = x1 + width_face
            y2 = y1 + height_face
        else:
            logger.error(f"❌ تنسيق إحداثيات الوجه غير متوقع: {bbox}")
            return

        # التأكد من أن الإحداثيات ضمن حدود الإطار
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(x1, min(x2, w))
        y2 = max(y1, min(y2, h))

        # إضافة هامش حول الوجه
        pad_x = int(width_face * FACE_PADDING_RATIO / 2)
        pad_y = int(height_face * FACE_PADDING_RATIO / 2)

        x1_padded = max(0, x1 - pad_x)
        y1_padded = max(0, y1 - pad_y)
        x2_padded = min(frame.shape[1], x2 + pad_x)
        y2_padded = min(frame.shape[0], y2 + pad_y)

        # قص صورة الوجه
        face_img = frame[y1_padded:y2_padded, x1_padded:x2_padded]
        TARGET_FACE_WIDTH = 228
        TARGET_FACE_HEIGHT = 228
        if face_img.shape[0] > 0 and face_img.shape[1] > 0:
            current_height, current_width = face_img.shape[:2]
            if current_width < TARGET_FACE_WIDTH or current_height < TARGET_FACE_HEIGHT:
                face_img = cv2.resize(face_img, (
                    max(current_width * 2, TARGET_FACE_WIDTH), max(current_height * 2, TARGET_FACE_HEIGHT)),
                                      interpolation=cv2.INTER_CUBIC)
            face_img_resized = cv2.resize(face_img, (TARGET_FACE_WIDTH, TARGET_FACE_HEIGHT),
                                          interpolation=cv2.INTER_AREA)

            face_filename = f"face_{frame_number}_{face.get('face_id', 0)}.jpg"
            face_path = OUTPUTS_DIR / process_id / "faces" / face_filename
            cv2.imwrite(str(face_path), face_img_resized)
            face["image_path"] = face_filename
            logger.info(f"✅ تم حفظ صورة الوجه: {face_filename}")
    except Exception as e:
        logger.error(f"❌ خطأ في حفظ صورة الوجه: {e}")
        logger.error(f"❌ تفاصيل الوجه: {face}")


async def get_video_info(video_path: str) -> dict:
    """الحصول على معلومات الفيديو الأساسية"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        cap.release()

        return {
            "total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "duration_seconds": duration,
            "resolution": f"{width}x{height}"
        }
    except Exception as e:
        logger.error(f"❌ خطأ في الحصول على معلومات الفيديو: {e}")
        return None


async def save_detection_results(process_id: str, faces: list, texts: list, objects: list, tracks: list):
    """حفظ نتائج الكشف في ملفات"""
    try:
        process_dir = OUTPUTS_DIR / process_id

        # حفظ الوجوه
        if faces:
            faces_file = process_dir / "faces_data.json"
            with open(faces_file, 'w', encoding='utf-8') as f:
                json.dump(faces, f, ensure_ascii=False, indent=2)

        # حفظ النصوص
        if texts:
            texts_file = process_dir / "texts_data.json"
            with open(texts_file, 'w', encoding='utf-8') as f:
                json.dump(texts, f, ensure_ascii=False, indent=2)

        # حفظ الكائنات والتتبع
        if objects or tracks:
            detection_data = {
                "objects": objects,
                "tracks": tracks
            }
            detection_file = process_dir / "detection_data.json"
            with open(detection_file, 'w', encoding='utf-8') as f:
                json.dump(convert_serializable_types(detection_data), f, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"❌ خطأ في حفظ نتائج الكشف: {e}")


def generate_color_from_id(track_id, saturation=0.8, value=0.8):
    """توليد لون فريد بناءً على track_id باستخدام HSV"""
    import math

    # استخدام الزاوية في دائرة الألوان بناءً على track_id
    hue = (track_id * 137.5) % 360  # زاوية ذهبية لتوزيع الألوان
    hue = hue / 360.0  # تحويل إلى نطاق [0, 1]

    # تحويل HSV إلى RGB
    h = hue
    s = saturation
    v = value

    i = math.floor(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    if i % 6 == 0:
        r, g, b = v, t, p
    elif i % 6 == 1:
        r, g, b = q, v, p
    elif i % 6 == 2:
        r, g, b = p, v, t
    elif i % 6 == 3:
        r, g, b = p, q, v
    elif i % 6 == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q

    # تحويل إلى BGR لـ OpenCV
    return (int(b * 255), int(g * 255), int(r * 255))


def get_color_for_track(track_id, color_cache={}):
    """الحصول على لون ثابت لكل track_id مع التخزين المؤقت"""
    if track_id not in color_cache:
        color_cache[track_id] = generate_color_from_id(track_id)
    return color_cache[track_id]


def get_color_for_class(class_name, color_cache={}):
    """الحصول على لون ثابت لكل فئة كائن مع التخزين المؤقت"""
    if class_name not in color_cache:
        # ألوان محددة مسبقاً للفئات الشائعة
        class_colors = {
            "person": (255, 0, 0),  # أحمر
            "car": (0, 255, 0),  # أخضر
            "bicycle": (0, 255, 255),  # أصفر
            "motorcycle": (255, 255, 0),  # سماوي
            "bus": (255, 0, 255),  # وردي
            "truck": (0, 0, 255),  # أزرق
            "cat": (128, 0, 128),  # بنفسجي
            "dog": (128, 128, 0),  # زيتوني
            "chair": (0, 128, 128),  # تركواز
            "book": (128, 128, 128),  # رمادي
        }
        color_cache[class_name] = class_colors.get(class_name, generate_color_from_id(hash(class_name) % 1000))

    return color_cache[class_name]



async def save_final_results(process_id: str, all_results: dict):
    """حفظ النتائج النهائية"""
    try:
        results_file = OUTPUTS_DIR / process_id / "final_results.json"

        # ✅ إنشاء نسخة نظيفة من النتائج
        final_results = {
            "process_id": process_id,
            "processing_date": all_results.get("processing_date"),
            "processing_options": all_results.get("processing_options"),
            "total_frames": all_results.get("total_frames"),
            "fps": all_results.get("fps"),
            "width": all_results.get("width"),
            "height": all_results.get("height"),
            "duration_seconds": all_results.get("duration_seconds"),
            "resolution": all_results.get("resolution"),
            "frames_processed": all_results.get("frames_processed"),
            "faces_detected": all_results.get("faces_detected"),
            "texts_detected": all_results.get("texts_detected"),
            "objects_detected": all_results.get("objects_detected"),
            "tracks_detected": all_results.get("tracks_detected"),
            "processing_duration_seconds": all_results.get("processing_duration_seconds"),
            "transcription": all_results.get("transcription"),
            "activity_analysis": all_results.get("activity_analysis"),
            "video_filename":all_results.get("activity_analysis")
        }

        # ✅ استخدام aiofiles للحفظ
        async with aiofiles.open(results_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(final_results, ensure_ascii=False, indent=2))

        logger.info(f"✅ تم حفظ النتائج النهائية في {results_file}")
    except Exception as e:
        logger.error(f"❌ خطأ في حفظ النتائج النهائية: {e}")


def draw_faces_on_frame(frame: np.ndarray, faces: list):
    """رسم الوجوه المكتشفة على الإطار مع ألوان مختلفة"""
    for i, face in enumerate(faces):
        bbox = face.get('bbox', [])
        if len(bbox) == 4:  # [x1, y1, width, height]
            x1, y1, width_face, height_face = map(int, bbox)
            x2 = x1 + width_face
            y2 = y1 + height_face
            confidence = face.get('confidence', 0)

            # توليد لون فريد لكل وجه بناءً على الفهرس
            face_color = generate_color_from_id(i, saturation=0.9, value=0.9)

            # رسم مربع الوجه
            cv2.rectangle(frame, (x1, y1), (x2, y2), face_color, 2)

            # إضافة نص الثقة مع خلفية
            label = f"Face {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1), face_color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame


def draw_texts_on_frame(frame: np.ndarray, texts: list):
    """رسم النصوص المكتشفة على الإطار مع ألوان مختلفة"""
    for i, text in enumerate(texts):
        bbox = text.get('bbox', [])
        if len(bbox) == 4:  # [x, y, width, height]
            x, y, w, h = bbox
            text_content = text.get('text', '')

            # توليد لون فريد لكل نص
            text_color = generate_color_from_id(i + 1000, saturation=0.7, value=0.9)

            # رسم مربع النص
            cv2.rectangle(frame, (x, y), (x + w, y + h), text_color, 2)

            # إضافة النص مع خلفية
            label = f"Text: {text_content[:20]}{'...' if len(text_content) > 20 else ''}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 10),
                          (x + label_size[0], y), text_color, -1)
            cv2.putText(frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame


def draw_objects_on_frame(frame: np.ndarray, objects_data: dict):
    """رسم الكائنات والمسارات على الإطار"""
    objects = objects_data.get('objects', [])
    tracks = objects_data.get('tracks', [])

    # رسم الكائنات
    for obj in objects:
        bbox = obj.get('bbox', [])
        if len(bbox) >= 4:  # دعم تنسيقات مختلفة [x1, y1, x2, y2] أو [x1, y1, width, height]
            if len(bbox) == 4:
                # إذا كان الطول 4، نفترض [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, bbox)
            else:
                # إذا كان الطول مختلف، نأخذ أول 4 عناصر
                x1, y1, x2, y2 = map(int, bbox[:4])

            class_name = obj.get('class_name', 'object')
            confidence = obj.get('confidence', 0)

            # رسم مربع الكائن
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # إضافة التسمية
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame


def draw_trajectories_on_frame(frame: np.ndarray, tracks: list, trajectory_history: dict):
    """رسم مسارات تحرك الأشخاص على الإطار مع ألوان مختلفة"""
    try:
        for track in tracks:
            track_id = track.get('track_id')
            if track_id is None:
                continue

            bbox = track.get('bbox', [])
            if len(bbox) >= 4:
                if len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                else:
                    x1, y1, x2, y2 = map(int, bbox[:4])

                # الحصول على لون فريد للمسار
                track_color = get_color_for_track(track_id)

                # حساب مركز المربع
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # إضافة النقطة إلى تاريخ المسار
                if track_id not in trajectory_history:
                    trajectory_history[track_id] = []

                trajectory_history[track_id].append((center_x, center_y))

                # الاحتفاظ بآخر 30 نقطة فقط للمسار (زيادة للرسم الأفضل)
                if len(trajectory_history[track_id]) > 30:
                    trajectory_history[track_id].pop(0)

                # رسم المسار بخط متدرج الشفافية
                points = trajectory_history[track_id]
                for i in range(1, len(points)):
                    # زيادة الشفافية تدريجياً للنقاط القديمة
                    alpha = i / len(points)
                    thickness = max(1, int(3 * alpha))
                    cv2.line(frame, points[i - 1], points[i], track_color, thickness)

                # رسم نقطة المركز الحالية
                cv2.circle(frame, (center_x, center_y), 4, track_color, -1)
                cv2.circle(frame, (center_x, center_y), 6, (255, 255, 255), 1)  # حد أبيض

                # رسم مربع التتبع باللون المطابق للمسار
                cv2.rectangle(frame, (x1, y1), (x2, y2), track_color, 2)

                # إضافة معرف المسار مع خلفية
                label = f"ID:{track_id}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                              (x1 + label_size[0], y1), track_color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame, trajectory_history
    except Exception as e:
        logger.error(f"❌ خطأ في رسم المسارات: {e}")
        return frame, trajectory_history

async def extract_audio_from_video(video_path: str, output_audio_path: str) -> bool:
    """استخراج الصوت من الفيديو باستخدام ffmpeg"""
    try:
        command = [
            'ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a',
            output_audio_path, '-y', '-loglevel', 'error'
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ خطأ في استخراج الصوت: {e}")
        return False

async def get_processing_status(process_id: str):
    """الحصول على حالة المعالجة"""
    try:
        process_info = db.get_process_status(process_id)
        if not process_info:
            return {}, "not_found", "لم يتم العثور على العملية"

        process_dir = OUTPUTS_DIR / process_id
        results_file = process_dir / "final_results.json"

        results = {}
        if results_file.exists():
            try:
                # ✅ استخدام aiofiles للقراءة غير المتزامنة
                async with aiofiles.open(results_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    results = json.loads(content)
            except Exception as e:
                logger.error(f"❌ خطأ في قراءة final_results.json: {e}")
                results = {}

        results["status"] = process_info["status"]
        results["message"] = process_info["message"]
        results["progress"] = process_info["progress"]

        return results, process_info["status"], process_info["message"]

    except Exception as e:
        logger.error(f"❌ خطأ في الحصول على حالة المعالجة: {e}")
        return {}, "error", f"خطأ في الحصول على الحالة: {str(e)}"


async def stop_video_processing(process_id: str):
    """إيقاف معالجة الفيديو (للتوافق مع الكود القديم)"""
    # في التنفيذ الجديد، يمكننا إضافة آلية لإيقاف المعالجة إذا لزم الأمر
    print(f"⏹️ طلب إيقاف العملية: {process_id}")
    return True


async def cleanup_processing():
    """تنظيف المعالجة (للتوافق مع الكود القديم)"""
    print("🧹 تنظيف موارد المعالجة")

@app.post("/analyze-video")
async def analyze_video_endpoint(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        enable_audio_transcription: bool = Form(True),
        enable_face_detection: bool = Form(True),
        enable_text_detection: bool = Form(True),
        enable_tracking: bool = Form(True),
        enable_activity_recognition: bool = Form(True),
        enable_video_enhancement: bool = Form(False),
        enhancement_strength: int = Form(2),
        activity_prompt: Optional[str] = Form("You are a video surveillance expert, and your task is to describe the key activities in the video and the environment in which the video events take place, while analyzing the surveillance records provided for each frame. Your goal is to describe unusual activities and notable events, such as numbers, times, and dates, the presence of weapons, masked individuals, or people with unusual appearances, and exceptional incidents such as shootings, thefts, break-ins, and rapid or sudden movements, based on the descriptions provided for each frame. Highlight any unusual activities or problems while maintaining continuity of context. Your summary style should focus on identifying specific incidents, such as potential police activity, accidents, or unusual gatherings, and highlight normal events to provide context about the environment. For example, someone steals from a store, places merchandise in their bag, assaults someone, breaks into a place, fires a gun, is kidnapped, or breaks or removes a window. Summarize what happened in the video. Answer concisely.."),
        activity_fps: Optional[float] = Form(1.0),
        face_threshold: float = Form(0.3),
        text_threshold: float = Form(0.3),
        object_threshold: float = Form(0.5),
        detection_step: int = Form(1),
        advanced_settings: bool = Form(False),
        max_new_tokens: Optional[int] = Form(600),
        temperature: Optional[float] = Form(0.3),
        top_p: Optional[float] = Form(0.9),
        top_k: Optional[int] = Form(50),
        do_sample: Optional[bool] = Form(True),
):
    try:
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="الملف يجب أن يكون فيديو")

        # التحقق من حجم الملف
        file.file.seek(0, 2)  # الانتقال إلى نهاية الملف
        file_size = file.file.tell()
        file.file.seek(0)  # العودة إلى بداية الملف

        if file_size > APP_CONFIG["max_file_size"]:
            raise HTTPException(status_code=400, detail="حجم الملف يتجاوز الحد المسموح (100MB)")

        # إنشاء معرف فريد للمعالجة
        process_id = str(uuid.uuid4())
        input_path = UPLOAD_DIR / f"{process_id}_{file.filename}"

        # حفظ الملف المرفوع
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        max_new_tokens_value = max_new_tokens if max_new_tokens is not None else 600
        temperature_value = temperature if temperature is not None else 0.3
        top_p_value = top_p if top_p is not None else 0.9
        top_k_value = top_k if top_k is not None else 50
        do_sample_value = do_sample if do_sample is not None else True

        print(f"🔧 معاملات النموذج المستلمة في endpoint:")
        print(f"max_new_tokens: {max_new_tokens} -> {max_new_tokens_value}")
        print(f"temperature: {temperature} -> {temperature_value}")
        print(f"top_p: {top_p} -> {top_p_value}")
        print(f"top_k: {top_k} -> {top_k_value}")
        print(f"do_sample: {do_sample} -> {do_sample_value}")

        # إعداد خيارات المعالجة
        processing_options = {
            "enable_audio_transcription": enable_audio_transcription,
            "enable_face_detection": enable_face_detection,
            "enable_text_detection": enable_text_detection,
            "enable_tracking": enable_tracking,
            "enable_activity_recognition": enable_activity_recognition,
            "original_filename": file.filename,
            "activity_prompt": activity_prompt,
            "activity_fps": activity_fps,
            "face_threshold": face_threshold,
            "text_threshold": text_threshold,
            "object_threshold": object_threshold,
            "detection_step": detection_step,
            "advanced_settings": advanced_settings,
            # تمرير القيم المحسنة والمعالجة
            "max_new_tokens": max_new_tokens_value,
            "temperature": temperature_value,
            "top_p": top_p_value,
            "top_k": top_k_value,
            "do_sample": do_sample_value,
            "enable_video_enhancement": enable_video_enhancement,
            "enhancement_strength": enhancement_strength,
        }

        # إضافة العملية إلى القائمة النشطة
        active_processes[process_id] = {
            "start_time": datetime.now(),
            "status": "starting",
            "options": processing_options
        }

        # إضافة العملية إلى قاعدة البيانات
        db.add_process(process_id, file.filename, str(input_path), processing_options)

        # إضافة المهمة للمعالجة في الخلفية
        background_tasks.add_task(
            process_video,
            input_path=str(input_path),
            process_id=process_id,
            options=processing_options
        )

        monitor.add_process(process_id)

        return JSONResponse({
            "process_id": process_id,
            "status": "processing",
            "message": "جاري معالجة الفيديو",
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        print(f"❌ خطأ في analyze-video: {str(e)}")
        if 'process_id' in locals():
            monitor.remove_process(process_id)
        raise HTTPException(status_code=500, detail=f"خطأ في معالجة الفيديو: {str(e)}")


@app.post("/enhance-face")
async def enhance_face_endpoint(data: dict):
    try:
        process_id = data.get("process_id")
        image_path = data.get("image_path")
        enhancement_type = data.get("enhancement_type", "super_resolution")
        parameters = data.get("parameters", {})

        if not process_id or not image_path:
            raise HTTPException(status_code=400, detail="معرف العملية ومسار الصورة مطلوبان")

        # مسار الصورة الأصلية
        original_image_path = OUTPUTS_DIR / process_id / "faces" / Path(image_path).name

        if not original_image_path.exists():
            raise HTTPException(status_code=404, detail="الصورة غير موجودة")

        # تحسين الصورة
        enhanced_filename = await enhance_face_image(
            str(original_image_path),
            process_id,
            enhancement_type,
            parameters
        )

        return JSONResponse({
            "status": "success",
            "enhanced_filename": enhanced_filename,
            "enhancement_type": enhancement_type,
            "message": f"تم تطبيق {get_enhancement_name(enhancement_type)} بنجاح"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في تحسين الصورة: {str(e)}")


def get_enhancement_name(enhancement_type: str) -> str:
    names = {
        "super_resolution": "تحسين الدقة",
        "sharpen": "زيادة الحدة",
        "contrast": "زيادة التباين",
        "smooth": "تنعيم الصورة"
    }
    return names.get(enhancement_type, enhancement_type)


@app.post("/stop-analysis/{process_id}")
async def stop_analysis_endpoint(process_id: str):
    """إيقاف معالجة الفيديو"""
    try:
        success = await stop_video_processing()

        if success:
            # تحديث حالة العملية في قاعدة البيانات
            process_info = db.get_process_status(process_id)
            if process_info:
                db.update_process_status(process_id, "stopped", process_info["progress"], "تم إيقاف التحليل يدوياً")
                monitor.remove_process(process_id)

            # إزالة العملية من القائمة النشطة
            if process_id in active_processes:
                del active_processes[process_id]

            return JSONResponse({
                "status": "success",
                "message": "تم إيقاف التحليل بنجاح",
                "process_id": process_id,
                "timestamp": datetime.now().isoformat()
            })
        else:
            raise HTTPException(status_code=404, detail="لم يتم العثور على العملية أو أنها غير نشطة")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في إيقاف التحليل: {str(e)}")


@app.post("/enhance-face-sequence")
async def enhance_face_sequence_endpoint(data: dict):
    """تطبيق سلسلة من التحسينات على الصورة بشكل تسلسلي"""
    try:
        process_id = data.get("process_id")
        original_path = data.get("original_path")
        enhancement_stack = data.get("enhancement_stack", [])

        if not process_id or not original_path:
            raise HTTPException(status_code=400, detail="معرف العملية ومسار الصورة مطلوبان")

        # مسار الصورة الأصلية
        original_image_path = OUTPUTS_DIR / process_id / "faces" / Path(original_path).name

        if not original_image_path.exists():
            raise HTTPException(status_code=404, detail="الصورة الأصلية غير موجودة")

        # تطبيق التحسينات بشكل تسلسلي
        enhanced_filename = await apply_enhancement_sequence(
            str(original_image_path),
            process_id,
            enhancement_stack
        )

        return JSONResponse({
            "status": "success",
            "enhanced_filename": enhanced_filename,
            "message": f"تم تطبيق {len(enhancement_stack)} تحسين بشكل تسلسلي"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في التحسين التسلسلي: {str(e)}")


async def apply_enhancement_sequence(image_path: str, process_id: str, enhancement_stack: list) -> str:
    """تطبيق سلسلة من التحسينات على الصورة"""
    try:
        import cv2
        import numpy as np

        # قراءة الصورة الأصلية
        current_image = cv2.imread(image_path)
        if current_image is None:
            raise Exception("تعذر قراءة الصورة الأصلية")

        # تطبيق كل تحسين في السلسلة
        for i, enhancement in enumerate(enhancement_stack):
            enhancement_type = enhancement.get("type")
            parameters = enhancement.get("parameters", {})

            # تطبيق التحسين الحالي
            if enhancement_type == "super_resolution":
                current_image = apply_super_resolution(current_image)
            elif enhancement_type == "sharpen":
                strength = parameters.get("strength", 2.0)
                current_image = apply_sharpening(current_image, strength)
            elif enhancement_type == "contrast":
                strength = parameters.get("strength", 1.5)
                current_image = apply_contrast_enhancement(current_image, strength)
            elif enhancement_type == "smooth":
                strength = parameters.get("strength", 3)
                current_image = apply_smoothing(current_image, strength)

        # حفظ الصورة النهائية
        original_path = Path(image_path)
        stack_description = "_".join([enh["type"] for enh in enhancement_stack])
        enhanced_filename = f"enhanced_sequence_{stack_description}_{original_path.name}"
        enhanced_path = OUTPUTS_DIR / process_id / "faces" / enhanced_filename

        cv2.imwrite(str(enhanced_path), current_image, [cv2.IMWRITE_JPEG_QUALITY, 95])

        return enhanced_filename

    except Exception as e:
        raise Exception(f"فشل في تطبيق السلسلة التحسينية: {str(e)}")


@app.post("/save-enhanced-face")
async def save_enhanced_face_endpoint(data: dict):
    """حفظ الصورة المحسنة بشكل منفصل"""
    try:
        process_id = data.get("process_id")
        original_path = data.get("original_path")
        enhanced_path = data.get("enhanced_path")
        enhancement_stack = data.get("enhancement_stack", [])

        if not all([process_id, original_path, enhanced_path]):
            raise HTTPException(status_code=400, detail="جميع الحقول مطلوبة")

        # إنشاء مجلد الوجوه المحسنة
        enhanced_faces_dir = OUTPUTS_DIR / process_id / "enhanced_faces"
        enhanced_faces_dir.mkdir(exist_ok=True, parents=True)

        # مسار الصورة المحسنة الحالية
        current_enhanced_path = OUTPUTS_DIR / process_id / "faces" / enhanced_path

        if not current_enhanced_path.exists():
            raise HTTPException(status_code=404, detail="الصورة المحسنة غير موجودة")

        # إنشاء اسم وصفي للصورة المحفوظة
        original_name = Path(original_path).stem
        extension = Path(original_path).suffix

        # بناء وصف التحسينات من السلسلة
        if enhancement_stack:
            enhancements_desc = "_".join([enh["type"] for enh in enhancement_stack])
            new_filename = f"{original_name}_enhanced_{enhancements_desc}{extension}"
        else:
            new_filename = f"{original_name}_enhanced{extension}"

        saved_path = enhanced_faces_dir / new_filename

        # نسخ الصورة المحسنة
        import shutil
        shutil.copy2(current_enhanced_path, saved_path)

        return JSONResponse({
            "status": "success",
            "saved_path": str(saved_path.relative_to(OUTPUTS_DIR / process_id)),
            "filename": new_filename,
            "message": "تم حفظ الصورة المحسنة بنجاح"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في حفظ الصورة المحسنة: {str(e)}")


@app.get("/results/{process_id}")
async def get_results(process_id: str):
    """الحصول على نتائج المعالجة"""
    try:
        results, status, message = await get_processing_status(process_id)

        if status == "not_found":
            raise HTTPException(status_code=404, detail="لم يتم العثور على العملية")

        # تحميل النتائج الإضافية من الملفات إذا كانت العملية مكتملة أو متوقفة
        if status in ["completed", "stopped"]:
            process_dir = OUTPUTS_DIR / process_id

            # تحميل النتائج النهائية من JSON - مع معالجة أفضل
            results_file = process_dir / "final_results.json"
            if results_file.exists():
                try:
                    async with aiofiles.open(results_file, 'r', encoding='utf-8') as f:
                        file_results = json.loads(await f.read())

                    # ✅ دمج النتائج بشكل انتقائي
                    for key, value in file_results.items():
                        if key not in ["status", "message", "progress"]:
                            results[key] = value
                except Exception as e:
                    logger.error(f"❌ خطأ في تحميل final_results.json: {e}")
                    results["final_results_error"] = str(e)

            # تحميل بيانات الوجوه إذا كانت موجودة
            faces_file = process_dir / "faces_data.json"
            if faces_file.exists():
                try:
                    async with aiofiles.open(faces_file, 'r', encoding='utf-8') as f:
                        results["faces_data"] = json.loads(await f.read())
                except Exception as e:
                    logger.error(f"❌ خطأ في تحميل faces_data.json: {e}")
                    results["faces_data"] = []

            # تحميل بيانات النصوص إذا كانت موجودة
            texts_file = process_dir / "texts_data.json"
            if texts_file.exists():
                try:
                    async with aiofiles.open(texts_file, 'r', encoding='utf-8') as f:
                        results["extracted_texts"] = json.loads(await f.read())
                except Exception as e:
                    logger.error(f"❌ خطأ في تحميل texts_data.json: {e}")
                    results["extracted_texts"] = []

            # تحميل بيانات التتبع إذا كانت موجودة
            tracking_file = process_dir / "tracking_data.json"
            if tracking_file.exists():
                try:
                    async with aiofiles.open(tracking_file, 'r', encoding='utf-8') as f:
                        results["tracking_data"] = json.loads(await f.read())
                except Exception as e:
                    logger.error(f"❌ خطأ في تحميل tracking_data.json: {e}")
                    results["tracking_data"] = []

            # ✅ تحميل بيانات النشاط بشكل صحيح
            activity_file = process_dir / "activity_analysis.json"
            if activity_file.exists():
                try:
                    async with aiofiles.open(activity_file, 'r', encoding='utf-8') as f:
                        activity_data = json.loads(await f.read())
                    results["activity_analysis"] = activity_data  # ✅ استخدام نفس المفتاح
                except Exception as e:
                    logger.error(f"❌ خطأ في تحميل activity_analysis.json: {e}")
                    results["activity_analysis"] = {"error": str(e)}
            else:
                results["activity_analysis"] = {}  # ✅ قيمة افتراضية متسقة

        return JSONResponse({
            "process_id": process_id,
            "status": status,
            "message": message,
            "results": results,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"❌ خطأ في الحصول على النتائج: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"خطأ في الحصول على النتائج: {str(e)}")


@app.get("/outputs/{process_id}/{filename:path}")
async def download_file(process_id: str, filename: str):
    """تحميل ملف من مجلد المخرجات"""
    try:
        file_path = OUTPUTS_DIR / process_id / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="الملف غير موجود")

        if not file_path.is_file():
            raise HTTPException(status_code=400, detail="المسار لا يشير إلى ملف")

        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="application/octet-stream"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في تحميل الملف: {str(e)}")


@app.get("/outputs_list/{process_id}")
async def get_outputs_list(process_id: str, path: Optional[str] = ""):
    """
    الحصول على قائمة بالملفات والمجلدات داخل مجلد مخرجات عملية معينة.
    يمكن استخدام المعامل 'path' لتصفح المجلدات الفرعية.
    """
    base_dir = OUTPUTS_DIR / process_id
    current_dir = base_dir / path

    if not current_dir.exists() or not current_dir.is_dir():
        raise HTTPException(status_code=404, detail="المجلد غير موجود")

    files_list = []
    for item in current_dir.iterdir():
        item_info = {
            "name": item.name,
            "type": "directory" if item.is_dir() else "file",
            "size": item.stat().st_size if item.is_file() else None,
            "last_modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
        }
        files_list.append(item_info)

    # فرز المجلدات أولاً ثم الملفات أبجدياً
    files_list.sort(key=lambda x: (x['type'] == 'file', x['name']))

    return JSONResponse(files_list)


@app.get("/api-docs")
async def api_docs_redirect():
    """إعادة التوجيه إلى وثائق API"""
    return RedirectResponse(url="/docs")


@app.get("/api-redoc")
async def api_redoc_redirect():
    """إعادة التوجيه إلى Redoc"""
    return RedirectResponse(url="/redoc")


async def enhance_face_image(image_path: str, process_id: str, enhancement_type: str, parameters: dict = None) -> str:
    """تحسين جودة صورة الوجه حسب النوع المطلوب"""
    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageEnhance, ImageFilter

        if parameters is None:
            parameters = {}

        # قراءة الصورة
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("تعذر قراءة الصورة")

        # تطبيق التحسين المطلوب
        if enhancement_type == "super_resolution":
            enhanced = apply_super_resolution(image)
        elif enhancement_type == "sharpen":
            enhanced = apply_sharpening(image, parameters.get("strength", 2.0))
        elif enhancement_type == "contrast":
            enhanced = apply_contrast_enhancement(image, parameters.get("strength", 1.5))
        elif enhancement_type == "smooth":
            enhanced = apply_smoothing(image, parameters.get("strength", 3))
        else:
            enhanced = image  # إذا كان النوع غير معروف، نعود للصورة الأصلية

        # حفظ الصورة المحسنة
        original_path = Path(image_path)
        enhanced_filename = f"enhanced_{enhancement_type}_{original_path.name}"
        enhanced_path = OUTPUTS_DIR / process_id / "faces" / enhanced_filename

        cv2.imwrite(str(enhanced_path), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])

        return enhanced_filename

    except Exception as e:
        raise Exception(f"فشل في تحسين الصورة: {str(e)}")


def apply_super_resolution(image):
    """تحسين دقة الصورة"""
    height, width = image.shape[:2]

    # زيادة الدقة باستخدام resize عالي الجودة
    enhanced = cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)

    # تقليل الضوضاء
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)

    return enhanced


def apply_sharpening(image, strength=2.0):
    """زيادة حدة الصورة"""
    # إنشاء kernel للحدة
    kernel = np.array([[-1, -1, -1],
                       [-1, 8 + strength, -1],
                       [-1, -1, -1]]) / strength
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened


def apply_contrast_enhancement(image, strength=1.5):
    """زيادة تباين الصورة"""
    # تحسين التباين باستخدام CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))
    clahe = cv2.createCLAHE(clipLimit=strength, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced


def apply_smoothing(image, strength=3):
    """تنعيم الصورة"""
    # تطبيق تنعيم Gaussian
    smoothed = cv2.GaussianBlur(image, (0, 0), strength)
    return smoothed


if __name__ == "__main__":
    import uvicorn

    print("🌐 جاري تشغيل خادم FastAPI...")
    print(f"📊 يمكنك الوصول إلى التطبيق على: http://{APP_CONFIG['host']}:{APP_CONFIG['port']}/docs")
    print("⏹️  اضغط Ctrl+C لإيقاف الخادم")

    uvicorn.run(
        app,
        host=APP_CONFIG["host"],
        port=APP_CONFIG["port"],
        log_level="info"
    )

