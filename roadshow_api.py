#!/usr/bin/env python3
"""
Roadshow V13 API - Professional Audio Enhancement
FastAPI backend for V13 Simplified processor
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import numpy as np
import soundfile as sf
import os
import uuid
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# Import V13 processor
try:
    from v13_processor import V13SimplifiedProcessor
    v13_available = True
    print("🚀 V13 Simplified mode activated!")
except ImportError as e:
    print(f"V13 not available: {e}")
    v13_available = False

# Create FastAPI app
app = FastAPI(title="Roadshow V13 API", version="V13-Simplified")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("processed", exist_ok=True)

# Thread pool for processing
executor = ThreadPoolExecutor(max_workers=4)

# Initialize V13 processor
if v13_available:
    try:
        processor = V13SimplifiedProcessor()
        processing_mode = "V13 Simplified - Professional Grade"
        logger = logging.getLogger(__name__)
        print("✅ V13 Simplified processor initialized successfully!")
    except Exception as e:
        print(f"V13 initialization failed: {e}")
        v13_available = False

if not v13_available:
    # Fallback basic processor
    class BasicProcessor:
        def process_audio(self, input_path, output_path, **kwargs):
            # Simple passthrough for testing
            import shutil
            shutil.copy2(input_path, output_path)
            return output_path, [], []
        
        def generate_waveform_data(self, audio, points=500):
            return []
    
    processor = BasicProcessor()
    processing_mode = "Basic Fallback (V13 Failed)"

def convert_audio_format(input_path):
    """Convert various audio formats to WAV"""
    try:
        from pydub import AudioSegment
        
        file_ext = input_path.lower().split('.')[-1]
        temp_wav = None
        
        if file_ext == 'mp3':
            audio_segment = AudioSegment.from_mp3(input_path)
            temp_wav = input_path.replace('.mp3', '_temp.wav')
            audio_segment.export(temp_wav, format='wav')
        elif file_ext == 'm4a':
            audio_segment = AudioSegment.from_file(input_path, format='m4a')
            temp_wav = input_path.replace('.m4a', '_temp.wav')
            audio_segment.export(temp_wav, format='wav')
        elif file_ext == 'flac':
            audio_segment = AudioSegment.from_file(input_path, format='flac')
            temp_wav = input_path.replace('.flac', '_temp.wav')
            audio_segment.export(temp_wav, format='wav')
        elif file_ext == 'wav':
            return input_path, None
        else:
            raise ValueError(f"Unsupported format: {file_ext}")
        
        return temp_wav, temp_wav
        
    except Exception as e:
        print(f"Audio conversion failed: {e}")
        return input_path, None

@app.get("/")
async def root():
    return {
        "message": "Roadshow V13 API - Professional Audio Enhancement",
        "version": "V13-Simplified",
        "processing_mode": processing_mode,
        "v13_available": v13_available,
        "supported_formats": ["WAV", "MP3", "M4A", "FLAC"],
        "microphone_models": ["u87", "u67", "u47", "c12", "251"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "processor": "v13_simplified_ready" if v13_available else "fallback_mode",
        "processing_mode": processing_mode,
        "supported_formats": ["WAV", "MP3", "M4A", "FLAC"],
        "version": "V13-Simplified",
        "v13_available": v13_available
    }

@app.post("/process")
async def process_audio(
    file: UploadFile = File(...),
    mic_type: str = "u87",
    character: Optional[float] = None,
    clarity: Optional[float] = None,
    vintage: Optional[float] = None
):
    """Process audio with V13 Simplified pipeline"""
    
    # Validate file format
    if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
        raise HTTPException(400, "Invalid file format. Supported: WAV, MP3, M4A, FLAC")
    
    # Default values
    character = character or 0.7
    clarity = clarity or 0.5
    vintage = vintage or 0.5
    
    # Validate parameters
    for param, value in [("character", character), ("clarity", clarity), ("vintage", vintage)]:
        if not (0 <= value <= 1):
            raise HTTPException(400, f"{param} must be between 0 and 1")
    
    # Save uploaded file
    upload_id = str(uuid.uuid4())
    file_extension = file.filename.split('.')[-1].lower()
    input_path = f"uploads/{upload_id}_{file.filename}"
    
    try:
        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Convert to WAV if needed
        wav_path, temp_wav = convert_audio_format(input_path)
        
        # Process with V13 (or fallback)
        loop = asyncio.get_event_loop()
        output_filename = f"v13_{mic_type}_{file.filename.rsplit('.', 1)[0]}.wav"
        output_path = f"processed/{upload_id}_{output_filename}"
        
        if v13_available:
            # Use V13 Simplified processor
            result = await loop.run_in_executor(
                executor,
                lambda: processor.process_audio(
                    wav_path, output_path,
                    character=character,
                    clarity=clarity,
                    vintage=vintage
                )
            )
            output_path, original_waveform, processed_waveform = result
        else:
            # Fallback processing
            result = await loop.run_in_executor(
                executor,
                lambda: processor.process_audio(wav_path, output_path)
            )
            output_path, original_waveform, processed_waveform = result
        
        # Clean up temp files
        if temp_wav and os.path.exists(temp_wav):
            os.remove(temp_wav)
        if os.path.exists(input_path):
            os.remove(input_path)
        
        return {
            "success": True,
            "download_url": f"/download/{upload_id}/{output_filename}",
            "preview_url": f"/preview/{upload_id}/{output_filename}",
            "filename": output_filename,
            "input_format": file_extension.upper(),
            "output_format": "WAV",
            "processing_mode": processing_mode,
            "microphone_model": mic_type,
            "v13_processed": v13_available,
            "waveforms": {
                "original": original_waveform,
                "processed": processed_waveform
            },
            "settings": {
                "character": character,
                "clarity": clarity,
                "vintage": vintage,
                "mic_type": mic_type
            }
        }
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(input_path):
            os.remove(input_path)
        raise HTTPException(500, f"V13 processing error: {str(e)}")

@app.get("/preview/{upload_id}/{filename}")
async def preview_file(upload_id: str, filename: str):
    """Stream processed file for preview"""
    file_path = f"processed/{upload_id}_{filename}"
    
    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found")
    
    return FileResponse(
        file_path,
        media_type='audio/wav',
        filename=filename,
        headers={"Cache-Control": "no-cache"}
    )

@app.get("/download/{upload_id}/{filename}")
async def download_file(upload_id: str, filename: str):
    """Download processed file"""
    file_path = f"processed/{upload_id}_{filename}"
    
    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found")
    
    # Schedule cleanup after download
    def cleanup():
        try:
            os.remove(file_path)
        except:
            pass
    
    loop = asyncio.get_event_loop()
    loop.call_later(3600, cleanup)  # Clean up after 1 hour
    
    return FileResponse(
        file_path,
        media_type='audio/wav',
        filename=filename
    )

# Cleanup old files on startup
def cleanup_old_files():
    """Remove files older than 24 hours"""
    import time
    current_time = time.time()
    
    for folder in ['uploads', 'processed']:
        if not os.path.exists(folder):
            continue
            
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                if current_time - os.path.getmtime(file_path) > 86400:
                    try:
                        os.remove(file_path)
                    except:
                        pass

# Run cleanup on startup
cleanup_old_files()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)