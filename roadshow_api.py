# Add this import at the top of your roadshow_api.py
from v13_processor import V13SimplifiedProcessor

# Replace the processor initialization section (around line 200+) with this:
try:
    # Try to use V13 Simplified processor
    processor = V13SimplifiedProcessor()
    processing_mode = "V13 Simplified - Professional Grade"
    print("🚀 V13 Simplified mode activated!")
    
except Exception as e:
    print(f"V13 not available ({e}), using fallback mode")
    # Keep your existing processor as fallback
    processor = RoadshowProcessor()
    processing_mode = "Fallback DSP"

# Update the process_audio method to use V13
# Find your existing process_audio endpoint and update the processing section:

@app.post("/process")
async def process_audio(
    file: UploadFile = File(...),
    preset: str = "natural",
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
    
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Process with V13
    loop = asyncio.get_event_loop()
    output_filename = f"v13_{mic_type}_{file.filename.rsplit('.', 1)[0]}.wav"
    output_path = f"processed/{upload_id}_{output_filename}"
    
    try:
        # Use V13 Simplified processor
        result = await loop.run_in_executor(
            executor,
            lambda: processor.process_audio(
                input_path, output_path,
                character=character,
                clarity=clarity, 
                vintage=vintage
            )
        )
        
        output_path, original_waveform, processed_waveform = result
        
    except Exception as e:
        if os.path.exists(input_path):
            os.remove(input_path)
        raise HTTPException(500, f"V13 processing error: {str(e)}")
    
    # Clean up input file
    if os.path.exists(input_path):
        os.remove(input_path)
    
    return {
        "success": True,
        "download_url": f"/download/{upload_id}/{output_filename}",
        "preview_url": f"/preview/{upload_id}/{output_filename}",
        "filename": output_filename,
        "input_format": file_extension.upper(),
        "output_format": "WAV",
        "processing_mode": "V13 Simplified Professional Grade",
        "microphone_model": mic_type,
        "waveforms": {
            "original": original_waveform,
            "processed": processed_waveform
        },
        "settings": {
            "character": character,
            "clarity": clarity,
            "vintage": vintage,
            "mic_type": mic_type
        },
        "algorithms_applied": [
            "conservative_denoising",
            "gentle_eq_shaping", 
            "u87_character_modeling"
        ]
    }

# Update your health endpoint to reflect V13:
@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "processor": "v13_simplified_ready",
        "processing_mode": processing_mode,
        "supported_formats": ["WAV", "MP3", "M4A", "FLAC"],
        "version": "V13-Simplified",
        "available_features": [
            "conservative_denoising",
            "gentle_eq_shaping",
            "u87_character_modeling",
            "real_time_processing"
        ]
    }