# Roadshow Backend V13

AI-powered audio enhancement API using V13 Simplified processor.

## Features
- V13 Simplified professional audio enhancement
- Conservative denoising + gentle EQ
- Multiple microphone models (U87, U67, U47, C12, 251)
- Real-time processing (~1 second per file)

## Endpoints
- POST /process - Upload audio for V13 enhancement
- GET /health - Check V13 API status
- GET /download/{id}/{filename} - Download processed audio

## Supported Formats
WAV, MP3, M4A, FLAC