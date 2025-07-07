#!/usr/bin/env python3
"""
Roadshow ML Inference Module
Real-time audio transformation using trained U87 model
"""

import numpy as np
import torch
import torch.nn as nn
import librosa
import soundfile as sf
import os

class U87TransformNet(nn.Module):
    """Neural network to transform iPhone audio to U87 quality"""
    
    def __init__(self, n_fft=1025):
        super().__init__()
        
        # Encoder (iPhone -> Features)
        self.encoder = nn.Sequential(
            nn.Conv1d(n_fft, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        
        # Attention mechanism for frequency-specific processing
        self.attention = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
        
        # Decoder (Features -> U87)
        self.decoder = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, n_fft, kernel_size=3, padding=1),
        )
        
        # Skip connection for preserving details
        self.skip_weight = nn.Parameter(torch.tensor(0.2))
    
    def forward(self, x):
        # x shape: [batch, freq_bins, time_frames]
        
        # Encode
        encoded = self.encoder(x)
        
        # Apply attention across frequency dimension
        b, c, t = encoded.shape
        encoded_reshaped = encoded.permute(0, 2, 1)  # [batch, time, channels]
        attended, _ = self.attention(encoded_reshaped, encoded_reshaped, encoded_reshaped)
        attended = attended.permute(0, 2, 1)  # Back to [batch, channels, time]
        
        # Decode
        decoded = self.decoder(attended)
        
        # Skip connection
        output = decoded + self.skip_weight * x
        
        return output

class U87MLProcessor:
    """ML-based U87 transformation processor"""
    
    def __init__(self, model_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = 48000
        
        # Load the trained model
        self.model = self._load_model(model_path)
        
        # STFT parameters (must match training)
        self.n_fft = 2048
        self.hop_length = 512
        
        print(f"ML Processor initialized on {self.device}")
    
    def _load_model(self, model_path):
        """Load the trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Initialize model architecture
        model = U87TransformNet().to(self.device)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def process_audio(self, audio, strength=1.0):
        """
        Process audio through the ML model
        
        Args:
            audio: Input audio array
            strength: Blend factor (0=original, 1=full ML processing)
        
        Returns:
            Processed audio array
        """
        # Store original for blending
        original_audio = audio.copy()
        
        # Handle empty audio
        if len(audio) == 0:
            return audio
        
        # Ensure audio is float32
        audio = audio.astype(np.float32)
        
        # Compute STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Prepare for model (log scale)
        log_mag = np.log1p(magnitude)
        
        # Convert to tensor and add batch dimension
        mag_tensor = torch.FloatTensor(log_mag).unsqueeze(0).to(self.device)
        
        # Process through model in chunks to handle any length
        chunk_size = 1000  # frames
        processed_chunks = []
        
        with torch.no_grad():
            for i in range(0, mag_tensor.shape[2], chunk_size):
                chunk = mag_tensor[:, :, i:i+chunk_size]
                
                # Skip if chunk is too small
                if chunk.shape[2] < 10:
                    processed_chunks.append(chunk)
                    continue
                
                # Process through model
                processed_chunk = self.model(chunk)
                processed_chunks.append(processed_chunk)
        
        # Concatenate all chunks
        processed_mag = torch.cat(processed_chunks, dim=2)
        
        # Convert back to numpy
        processed_log_mag = processed_mag.squeeze(0).cpu().numpy()
        
        # Inverse log scale
        processed_magnitude = np.expm1(processed_log_mag)
        
        # Reconstruct complex STFT
        processed_stft = processed_magnitude * np.exp(1j * phase)
        
        # Inverse STFT
        processed_audio = librosa.istft(processed_stft, hop_length=self.hop_length)
        
        # Ensure same length as input
        if len(processed_audio) > len(original_audio):
            processed_audio = processed_audio[:len(original_audio)]
        elif len(processed_audio) < len(original_audio):
            processed_audio = np.pad(processed_audio, (0, len(original_audio) - len(processed_audio)))
        
        # Blend with original based on strength
        final_audio = (1 - strength) * original_audio + strength * processed_audio
        
        # Normalize to prevent clipping
        peak = np.max(np.abs(final_audio))
        if peak > 0.95:
            final_audio = final_audio * (0.95 / peak)
        
        return final_audio.astype(np.float32)
    
    def process_file(self, input_path, output_path, strength=1.0):
        """Process an audio file"""
        # Load audio
        audio, sr = librosa.load(input_path, sr=self.sample_rate, mono=True)
        
        # Process
        processed_audio = self.process_audio(audio, strength)
        
        # Save
        sf.write(output_path, processed_audio, self.sample_rate)
        
        return output_path

# Integration with existing API
class HybridRoadshowProcessor:
    """Combines ML and DSP processing for best results"""
    
    def __init__(self, model_path=None):
        self.has_ml = False
        
        # Try to load ML model
        if model_path and os.path.exists(model_path):
            try:
                self.ml_processor = U87MLProcessor(model_path)
                self.has_ml = True
                print("ML processing enabled!")
            except Exception as e:
                print(f"ML model loading failed: {e}")
                print("Falling back to DSP-only processing")
        else:
            print(f"ML model not found at {model_path}, using DSP-only mode")
        
        # Initialize DSP processor (import here to avoid circular imports)
        from roadshow_api import EnhancedRoadshowProcessor
        self.dsp_processor = EnhancedRoadshowProcessor()
    
    def generate_waveform_data(self, audio, points=500):
        """Generate waveform data for visualization"""
        if len(audio) == 0:
            return []
        
        # Downsample for visualization
        step = max(1, len(audio) // points)
        downsampled = audio[::step]
        
        # Convert to list for JSON serialization
        return downsampled.tolist()
    
    def process_audio(self, input_path, output_path, character=0.7, clarity=0.5, room=0.7):
        """
        Process audio with hybrid ML/DSP approach
        
        ML handles: Core iPhone->U87 transformation
        DSP handles: Fine-tuning, clarity, room treatment
        """
        
        # Convert input file if needed (using DSP processor's method)
        wav_path, temp_wav = self.dsp_processor._convert_to_wav(input_path)
        
        try:
            if self.has_ml and character > 0.3:
                # ML processing for main transformation
                print("Applying ML transformation...")
                
                # Load audio
                audio, sr = sf.read(wav_path)
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                
                # Store original waveform
                original_waveform = self.generate_waveform_data(audio)
                
                # ML transformation (strength based on character)
                ml_strength = min(character * 1.2, 1.0)  # Scale character to ML strength
                processed_audio = self.ml_processor.process_audio(audio, ml_strength)
                
                # Save intermediate result
                temp_path = input_path.replace('.', '_ml_temp.')
                sf.write(temp_path, processed_audio, sr)
                
                # Apply DSP for fine-tuning
                # Reduce character since ML already did the heavy lifting
                dsp_character = character * 0.3
                final_path, _, processed_waveform = self.dsp_processor.process_audio(
                    temp_path, output_path, 
                    dsp_character, clarity, room
                )
                
                # Clean up temp files
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                if temp_wav and os.path.exists(temp_wav):
                    os.remove(temp_wav)
                
                return final_path, original_waveform, processed_waveform
            else:
                # Fallback to pure DSP processing
                print("Using DSP-only processing...")
                result = self.dsp_processor.process_audio(
                    wav_path, output_path,
                    character, clarity, room
                )
                
                # Clean up temp files
                if temp_wav and os.path.exists(temp_wav):
                    os.remove(temp_wav)
                
                return result
                
        except Exception as e:
            # Clean up on error
            if temp_wav and os.path.exists(temp_wav):
                os.remove(temp_wav)
            raise e

# Example usage and testing
if __name__ == "__main__":
    import sys
    
    print("Roadshow ML Inference Module")
    print("-" * 50)
    
    if len(sys.argv) < 2:
        print("Usage: python roadshow_ml_inference.py <model_path> [test_audio.wav]")
        print("\nExample:")
        print("  python roadshow_ml_inference.py models/roadshow_u87_model.pt test_iphone.wav")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    # Test model loading
    try:
        processor = U87MLProcessor(model_path)
        print(f"✓ Model loaded successfully from {model_path}")
        print(f"✓ Using device: {processor.device}")
        
        # Test processing if audio file provided
        if len(sys.argv) > 2:
            test_input = sys.argv[2]
            test_output = test_input.replace('.', '_ml_processed.')
            
            print(f"\nProcessing: {test_input}")
            processor.process_file(test_input, test_output, strength=0.8)
            print(f"✓ Output saved to: {test_output}")
            
    except Exception as e:
        print(f"✗ Error: {e}")