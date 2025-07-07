#!/usr/bin/env python3
"""
V13 Simplified Backend Processor
Direct processing without dry/wet mixing for Railway deployment
"""

import os
import numpy as np
import torch
import soundfile as sf
import librosa
from scipy import signal
import logging

logger = logging.getLogger(__name__)

def diagnose_audio_simple(audio, label="Audio"):
    """Simple audio diagnostics"""
    if hasattr(audio, '__len__') and len(audio) > 0:
        rms = np.sqrt(np.mean(audio**2))
        logger.info(f"{label}: {len(audio)} samples, RMS: {rms:.6f}, Range: [{np.min(audio):.3f}, {np.max(audio):.3f}]")
        return len(audio) > 0 and rms > 1e-6
    else:
        logger.warning(f"{label}: Empty or invalid audio")
        return False

class ConservativeDenoiserModule:
    """Conservative denoising - just enough to clean, not destroy character"""
    
    def __init__(self):
        self.name = "Conservative Denoiser"
        self.is_available = False
        self.model = None
        
    def check_availability(self):
        try:
            from denoiser.pretrained import dns64
            self.is_available = True
            logger.info("✅ Conservative Denoiser available")
            return True
        except ImportError:
            logger.warning("❌ Meta Denoiser not available - using built-in denoising")
            self.is_available = True
            return True
    
    def initialize(self):
        if self.is_available:
            try:
                from denoiser.pretrained import dns64
                self.model = dns64(pretrained=True)
                self.model.eval()
                logger.info("Conservative Denoiser initialized (Meta DNS64)")
            except Exception as e:
                logger.warning(f"Meta Denoiser failed, using built-in: {e}")
                self.model = "builtin"
                logger.info("Conservative Denoiser initialized (built-in)")
    
    def process(self, audio, sr):
        """Process audio - NO DRY/WET MIXING"""
        if not diagnose_audio_simple(audio, "Denoiser Input"):
            return audio, sr
            
        try:
            if self.model == "builtin":
                result = self._builtin_denoise(audio, sr)
            else:
                result = self._meta_denoise(audio, sr)
            
            if not diagnose_audio_simple(result, "Denoiser Output"):
                logger.warning("Denoiser produced invalid output, using original")
                result = audio
            
            logger.info("Conservative Denoiser processed successfully")
            return result, sr
            
        except Exception as e:
            logger.error(f"Conservative Denoiser failed: {e}")
            return audio, sr
    
    def _builtin_denoise(self, audio, sr):
        """Very gentle built-in denoising"""
        try:
            D = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(D)
            phase = np.angle(D)
            
            # Very conservative noise reduction
            noise_floor = np.percentile(magnitude, 15, axis=1, keepdims=True)
            noise_threshold = noise_floor * 2.5
            
            # Gentle soft mask
            soft_mask = np.minimum(magnitude / (noise_threshold + 1e-10), 1.0)
            soft_mask = np.maximum(soft_mask, 0.5)  # Don't suppress below 50%
            
            denoised_magnitude = magnitude * soft_mask
            D_denoised = denoised_magnitude * np.exp(1j * phase)
            denoised = librosa.istft(D_denoised, hop_length=512, length=len(audio))
            
            return denoised.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Built-in denoising failed: {e}")
            return audio
    
    def _meta_denoise(self, audio, sr):
        """Meta Denoiser processing"""
        try:
            if sr != 16000:
                audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            else:
                audio_16k = audio
            
            audio_tensor = torch.from_numpy(audio_16k).float().unsqueeze(0)
            
            with torch.no_grad():
                denoised = self.model(audio_tensor)
            
            denoised_audio = denoised.squeeze().numpy()
            
            if sr != 16000:
                denoised_audio = librosa.resample(denoised_audio, orig_sr=16000, target_sr=sr)
            
            # Ensure same length as input
            if len(denoised_audio) != len(audio):
                if len(denoised_audio) > len(audio):
                    denoised_audio = denoised_audio[:len(audio)]
                else:
                    denoised_audio = np.pad(denoised_audio, (0, len(audio) - len(denoised_audio)))
            
            return denoised_audio.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Meta denoising failed: {e}")
            return audio

class SimpleEQModule:
    """Simple, gentle EQ for professional character"""
    
    def __init__(self, eq_type="u87"):
        self.name = f"Simple EQ ({eq_type.upper()})"
        self.eq_type = eq_type
        self.is_available = True
        
    def process(self, audio, sr):
        """Apply gentle EQ - NO DRY/WET MIXING"""
        if not diagnose_audio_simple(audio, "EQ Input"):
            return audio, sr
            
        try:
            result = self._apply_gentle_eq(audio, sr)
            
            if not diagnose_audio_simple(result, "EQ Output"):
                logger.warning("EQ produced invalid output, using original")
                result = audio
            
            logger.info("Simple EQ processed successfully")
            return result, sr
            
        except Exception as e:
            logger.error(f"Simple EQ failed: {e}")
            return audio, sr
    
    def _apply_gentle_eq(self, audio, sr):
        """Very gentle EQ curve"""
        try:
            D = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(D)
            phase = np.angle(D)
            
            freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
            eq_curve = np.ones_like(freqs)
            
            if self.eq_type == "u87":
                # Very gentle U87-inspired curve
                warmth_idx = np.where((freqs >= 150) & (freqs <= 300))[0]
                if len(warmth_idx) > 0:
                    eq_curve[warmth_idx] *= 1.05  # +0.4dB
                
                presence_idx = np.where((freqs >= 4000) & (freqs <= 6000))[0]
                if len(presence_idx) > 0:
                    eq_curve[presence_idx] *= 1.1  # +0.8dB
                
                air_idx = np.where((freqs >= 10000) & (freqs <= 15000))[0]
                if len(air_idx) > 0:
                    eq_curve[air_idx] *= 1.05  # +0.4dB
            
            # Apply EQ
            magnitude *= eq_curve[:, np.newaxis]
            
            # Reconstruct
            D_eq = magnitude * np.exp(1j * phase)
            audio_eq = librosa.istft(D_eq, hop_length=512, length=len(audio))
            
            return audio_eq.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Gentle EQ failed: {e}")
            return audio

class V13SimplifiedProcessor:
    """V13 Simplified - Direct processing for web deployment"""
    
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        self.modules = []
        self.setup_pipeline()
    
    def setup_pipeline(self):
        """Setup simplified pipeline"""
        logger.info("🎵 Initializing V13 Simplified Pipeline...")
        
        # Stage 1: Conservative denoising
        denoiser = ConservativeDenoiserModule()
        if denoiser.check_availability():
            denoiser.initialize()
            self.modules.append(denoiser)
        
        # Stage 2: Gentle EQ
        eq = SimpleEQModule("u87")
        self.modules.append(eq)
        
        logger.info(f"🎛️ V13 Simplified Pipeline: {len(self.modules)} modules ready")
    
    def generate_waveform_data(self, audio, points=500):
        """Generate waveform data for visualization"""
        if len(audio) == 0:
            return []
        
        step = max(1, len(audio) // points)
        downsampled = audio[::step]
        return downsampled.tolist()
    
    def process_audio(self, input_path, output_path, character=0.7, clarity=0.5, vintage=0.5):
        """Process audio file with V13 Simplified"""
        logger.info(f"🎵 V13 Simplified Processing: {os.path.basename(input_path)}")
        
        try:
            # Load audio
            audio, sr = librosa.load(input_path, sr=self.sample_rate, mono=True)
            logger.info(f"Input: {len(audio)} samples @ {sr}Hz")
            
            # Validate loaded audio
            if not diagnose_audio_simple(audio, "Loaded Audio"):
                raise ValueError("Invalid input audio")
            
            # Store original for waveform
            original_waveform = self.generate_waveform_data(audio)
            
            # Process through pipeline
            for i, module in enumerate(self.modules):
                logger.info(f"Stage {i+1}: {module.name}")
                audio, sr = module.process(audio, sr)
                
                # Validate each stage
                if not diagnose_audio_simple(audio, f"Stage {i+1} Output"):
                    raise ValueError(f"Stage {i+1} produced invalid audio")
            
            # Final validation
            if not diagnose_audio_simple(audio, "Final Audio"):
                raise ValueError("Final audio is invalid")
            
            # Gentle normalization
            max_val = np.max(np.abs(audio))
            if max_val > 0.95:
                audio = audio / max_val * 0.95
                logger.info("Applied gentle normalization")
            
            # Generate processed waveform
            processed_waveform = self.generate_waveform_data(audio)
            
            # Create output directory
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Save result
            sf.write(output_path, audio, sr)
            
            # Verify saved file
            verify_audio, verify_sr = librosa.load(output_path, sr=None)
            if not diagnose_audio_simple(verify_audio, "Saved Audio"):
                raise ValueError("Saved file verification failed")
            
            logger.info(f"✅ V13 Simplified: {os.path.basename(output_path)} @ {sr}Hz")
            
            return output_path, original_waveform, processed_waveform
            
        except Exception as e:
            logger.error(f"V13 simplified processing failed: {e}")
            raise e

# Test the processor
if __name__ == "__main__":
    processor = V13SimplifiedProcessor()
    print("V13 Simplified Processor initialized successfully!")
    print(f"Modules: {[m.name for m in processor.modules]}")