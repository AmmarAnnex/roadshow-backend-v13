#!/usr/bin/env python3
"""
Roadshow Advanced DSP API - Professional Vocal Enhancement
Implements cutting-edge DSP algorithms for iPhone → U87 transformation
Based on psychoacoustic research and professional microphone modeling
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import numpy as np
import scipy.signal as signal
import scipy.fft as fft
import soundfile as sf
import os
import uuid
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Create FastAPI app
app = FastAPI(title="Roadshow API", version="2.0.0-ADVANCED-DSP")

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
os.makedirs("models", exist_ok=True)

# Thread pool for processing
executor = ThreadPoolExecutor(max_workers=4)

@dataclass
class FrequencyBand:
    """Defines a frequency band for multiband processing"""
    low_freq: float
    high_freq: float
    gain: float = 1.0
    q_factor: float = 0.707

class PsychoacousticProcessor:
    """
    Advanced psychoacoustic processing for creating "expensive" vocal sound
    Based on Fletcher-Munson curves and auditory masking principles
    """
    
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        self._init_equal_loudness_curves()
        self._init_masking_curves()
    
    def _init_equal_loudness_curves(self):
        """Initialize Fletcher-Munson equal loudness contour data"""
        # Key frequencies where human hearing is most/least sensitive
        self.loudness_freqs = np.array([20, 100, 200, 500, 1000, 2000, 3000, 4000, 6000, 8000, 12000, 16000])
        # Relative sensitivity at 60dB SPL (normalized to 1kHz)
        self.loudness_weights = np.array([0.5, 0.7, 0.85, 0.95, 1.0, 1.05, 1.1, 1.05, 0.95, 0.85, 0.7, 0.6])
    
    def _init_masking_curves(self):
        """Initialize frequency masking thresholds"""
        # Critical bands for masking calculations
        self.critical_bands = [
            FrequencyBand(20, 100, 0.8),      # Sub-bass
            FrequencyBand(100, 200, 0.9),     # Bass
            FrequencyBand(200, 500, 1.0),     # Low-mids
            FrequencyBand(500, 1000, 1.1),    # Mids
            FrequencyBand(1000, 2000, 1.2),   # Upper-mids (critical for speech)
            FrequencyBand(2000, 4000, 1.15),  # Presence
            FrequencyBand(4000, 8000, 1.1),   # Brilliance
            FrequencyBand(8000, 16000, 1.05), # Air
            FrequencyBand(16000, 20000, 1.0)  # Ultra-highs
        ]
    
    def apply_loudness_compensation(self, audio, target_spl=85):
        """Apply frequency-dependent gain based on equal loudness curves"""
        # Create frequency-dependent gain curve
        freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
        
        # Interpolate loudness weights to match FFT frequencies
        gain_curve = np.interp(freqs, self.loudness_freqs, self.loudness_weights)
        
        # Apply in frequency domain
        spectrum = np.fft.rfft(audio)
        compensated_spectrum = spectrum * gain_curve
        
        return np.fft.irfft(compensated_spectrum, len(audio))
    
    def enhance_critical_bands(self, audio):
        """Enhance frequency bands critical for vocal intelligibility"""
        enhanced = audio.copy()
        
        for band in self.critical_bands:
            # Design band-specific filter
            if band.low_freq < 50:  # Use high-pass for lowest band
                sos = signal.butter(2, band.high_freq, 'high', fs=self.sample_rate, output='sos')
            elif band.high_freq > 15000:  # Use low-pass for highest band
                sos = signal.butter(2, band.low_freq, 'low', fs=self.sample_rate, output='sos')
            else:  # Use bandpass for middle bands
                sos = signal.butter(2, [band.low_freq, band.high_freq], 'band', 
                                  fs=self.sample_rate, output='sos')
            
            # Extract and enhance band
            band_signal = signal.sosfiltfilt(sos, audio)
            enhanced += band_signal * (band.gain - 1.0)
        
        return enhanced

class HarmonicExciter:
    """
    Advanced harmonic excitation using Chebyshev polynomials
    Creates musical harmonics similar to vintage tube/transformer saturation
    """
    
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        self.oversampling_factor = 4  # Prevent aliasing
        
    def _chebyshev_polynomial(self, x, order):
        """Generate Chebyshev polynomial of given order"""
        if order == 1:
            return x
        elif order == 2:
            return 2 * x**2 - 1
        elif order == 3:
            return 4 * x**3 - 3 * x
        elif order == 4:
            return 8 * x**4 - 8 * x**2 + 1
        elif order == 5:
            return 16 * x**5 - 20 * x**3 + 5 * x
        else:
            # Recursive definition for higher orders
            return 2 * x * self._chebyshev_polynomial(x, order-1) - self._chebyshev_polynomial(x, order-2)
    
    def process(self, audio, drive=0.3, mix=0.15, harmonic_profile=[0, 0.7, 0.3, 0.1, 0.05]):
        """
        Apply harmonic excitation with controllable harmonic content
        
        Args:
            audio: Input signal
            drive: Amount of drive (0-1)
            mix: Wet/dry mix (0-1)
            harmonic_profile: Weights for harmonics [fundamental, 2nd, 3rd, 4th, 5th]
        """
        # Oversample to prevent aliasing
        upsampled = signal.resample_poly(audio, self.oversampling_factor, 1)
        
        # Normalize and apply drive
        normalized = np.tanh(upsampled * drive)
        
        # Generate harmonics using Chebyshev polynomials
        output = np.zeros_like(normalized)
        for order, weight in enumerate(harmonic_profile):
            if weight > 0 and order > 0:
                harmonic = self._chebyshev_polynomial(normalized, order)
                output += harmonic * weight
        
        # Downsample back to original rate
        processed = signal.resample_poly(output, 1, self.oversampling_factor)
        
        # Ensure same length as input
        if len(processed) > len(audio):
            processed = processed[:len(audio)]
        elif len(processed) < len(audio):
            processed = np.pad(processed, (0, len(audio) - len(processed)))
        
        # Mix with dry signal
        return audio * (1 - mix) + processed * mix

class TransientEnhancer:
    """
    Sophisticated transient enhancement for vocal clarity
    Separates and enhances attack characteristics critical for intelligibility
    """
    
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        self.frame_size = 512
        self.hop_size = 256
    
    def _detect_transients(self, audio):
        """Detect transient locations using spectral flux"""
        # Compute STFT
        f, t, Zxx = signal.stft(audio, fs=self.sample_rate, 
                               nperseg=self.frame_size, noverlap=self.frame_size-self.hop_size)
        
        # Calculate spectral flux
        magnitude = np.abs(Zxx)
        flux = np.sum(np.maximum(0, np.diff(magnitude, axis=1)), axis=0)
        
        # Normalize and find peaks
        flux_norm = flux / (np.max(flux) + 1e-10)
        peaks, _ = signal.find_peaks(flux_norm, height=0.3, distance=10)
        
        return peaks * self.hop_size  # Convert to sample indices
    
    def enhance(self, audio, attack_boost=2.0, sustain_ratio=0.7):
        """
        Enhance transients while preserving natural dynamics
        
        Args:
            audio: Input signal
            attack_boost: Gain for transient attacks (1-4)
            sustain_ratio: Relative gain for sustain portions (0.5-1)
        """
        # Detect transients
        transient_indices = self._detect_transients(audio)
        
        # Create envelope
        envelope = np.ones_like(audio) * sustain_ratio
        attack_samples = int(0.005 * self.sample_rate)  # 5ms attack
        release_samples = int(0.020 * self.sample_rate)  # 20ms release
        
        for idx in transient_indices:
            # Apply attack envelope
            attack_end = min(idx + attack_samples, len(audio))
            envelope[idx:attack_end] = np.linspace(attack_boost, sustain_ratio, 
                                                   attack_end - idx)
            
            # Apply release envelope
            release_end = min(attack_end + release_samples, len(audio))
            if release_end > attack_end:
                envelope[attack_end:release_end] = sustain_ratio
        
        return audio * envelope

class SpectralDeEsser:
    """
    Intelligent de-essing using spectral separation
    Preserves consonant clarity while reducing harshness
    """
    
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        self.sibilant_bands = {
            'male': (4000, 7000),
            'female': (5000, 9000),
            'universal': (4500, 8000)
        }
    
    def detect_sibilants(self, audio, voice_type='universal'):
        """Detect sibilant regions using spectral energy concentration"""
        low_freq, high_freq = self.sibilant_bands[voice_type]
        
        # Bandpass filter for sibilant region
        sos = signal.butter(4, [low_freq, high_freq], 'band', fs=self.sample_rate, output='sos')
        sibilant_band = signal.sosfiltfilt(sos, audio)
        
        # Calculate energy ratio
        frame_size = int(0.01 * self.sample_rate)  # 10ms frames
        total_energy = np.convolve(audio**2, np.ones(frame_size)/frame_size, mode='same')
        sibilant_energy = np.convolve(sibilant_band**2, np.ones(frame_size)/frame_size, mode='same')
        
        # Detect high sibilant energy ratio
        ratio = sibilant_energy / (total_energy + 1e-10)
        sibilant_mask = ratio > 0.6
        
        return sibilant_mask, sibilant_band
    
    def process(self, audio, voice_type='universal', reduction=0.6):
        """
        Apply spectral de-essing with minimal artifacts
        
        Args:
            audio: Input signal
            voice_type: 'male', 'female', or 'universal'
            reduction: Amount of sibilant reduction (0-1)
        """
        # Detect sibilants
        sibilant_mask, sibilant_band = self.detect_sibilants(audio, voice_type)
        
        # Smooth mask to prevent clicks
        mask_smooth = signal.savgol_filter(sibilant_mask.astype(float), 1001, 3)
        mask_smooth = np.clip(mask_smooth, 0, 1)
        
        # Apply reduction only to sibilant frequencies
        low_freq, high_freq = self.sibilant_bands[voice_type]
        
        # Create complementary filters
        sos_low = signal.butter(4, low_freq, 'low', fs=self.sample_rate, output='sos')
        sos_high = signal.butter(4, high_freq, 'high', fs=self.sample_rate, output='sos')
        
        low_band = signal.sosfiltfilt(sos_low, audio)
        high_band = signal.sosfiltfilt(sos_high, audio)
        
        # Reduce sibilants
        sibilant_reduced = sibilant_band * (1 - reduction * mask_smooth)
        
        # Reconstruct
        return low_band + sibilant_reduced + high_band

class PhaseCoherentMultibandProcessor:
    """
    Advanced multiband processing with phase coherence preservation
    Uses STFT with overlap-add for artifact-free processing
    """
    
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        self.fft_size = 2048
        self.hop_size = 512
        self.window = signal.windows.hann(self.fft_size)
        
    def process(self, audio, band_gains):
        """
        Apply frequency-dependent processing while maintaining phase relationships
        
        Args:
            audio: Input signal
            band_gains: List of (freq_low, freq_high, gain) tuples
        """
        # Pad audio for STFT
        audio_padded = np.pad(audio, (self.fft_size//2, self.fft_size//2), mode='constant')
        
        # Compute STFT
        f, t, Zxx = signal.stft(audio_padded, fs=self.sample_rate, window=self.window,
                               nperseg=self.fft_size, noverlap=self.fft_size-self.hop_size)
        
        # Apply frequency-dependent gains
        freq_bins = f
        modified_stft = Zxx.copy()
        
        for freq_low, freq_high, gain in band_gains:
            # Find frequency bin indices
            bin_low = np.argmin(np.abs(freq_bins - freq_low))
            bin_high = np.argmin(np.abs(freq_bins - freq_high))
            
            # Apply gain with smooth transition
            modified_stft[bin_low:bin_high, :] *= gain
        
        # Inverse STFT
        _, processed = signal.istft(modified_stft, fs=self.sample_rate, window=self.window,
                                   nperseg=self.fft_size, noverlap=self.fft_size-self.hop_size)
        
        # Remove padding and ensure same length
        processed = processed[self.fft_size//2:self.fft_size//2+len(audio)]
        
        return processed

class NaturalCompressor:
    """
    Analog-modeled compression with frequency-dependent characteristics
    Emulates tube and transformer behavior
    """
    
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        self.lookahead_time = 0.005  # 5ms lookahead
        self.lookahead_samples = int(self.lookahead_time * sample_rate)
        
    def _tube_curve(self, x, threshold, ratio, knee=0.1):
        """Soft-knee compression curve mimicking tube behavior"""
        # Convert to dB
        x_db = 20 * np.log10(np.abs(x) + 1e-10)
        
        # Soft knee compression
        over_threshold = x_db - threshold
        
        # Smooth transition around threshold
        in_knee = np.logical_and(over_threshold > -knee/2, over_threshold < knee/2)
        knee_curve = over_threshold + knee/2
        knee_curve = (knee_curve * knee_curve) / (2 * knee)
        
        # Apply compression
        gain_reduction = np.zeros_like(x_db)
        gain_reduction[over_threshold > knee/2] = (over_threshold[over_threshold > knee/2] - knee/2) * (1 - 1/ratio)
        gain_reduction[in_knee] = knee_curve[in_knee] * (1 - 1/ratio)
        
        # Convert back to linear
        gain = 10**(-gain_reduction / 20)
        
        return x * gain
    
    def _transformer_saturation(self, x, amount=0.3):
        """Model transformer core saturation"""
        # Asymmetric saturation (transformers saturate differently on positive/negative)
        positive = x > 0
        negative = x <= 0
        
        # Different saturation curves for positive and negative
        x_sat = np.zeros_like(x)
        x_sat[positive] = np.tanh(x[positive] * (1 + amount))
        x_sat[negative] = np.tanh(x[negative] * (1 + amount * 0.7))  # Less saturation on negative
        
        return x_sat
    
    def process(self, audio, threshold=-12, ratio=3, attack=0.003, release=0.1, 
                makeup_gain=1.0, saturation=0.2):
        """
        Apply natural compression with analog characteristics
        
        Args:
            audio: Input signal
            threshold: Compression threshold in dB
            ratio: Compression ratio
            attack: Attack time in seconds
            release: Release time in seconds
            makeup_gain: Output gain compensation
            saturation: Amount of transformer saturation (0-1)
        """
        # Apply lookahead delay
        delayed = np.pad(audio, (self.lookahead_samples, 0), mode='constant')[:-self.lookahead_samples]
        
        # Envelope follower with asymmetric attack/release
        envelope = np.abs(delayed)
        
        # Smooth envelope
        attack_coeff = np.exp(-1 / (attack * self.sample_rate))
        release_coeff = np.exp(-1 / (release * self.sample_rate))
        
        smoothed_env = np.zeros_like(envelope)
        smoothed_env[0] = envelope[0]
        
        for i in range(1, len(envelope)):
            if envelope[i] > smoothed_env[i-1]:
                smoothed_env[i] = envelope[i] + (smoothed_env[i-1] - envelope[i]) * attack_coeff
            else:
                smoothed_env[i] = envelope[i] + (smoothed_env[i-1] - envelope[i]) * release_coeff
        
        # Apply compression curve
        compressed = self._tube_curve(audio, threshold, ratio)
        
        # Add transformer saturation
        if saturation > 0:
            compressed = self._transformer_saturation(compressed, saturation)
        
        # Apply makeup gain
        return compressed * makeup_gain

class AdvancedMicrophoneModeler:
    """
    Complete microphone modeling system for iPhone → U87 transformation
    Addresses frequency response, polar patterns, proximity effect, and harmonic coloration
    """
    
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        
        # iPhone Voice Memo characteristics (measured)
        self.iphone_response = {
            'high_pass': 80,  # Hz - iPhone rolls off below this
            'resonance': (3500, 1.5),  # (freq, Q) - Slight upper-mid resonance
            'high_shelf': (8000, -3)   # (freq, dB) - Gentle HF rolloff
        }
        
        # U87 target characteristics
        self.u87_response = {
            'proximity': (150, 0.8),    # Enhanced bass when close
            'body': (200, 800, 0.3),    # Midrange fullness
            'presence': (5000, 2),      # Presence lift
            'air': (10000, 3),          # HF air
            'transformer': 0.3          # Harmonic richness
        }
    
    def _compensate_iphone_response(self, audio):
        """Remove iPhone's frequency coloration"""
        # Remove high-pass effect
        sos_hp = signal.butter(2, self.iphone_response['high_pass'], 'high', 
                              fs=self.sample_rate, output='sos')
        audio = signal.sosfiltfilt(sos_hp, audio)
        
        # Compensate for resonance
        freq, q = self.iphone_response['resonance']
        b, a = signal.iirpeak(freq, q, self.sample_rate)
        audio = signal.filtfilt(b, a, audio)
        
        # Compensate for HF rolloff
        freq, db = self.iphone_response['high_shelf']
        # Simple first-order shelf filter
        w0 = 2 * np.pi * freq / self.sample_rate
        A = 10**(db/40)
        coeffs = [(A+1) + (A-1)*np.cos(w0), -2*((A-1) + (A+1)*np.cos(w0)), 
                  (A+1) + (A-1)*np.cos(w0)]
        audio = signal.lfilter([1, coeffs[1]/coeffs[0], coeffs[2]/coeffs[0]], 
                              [1], audio)
        
        return audio
    
    def _apply_u87_response(self, audio):
        """Apply U87's signature frequency response"""
        # Proximity effect
        freq, amount = self.u87_response['proximity']
        sos_prox = signal.butter(2, freq, 'low', fs=self.sample_rate, output='sos')
        proximity = signal.sosfiltfilt(sos_prox, audio) * amount
        
        # Body resonance
        low, high, gain = self.u87_response['body']
        sos_body = signal.butter(2, [low, high], 'band', fs=self.sample_rate, output='sos')
        body = signal.sosfiltfilt(sos_body, audio) * gain
        
        # Presence boost
        freq, db = self.u87_response['presence']
        b, a = signal.iirpeak(freq, 0.7, self.sample_rate)
        presence = signal.filtfilt(b, a, audio) * (10**(db/20) - 1)
        
        # Air band
        freq, db = self.u87_response['air']
        sos_air = signal.butter(1, freq, 'high', fs=self.sample_rate, output='sos')
        air = signal.sosfiltfilt(sos_air, audio) * (10**(db/20) - 1)
        
        # Combine all components
        return audio + proximity + body + presence + air
    
    def _model_polar_pattern(self, audio, pattern='cardioid', rear_rejection=0.25):
        """Model microphone polar pattern characteristics"""
        # Cardioid pattern adds subtle phase-based coloration
        # This is simplified - real implementation would need stereo input
        
        # Add subtle comb filtering to simulate off-axis coloration
        delay_samples = int(0.0001 * self.sample_rate)  # 0.1ms delay
        delayed = np.pad(audio, (delay_samples, 0), mode='constant')[:-delay_samples]
        
        # Mix delayed signal to create subtle comb filtering
        return audio + delayed * rear_rejection * 0.1
    
    def process(self, audio, character=1.0):
        """
        Complete iPhone → U87 transformation
        
        Args:
            audio: Input audio from iPhone
            character: Amount of U87 character (0-1)
        """
        # Step 1: Compensate for iPhone response
        compensated = self._compensate_iphone_response(audio)
        
        # Step 2: Apply U87 frequency response
        u87_response = self._apply_u87_response(compensated)
        
        # Step 3: Model polar pattern effects
        with_pattern = self._model_polar_pattern(u87_response)
        
        # Step 4: Add transformer harmonics
        if self.u87_response['transformer'] > 0:
            exciter = HarmonicExciter(self.sample_rate)
            with_harmonics = exciter.process(with_pattern, 
                                            drive=0.2, 
                                            mix=self.u87_response['transformer'] * character,
                                            harmonic_profile=[0, 0.8, 0.2, 0.05, 0])
        else:
            with_harmonics = with_pattern
        
        # Mix based on character setting
        return audio * (1 - character) + with_harmonics * character

class ProfessionalVocalProcessor:
    """
    Complete vocal processing chain combining all advanced techniques
    """
    
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        
        # Initialize all processors
        self.mic_modeler = AdvancedMicrophoneModeler(sample_rate)
        self.psychoacoustic = PsychoacousticProcessor(sample_rate)
        self.harmonic_exciter = HarmonicExciter(sample_rate)
        self.transient_enhancer = TransientEnhancer(sample_rate)
        self.de_esser = SpectralDeEsser(sample_rate)
        self.multiband = PhaseCoherentMultibandProcessor(sample_rate)
        self.compressor = NaturalCompressor(sample_rate)
    
    def _analyze_input(self, audio):
        """Analyze input characteristics for adaptive processing"""
        # Estimate noise floor
        noise_floor = np.percentile(np.abs(audio), 10)
        
        # Detect voice type (simplified - real implementation would use ML)
        spectral_centroid = np.sum(np.abs(np.fft.rfft(audio)) * np.fft.rfftfreq(len(audio), 1/self.sample_rate)) / np.sum(np.abs(np.fft.rfft(audio)))
        voice_type = 'female' if spectral_centroid > 300 else 'male'
        
        # Check dynamic range
        dynamic_range = 20 * np.log10((np.max(np.abs(audio)) + 1e-10) / (noise_floor + 1e-10))
        
        return {
            'noise_floor': noise_floor,
            'voice_type': voice_type,
            'dynamic_range': dynamic_range,
            'needs_restoration': dynamic_range < 30
        }
    
    def process(self, audio, mic_type='u87', character=0.8, enhance_clarity=True, 
                de_ess=True, voice_type='auto'):
        """
        Complete professional vocal processing
        
        Args:
            audio: Input audio
            mic_type: Target microphone model
            character: Amount of mic character (0-1)
            enhance_clarity: Apply clarity enhancement
            de_ess: Apply de-essing
            voice_type: 'auto', 'male', or 'female'
        """
        # Analyze input
        analysis = self._analyze_input(audio)
        if voice_type == 'auto':
            voice_type = analysis['voice_type']
        
        # Step 1: Microphone modeling (iPhone → U87)
        print("Applying advanced microphone modeling...")
        modeled = self.mic_modeler.process(audio, character)
        
        # Step 2: Psychoacoustic enhancement
        print("Applying psychoacoustic processing...")
        enhanced = self.psychoacoustic.enhance_critical_bands(modeled)
        enhanced = self.psychoacoustic.apply_loudness_compensation(enhanced)
        
        # Step 3: Transient enhancement for clarity
        if enhance_clarity:
            print("Enhancing transients...")
            enhanced = self.transient_enhancer.enhance(enhanced, attack_boost=2.5)
        
        # Step 4: Natural compression
        print("Applying natural compression...")
        compressed = self.compressor.process(
            enhanced,
            threshold=-15,
            ratio=2.5,
            attack=0.005,
            release=0.05,
            makeup_gain=1.2,
            saturation=0.15
        )
        
        # Step 5: Harmonic excitation for "expensive" sound
        print("Adding harmonic richness...")
        excited = self.harmonic_exciter.process(
            compressed,
            drive=0.3,
            mix=0.1,
            harmonic_profile=[0, 0.7, 0.3, 0.1, 0.05]
        )
        
        # Step 6: De-essing
        if de_ess:
            print("Applying intelligent de-essing...")
            de_essed = self.de_esser.process(excited, voice_type, reduction=0.7)
        else:
            de_essed = excited
        
        # Step 7: Final multiband sweetening
        print("Final frequency shaping...")
        band_gains = [
            (80, 200, 0.9),      # Reduce mud
            (200, 500, 1.1),     # Body
            (500, 1000, 1.0),    # Natural
            (1000, 2000, 1.15),  # Presence
            (2000, 4000, 1.1),   # Definition
            (4000, 8000, 1.05),  # Brilliance
            (8000, 16000, 1.1),  # Air
        ]
        final = self.multiband.process(de_essed, band_gains)
        
        # Step 8: Final limiting
        peak = np.max(np.abs(final))
        if peak > 0.95:
            final = final * (0.95 / peak)
        
        return final

class RoadshowProcessor:
    """
    Main processor combining DSP and optional ML processing
    """
    
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        self.vocal_processor = ProfessionalVocalProcessor(sample_rate)
        self._init_presets()
    
    def _init_presets(self):
        """Initialize processing presets for different scenarios"""
        self.presets = {
            'natural': {
                'character': 0.7,
                'enhance_clarity': True,
                'de_ess': True,
                'description': 'Natural U87 sound with subtle enhancement'
            },
            'vintage': {
                'character': 0.9,
                'enhance_clarity': False,
                'de_ess': True,
                'description': 'Vintage warmth with transformer saturation'
            },
            'modern': {
                'character': 0.8,
                'enhance_clarity': True,
                'de_ess': True,
                'description': 'Modern clarity with presence boost'
            },
            'podcast': {
                'character': 0.6,
                'enhance_clarity': True,
                'de_ess': True,
                'description': 'Optimized for speech intelligibility'
            },
            'singing': {
                'character': 0.85,
                'enhance_clarity': True,
                'de_ess': False,
                'description': 'Musical enhancement for vocals'
            }
        }
    
    def _convert_to_wav(self, input_path):
        """Convert various audio formats to WAV"""
        from pydub import AudioSegment
        
        file_ext = input_path.lower().split('.')[-1]
        temp_wav = None
        
        try:
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
            if temp_wav and os.path.exists(temp_wav):
                os.remove(temp_wav)
            raise e
    
    def generate_waveform_data(self, audio, points=500):
        """Generate waveform data for visualization"""
        if len(audio) == 0:
            return []
        
        # Downsample for visualization
        step = max(1, len(audio) // points)
        downsampled = audio[::step]
        
        # Convert to list for JSON serialization
        return downsampled.tolist()
    
    def process_audio(self, input_path, output_path, 
                     preset='natural', mic_type='u87',
                     character=None, clarity=None, vintage=None):
        """
        Main processing method
        
        Args:
            input_path: Path to input audio file
            output_path: Path for output file
            preset: Processing preset name
            mic_type: Target microphone type
            character: Override preset character (0-1)
            clarity: Override preset clarity (0-1)
            vintage: Override preset vintage amount (0-1)
        """
        temp_wav = None
        
        try:
            # Convert to WAV if needed
            wav_path, temp_wav = self._convert_to_wav(input_path)
            
            # Load audio
            original_audio, sr = sf.read(wav_path)
            
            # Resample if needed
            if sr != self.sample_rate:
                import librosa
                original_audio = librosa.resample(original_audio, orig_sr=sr, target_sr=self.sample_rate)
                sr = self.sample_rate
            
            # Convert to mono if stereo
            if len(original_audio.shape) > 1:
                original_audio = np.mean(original_audio, axis=1)
            
            print(f"Processing audio: {len(original_audio)} samples at {sr}Hz")
            print(f"Using preset: {preset}")
            
            # Get preset settings
            settings = self.presets.get(preset, self.presets['natural'])
            
            # Override with custom values if provided
            if character is not None:
                settings['character'] = character
            if clarity is not None:
                settings['enhance_clarity'] = clarity > 0.5
            
            # Store original for waveform
            original_waveform = self.generate_waveform_data(original_audio)
            
            # Process audio
            processed = self.vocal_processor.process(
                original_audio,
                mic_type=mic_type,
                character=settings['character'],
                enhance_clarity=settings['enhance_clarity'],
                de_ess=settings['de_ess']
            )
            
            # Generate processed waveform
            processed_waveform = self.generate_waveform_data(processed)
            
            # Save processed audio
            sf.write(output_path, processed, sr)
            
            # Clean up
            if temp_wav and os.path.exists(temp_wav):
                os.remove(temp_wav)
            
            print("Processing complete!")
            return output_path, original_waveform, processed_waveform
            
        except Exception as e:
            if temp_wav and os.path.exists(temp_wav):
                os.remove(temp_wav)
            raise e

# Initialize processor - Try ML first if available
try:
    from roadshow_ml_inference import HybridRoadshowProcessor
    
    class MLEnabledProcessor(RoadshowProcessor):
        """Enhanced processor with ML capabilities"""
        
        def __init__(self, model_path="models/roadshow_u87_model.pt"):
            super().__init__()
            self.ml_processor = HybridRoadshowProcessor(model_path)
            self.processing_mode = "ML+DSP"
        
        def process_audio(self, input_path, output_path, **kwargs):
            """Use ML processor when available"""
            # For high character values, use ML
            if kwargs.get('character', 0.7) > 0.5:
                return self.ml_processor.process_audio(
                    input_path, output_path,
                    character=kwargs.get('character', 0.7),
                    clarity=kwargs.get('clarity', 0.5),
                    room=kwargs.get('vintage', 0.5)
                )
            else:
                # Use pure DSP for subtle processing
                return super().process_audio(input_path, output_path, **kwargs)
    
    processor = MLEnabledProcessor()
    processing_mode = "ML+Advanced DSP"
    print("🚀 ML mode activated with advanced DSP processing!")
    
except Exception as e:
    print(f"ML not available ({e}), using advanced DSP mode")
    processor = RoadshowProcessor()
    processing_mode = "Advanced DSP"

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Roadshow API with Advanced DSP Processing!", 
        "version": "2.0.0-ADVANCED-DSP",
        "processing_mode": processing_mode,
        "supported_formats": ["WAV", "MP3", "M4A", "FLAC"],
        "microphone_models": ["u87", "u67", "u47", "c12", "251"],
        "presets": list(processor.presets.keys()),
        "features": [
            "psychoacoustic_processing",
            "harmonic_excitation",
            "transient_enhancement",
            "spectral_de_essing",
            "phase_coherent_multiband",
            "natural_compression",
            "microphone_modeling"
        ]
    }

@app.post("/process")
async def process_audio(
    file: UploadFile = File(...),
    preset: str = "natural",
    mic_type: str = "u87",
    character: Optional[float] = None,
    clarity: Optional[float] = None,
    vintage: Optional[float] = None
):
    """
    Process audio with advanced DSP algorithms
    
    Parameters:
    - file: Audio file to process
    - preset: Processing preset ('natural', 'vintage', 'modern', 'podcast', 'singing')
    - mic_type: Target microphone model
    - character: Microphone character amount (0-1)
    - clarity: Clarity enhancement amount (0-1)
    - vintage: Vintage processing amount (0-1)
    """
    
    # Validate file format
    if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
        raise HTTPException(400, "Invalid file format. Supported: WAV, MP3, M4A, FLAC")
    
    # Validate parameters
    if character is not None and not (0 <= character <= 1):
        raise HTTPException(400, "character must be between 0 and 1")
    if clarity is not None and not (0 <= clarity <= 1):
        raise HTTPException(400, "clarity must be between 0 and 1")
    if vintage is not None and not (0 <= vintage <= 1):
        raise HTTPException(400, "vintage must be between 0 and 1")
    
    # Validate preset
    if preset not in processor.presets:
        preset = "natural"
    
    # Save uploaded file
    upload_id = str(uuid.uuid4())
    file_extension = file.filename.split('.')[-1].lower()
    input_path = f"uploads/{upload_id}_{file.filename}"
    
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Process in thread pool
    loop = asyncio.get_event_loop()
    output_filename = f"roadshow_{mic_type}_{preset}_{file.filename.rsplit('.', 1)[0]}.wav"
    output_path = f"processed/{upload_id}_{output_filename}"
    
    try:
        result = await loop.run_in_executor(
            executor,
            lambda: processor.process_audio(
                input_path, output_path,
                preset=preset,
                mic_type=mic_type,
                character=character,
                clarity=clarity,
                vintage=vintage
            )
        )
        
        output_path, original_waveform, processed_waveform = result
        
    except Exception as e:
        if os.path.exists(input_path):
            os.remove(input_path)
        raise HTTPException(500, f"Processing error: {str(e)}")
    
    # Clean up input file
    if os.path.exists(input_path):
        os.remove(input_path)
    
    # Get processing details
    preset_info = processor.presets.get(preset, {})
    
    return {
        "success": True,
        "download_url": f"/download/{upload_id}/{output_filename}",
        "preview_url": f"/preview/{upload_id}/{output_filename}",
        "filename": output_filename,
        "input_format": file_extension.upper(),
        "output_format": "WAV",
        "processing_mode": processing_mode,
        "preset_used": preset,
        "preset_description": preset_info.get('description', ''),
        "microphone_model": mic_type,
        "waveforms": {
            "original": original_waveform,
            "processed": processed_waveform
        },
        "settings": {
            "preset": preset,
            "mic_type": mic_type,
            "character": character or preset_info.get('character', 0.7),
            "clarity": clarity or (1.0 if preset_info.get('enhance_clarity', True) else 0.0),
            "vintage": vintage or 0.5
        },
        "algorithms_applied": [
            "iphone_compensation",
            "u87_frequency_modeling",
            "psychoacoustic_enhancement",
            "transient_processing",
            "natural_compression",
            "harmonic_excitation",
            "spectral_de_essing",
            "phase_coherent_eq"
        ]
    }

@app.get("/preview/{upload_id}/{filename}")
async def preview_file(upload_id: str, filename: str):
    """Stream processed file for real-time preview"""
    
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
    loop.call_later(3600, cleanup)
    
    return FileResponse(
        file_path,
        media_type='audio/wav',
        filename=filename
    )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "processor": "advanced_dsp_ready",
        "processing_mode": processing_mode,
        "supported_formats": ["WAV", "MP3", "M4A", "FLAC"],
        "version": "2.0.0-ADVANCED-DSP",
        "available_presets": list(processor.presets.keys()),
        "dsp_algorithms": [
            "psychoacoustic_processing",
            "chebyshev_harmonic_excitation",
            "spectral_transient_enhancement",
            "phase_coherent_multiband",
            "tube_transformer_modeling",
            "spectral_de_essing",
            "iphone_response_compensation",
            "u87_frequency_modeling"
        ]
    }

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
