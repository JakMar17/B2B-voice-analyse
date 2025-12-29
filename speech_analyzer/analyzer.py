import librosa
import numpy as np
from scipy.signal import resample
import json

def analyze_speech_return_dict(audio_path, segments=5):
    y, sr = librosa.load(audio_path, sr=None)
    audio_length = librosa.get_duration(y=y, sr=sr)

    num_parts = segments
    # Create time intervals to match number of segments
    time_intervals = np.linspace(0, audio_length, num=num_parts).tolist()

    # ---------------- TONALITY ----------------
    def analyze_tonality(y, sr):
        # 1. Extract raw pitch (Hz)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        
        absolute_pitches = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if 70 < pitch < 400: # Human vocal range filter
                absolute_pitches.append(pitch)
        
        if not absolute_pitches:
            return np.zeros(num_parts)

        absolute_pitches = np.array(absolute_pitches)
        base_hz = np.mean(absolute_pitches)
        
        # 2. Calculate local percentage deviations (Relative to speaker's own base)
        # This allows us to judge the "effort" regardless of gender or natural pitch
        dev_pct = np.abs((absolute_pitches - base_hz) / base_hz) * 100
        
        # 3. Calculate "Vocal Energy" (Standard Deviation of the percentages)
        # This represents how much the speaker "moves" their tone.
        vocal_energy = np.std(dev_pct) 

        # 4. Map Vocal Energy to your 0-100 scale
        # Calibrated Logic:
        # - Low movement (0-2%) -> 0-30 (Monotone)
        # - Standard movement (approx 5%) -> 45 (Base Monotony Anchor)
        # - High movement (>12%) -> 70-100 (Engagement)
        
        # Transformation Multiplier (Linear Scaling for simplicity)
        # Adjust the 'scalar' to tune the sensitivity of the zones
        scalar = 9.0 
        offset = 10.0 # Ensures we don't start at literal zero
        
        zone_data = (dev_pct * (scalar / (vocal_energy + 1e-6))) + offset
        
        # Ensure results stay within the 0-100 constraints
        final_data = np.clip(zone_data, 0, 100)

        # 5. Resample to match your output length (num_parts)
        return resample(final_data, num_parts)

    # ---------------- PACE ----------------
    def analyze_pace(y, sr, num_parts):
        segment_length = len(y) // num_parts
        pace = []

        for i in range(num_parts):
            segment_y = y[i * segment_length:(i + 1) * segment_length]
            if len(segment_y) == 0:
                pace.append(0)
                continue

            onset_env = librosa.onset.onset_strength(y=segment_y, sr=sr)
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)

            duration_minutes = (len(segment_y) / sr) / 60
            words_per_minute = len(onset_times) / duration_minutes if duration_minutes else 0
            pace.append(words_per_minute)

        return np.array(pace)

    # ---------------- PAUSING ----------------
    def analyze_pauses(y, sr, num_parts):
        segment_length = len(y) // num_parts
        pauses = []

        for i in range(num_parts):
            segment_y = y[i * segment_length:(i + 1) * segment_length]
            if len(segment_y) == 0:
                pauses.append(0)
                continue

            rms = librosa.feature.rms(y=segment_y)[0]
            if len(rms) == 0:
                pauses.append(0)
                continue

            threshold = np.mean(rms) * 0.5
            silent_regions = rms < threshold
            times = librosa.times_like(rms)

            pause_durations = []
            current_pause = 0

            for j, is_silent in enumerate(silent_regions):
                dt = times[1] - times[0] if len(times) > 1 else 0.01
                if is_silent:
                    current_pause += dt
                elif current_pause > 0:
                    pause_durations.append(current_pause)
                    current_pause = 0

            if current_pause > 0:
                pause_durations.append(current_pause)

            pauses.append(np.mean(pause_durations) if pause_durations else 0)

        return np.array(pauses)

    # ---------------- VOCAL CHARACTERISTICS ----------------
    def analyze_vocal_characters(y, sr):
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

        if len(spectral_centroid) == 0:
            spectral_centroid_resampled = np.zeros(num_parts)
        else:
            spectral_centroid_resampled = resample(spectral_centroid, num_parts)

        mean_centroid = np.mean(spectral_centroid_resampled)
        min_centroid = 1000
        max_centroid = 4000

        masculinity_percentage = max(
            0, min(100, (max_centroid - mean_centroid) / (max_centroid - min_centroid) * 100)
        )
        femininity_percentage = 100 - masculinity_percentage

        return masculinity_percentage, femininity_percentage

    # ---------------- RUN ANALYSIS ----------------
    tonality_var = analyze_tonality(y, sr)
    pace_var = analyze_pace(y, sr, num_parts)
    pauses_var = analyze_pauses(y, sr, num_parts)
    masculinity_percentage, femininity_percentage = analyze_vocal_characters(y, sr)

    # ---------------- BUILD RESPONSE ----------------
    response = {
        "tonality": {
            "time": time_intervals,
            "data": tonality_var.tolist(),
            "average": float(np.mean(tonality_var))
        },
        "pace": {
            "time": time_intervals,
            "data": pace_var.tolist(),
            "average": float(np.mean(pace_var))
        },
        "pausing": {
            "time": time_intervals,
            "data": pauses_var.tolist(),
            "average": float(np.mean(pauses_var))
        },
        "vocalCharacters": {
            "masculinityPercentage": masculinity_percentage,
            "femininityPercentage": femininity_percentage
        }
    }

    return response
