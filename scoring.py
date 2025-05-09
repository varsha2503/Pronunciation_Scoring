import os
import torch
import torchaudio
import numpy as np
import wave
import random
from scipy.io.wavfile import write
from gtts import gTTS
from pydub import AudioSegment
import parselmouth
import tgt
from librosa.sequence import dtw
import subprocess
import librosa
import sys
import sounddevice as sd
import threading
import keyboard

# ========== Paths and Config ==========
AUDIO_DIR = "audio"
USER_AUDIO = os.path.join(AUDIO_DIR, "user.wav")
NATIVE_AUDIO = os.path.join(AUDIO_DIR, "native.wav")
TEXT_PATH = os.path.join(AUDIO_DIR, "text.txt")
TEXTGRID_PATH = "alignments/user.TextGrid"
os.makedirs(AUDIO_DIR, exist_ok=True)

# ========== Model Setup ==========
from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H
bundle = WAV2VEC2_ASR_BASE_960H
model = bundle.get_model()
labels = bundle.get_labels()
sample_rate = bundle.sample_rate

# ========== Audio Handling ==========
audio_data = []
recording = False
fs = 16000

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    if recording:
        audio_data.append(indata.copy())

def start_recording():
    global recording
    audio_data.clear()
    recording = True
    print("🎙️ Recording started... Speak now!")
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=fs, dtype='int16'):
        while recording:
            sd.sleep(100)

def stop_recording():
    global recording
    recording = False
    audio_data_np = np.concatenate(audio_data, axis=0)
    write(USER_AUDIO, fs, audio_data_np)
    print(f"✅ Recording stopped and saved to {USER_AUDIO}")

def play_audio(path):
    if os.name == 'nt':  # Windows
        os.system(f'start {path}')
    else:  # macOS or Linux
        os.system(f'afplay "{path}"' if sys.platform == "darwin" else f'aplay "{path}"')



def listen_for_keys():
    print("\nPress 'r' to start recording and 's' to stop.")
    while True:
        if keyboard.is_pressed('r') and not recording:
            threading.Thread(target=start_recording).start()
        if keyboard.is_pressed('s') and recording:
            stop_recording()
            break

def resample_audio(path):
    waveform, sr = torchaudio.load(path)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    return waveform

def get_emissions(waveform):
    with torch.inference_mode():
        emissions, _ = model(waveform)
    return torch.nn.functional.softmax(emissions[0], dim=-1)

# ========== Context-aware GOP ==========
def context_aware_gop(audio_path):
    waveform = resample_audio(audio_path)
    probs = get_emissions(waveform)
    top_probs, _ = probs.max(dim=-1)
    log_post = torch.log(top_probs)
    S = -log_post.mean().item()
    return round((2 * np.exp(-3 * S)) / (1 + np.exp(-3 * S)) * 100, 2)

# ========== Traditional GOP ==========
def traditional_gop(audio_path):
    waveform = resample_audio(audio_path)
    probs = get_emissions(waveform)
    max_probs, _ = probs.max(dim=-1)
    return round(max_probs.mean().item() * 100, 2)

# ========== Pitch using Autocorrelation ==========

def auto_correlation_pitch(signal, sr=16000, frame_size=1024, hop_size=512):
    pitches = []

    # Convert to float32 and normalize
    if signal.dtype == np.int16:
        signal = signal.astype(np.float32) / 32768.0
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))

    for i in range(0, len(signal) - frame_size, hop_size):
        frame = signal[i:i + frame_size]

        # Energy threshold to ignore silent frames
        energy = np.sum(frame ** 2)
        if energy < 1e-5:
            pitches.append(0)
            continue

        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr) // 2:]

        peak = np.argmax(corr[1:]) + 1
        if peak == 0:
            pitches.append(0)
        else:
            f0 = sr / peak
            # Filter valid human pitch range
            if 50 <= f0 <= 500:
                pitches.append(f0)
            else:
                pitches.append(0)

    return np.array(pitches)


def autocorr_dtw_score(user_audio_path, native_audio_path, threshold=750):
    user_waveform, sr_user = torchaudio.load(user_audio_path)
    native_waveform, sr_native = torchaudio.load(native_audio_path)

    user_signal = user_waveform[0].numpy()
    native_signal = native_waveform[0].numpy()

    user_pitch = auto_correlation_pitch(user_signal, sr=sr_user)
    native_pitch = auto_correlation_pitch(native_signal, sr=sr_native)

    if len(user_pitch) == 0 or len(native_pitch) == 0:
        return 0.0

    D, _ = librosa.sequence.dtw(X=user_pitch.reshape(1, -1), Y=native_pitch.reshape(1, -1), metric='euclidean')
    distance = D[-1, -1]

    score = np.exp((threshold - distance) / threshold)
    score = min(max(score, 0), 1)
    return round(score * 100, 2)




def extract_pitch_deltas(audio_path):
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values == 0] = np.nan
    pitch_deltas = np.diff(pitch_values)
    return np.nan_to_num(pitch_deltas)


# ========== Improved DTW Score ==========
def pitch_dtw_score(user_audio, native_audio, threshold=750):
    d1 = extract_pitch_deltas(user_audio)
    d2 = extract_pitch_deltas(native_audio)
    D, _ = librosa.sequence.dtw(X=d1.reshape(1, -1), Y=d2.reshape(1, -1), metric='euclidean')
    distance = D[-1, -1]

    # Use logistic-like transformation to clamp score between 0 and 100
    score = 1 / (1 + np.exp((distance - threshold) / threshold))
    return round(score * 100, 2)

    



def compute_fluency_dtw(user_audio_path, native_audio_path, threshold=750):
    # Extract pitch contours (or better: intensity or energy envelope)
    def get_energy_deltas(audio_path):
        snd = parselmouth.Sound(audio_path)
        intensity = snd.to_intensity()
        duration = snd.duration
        frame_times = np.arange(0, duration, 0.01)
        energies = np.array([intensity.get_value(t) or 0 for t in frame_times])
        deltas = np.diff(energies)
        return np.nan_to_num(deltas)

    u_deltas = get_energy_deltas(user_audio_path)
    n_deltas = get_energy_deltas(native_audio_path)

    # DTW on energy deltas to capture pacing/pause alignment
    D, _ = librosa.sequence.dtw(X=u_deltas.reshape(1, -1), Y=n_deltas.reshape(1, -1), metric='euclidean')
    distance = D[-1, -1]

    # Logistic-based score transformation
    score = 1 / (1 + np.exp((distance - threshold) / threshold))
    return round(score * 100, 2)



# ========== Combined Score ==========
def final_score(gop_score, pitch_score, alpha=0.5):
    return round(alpha * gop_score + (1 - alpha) * pitch_score, 2)

# ========== Forced Alignment ==========
def run_mfa():
    if not os.path.exists(AUDIO_DIR):
        raise FileNotFoundError(f"❌ Corpus directory does not exist: {AUDIO_DIR}")
    try:
        result = subprocess.run([
            "mfa", "align",
            os.path.abspath(AUDIO_DIR),                      # Ensure full path
            os.path.abspath("dict/english_us_arpa.dict"),      # Dictionary path
            os.path.abspath("models/english_us_arpa.zip"),     # Pretrained model path
            os.path.abspath("alignments"),                     # Output directory
            "--clean", "--overwrite"
        ], check=True)
        print("✅ MFA alignment completed successfully")
    except subprocess.CalledProcessError as e:
        raise RuntimeError("❌ MFA alignment failed") from e
    except FileNotFoundError:
        raise RuntimeError("❌ MFA is not installed or not in PATH. Run `mfa --version` to check.")

# ========== Phoneme Scores ==========
def load_textgrid_phonemes(textgrid_path):
    tg = tgt.read_textgrid(textgrid_path)
    phones = tg.get_tier_by_name("phones")
    return [(interval.text, interval.start_time, interval.end_time) for interval in phones.intervals if interval.text.strip()]

def phoneme_confidence_from_emissions(waveform, phones, sr=sample_rate):
    emissions = get_emissions(waveform)
    scores = []
    for phoneme, start, end in phones:
        start_frame = int(start * sr / 320)
        end_frame = int(end * sr / 320)
        segment_probs = emissions[start_frame:end_frame]
        if len(segment_probs) == 0:
            continue
        top_probs, _ = segment_probs.max(dim=-1)
        avg_log_prob = torch.log(top_probs).mean().item()
        p = (2 * np.exp(-3 * -avg_log_prob)) / (1 + np.exp(-3 * -avg_log_prob))
        scores.append((phoneme, round(p * 100, 2)))
    return scores

def feedback_report(phonemes, threshold=60):
    weak = [f"{p}: {s}%" for p, s in phonemes if s < threshold]
    if not weak:
        return "✅ Excellent pronunciation!"
    return (
        "⚠️ Phonemes needing improvement:\n" +
        "\n".join(f" - {w}" for w in weak) +
        "\n\n💡 Tips:\n - Speak slowly\n - Mimic native tone\n - Use mirror practice"
    )

# ========== Generate TTS Native Audio ==========
def generate_native_audio():
    librispeech_path = os.path.join(".", "LIBRISPEECH", "test-clean")
    if not os.path.exists(librispeech_path):
        print("⬇️ Downloading LibriSpeech test-clean subset...")
        torchaudio.datasets.LIBRISPEECH(".", url="test-clean", download=True)
    else:
        print("📂 LibriSpeech test-clean already available. Skipping download.")

    dataset = torchaudio.datasets.LIBRISPEECH(".", url="test-clean", download=False)
    waveform, sr, transcript, *_ = dataset[random.randint(0, len(dataset) - 1)]

    # Save audio
    torchaudio.save(NATIVE_AUDIO, waveform, sr)

    # Save transcript to .lab file
    lab_path = os.path.splitext("audio/native")[0] + ".lab"
    with open(lab_path, "w", encoding="utf-8") as f:
        f.write(transcript.lower().strip())
    lab_path = os.path.splitext("audio/user")[0] + ".lab"
    with open(lab_path, "w", encoding="utf-8") as f:
        f.write(transcript.lower().strip())

    print(f"✅ Native audio saved.\n📜 Transcript: {transcript.strip()}")
    return transcript.strip()

def compute_accuracy(phoneme_scores, threshold=60):
    if not phoneme_scores:
        return 0.0
    correct = sum(1 for _, score in phoneme_scores if score >= threshold)
    total = len(phoneme_scores)
    return round((correct / total) * 100, 2)

def load_textgrid_words(textgrid_path):
    tg = tgt.read_textgrid(textgrid_path)
    words = tg.get_tier_by_name("words")
    return [(interval.text, interval.start_time, interval.end_time) for interval in words.intervals if interval.text.strip()]

def compute_word_level_scores(waveform, phones, words, sr=sample_rate):
    emissions = get_emissions(waveform)
    word_scores = []
    for word, w_start, w_end in words:
        word_phonemes = [(p, s, e) for p, s, e in phones if s >= w_start and e <= w_end]
        scores = []
        for phoneme, start, end in word_phonemes:
            start_frame = int(start * sr / 320)
            end_frame = int(end * sr / 320)
            segment_probs = emissions[start_frame:end_frame]
            if len(segment_probs) == 0:
                continue
            top_probs, _ = segment_probs.max(dim=-1)
            avg_log_prob = torch.log(top_probs).mean().item()
            p = (2 * np.exp(-3 * -avg_log_prob)) / (1 + np.exp(-3 * -avg_log_prob))
            scores.append(p * 100)
        if scores:
            word_scores.append((word, round(np.mean(scores), 2)))
    return word_scores

def compute_sentence_score(scores):
    if not scores:
        return 0.0
    values = [s for _, s in scores]
    return round(np.mean(values), 2)


# ========== Final Report ==========
def final_report(alpha=0.5):
    print("\n📊 Generating Pronunciation Report...\n")
    waveform = resample_audio(USER_AUDIO)

    gop_trad = traditional_gop(USER_AUDIO)
    # dtw_trad = traditional_dtw_score(USER_AUDIO, NATIVE_AUDIO)

    gop_impr = context_aware_gop(USER_AUDIO)
    dtw_impr = pitch_dtw_score(USER_AUDIO, NATIVE_AUDIO)
    final = final_score(gop_impr, dtw_impr, alpha)

    phonemes = load_textgrid_phonemes(TEXTGRID_PATH)
    phoneme_scores = phoneme_confidence_from_emissions(waveform, phonemes)
    feedback = feedback_report(phoneme_scores)
    fluency = compute_fluency_dtw(USER_AUDIO, NATIVE_AUDIO)
    fluencyy = pitch_dtw_score(USER_AUDIO, NATIVE_AUDIO)
    fluencyyy = autocorr_dtw_score(USER_AUDIO, NATIVE_AUDIO)
    accuracy = compute_accuracy(phoneme_scores)

    # final_combined_score = final_combined_score(gop_impr, dtw_impr, fluency)
    print(f"💬 Fluency Score DTW (intensity)   : {fluency}% ✅")
    print(f"💬 Fluency Score DTW (pitch delta) : {fluencyy}% ✅")
    print(f"💬 Fluency Score DTW (autocorr DTW): {fluencyyy}% ✅")
    print(f"💬 Accuracy Score                  : {accuracy}% ✅")

    print(f"🧪 Traditional GOP Score           : {gop_trad}%")
    print(f"🧪 Improved GOP Score              : {gop_impr}% ✅")
    # print(f"🎵 Traditional DTW Score      : {dtw_trad}%")
    print(f"🎵 Improved DTW Score              : {dtw_impr}% ✅")
    print(f"🎯 Final Combined Score (α={alpha}): {final}% ✅")
    # print(f"🎯 Final Combined Score with fluency (α={alpha}), (β={beta}), (γ={gamma}): {final_combined_score}% ✅")
    print("\n🔤 Phoneme Scores:")
    for p, s in phoneme_scores:
        print(f" - {p}: {s}% {'✅' if s >= 60 else '⚠️'}")
    print("\n📌 Feedback:")

    words = load_textgrid_words(TEXTGRID_PATH)
    word_scores = compute_word_level_scores(waveform, phonemes, words)
    sentence_score = compute_sentence_score(word_scores)

    print("\n📖 Word Scores:")
    for word, score in word_scores:
        print(f" - {word}: {score}% {'✅' if score >= 60 else '⚠️'}")

    print(f"\n📄 Sentence Score: {sentence_score}% ✅")

    print(feedback)



# ========== MAIN ==========
if __name__ == "__main__":
    print("🗣️ Pronunciation Scoring System")
    generate_native_audio()
    print("\n🔊 Listen to how it should be pronounced...")
    play_audio(NATIVE_AUDIO)
    listen_for_keys()
    run_mfa()
    final_report(alpha=0.5)
    print("✅ Process completed successfully.")