# October 4, 2025 (Windows-friendly, cleaned version)
import os
import sys
import time
import wave
import tempfile
import subprocess
import datetime
import argparse
import msvcrt  # Windows keypress
import numpy as np
from scipy.signal import resample_poly
import pyaudio
import soundfile as sf
from scipy.io import wavfile
import noisereduce as nr
import whisper


# === CONFIG ===
RECORD_SECONDS = 5           # default capture length when pressing 'R'
SAMPLE_RATE = 44100
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024
DEFAULT_MODEL = "tiny"       # "tiny" for speed on low-power devices

TARGET_SR = 16000

def _ffmpeg_bin() -> str:
    """
    Returns the ffmpeg executable to use.
    - If you set env var FFMPEG_BIN, we use that.
    - Otherwise, we assume 'ffmpeg' is on PATH.
    """
    return os.environ.get("FFMPEG_BIN", "ffmpeg")


def convert_to_wav_ffmpeg(src_path: str, target_sr: int = TARGET_SR, mono: bool = True) -> str:
    """
    Convert any audio (e.g., .m4a, .mp3, .aac) to a temp WAV using ffmpeg.
    Returns path to the temp WAV.
    """
    tmp_fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)  # we'll let ffmpeg write to it

    cmd = [
        _ffmpeg_bin(),
        "-y",               # overwrite temp file if exists
        "-i", src_path,     # input
    ]
    if mono:
        cmd += ["-ac", "1"]
    if target_sr:
        cmd += ["-ar", str(target_sr)]

    cmd += ["-vn", tmp_wav]  # no video

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return tmp_wav
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Either add it to PATH or set FFMPEG_BIN to the full path of ffmpeg.exe."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed to convert '{src_path}': {e.stderr.decode(errors='ignore')}")

def load_audio_np_any(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """
    Load audio into a float32 mono NumPy array at target_sr.
    Strategy:
      1) Try reading with soundfile.
      2) If that fails (common for .m4a), convert with ffmpeg to WAV, then read.
    """
    # First try direct load via soundfile
    try:
        audio, sr = sf.read(path, always_2d=False)
    except Exception:
        # Fallback: convert to WAV with ffmpeg then read
        tmp_wav = convert_to_wav_ffmpeg(path, target_sr=target_sr, mono=True)
        try:
            audio, sr = sf.read(tmp_wav, always_2d=False)
        finally:
            try:
                os.remove(tmp_wav)
            except Exception:
                pass

    # Mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Resample if needed (in case soundfile read wasn’t at target_sr)
    if sr != target_sr:
        audio = resample_poly(audio, target_sr, sr)

    return np.ascontiguousarray(audio)


def load_audio_np(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """Load audio without ffmpeg: read with soundfile, mono, float32, resample to 16 kHz."""
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if sr != target_sr:
        audio = resample_poly(audio, target_sr, sr)
    return np.ascontiguousarray(audio)

def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def record_audio(filename: str, duration: int):
    """Record from the default input device into a WAV file."""
    p = pyaudio.PyAudio()
    print(f"🎙️ Using input device: {p.get_default_input_device_info().get('name', 'Unknown')}")
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print(f"🎙️ Recording for {duration} seconds...")
    frames = []
    for _ in range(int(SAMPLE_RATE / CHUNK * duration)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Write WAV
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))

    print(f"✅ Saved raw audio: {filename}")


def clean_audio(input_file: str, output_file: str):
    """Reduce background noise and save as a clean WAV."""
    print("🧼 Cleaning background noise...")

    # Try loading with scipy; fallback to soundfile
    try:
        sr, data = wavfile.read(input_file)
        if data.dtype != np.int16:
            if np.issubdtype(data.dtype, np.floating):
                data = np.clip(data, -1.0, 1.0)
                data = (data * 32767).astype(np.int16)
            else:
                data = data.astype(np.int16)
        audio = data.astype(np.float32) / 32768.0
    except Exception:
        audio, sr = sf.read(input_file, always_2d=False)
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = audio.astype(np.float32)

    # Noise reduction
    noise_len = min(sr // 2, len(audio))
    noise_clip = audio[:noise_len] if noise_len > 0 else audio
    reduced = nr.reduce_noise(y=audio, y_noise=noise_clip, sr=sr, prop_decrease=0.9)

    reduced_i16 = np.clip(reduced * 32768.0, -32768, 32767).astype(np.int16)
    wavfile.write(output_file, sr, reduced_i16)
    print(f"✅ Cleaned audio saved as: {output_file}")


def transcribe_audio(audio_file: str, output_text_file: str, model_size: str = DEFAULT_MODEL):
    """Transcribe without ffmpeg by passing a NumPy waveform directly."""
    print(f"📝 Transcribing with Whisper ({model_size})...")
    model = whisper.load_model(model_size)

    # Load/resample ourselves so Whisper doesn't call ffmpeg
    audio_np = load_audio_np_any(audio_file, TARGET_SR)

    # On CPU, ensure fp16=False
    result = model.transcribe(audio_np, fp16=False)
    text = result.get("text", "").strip()

    print("🗒️ Transcript:")
    print(text if text else "[No speech detected]")

    with open(output_text_file, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"✅ Transcript saved to: {output_text_file}")


def process_existing(input_path: str, model_size: str):
    """Process an existing audio file: clean → transcribe."""
    base = os.path.splitext(os.path.basename(input_path))[0]
    t = timestamp()
    clean_file = f"clean_{base}_{t}.wav"
    transcript_file = f"transcript_{base}_{t}.txt"

    clean_audio(input_path, clean_file)
    transcribe_audio(clean_file, transcript_file, model_size)


def interactive_loop(record_seconds: int, model_size: str):
    """Interactive mode for recording and transcribing with key press."""
    print("📦 Ready. Press 'R' to record, 'Q' to quit.")
    print("   Using default input (built-in microphone).")

    while True:
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            try:
                key = ch.decode("utf-8").lower()
            except Exception:
                key = ''

            if key == 'r':
                t = timestamp()
                raw_file = f"raw_{t}.wav"
                clean_file = f"clean_{t}.wav"
                transcript_file = f"transcript_{t}.txt"

                record_audio(raw_file, record_seconds)
                clean_audio(raw_file, clean_file)
                transcribe_audio(clean_file, transcript_file, model_size)
                print("✅ Done. Press 'R' to record again, or 'Q' to quit.\n")

            elif key == 'q':
                print("👋 Exiting...")
                break

        time.sleep(0.02)


def main():
    parser = argparse.ArgumentParser(description="Record or process audio on Windows, clean it, and transcribe with Whisper.")
    parser.add_argument("--input", type=str, help="Path to an existing audio file instead of recording.")
    parser.add_argument("--seconds", type=int, default=RECORD_SECONDS, help="Record duration when pressing 'R'. Default: 5")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Whisper model size (tiny, base, small, medium, large). Default: tiny")

    args = parser.parse_args()

    if args.input:
        if not os.path.exists(args.input):
            print(f"❌ Input file not found: {args.input}")
            sys.exit(1)
        process_existing(args.input, args.model)
    else:
        interactive_loop(args.seconds, args.model)


if __name__ == "__main__":
    main()
