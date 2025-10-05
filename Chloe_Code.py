# Raspberry Pi friendly version
# - Keypress on Linux (no msvcrt)
# - Optional GPIO button (pass --button-pin N)
# - M4A/MP3 accepted via ffmpeg convert fallback
# - Whisper transcription via NumPy array (no ffmpeg call inside whisper)

import os
import sys
import time
import wave
import tempfile
import subprocess
import datetime
import argparse
import select
import tty
import termios

import numpy as np
from scipy.signal import resample_poly

import pyaudio
import soundfile as sf
from scipy.io import wavfile
import noisereduce as nr
import whisper

# === Optional GPIO import (only if used) ===
try:
    import RPi.GPIO as GPIO
    _HAS_GPIO = True
except Exception:
    _HAS_GPIO = False

# === CONFIG ===
RECORD_SECONDS = 5
SAMPLE_RATE = 44100
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024
DEFAULT_MODEL = "tiny"       # tiny is best for speed on Pi
TARGET_SR = 16000            # Whisper input sample rate

def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------- Input helpers ----------

def _ffmpeg_bin() -> str:
    return os.environ.get("FFMPEG_BIN", "ffmpeg")

def convert_to_wav_ffmpeg(src_path: str, target_sr: int = TARGET_SR, mono: bool = True) -> str:
    """Convert any audio (e.g., .m4a/.mp3) to a temp WAV using ffmpeg; returns path to temp WAV."""
    tmp_fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)

    cmd = [_ffmpeg_bin(), "-y", "-i", src_path]
    if mono:
        cmd += ["-ac", "1"]
    if target_sr:
        cmd += ["-ar", str(target_sr)]
    cmd += ["-vn", tmp_wav]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return tmp_wav
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Install it with: sudo apt install ffmpeg")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed to convert '{src_path}': {e.stderr.decode(errors='ignore')}")

def load_audio_np_any(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """
    Load audio to float32 mono at target_sr.
    Try soundfile first; if it fails (e.g., m4a), convert via ffmpeg and retry.
    """
    try:
        audio, sr = sf.read(path, always_2d=False)
    except Exception:
        tmp_wav = convert_to_wav_ffmpeg(path, target_sr=target_sr, mono=True)
        try:
            audio, sr = sf.read(tmp_wav, always_2d=False)
        finally:
            try:
                os.remove(tmp_wav)
            except Exception:
                pass

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if sr != target_sr:
        audio = resample_poly(audio, target_sr, sr)

    return np.ascontiguousarray(audio)

# ---------- Recording / cleaning ----------

def record_audio(filename: str, duration: int):
    """Record from default ALSA input into WAV."""
    p = pyaudio.PyAudio()
    try:
        dev = p.get_default_input_device_info()
        print(f"🎙️ Input device: {dev.get('name', 'Unknown')}")
    except Exception:
        print("🎙️ Input device: (unknown)")

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE,
                    input=True, frames_per_buffer=CHUNK)

    print(f"🎙️ Recording for {duration} s...")
    frames = []
    for _ in range(int(SAMPLE_RATE / CHUNK * duration)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))

    print(f"✅ Saved raw audio: {filename}")

def clean_audio(input_file: str, output_file: str):
    """Reduce background noise and save as clean WAV."""
    print("🧼 Cleaning background noise...")
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

    # Estimate noise from first 0.5s (cap at clip length)
    noise_len = min(sr // 2, len(audio))
    noise_clip = audio[:noise_len] if noise_len > 0 else audio
    reduced = nr.reduce_noise(y=audio, y_noise=noise_clip, sr=sr, prop_decrease=0.9)

    reduced_i16 = np.clip(reduced * 32768.0, -32768, 32767).astype(np.int16)
    wavfile.write(output_file, sr, reduced_i16)
    print(f"✅ Cleaned audio saved as: {output_file}")

# ---------- Transcription ----------

def transcribe_audio(audio_file: str, output_text_file: str, model_size: str = DEFAULT_MODEL):
    """Transcribe by passing a NumPy waveform (no ffmpeg inside whisper)."""
    print(f"📝 Transcribing with Whisper ({model_size})...")
    model = whisper.load_model(model_size)
    audio_np = load_audio_np_any(audio_file, TARGET_SR)

    result = model.transcribe(audio_np, fp16=False)
    text = result.get("text", "").strip()

    print("🗒️ Transcript:")
    print(text if text else "[No speech detected]")

    with open(output_text_file, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"✅ Transcript saved to: {output_text_file}")

# ---------- Modes ----------

def process_existing(input_path: str, model_size: str):
    base = os.path.splitext(os.path.basename(input_path))[0]
    t = timestamp()
    clean_file = f"clean_{base}_{t}.wav"
    transcript_file = f"transcript_{base}_{t}.txt"

    clean_audio(input_path, clean_file)
    transcribe_audio(clean_file, transcript_file, model_size)

def _stdin_key_available() -> bool:
    """Non-blocking check for keypress on Linux."""
    return select.select([sys.stdin], [], [], 0)[0] != []

def _read_single_key_nonblocking() -> str:
    """Return a single lowercased key if available; else ''."""
    if not _stdin_key_available():
        return ''
    ch = sys.stdin.read(1)
    return ch.lower() if ch else ''

def interactive_loop_keypress(record_seconds: int, model_size: str):
    """Interactive mode (keypress): 'r' to record, 'q' to quit."""
    print("📦 Ready. Press 'r' to record, 'q' to quit.")
    print("   Using default ALSA input (USB/built-in mic).")

    # Put stdin into cbreak mode so keypresses are immediate
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            key = _read_single_key_nonblocking()
            if key == 'r':
                t = timestamp()
                raw_file = f"raw_{t}.wav"
                clean_file = f"clean_{t}.wav"
                transcript_file = f"transcript_{t}.txt"

                record_audio(raw_file, record_seconds)
                clean_audio(raw_file, clean_file)
                transcribe_audio(clean_file, transcript_file, model_size)
                print("✅ Done. Press 'r' to record again, or 'q' to quit.\n")

            elif key == 'q':
                print("👋 Exiting...")
                break

            time.sleep(0.02)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def interactive_loop_gpio(record_seconds: int, model_size: str, button_pin: int):
    """Interactive mode (GPIO button): press to record."""
    if not _HAS_GPIO:
        print("❌ RPi.GPIO not installed. Install with: pip install RPi.GPIO")
        sys.exit(1)

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    print(f"📦 Ready. Press the GPIO button on pin {button_pin} to record. Ctrl+C to quit.")

    try:
        while True:
            if GPIO.input(button_pin) == GPIO.LOW:  # active low button
                t = timestamp()
                raw_file = f"raw_{t}.wav"
                clean_file = f"clean_{t}.wav"
                transcript_file = f"transcript_{t}.txt"

                print("🔘 Button pressed!")
                record_audio(raw_file, record_seconds)
                clean_audio(raw_file, clean_file)
                transcribe_audio(clean_file, transcript_file, model_size)
                print("✅ Done. Waiting for next press...\n")
                time.sleep(1.0)  # debounce

            time.sleep(0.02)
    except KeyboardInterrupt:
        print("👋 Exiting...")
    finally:
        GPIO.cleanup()

# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Record/process audio on Raspberry Pi, clean it, and transcribe with Whisper.")
    parser.add_argument("--input", type=str, help="Path to an existing audio file instead of recording.")
    parser.add_argument("--seconds", type=int, default=RECORD_SECONDS, help="Record duration when triggered. Default: 5")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Whisper model size (tiny, base, small, medium, large). Default: tiny")
    parser.add_argument("--button-pin", type=int, default=None,
                        help="BCM pin number for a GPIO button. If omitted, uses keyboard ('r' to record).")

    args = parser.parse_args()

    if args.input:
        if not os.path.exists(args.input):
            print(f"❌ Input file not found: {args.input}")
            sys.exit(1)
        process_existing(args.input, args.model)
        return

    if args.button_pin is not None:
        interactive_loop_gpio(args.seconds, args.model, args.button_pin)
    else:
        # Ensure stdin is a TTY for keypress mode
        if not sys.stdin.isatty():
            print("⚠️ stdin is not a TTY; falling back to simple prompt mode.")
            while True:
                resp = input("Press Enter to record (or type q to quit): ").strip().lower()
                if resp == 'q':
                    break
                t = timestamp()
                raw_file = f"raw_{t}.wav"
                clean_file = f"clean_{t}.wav"
                transcript_file = f"transcript_{t}.txt"
                record_audio(raw_file, args.seconds)
                clean_audio(raw_file, clean_file)
                transcribe_audio(clean_file, transcript_file, args.model)
        else:
            interactive_loop_keypress(args.seconds, args.model)

if __name__ == "__main__":
    main()
