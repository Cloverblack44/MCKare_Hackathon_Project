# October 4, 2025

# When you press the button:

# 1. Records audio (raw.wav)
# 2. Cleans audio to remove background noise (clean.wav)
# 3. Whispr AI transcribes (transcript.txt)

# You'll also see the transcript printed in the terminal.

import RPi.GPIO as GPIO
import time
import pyaudio
import wave
import subprocess
import whisper
import datetime

# === CONFIG ===
BUTTON_PIN = 17              # GPIO pin connected to button
RECORD_SECONDS = 5           # Duration to record
RAW_FILE = "raw.wav"
CLEAN_FILE = "clean.wav"
TRANSCRIPT_FILE = "transcript.txt"

# === SETUP ===
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# === AUDIO RECORDING ===
def record_audio(filename, duration):
    RATE = 44100
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("🎙️ Recording for", duration, "seconds...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print("✅ Saved raw audio:", filename)

# === NOISE REDUCTION ===
def clean_audio(input_file, output_file):
    print("🧼 Cleaning background noise...")
    subprocess.run([
        "sox", input_file, "-n", "trim", "0", "0.5", "noiseprof", "noise.prof"
    ])
    subprocess.run([
        "sox", input_file, output_file, "noisered", "noise.prof", "0.21"
    ])
    print("✅ Cleaned audio saved as:", output_file)

# === WHISPER TRANSCRIPTION ===
def transcribe_audio(audio_file, output_text_file):
    print("📝 Transcribing with Whisper...")
    model = whisper.load_model("tiny")  # Use "tiny" for speed on Raspberry Pi
    result = model.transcribe(audio_file)
    text = result["text"]

    print("🗒️ Transcript:")
    print(text)

    with open(output_text_file, "w") as f:
        f.write(text)

    print("✅ Transcript saved to:", output_text_file)

# === MAIN LOOP ===
print("📦 Ready. Press the button to record.")

try:
    while True:
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:
            print("🔘 Button pressed!")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_file = f"raw_{timestamp}.wav"
            clean_file = f"clean_{timestamp}.wav"
            transcript_file = f"transcript_{timestamp}.txt"

            record_audio(raw_file, RECORD_SECONDS)
            clean_audio(raw_file, clean_file)
            transcribe_audio(clean_file, transcript_file)

            print("✅ Done. Waiting for next press...\n")
            time.sleep(1)  # debounce
        time.sleep(0.05)

except KeyboardInterrupt:
    print("👋 Exiting...")

finally:
    GPIO.cleanup()

# === Using Whispr tiny model for Raspberry Pi ===
model = whisper.load_model("tiny")
result = model.transcribe("clean.wav")
print(result["text"])