_# Note: Make sure you have installed the following dependencies in the terminal:
"""
sudo apt update
sudo apt install sox ffmpeg -y
pip3 install git+https://github.com/openai/whisper.git
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip3 install flask

"""
import subprocess
import whisper
import os
import sys

# ========== Get rid of warning from Whispr ==========
import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU*")

# ========== Configuration ==========
input_file = sys.argv[1] if len(sys.argv) > 1 else "input.wav"
clean_file = "clean.wav"
transcript_file = "transcript.txt"

# ========== Step 1: Check File ==========
if not os.path.exists(input_file):
    print(f"❌ Error: File '{input_file}' not found.")
    sys.exit(1)

# ========== Step 2: Generate Noise Profile ==========
print("🔍 Generating noise profile from first 0.5 seconds...")
subprocess.run([
    "sox", input_file, "-n", "trim", "0", "0.5", "noiseprof", "noise.prof"
], check=True)

# ========== Step 3: Apply Noise Reduction ==========
print("🧼 Applying noise reduction...")
subprocess.run([
    "sox", input_file, clean_file, "noisered", "noise.prof", "0.21"
], check=True)

# ========== Step 4: Transcribe with Whisper ==========
print("📝 Transcribing cleaned audio...")
model = whisper.load_model("base")  # Can also use 'base', 'small', etc.
result = model.transcribe(clean_file)
text = result["text"]

# ========== Step 5: Save Transcript ==========
with open(transcript_file, "w") as f:
    f.write(text)

print(f"✅ Transcription complete! Saved to: {transcript_file}")

"""
How to run it:
python3 no_mic.py my_audio.wav

"""
