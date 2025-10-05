#! python3.7
"""
Unified Audio Transcription System
Combines real-time transcription, noise reduction, and Raspberry Pi support

Features:
- Real-time microphone transcription (Kayla's approach)
- Efficient noise reduction (Meilin's approach)
- Raspberry Pi GPIO button support (Chloe's approach)
- Audio file transcription with optional playback
- Cross-platform support (Windows/Linux/Raspberry Pi)
"""

import argparse
import os
import sys
import numpy as np
import whisper
import torch
import queue
import soundfile as sf
from datetime import datetime, timedelta
from time import sleep
from threading import Thread
import warnings

# Platform-specific imports
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False
    print("Warning: sounddevice not available. Real-time transcription disabled.")

try:
    import pyaudio
    import wave
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False

try:
    import RPi.GPIO as GPIO
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False

# Noise reduction imports (Meilin's approach)
try:
    import subprocess
    HAS_SOX = subprocess.run(["sox", "--version"], capture_output=True).returncode == 0
except:
    HAS_SOX = False

try:
    from scipy.io import wavfile
    import noisereduce as nr
    from scipy.signal import resample_poly
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False

# Suppress FP16 warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU*")

# ============================================================================
# NOISE REDUCTION MODULE (Meilin's approach)
# ============================================================================

def clean_audio_sox(input_file, output_file, noise_amount=0.21):
    """
    Clean audio using SOX (Meilin's method - very efficient)
    Uses first 0.5 seconds as noise profile
    """
    if not HAS_SOX:
        print("⚠️ SOX not available, skipping noise reduction")
        return input_file
    
    print("🧼 Cleaning background noise with SOX...")
    try:
        # Generate noise profile from first 0.5 seconds
        subprocess.run([
            "sox", input_file, "-n", "trim", "0", "0.5", "noiseprof", "noise.prof"
        ], check=True, capture_output=True)
        
        # Apply noise reduction
        subprocess.run([
            "sox", input_file, output_file, "noisered", "noise.prof", str(noise_amount)
        ], check=True, capture_output=True)
        
        print(f"✅ Cleaned audio saved as: {output_file}")
        return output_file
    except Exception as e:
        print(f"⚠️ SOX cleaning failed: {e}")
        return input_file


def clean_audio_noisereduce(input_file, output_file, prop_decrease=0.9):
    """
    Clean audio using noisereduce library (Chloe's method - no SOX needed)
    Fallback when SOX is not available
    """
    if not HAS_NOISEREDUCE:
        print("⚠️ noisereduce not available, skipping noise reduction")
        return input_file
    
    print("🧼 Cleaning background noise with noisereduce...")
    try:
        # Load audio
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
        
        # Use first half second as noise sample
        noise_len = min(sr // 2, len(audio))
        noise_clip = audio[:noise_len] if noise_len > 0 else audio
        
        # Apply noise reduction
        reduced = nr.reduce_noise(y=audio, y_noise=noise_clip, sr=sr, prop_decrease=prop_decrease)
        
        # Save cleaned audio
        reduced_i16 = np.clip(reduced * 32768.0, -32768, 32767).astype(np.int16)
        wavfile.write(output_file, sr, reduced_i16)
        
        print(f"✅ Cleaned audio saved as: {output_file}")
        return output_file
    except Exception as e:
        print(f"⚠️ Noise reduction failed: {e}")
        return input_file


def clean_audio(input_file, output_file, method='auto'):
    """
    Clean audio with automatic method selection
    Tries SOX first (faster), falls back to noisereduce
    """
    if method == 'sox' or (method == 'auto' and HAS_SOX):
        return clean_audio_sox(input_file, output_file)
    elif method == 'noisereduce' or (method == 'auto' and HAS_NOISEREDUCE):
        return clean_audio_noisereduce(input_file, output_file)
    else:
        print("⚠️ No noise reduction method available")
        return input_file


# ============================================================================
# RECORDING MODULE (Chloe's PyAudio approach for button/simple recording)
# ============================================================================

def record_audio_simple(filename, duration, sample_rate=44100):
    """
    Simple recording using PyAudio (Chloe's approach)
    Good for button-triggered recording on Raspberry Pi
    """
    if not HAS_PYAUDIO:
        print("❌ PyAudio not available")
        return False
    
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    
    p = pyaudio.PyAudio()
    print(f"🎙️ Using input device: {p.get_default_input_device_info().get('name', 'Unknown')}")
    
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    print(f"🎙️ Recording for {duration} seconds...")
    frames = []
    for _ in range(int(sample_rate / CHUNK * duration)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Write WAV
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
    
    print(f"✅ Saved raw audio: {filename}")
    return True


# ============================================================================
# REAL-TIME TRANSCRIPTION MODULE (Kayla's approach)
# ============================================================================

def transcribe_microphone_realtime(args, audio_model, output_file):
    """
    Real-time transcription from microphone (Kayla's approach)
    Displays transcription as you speak with continuous updates
    """
    if not HAS_SOUNDDEVICE:
        print("❌ sounddevice not available for real-time transcription")
        return
    
    # Configuration from args
    phrase_time = None
    data_queue = queue.Queue()
    phrase_audio = np.array([], dtype=np.float32)
    
    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout
    sample_rate = args.sample_rate
    energy_threshold = args.energy_threshold
    
    transcription = ['']
    
    def audio_callback(indata, frames, time, status):
        """Callback to receive audio data from microphone"""
        if status:
            print(status)
        
        # Calculate RMS energy to detect speech
        audio_data = indata.copy().flatten()
        energy = np.sqrt(np.mean(audio_data**2))
        
        # Only add to queue if energy exceeds threshold
        if energy > energy_threshold:
            data_queue.put(audio_data.copy())
    
    # Start the audio stream
    print("Listening... (Press Ctrl+C to stop)\n")
    
    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype=np.float32,
        blocksize=int(sample_rate * record_timeout),
        callback=audio_callback
    )
    
    with stream:
        while True:
            try:
                now = datetime.utcnow()
                
                # Pull raw recorded audio from the queue
                if not data_queue.empty():
                    phrase_complete = False
                    
                    # If enough time has passed, consider the phrase complete
                    if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                        phrase_audio = np.array([], dtype=np.float32)
                        phrase_complete = True
                    
                    phrase_time = now
                    
                    # Combine audio data from queue
                    audio_chunks = []
                    while not data_queue.empty():
                        audio_chunks.append(data_queue.get())
                    
                    if audio_chunks:
                        audio_data = np.concatenate(audio_chunks)
                        phrase_audio = np.concatenate([phrase_audio, audio_data])
                        
                        # Transcribe
                        result = audio_model.transcribe(phrase_audio, fp16=torch.cuda.is_available())
                        text = result['text'].strip()
                        
                        # Update transcription
                        if phrase_complete:
                            transcription.append(text)
                        else:
                            transcription[-1] = text
                        
                        # Display
                        os.system('cls' if os.name=='nt' else 'clear')
                        for line in transcription:
                            print(line)
                        print('', end='', flush=True)
                        
                        # Save to file
                        with open(output_file, 'w', encoding='utf-8') as f:
                            for line in transcription:
                                if line:
                                    f.write(line + '\n')
                else:
                    sleep(0.1)
                    
            except KeyboardInterrupt:
                break
    
    print("\n\nFinal Transcription:")
    for line in transcription:
        print(line)
    
    print(f"\n✅ Transcription saved to: {output_file}")


# ============================================================================
# AUDIO FILE TRANSCRIPTION (Kayla's approach with playback)
# ============================================================================

def play_audio_file(audio_file):
    """Play an audio file in a separate thread"""
    try:
        data, samplerate = sf.read(audio_file)
        sd.play(data, samplerate)
        sd.wait()
    except Exception as e:
        print(f"Error playing audio: {e}")


def transcribe_audio_file(audio_file, audio_model, output_file, play_audio=False):
    """
    Transcribe an audio file with optional playback (Kayla's approach)
    Can display transcription in sync with audio
    """
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found.")
        return
    
    print(f"📝 Transcribing audio file: {audio_file}")
    
    # Start playing audio if requested
    audio_thread = None
    if play_audio and HAS_SOUNDDEVICE:
        print("Playing audio...\n")
        audio_thread = Thread(target=play_audio_file, args=(audio_file,))
        audio_thread.daemon = False
        audio_thread.start()
        sleep(0.5)
    else:
        print("Processing...\n")
    
    # Transcribe with word-level timestamps
    result = audio_model.transcribe(
        audio_file,
        fp16=torch.cuda.is_available(),
        word_timestamps=True,
        verbose=False
    )
    
    segments = result.get('segments', [])
    
    print("Transcription:")
    print("=" * 50)
    
    transcription_lines = []
    if play_audio:
        start_time = datetime.now()
    
    if segments:
        for segment in segments:
            words = segment.get('words', [])
            
            if words:
                line_text = ""
                for word_info in words:
                    word = word_info['word']
                    word_start = word_info.get('start', 0)
                    
                    if play_audio:
                        elapsed = (datetime.now() - start_time).total_seconds()
                        wait_time = word_start - elapsed
                        if wait_time > 0:
                            sleep(wait_time)
                    
                    line_text += word
                    print(word, end='', flush=True)
                print()
                transcription_lines.append(line_text.strip())
            else:
                line = segment['text'].strip()
                print(line)
                transcription_lines.append(line)
    else:
        full_text = result['text'].strip()
        print(full_text)
        transcription_lines = [full_text]
    
    print("=" * 50)
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in transcription_lines:
            if line:
                f.write(line + '\n')
    
    if play_audio and audio_thread:
        print("\nWaiting for audio to finish...")
        audio_thread.join()
    
    print(f"\n✅ Transcription saved to: {output_file}")


# ============================================================================
# RASPBERRY PI BUTTON MODE (Chloe's approach with start/stop toggle)
# ============================================================================

def raspberry_pi_button_mode(args, audio_model):
    """
    Raspberry Pi button-triggered recording mode (Chloe's approach)
    Press button → record → clean → transcribe (original behavior)
    """
    if not HAS_GPIO:
        print("❌ RPi.GPIO not available. Not running on Raspberry Pi?")
        return
    
    if not HAS_PYAUDIO:
        print("❌ PyAudio not available for recording")
        return
    
    BUTTON_PIN = args.gpio_pin
    RECORD_SECONDS = args.record_duration
    
    # Setup GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    
    print(f"🔘 Ready. Press button on GPIO pin {BUTTON_PIN} to record.")
    
    try:
        while True:
            if GPIO.input(BUTTON_PIN) == GPIO.LOW:
                print("🔘 Button pressed!")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                raw_file = os.path.join(args.output_dir, f"raw_{timestamp}.wav")
                clean_file = os.path.join(args.output_dir, f"clean_{timestamp}.wav")
                transcript_file = os.path.join(args.output_dir, f"transcript_{timestamp}.txt")
                
                # Record
                if record_audio_simple(raw_file, RECORD_SECONDS):
                    # Clean
                    cleaned = clean_audio(raw_file, clean_file)
                    
                    # Transcribe
                    print("📝 Transcribing with Whisper...")
                    result = audio_model.transcribe(cleaned, fp16=False)
                    text = result["text"].strip()
                    
                    print("🗣️ Transcript:")
                    print(text)
                    
                    with open(transcript_file, "w", encoding='utf-8') as f:
                        f.write(text)
                    
                    print(f"✅ Transcript saved to: {transcript_file}")
                
                print("✅ Done. Waiting for next press...\n")
                sleep(1)  # debounce
            
            sleep(0.05)
    
    except KeyboardInterrupt:
        print("👋 Exiting...")
    finally:
        GPIO.cleanup()


def raspberry_pi_button_toggle_mode(args, audio_model):
    """
    Raspberry Pi button START/STOP toggle mode
    First press → start listening (realtime transcription)
    Second press → stop listening and save
    
    Falls back to keyboard control ('s' to start/stop) if GPIO unavailable
    """
    use_gpio = HAS_GPIO
    use_keyboard = False
    
    if not HAS_GPIO:
        print("⚠️ RPi.GPIO not available. Falling back to keyboard control.")
        print("   Press 's' to START/STOP listening (instead of button)")
        use_keyboard = True
        
        # Import keyboard detection
        if os.name == 'nt':  # Windows
            import msvcrt
        else:  # Linux/Mac
            import sys
            import tty
            import termios
    
    if not HAS_SOUNDDEVICE:
        print("❌ sounddevice not available for realtime transcription")
        return
    
    BUTTON_PIN = args.gpio_pin
    
    # Setup GPIO if available
    if use_gpio:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        print(f"🔘 Ready. Press button on GPIO pin {BUTTON_PIN} to START/STOP listening.")
    else:
        print("⌨️  Ready. Press 's' key to START/STOP listening.")
    
    print("   First press = start listening, second press = stop and save")
    
    is_recording = False
    stream = None
    phrase_audio = np.array([], dtype=np.float32)
    data_queue = queue.Queue()
    transcription = ['']
    output_file = None
    
    def check_keyboard_press():
        """Check if 's' key was pressed (non-GPIO fallback)"""
        if os.name == 'nt':  # Windows
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                try:
                    return ch.decode('utf-8').lower() == 's'
                except:
                    return False
        else:  # Linux/Mac
            import select
            if select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                return ch.lower() == 's'
        return False
    
    def audio_callback(indata, frames, time, status):
        """Callback to receive audio data from microphone"""
        if status:
            print(status)
        
        # Calculate RMS energy to detect speech
        audio_data = indata.copy().flatten()
        energy = np.sqrt(np.mean(audio_data**2))
        
        # Only add to queue if energy exceeds threshold
        if energy > args.energy_threshold:
            data_queue.put(audio_data.copy())
    
    def start_listening():
        """Start the audio stream and transcription"""
        nonlocal stream, phrase_audio, transcription, output_file
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(args.output_dir, f"transcription_{timestamp}.txt")
        
        # Reset state
        phrase_audio = np.array([], dtype=np.float32)
        transcription = ['']
        
        # Clear queue
        while not data_queue.empty():
            data_queue.get()
        
        # Start stream
        stream = sd.InputStream(
            samplerate=args.sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=int(args.sample_rate * args.record_timeout),
            callback=audio_callback
        )
        stream.start()
        
        if use_gpio:
            print("🎙️ LISTENING... (press button again to stop)")
        else:
            print("🎙️ LISTENING... (press 's' again to stop)")
        print(f"   Saving to: {output_file}\n")
    
    def stop_listening():
        """Stop the audio stream and save transcription"""
        nonlocal stream
        
        if stream:
            stream.stop()
            stream.close()
            stream = None
        
        print("\n🛑 STOPPED listening.")
        print("\nFinal Transcription:")
        print("=" * 50)
        for line in transcription:
            if line:
                print(line)
        print("=" * 50)
        
        # Save final transcription
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in transcription:
                if line:
                    f.write(line + '\n')
        
        print(f"\n✅ Transcription saved to: {output_file}")
        if use_gpio:
            print("🔘 Ready. Press button to start new recording.\n")
        else:
            print("⌨️  Ready. Press 's' to start new recording.\n")
    
    last_button_time = 0
    phrase_time = None
    
    # Setup terminal for non-blocking keyboard input on Linux/Mac
    if use_keyboard and os.name != 'nt':
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
        except:
            pass
    
    try:
        while True:
            toggle_triggered = False
            
            # Check for button press (GPIO) or keyboard press
            if use_gpio:
                if GPIO.input(BUTTON_PIN) == GPIO.LOW:
                    current_time = time.time()
                    if current_time - last_button_time > 0.5:  # 500ms debounce
                        last_button_time = current_time
                        toggle_triggered = True
                        sleep(0.5)  # Give user time to release button
            
            elif use_keyboard:
                if check_keyboard_press():
                    current_time = time.time()
                    if current_time - last_button_time > 0.5:  # 500ms debounce
                        last_button_time = current_time
                        toggle_triggered = True
                        sleep(0.3)  # Brief delay
            
            # Handle toggle
            if toggle_triggered:
                if not is_recording:
                    start_listening()
                    is_recording = True
                else:
                    stop_listening()
                    is_recording = False
                    phrase_time = None
            
            # If recording, process audio queue
            if is_recording and not data_queue.empty():
                now = datetime.utcnow()
                phrase_complete = False
                
                # If enough time has passed, consider the phrase complete
                if phrase_time and now - phrase_time > timedelta(seconds=args.phrase_timeout):
                    phrase_audio = np.array([], dtype=np.float32)
                    phrase_complete = True
                
                phrase_time = now
                
                # Combine audio data from queue
                audio_chunks = []
                while not data_queue.empty():
                    audio_chunks.append(data_queue.get())
                
                if audio_chunks:
                    audio_data = np.concatenate(audio_chunks)
                    phrase_audio = np.concatenate([phrase_audio, audio_data])
                    
                    # Transcribe
                    result = audio_model.transcribe(phrase_audio, fp16=torch.cuda.is_available())
                    text = result['text'].strip()
                    
                    # Update transcription
                    if phrase_complete:
                        transcription.append(text)
                    else:
                        transcription[-1] = text
                    
                    # Display (no clear screen on RPi, just append)
                    print(f"\r{transcription[-1]}", end='', flush=True)
                    
                    # Save to file incrementally
                    with open(output_file, 'w', encoding='utf-8') as f:
                        for line in transcription:
                            if line:
                                f.write(line + '\n')
            
            sleep(0.05)
    
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
        if stream:
            stream.stop()
            stream.close()
    finally:
        if use_gpio:
            GPIO.cleanup()
        if use_keyboard and os.name != 'nt':
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except:
                pass


# ============================================================================
# SIMPLE RECORDING MODE (Non-realtime, with cleaning)
# ============================================================================

def simple_record_and_transcribe(args, audio_model):
    """
    Simple record → clean → transcribe workflow (Meilin/Chloe style)
    Good for when you don't need real-time display
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    raw_file = os.path.join(args.output_dir, f"raw_{timestamp}.wav")
    clean_file = os.path.join(args.output_dir, f"clean_{timestamp}.wav")
    transcript_file = os.path.join(args.output_dir, f"transcript_{timestamp}.txt")
    
    # Record
    print(f"Recording for {args.record_duration} seconds...")
    if not record_audio_simple(raw_file, args.record_duration):
        return
    
    # Clean
    cleaned = clean_audio(raw_file, clean_file)
    
    # Transcribe
    print("📝 Transcribing with Whisper...")
    result = audio_model.transcribe(cleaned, fp16=False)
    text = result["text"].strip()
    
    print("🗣️ Transcript:")
    print(text if text else "[No speech detected]")
    
    with open(transcript_file, "w", encoding='utf-8') as f:
        f.write(text)
    
    print(f"✅ Transcript saved to: {transcript_file}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="""
╔══════════════════════════════════════════════════════════════════╗
║   Unified Audio Transcription System                            ║
║   Combines real-time transcription, noise reduction,             ║
║   and Raspberry Pi support in one modular script                 ║
╚══════════════════════════════════════════════════════════════════╝

MODES:
  realtime  - Live microphone transcription with streaming display
  file      - Transcribe existing audio files (with optional playback)
  simple    - Record audio, clean it, then transcribe (no realtime)
  button    - Raspberry Pi GPIO button-triggered recording (press → record N seconds)
  button_toggle - Raspberry Pi GPIO start/stop toggle (press → start, press → stop)

EXAMPLES:
  # Real-time transcription as you speak
  python nightingale.py --mode realtime --model base

  # Transcribe a file with noise cleaning
  python nightingale.py --mode file --audio_file lecture.wav

  # Simple recording (5 seconds, then transcribe)
  python nightingale.py --mode simple --record_duration 5

  # Raspberry Pi button mode (each press records 5 seconds)
  python nightingale.py --mode button --gpio_pin 17

  # Raspberry Pi toggle mode (first press starts, second press stops)
  python nightingale.py --mode button_toggle --gpio_pin 17
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
TIPS:
  - Use --model tiny for fastest processing on low-power devices
  - Use --model large for highest accuracy (requires more RAM/CPU)
  - Lower --energy_threshold if speech isn't being detected
  - Use --no_clean to skip noise reduction for already-clean audio
  - SOX cleaning is faster than noisereduce but requires installation

For detailed documentation, see design_README.md
        """
    )
    
    # Mode selection
    parser.add_argument("--mode", default="realtime",
                        choices=["realtime", "file", "simple", "button", "button_toggle"],
                        metavar="MODE",
                        help="Transcription mode (default: realtime)\n"
                             "  realtime = Live mic with streaming display\n"
                             "  file = Transcribe existing audio file\n"
                             "  simple = Record then transcribe (no realtime)\n"
                             "  button = RPi GPIO trigger (press=record N sec)\n"
                             "  button_toggle = RPi GPIO (press=start, press=stop)")
    
    # Model settings
    parser.add_argument("--model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        metavar="SIZE",
                        help="Whisper model size (default: base)\n"
                             "  tiny = Fastest, least accurate (~1GB RAM)\n"
                             "  base = Good balance (~1GB RAM)\n"
                             "  small = Better accuracy (~2GB RAM)\n"
                             "  medium = High accuracy (~5GB RAM)\n"
                             "  large = Best accuracy (~10GB RAM)")
    parser.add_argument("--non_english", action='store_true',
                        help="Use multilingual model instead of English-only\n"
                             "(English-only models are faster for English speech)")
    
    # Audio file settings
    parser.add_argument("--audio_file", type=str,
                        metavar="PATH",
                        help="Path to audio file (required for file mode)\n"
                             "Supports: .wav, .mp3, .m4a, .flac, etc.")
    parser.add_argument("--play_audio", action='store_true',
                        help="Play audio file while transcribing (file mode)\n"
                             "Shows words synced to playback timing")
    
    # Real-time settings (Kayla's parameters)
    parser.add_argument("--energy_threshold", default=0.01, type=float,
                        metavar="FLOAT",
                        help="Energy level for mic to detect speech (default: 0.01)\n"
                             "Lower = more sensitive (picks up quieter speech)\n"
                             "Higher = less sensitive (ignores background noise)\n"
                             "Try 0.005 for quiet environments, 0.02 for noisy")
    parser.add_argument("--record_timeout", default=2, type=float,
                        metavar="SECONDS",
                        help="Recording chunk size in seconds (default: 2)\n"
                             "How often to process audio in realtime mode\n"
                             "Lower = more responsive, higher CPU usage")
    parser.add_argument("--phrase_timeout", default=3, type=float,
                        metavar="SECONDS",
                        help="Silence duration to consider phrase complete (default: 3)\n"
                             "After this many seconds of silence, starts new line\n"
                             "Lower = more line breaks, higher = fewer breaks")
    parser.add_argument("--sample_rate", default=16000, type=int,
                        metavar="HZ",
                        help="Sample rate for audio recording (default: 16000)\n"
                             "Whisper uses 16kHz internally, higher may help quality")
    
    # Simple/button recording settings
    parser.add_argument("--record_duration", default=5, type=int,
                        metavar="SECONDS",
                        help="Recording duration in seconds (default: 5)\n"
                             "Used in simple and button modes\n"
                             "How long to record when triggered")
    
    # Raspberry Pi settings (Chloe's parameters)
    parser.add_argument("--gpio_pin", default=17, type=int,
                        metavar="PIN",
                        help="GPIO pin number for button (default: 17)\n"
                             "Uses BCM numbering (not physical pin numbers)\n"
                             "Connect button between this pin and ground")
    
    # Noise reduction
    parser.add_argument("--no_clean", action='store_true',
                        help="Skip noise reduction/cleaning step\n"
                             "Use this if audio is already clean or to save time")
    parser.add_argument("--clean_method", default="auto",
                        choices=["auto", "sox", "noisereduce"],
                        metavar="METHOD",
                        help="Noise reduction method (default: auto)\n"
                             "  auto = Try SOX first, fallback to noisereduce\n"
                             "  sox = Use SOX (faster, requires installation)\n"
                             "  noisereduce = Pure Python (slower, no deps)")
    
    # Output settings
    parser.add_argument("--output_dir", default="transcriptions",
                        metavar="DIR",
                        help="Directory to save transcription files (default: transcriptions)\n"
                             "Creates timestamped .txt files for each transcription")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Whisper model
    model_name = args.model
    if args.model != "large" and not args.non_english:
        model_name = model_name + ".en"
    
    print(f"Loading Whisper model: {model_name}")
    audio_model = whisper.load_model(model_name)
    print("Model loaded.\n")
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.audio_file and args.mode == "file":
        base_name = os.path.splitext(os.path.basename(args.audio_file))[0]
        output_file = os.path.join(args.output_dir, f"transcription_{base_name}_{timestamp}.txt")
    else:
        output_file = os.path.join(args.output_dir, f"transcription_{timestamp}.txt")
    
    # Route to appropriate mode
    if args.mode == "realtime":
        print("🎙️ Real-time Microphone Mode")
        print(f"Transcription will be saved to: {output_file}\n")
        transcribe_microphone_realtime(args, audio_model, output_file)
    
    elif args.mode == "file":
        if not args.audio_file:
            print("❌ --audio_file required for file mode")
            sys.exit(1)
        
        # Optionally clean the file first
        if not args.no_clean:
            clean_file = os.path.join(args.output_dir, f"clean_{timestamp}.wav")
            audio_to_transcribe = clean_audio(args.audio_file, clean_file, args.clean_method)
        else:
            audio_to_transcribe = args.audio_file
        
        transcribe_audio_file(audio_to_transcribe, audio_model, output_file, args.play_audio)
    
    elif args.mode == "simple":
        print("🎙️ Simple Record & Transcribe Mode")
        simple_record_and_transcribe(args, audio_model)
    
    elif args.mode == "button":
        print("🔘 Raspberry Pi Button Mode (Press to Record)")
        raspberry_pi_button_mode(args, audio_model)
    
    elif args.mode == "button_toggle":
        print("🔘 Raspberry Pi Button Toggle Mode (Press to Start/Stop)")
        raspberry_pi_button_toggle_mode(args, audio_model)


if __name__ == "__main__":
    main()