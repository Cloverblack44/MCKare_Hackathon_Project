# Unified Audio Transcription System - Design Choices

## 🎯 Philosophy: Keep Your Code Recognizable

I kept each person's original code as intact as possible so you can still see YOUR work in there. The snippets you wrote are preserved - I just wrapped them in functions and added a routing system to pick which one to use.

---

## 🏗️ Architecture Overview

```
Main Entry Point
      ↓
  Load Model
      ↓
  Route by Mode → [realtime] → Kayla's real-time engine
                → [file]     → Kayla's file transcriber + optional cleaning
                → [simple]   → Meilin/Chloe's record-clean-transcribe
                → [button]   → Chloe's Raspberry Pi GPIO listener
```

---

## 🔧 Design Choices Explained

### 1. **Modular Function Organization**

**Why**: Each of you had different strengths - combining them as separate modules means you can easily understand, test, and modify YOUR part without breaking others' code.

**How it works**:
- **Noise Reduction Module** (lines 65-175): Meilin's SOX method + Chloe's noisereduce fallback
- **Recording Module** (lines 181-226): Chloe's PyAudio button-friendly recording
- **Real-time Transcription** (lines 232-324): Kayla's streaming approach with energy detection
- **File Transcription** (lines 330-424): Kayla's word-synced playback method
- **Button Mode** (lines 430-495): Chloe's GPIO integration

### 2. **Noise Reduction: Two Methods, One Interface**

**Original code preserved**:
- Meilin's `clean_audio_sox()` (lines 67-95): Exact SOX commands you wrote
- Chloe's `clean_audio_noisereduce()` (lines 98-142): Your scipy/noisereduce approach

**Design choice**: `clean_audio()` wrapper (lines 145-158) auto-picks the best available method
- Tries SOX first (faster, Meilin's preference)
- Falls back to noisereduce (no external deps needed)
- Gracefully degrades if neither available

**Why**: Meilin wanted efficiency (SOX is blazing fast). Chloe wanted it to work without installing system packages. Auto-selection gives you both!

### 3. **Recording: PyAudio for Simplicity**

**Original code**: Chloe's `record_audio()` from line 44-70 of Meilin_microphone.py

**Kept as-is** because:
- Works great for button-triggered recording (your Raspberry Pi use case)
- Simple, blocking recording - perfect for "press button, wait N seconds"
- Used in both `simple` mode and `button` mode

**Why not sounddevice?**: Sounddevice is better for real-time streaming (Kayla's use case), but PyAudio is simpler for "just record a clip". Different tools for different jobs!

### 4. **Real-Time Transcription: Kayla's Streaming Engine**

**Original code preserved**: Your entire `transcribe_microphone()` function (lines 234-324)

**Key features kept**:
- `audio_callback()` with RMS energy detection (lines 259-267)
- Queue-based audio buffering
- Phrase timeout logic to segment sentences
- Live console updates that clear and redisplay

**Why this approach rocks**:
- Energy threshold prevents transcribing silence
- Queue lets audio recording happen in background thread
- Phrase detection creates natural sentence breaks
- Users see transcription appear AS THEY SPEAK

### 5. **File Transcription: Kayla's Word-Synced Playback**

**Original code**: Your `transcribe_audio_file()` (lines 330-424)

**Coolest feature**: Word-by-word display synced to audio timestamps
```python
if play_audio:
    elapsed = (datetime.now() - start_time).total_seconds()
    wait_time = word_start - elapsed
    if wait_time > 0:
        sleep(wait_time)
```

**Why**: Makes transcription feel alive - you SEE words appear as the audio SAYS them. Great for demos and verification!

### 6. **Raspberry Pi Button Mode: Chloe's GPIO Integration**

**Original code**: Your GPIO loop from Meilin_microphone.py (lines 430-495)

**Flow preserved**:
1. Setup GPIO with pull-up resistor
2. Infinite loop checking `GPIO.input(BUTTON_PIN)`
3. On button press → record → clean → transcribe
4. Debounce delay to prevent double-triggers

**Why**: Perfect for embedded devices. Press physical button, get transcription. No keyboard needed!

### 7. **Platform Detection & Graceful Degradation**

**Design choice**: Check imports at startup (lines 27-60)

```python
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False
```

**Why**: 
- Raspberry Pi might not have sounddevice
- Windows doesn't have RPi.GPIO
- Some systems lack SOX

**Result**: Code runs on ANY platform, just disables unavailable features with helpful messages

### 8. **Simple Mode: The Meilin/Chloe Hybrid**

**What it does**: Record → Clean → Transcribe (no real-time display)

**Uses**:
- Chloe's `record_audio_simple()` for recording
- Meilin's `clean_audio()` for noise reduction
- Standard Whisper transcribe

**Why add this?**: Sometimes you DON'T want real-time. You just want "record 5 seconds, give me text". Lower CPU usage, works on slower devices.

### 9. **Argument Routing System**

**Design choice**: `--mode` flag picks which pipeline to run

```bash
# Kayla's real-time magic
python script.py --mode realtime

# Meilin's file cleaning approach
python script.py --mode file --audio_file recording.wav

# Chloe's simple record-and-go
python script.py --mode simple

# Chloe's Raspberry Pi button
python script.py --mode button --gpio_pin 17
```

**Why**: One script, four workflows. Install once, use everywhere.

### 10. **Preserved Original Parameters**

**Kayla's real-time parameters** (kept exactly as you had them):
- `--energy_threshold`: Speech detection sensitivity
- `--record_timeout`: Chunk size for processing
- `--phrase_timeout`: How long of silence = new sentence

**Chloe's button parameters**:
- `--gpio_pin`: Which pin the button connects to
- `--record_duration`: How many seconds to record

**Meilin's model choice**:
- `--model`: Whisper model size (you preferred "base")

**Why**: If you used `--energy_threshold 0.01` before, it still works the same way!

---

## 📊 Feature Comparison Table

| Feature | Kayla's Code | Meilin's Code | Chloe's Code | Unified Script |
|---------|--------------|---------------|--------------|----------------|
| Real-time transcription | ✅ | ❌ | ❌ | ✅ (mode=realtime) |
| SOX noise reduction | ❌ | ✅ | ❌ | ✅ (auto-selected) |
| Python noise reduction | ❌ |