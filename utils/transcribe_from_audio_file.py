


import argparse
import os
import numpy as np
import whisper
import torch
import sounddevice as sd
import queue
import soundfile as sf

from datetime import datetime, timedelta
from time import sleep
from threading import Thread
import soundfile as sf

def play_audio_file(audio_file):
    """
    Play an audio file in a separate thread.
    """
    try:
        data, samplerate = sf.read(audio_file)
        sd.play(data, samplerate)
        sd.wait()
    except Exception as e:
        print(f"Error playing audio: {e}")


def transcribe_audio_file(audio_file, audio_model, output_file, play_audio=False):
    """
    Transcribe an audio file with real-time-like display and optional audio playback.
    """
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found.")
        return
    
    print(f"Transcribing audio file: {audio_file}")

    audio_data = whisper.load_audio(audio_file)
    sample_rate = 16000  # Whisper always uses 16kHz
    
    # Start playing audio in a separate thread if requested
    audio_thread = None
    if play_audio:
        print("Playing audio...\n")
        audio_thread = Thread(target=play_audio_file, args=(audio_file,))
        audio_thread.daemon = False
        audio_thread.start()
        sleep(0.5)

    else:
        print("Processing...\n")
    
    # Transcribe with word-level timestamps for real-time display
    result = audio_model.transcribe(
        audio_file, 
        fp16=torch.cuda.is_available(),
        word_timestamps=True,
        verbose=False
    )
    
    # Get segments
    segments = result.get('segments', [])
    
    # Display transcription in real-time fashion
    print("Transcription:")
    print("=" * 50)
    
    transcription_lines = []
    if play_audio:
        start_time = datetime.now()

    if segments:
        for segment in segments:
            # Simulate real-time by displaying words as they would be spoken
            words = segment.get('words', [])
            
            if words:
                # Display word by word with timing
                line_text = ""
                for word_info in words:
                    word = word_info['word']
                    line_text += word
                    word_start = word_info.get('start', 0)
                    
                    if play_audio:
                        elapsed = (datetime.now() - start_time).total_seconds()
                        wait_time = word_start - elapsed
                        if wait_time > 0:
                            sleep(wait_time)
                    line_text += word
                    print(word, end='', flush=True)
                print()  # New line after segment
                transcription_lines.append(line_text.strip())
            else:
                # Fallback if no word timestamps
                line = segment['text'].strip()
                print(line)
                transcription_lines.append(line)
    else:
        # Fallback to full text
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

    print(f"\n✓ Transcription saved to: {output_file}")


