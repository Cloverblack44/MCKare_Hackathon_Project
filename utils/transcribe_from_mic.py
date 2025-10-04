

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



def transcribe_microphone(args, audio_model, output_file):
    """
    Transcribe audio from microphone in real-time.
    """
    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = queue.Queue()
    # Audio data for the current phrase
    phrase_audio = np.array([], dtype=np.float32)
    
    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout
    sample_rate = args.sample_rate
    energy_threshold = args.energy_threshold

    transcription = ['']

    def audio_callback(indata, frames, time, status):
        """
        Callback function to receive audio data from the microphone.
        """
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
                
                # Pull raw recorded audio from the queue.
                if not data_queue.empty():
                    phrase_complete = False
                    
                    # If enough time has passed between recordings, consider the phrase complete.
                    if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                        phrase_audio = np.array([], dtype=np.float32)
                        phrase_complete = True
                    
                    # This is the last time we received new audio data from the queue.
                    phrase_time = now
                    
                    # Combine audio data from queue
                    audio_chunks = []
                    while not data_queue.empty():
                        audio_chunks.append(data_queue.get())
                    
                    if audio_chunks:
                        audio_data = np.concatenate(audio_chunks)
                        
                        # Add the new audio data to the accumulated data for this phrase
                        phrase_audio = np.concatenate([phrase_audio, audio_data])
                        
                        # Read the transcription.
                        result = audio_model.transcribe(phrase_audio, fp16=torch.cuda.is_available())
                        text = result['text'].strip()

                        # If we detected a pause between recordings, add a new item to our transcription.
                        # Otherwise edit the existing one.
                        if phrase_complete:
                            transcription.append(text)
                        else:
                            transcription[-1] = text

                        # Clear the console to reprint the updated transcription.
                        os.system('cls' if os.name=='nt' else 'clear')
                        for line in transcription:
                            print(line)
                        # Flush stdout.
                        print('', end='', flush=True)
                        
                        # Save transcription to file after each update
                        with open(output_file, 'w', encoding='utf-8') as f:
                            for line in transcription:
                                if line:  # Skip empty lines
                                    f.write(line + '\n')
                else:
                    # Infinite loops are bad for processors, must sleep.
                    sleep(0.1)
                    
            except KeyboardInterrupt:
                break

    print("\n\nFinal Transcription:")
    for line in transcription:
        print(line)
    
    # Save final transcription
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in transcription:
            if line:
                f.write(line + '\n')
    
    print(f"\n✓ Transcription saved to: {output_file}")

