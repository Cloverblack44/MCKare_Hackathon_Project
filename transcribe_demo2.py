#! python3.7

import argparse
import os
import numpy as np
import whisper
import torch
import sounddevice as sd
import queue

from datetime import datetime, timedelta
from time import sleep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=0.01,
                        help="Energy level for mic to detect.", type=float)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    parser.add_argument("--sample_rate", default=16000,
                        help="Sample rate for audio recording.", type=int)
    parser.add_argument("--output_dir", default="transcriptions",
                        help="Directory to save transcription files.", type=str)
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"transcription_{timestamp}.txt")



    # The last time a recording was retrieved from the queue.
    phrase_time = None

    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = queue.Queue()

    # Bytes object which holds audio data for the current phrase
    phrase_audio = np.array([], dtype=np.float32)
    
    # Load / Download model
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    audio_model = whisper.load_model(model)

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
    print("Model loaded.\n")
    print(f"Transcription will be saved to: {output_file}\n")
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
                else:
                    # Infinite loops are bad for processors, must sleep.
                    sleep(0.1)
                    
            except KeyboardInterrupt:
                break

    print("\n\nFinal Transcription:")
    for line in transcription:
        print(line)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in transcription:
            if line:
                f.write(line + '\n')
    
    print(f"\n✓ Transcription saved to: {output_file}")

if __name__ == "__main__":
    main()