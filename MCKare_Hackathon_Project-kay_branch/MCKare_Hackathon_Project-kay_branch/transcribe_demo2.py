#! python3.7

import argparse
import os
import numpy as np
import whisper
import torch
import sounddevice as sd
import queue
import soundfile as sf

from utils.transcribe_from_mic import transcribe_microphone
from utils.transcribe_from_audio_file import transcribe_audio_file

from datetime import datetime, timedelta
from time import sleep
from threading import Thread
import soundfile as sf

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
    parser.add_argument("--audio_file", default=None,
                        help="Path to audio file to transcribe (instead of microphone).", type=str)
    parser.add_argument("--play_audio", action='store_true',
                        help="Play audio file while transcribing.")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # If audio file is provided, use its name in the output filename
    if args.audio_file:
        base_name = os.path.splitext(os.path.basename(args.audio_file))[0]
        output_file = os.path.join(args.output_dir, f"transcription_{base_name}_{timestamp}.txt")
    else:
        output_file = os.path.join(args.output_dir, f"transcription_{timestamp}.txt")
    
    # Load / Download model
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    audio_model = whisper.load_model(model)

    print("Model loaded.\n")
    print(f"Transcription will be saved to: {output_file}\n")

    # If audio file is provided, transcribe it directly
    if args.audio_file:
        transcribe_audio_file(args.audio_file, audio_model, output_file, args.play_audio)
    else:
        transcribe_microphone(args, audio_model, output_file)

if __name__ == "__main__":
    main()