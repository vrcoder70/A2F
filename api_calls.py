import os
import sys
import contextlib
import threading
import time
from typing import Union, Type
import speech_recognition as sr

is_recording = False
input_received = False

@contextlib.contextmanager
def ignoreStderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)

def keyboard_input_listener():
    global is_recording, input_received
    while True:
        try:
            input()
            input_received = True
        except KeyboardInterrupt:
            break

def speech_to_text(audio: sr.AudioData) -> tuple[bool, Union[str, Type[Exception]]]:
    """
    Convert speech audio to text using Google Web Speech API.
    
    Parameters:
        audio (sr.AudioData): Speech audio data.
        
    Returns:
        Tuple[bool, Union[str, Type[Exception]]]: A tuple containing a boolean indicating if the recognition
                                                 was successful (True) or not (False), and the recognized text
                                                 or the class of the exception if an error occurred.
    """
    try:
        # Use Google Web Speech API to recognize speech from audio data
        return True, asr.recognize_google(audio, language="en-US")
    except Exception as e:
        # If an error occurs during speech recognition, return False and the type of the exception
        return False, e.__class__

print("Press Enter to start recording...")
# Start keyboard input listener thread
keyboard_thread = threading.Thread(target=keyboard_input_listener)
keyboard_thread.daemon = True
keyboard_thread.start()

with ignoreStderr():
    
    with sr.Microphone() as source:
        asr = sr.Recognizer()
        asr.adjust_for_ambient_noise(source, duration=7)
        
        while True:
            if input_received:
                input_received = False
                is_recording = not is_recording
                if is_recording:
                    print('Recording started. Say something:')
                else:
                    print('Recording stopped.')
                continue

            if is_recording:
                try:
                    audio = asr.listen(source, timeout=5)
                    is_valid_input, _input = speech_to_text(audio)
                    if is_valid_input and _input != "":
                        print("User : ", _input)
                        # generate_A2F_response(_input)
                    else:
                        print("No valid input received.")
                except sr.WaitTimeoutError:
                    print("Error: No speech detected for 5 seconds.")
            input("Press Enter to continue...")
