import os
import io
import sys
import time
import openai
import requests
import threading
import contextlib
import numpy as np
from gtts import gTTS
from typing import Union, Type
from dotenv import load_dotenv
from pydub import AudioSegment
import speech_recognition as sr
from scipy.io.wavfile import read, write
from audio2face_streaming_utils import push_audio_track, push_audio_track_stream

# Load api key and model id.
load_dotenv()
openai.api_key = os.getenv("OPEN_AI_KEY")
model_id = os.getenv("MODEL_ID")

# Create the audio2face URL.
a2f_url = f'{os.getenv("DEFAULT_URL")}:{os.getenv("GRPC_PORT")}'

#Load A2F instance:
rest_api_url = f'http://{os.getenv("DEFAULT_URL")}:{os.getenv("REST_API_PORT")}'

# audio frame rate for the audio data.
framerate = 22050  # Replace '22050' with the desired audio frame rate (samples per second).

# the instance name of the avatar in audio2face service.
a2f_avatar_instance_streaming = '/World/audio2face/PlayerStreaming'
a2f_avatar_instance_regular = '/World/audio2face/Player'

asr = sr.Recognizer()

#Endpoints
load_instance = '/A2F/USD/Load'
set_emotions = '/A2F/A2E/SetEmotionByName'
request_player = '/A2F/Player/GetInstances'

# Avatar file:
streaming_player = '/home/avstech/Desktop/NvidiaAudio2Face/models/claire_streaming_player.usd'
regular_player = '/home/avstech/Desktop/NvidiaAudio2Face/models/claire_regular_player.usd'

# Emotions
emotions = {
    "a2f_instance": "/World/audio2face/CoreFullface",
    "emotions": { "amazement" : 0.5, "anger" : 0, "cheekiness" : 1, "disgust" : 0, "fear" : 0, "grief" : 0, "joy" : 0.5, "outofbreath" : 0, "pain" : 0, "sadness" : 0 }
}


def generate_A2F_response(text: str=""):
    """
    Generates a response based on the provided text using OpenAI's ChatGPT model, converts the response to text-to-speech (TTS) audio,
    and sends the audio to an avatar-to-facetracking (A2F) system for playback.

    Args:
        text (str): The text for which the response needs to be generated.
        framerate (int): The framerate for the audio output (default is 22050).

    Returns:
        bool: True if the response was successfully generated, False otherwise.
        None: None if the response was not successfully generated.
    """
    
    messages = [
        {"role": "system", "content" : "You are helpful assistant. Answer in very short and concise manner."},
        {"role": "user", "content" : text}
    ]

    response = openai.ChatCompletion.create(model=model_id,messages=messages).choices[0].message.content

    print("Avatar : ", response)

    # Generate TTS audio in mp3 format from the given text
    tts_result = io.BytesIO()
    tts = gTTS(text=response, lang='en-US', slow=False)
    tts.write_to_fp(tts_result)
    tts_result.seek(0)
    mp3_byte = tts_result.read()

    # Convert the TTS audio in mp3 format to WAV format and a numpy array of float32 values
    seg = AudioSegment.from_mp3(io.BytesIO(mp3_byte))
    seg = seg.set_frame_rate(framerate)
    seg = seg.set_channels(1)
    wavIO = io.BytesIO()
    seg.export(wavIO, format="wav")
    rate, wav = read(io.BytesIO(wavIO.getvalue()))
    
    # Convert the WAV audio bytes to a numpy array of float32 values
    tts_audio = wav.astype(np.float32, order='C') / 32768.0

    # Send audio A2F
    push_audio_track(a2f_url, tts_audio, framerate, a2f_avatar_instance_streaming)

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

succesful_code = 200
is_recording = False



with ignoreStderr():
    status = requests.get(rest_api_url+"/status")
    if status.status_code != succesful_code:
        print("Error: unable to reach A2F...")
        sys.exit()    
    with sr.Microphone() as source:
        asr.adjust_for_ambient_noise(source, duration=5)
    # Request to load avatar Execute when run for first time in day
    # a2f_instance = {"file_name" : streaming_player}
    # response = requests.post(rest_api_url+load_instance, json=a2f_instance)
    # time.sleep(5)
    # if response.status_code != succesful_code:
    #     print(f'Error: Unable to load {streaming_player}...')
    #     sys.exit()
    # print(f'A2F avatar uploaded')
    
    # Optional
    # Request Player 
    # players = requests.get(rest_api_url+request_player)
    # if players.status_code != response_code:
    #     print(f'Players not available...')
    #     sys.exit()
    # print(players['result'])

    # Set emotions
    emotions_response = requests.post(rest_api_url+set_emotions, json=emotions)
    if emotions_response.status_code != 200:
        print(f'Could not load emotions, please load manually...')
    else:
        print(f'Emotions are set')

    while True:
        with sr.Microphone() as source:
            if is_recording:
                try:
                    is_recording = False
                    print('Say Something:')
                    asr.pause_threshold = 0.5
                    audio = asr.listen(source)
                    is_valid_input, _input = speech_to_text(audio)
                    if is_valid_input and _input != "":
                        print("User : ", _input)
                        generate_A2F_response(_input)
                    else:
                        print("No valid input received.")
                except Exception as e:
                    print(f"Error: {e}")
                    time.sleep(5)
        print("Press Enter to continue...")     
        input()
        is_recording = True
                    
            
                
            
