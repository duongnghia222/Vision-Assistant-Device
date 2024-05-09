import speech_recognition as sr
from vosk import Model, KaldiRecognizer
import pyaudio

# Initialize recognizer
# r = sr.Recognizer()
model = Model(r"vosk-model-small-en-us-0.15")
recognizer = KaldiRecognizer(model, 16000)
mic = pyaudio.PyAudio()

stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
stream.start_stream()

while True:
    data = stream.read(4096)

    if recognizer.AcceptWaveform(data):
        text = recognizer.Result()
        print(f"' {text[14:-3]} '")
        if text[14:-3] == 'bottle':
            break





#
# # Using the default microphone as the audio source
# with sr.Microphone() as source:
#     print("Please say something:")
#     # Adjust the recognizer sensitivity to ambient noise
#     r.adjust_for_ambient_noise(source, duration=3)
#     # Listening for the first phrase and extracting it into audio data
#     print("Now")
#     audio = r.listen(source)
#
# try:
#     # Using Google Web Speech API to recognize audio
#     print("Google Speech Recognition thinks you said:")
#     print(r.recognize_google(audio))
# except sr.UnknownValueError:
#     # API was unable to understand the audio
#     print("Google Speech Recognition could not understand the audio")
# except sr.RequestError as e:
#     # API was unreachable or unresponsive
#     print(f"Could not request results from Google Speech Recognition service; {e}")
#
#
# try:
#     # Use Sphinx for offline speech recognition
#     print("Sphinx thinks you said:")
#     print(r.recognize_sphinx(audio))
# except sr.UnknownValueError:
#     print("Sphinx could not understand the audio")
# except sr.RequestError as e:
#     print(f"Sphinx error; {e}")