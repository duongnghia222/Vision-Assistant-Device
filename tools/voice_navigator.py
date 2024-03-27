import pyttsx3
import pyaudio
from vosk import Model, KaldiRecognizer


class TextToSpeech:
    def __init__(self, rate=150, volume=0.9):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)

    def speak(self, text):
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"An error occurred during speech: {e}")


class CommandRecognizer:
    def __init__(self, command_prompt, recognizer_model_path, voice):
        self.command_prompt = command_prompt
        self.recognizer_model_path = recognizer_model_path
        self.command = None
        self.audio = pyaudio.PyAudio()
        self.recognizer = KaldiRecognizer(Model(self.recognizer_model_path), 16000)
        self.voice = voice
        self.previous_text = None

    def __del__(self):
        self.audio.terminate()

    def recognize_command(self):
        self.voice.speak(self.command_prompt)
        stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
        stream.start_stream()

        while not self.command:
            data = stream.read(4096, exception_on_overflow=False)
            if self.recognizer.AcceptWaveform(data):
                text = self.recognizer.Result()
                text = text[14:-3].lower().strip()
                print(text)

                if text in ["ok", "k", "okay"]:
                    if self.command or not self.previous_text:
                        self.voice.speak("Please provide a command first")
                    else:
                        self.command = self.previous_text
                else:
                    self.command = text
                    self.previous_text = text
                    self.voice.speak(f"You want to {self.command}! Say 'ok' to confirm")

        stream.stop_stream()
        stream.close()
        return self.command

