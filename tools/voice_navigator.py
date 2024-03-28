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
    def __init__(self, recognizer_model_path, voice):
        self.recognizer_model_path = recognizer_model_path
        self.audio = pyaudio.PyAudio()
        self.recognizer = KaldiRecognizer(Model(self.recognizer_model_path), 16000)
        self.voice = voice

    def __del__(self):
        self.audio.terminate()

    def recognize_command(self, command_prompt="None", confirm_command="None"):
        command = None
        previous_text = None
        self.voice.speak(command_prompt)
        stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=2048)
        stream.start_stream()
        print("listening")
        while not command:
            data = stream.read(1024, exception_on_overflow=False)
            if self.recognizer.AcceptWaveform(data):
                print("Recognizing")
                text = self.recognizer.Result()
                print(text)
                text = text[14:-3].lower().strip()
                if "the" in text:
                    text = text.replace("the", "")
                if text:
                    if text in ["ok", "k", "okay"]:
                        if command or not previous_text:
                            # self.voice.speak("Please provide a command first")
                            print("Please provide a command first")
                        else:
                            command = previous_text
                    else:
                        previous_text = text
                        print(text)
                        # self.voice.speak(f"You want to {confirm_command} {text}! Say 'ok' to confirm")
                else:
                    print("Say it again")

        stream.stop_stream()
        stream.close()
        return command

