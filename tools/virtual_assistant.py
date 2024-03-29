import pyttsx3
import pyaudio
from vosk import Model, KaldiRecognizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os


def download_nltk_resources():
    nltk.data.path.append("./../nltk_data")
    if not os.path.exists("./../nltk_data/corpora/stopwords"):
        nltk.download('stopwords', download_dir="./../nltk_data")
    if not os.path.exists("./../nltk_data/tokenizers/punkt"):
        nltk.download('punkt', download_dir="./nltk_data")


def remove_stopwords(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_text = ' '.join(filtered_words)
    return filtered_text


class VirtualAssistant:
    def __init__(self, recognizer_model_path, words_per_minute=150, volume=0.9):
        self.recognizer_model_path = recognizer_model_path
        self.audio = pyaudio.PyAudio()
        self.recognizer = KaldiRecognizer(Model(self.recognizer_model_path), 16000)

        # VA Voice
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', words_per_minute)
        self.engine.setProperty('volume', volume)
        download_nltk_resources()

    def speak(self, text):
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"An error occurred during speech: {e}")

    def recognize_command(self, command_prompt="None", confirm_command="None"):
        command = None
        previous_text = None
        self.speak(command_prompt)
        stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=2048)
        stream.start_stream()
        while not command:
            data = stream.read(1024, exception_on_overflow=False)
            if self.recognizer.AcceptWaveform(data):
                text = self.recognizer.Result()
                print(text)
                text = text[14:-3].lower().strip()
                text = remove_stopwords(text)
                print(text)
                # remove duplicates
                text = ' '.join(dict.fromkeys(text.split()))
                print(text)
                if text:
                    if text in ["ok", "k", "okay"]:
                        if not command or not previous_text:
                            # self.speak("Please provide a command first")
                            print("Please provide a command first")
                        else:
                            command = previous_text
                            break
                    elif text in ["exit", "quit", "stop", "cancel"]:
                        break
                    else:
                        previous_text = text
                        # self.speak(f"You want to {confirm_command} {text}! Say 'ok' to confirm")
                        print(f"You want to {confirm_command} {text}! Say 'ok' to confirm")
                else:
                    # self.speak("Say it again")
                    print("Say it again")

        stream.stop_stream()
        stream.close()
        return command

    def close(self):
        self.audio.terminate()
        self.engine.stop()
        self.engine.runAndWait()
        print("Virtual Assistant closed")

    def hey_virtual_assistant(self):
        self.speak("Hello, I am your virtual assistant. How can I help you today?")
        command = self.recognize_command("What do you want to do?", "do")
        print(command)
        if command:
            self.speak(f"Ok, I will {command} for you")
        else:
            self.speak("Goodbye")
            self.close()
        return command

    def navigate_to_object(self, instruction, rotation_degrees, distance):
        if instruction == "stop":
            self.speak(instruction)
        elif instruction == "move forward":
            self.speak(instruction + " " + str(int(distance / 100)) + " meters away")
        else:
            self.speak(instruction + " " + str(rotation_degrees) + " degrees")

    def inform_object_location(self, direction, size, distance, obstacle_class, prob):
        if obstacle_class and prob:
            self.speak(f"Probably {obstacle_class} with confidence {int(prob)} percent")
        self.speak(f"{size} obstacle {distance // 100} meters away on your {direction}")
