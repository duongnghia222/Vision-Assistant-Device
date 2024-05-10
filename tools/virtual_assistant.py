import pyttsx3
import pyaudio
from vosk import Model, KaldiRecognizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
from thefuzz import process
import datetime
from tools.classifier import WeatherClassifier
import threading
from subprocess import call
from multiprocessing import Process
nltk.data.path.append("./../nltk_data")

def download_nltk_resources():
    if not os.path.exists("./../nltk_data/corpora/stopwords"):
        print("Downloading stopwords")
        nltk.download('stopwords', download_dir="./../nltk_data")
    if not os.path.exists("./../nltk_data/tokenizers/punkt"):
        print("Downloading punkt")
        nltk.download('punkt', download_dir="./../nltk_data")


def remove_stopwords(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_text = ' '.join(filtered_words)
    return filtered_text

def run_on_separate_thread(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    engine.stop()  
    print("stopped", text)


class VirtualAssistant:
    def __init__(self, recognizer_model_path, rs_camera, words_per_minute=150, volume=0.9):
        self.recognizer_model_path = recognizer_model_path
        self.rs_camera = rs_camera
        self.audio = pyaudio.PyAudio()
        self.recognizer = KaldiRecognizer(Model(self.recognizer_model_path), 16000)

        # VA Voice
        # self.engine = pyttsx3.init()
        # self.engine_is_running = False
        # self.engine.setProperty('rate', words_per_minute)
        # self.engine.setProperty('volume', volume)
        download_nltk_resources()


    # def speak(self, text):
    #     def run_on_separate_thread(text):
    #             self.engine = pyttsx3.init()
    #             self.engine.say(text)
    #             self.engine.runAndWait()
    #             self.engine.stop()  
    #             print("stopped", text)

    #     thread = threading.Thread(target=run_on_separate_thread, args=(text,))
    #     thread.start()
    

    def speak(self, text):
        threading.Thread(target=run_on_separate_thread, args=(text,)).start()
    #     thread.start()

    def receive_object(self):
        command = None
        previous_text = None
        self.speak("What object do you want to find?")
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
                        # print("loop 1:", '\ntext: ', text, '\nprev: ', previous_text, '\ncmd: ', command)
                        if not previous_text:
                            # self.speak("Please provide a command first")
                            print("Please provide a command first")
                        else:
                            command = previous_text
                            break
                    elif text in ["exit", "quit", "stop", "cancel"]:
                        break
                    else:
                        # print("loop 2:", '\ntext: ', text, '\nprev: ', previous_text, '\ncmd: ', command)
                        previous_text = text
                        # self.speak(f"You want to {confirm_command} {text}! Say 'ok' to confirm")
                        self.speak(f"You want to find {text}?")
                        print(f"You want to find {text}?")
                else:
                    self.speak("Say it again")
                    print("Say it again")

        stream.stop_stream()
        stream.close()
        return command

    def recognize_command(self, command_prompt="None", confirm_command="None"):
        choices = ["change mode to finding", "change mode to walking", "take note",
                   "quit program", "what time is it", "what's the weather like", "change setting"]
        command = None
        previous_text = None
        special_cases = {
            "tom" : "time",
            "thomas" : "time",
            "top" : "time",
            "whether" : "weather",
            "fighting" : "finding",

        }
        stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
        stream.start_stream()
        print("Listening...")
        while not command:
            data = stream.read(2048, exception_on_overflow=False)
            if self.recognizer.AcceptWaveform(data):
                text = self.recognizer.Result()
                print(text)
                text = text[14:-3].lower().strip()
                text = remove_stopwords(text)
                # if special cases exist, replace them
                text = ' '.join([special_cases[word] if word in special_cases else word for word in text.split()])
                print(text)
                # remove duplicates
                text = ' '.join(dict.fromkeys(text.split()))
                if text:
                    if text in ["ok", "k", "okay"]:
                        # print("loop 1:", '\ntext: ', text, '\nprev: ', previous_text, '\ncmd: ', command)
                        if not previous_text:
                            # self.speak("Please provide a command first")
                            print("Please provide a command first")
                        else:
                            command = previous_text
                            break
                    else:
                        # print("loop 2:", '\ntext: ', text, '\nprev: ', previous_text, '\ncmd: ', command)
                        choice = process.extractOne(text, choices)
                        confidence = choice[1]
                        # self.speak(f"You want to {confirm_command} {text}! Say 'ok' to confirm")
                        if confidence > 55:
                            previous_text = choice[0]
                            if previous_text == "what time is it":
                                self.speak("You want to know the current time! Say 'ok' to confirm")
                            elif previous_text == "what's the weather like":
                                self.speak("You want to know the current weather! Say 'ok' to confirm")
                            else:
                                self.speak(f"You want to {previous_text}! Say 'ok' to confirm")
                            print(f"You want to {previous_text}! Say 'ok' to confirm")
                            print(confidence)
                        else:
                            print("I didn't catch that")
                else:
                    # self.speak("Say it again")
                    print("Say it again")

        stream.stop_stream()
        stream.close()
        return command

    def get_time(self):
        # get current time and date
        current_time = datetime.datetime.now()
        time_str = current_time.strftime("%I:%M %p")
        self.speak(f"The current time is {time_str}.")

    def take_note(self):
        with open("note.txt", "a") as f:
            self.speak("What would you like to write?")
            note = self.recognize_command()
            f.write(note + "\n")

    def play_music(self):
        pass

    def change_setting(self):
        pass


    def weather_classify(self):
        weather_classifier = WeatherClassifier("models/vit-base-patch16-224-in21k-weather-images-classification")
        ret, color_frame, _, _ = self.rs_camera.get_frame_stream()
        if not ret:
            print("Error: Could not read frame.")
            return
        class_label, prob = weather_classifier.predict(color_frame)
        print(class_label, prob)

    def hey_virtual_assistant(self):
        mode = "assistant"
        while mode == "assistant":
            self.speak("Hello, I am your virtual assistant. How can I help you today?")
            command = self.recognize_command()
            if command == "change mode to finding":
                mode = "finding"
            elif command == "change mode to walking":
                mode = "walking"
            elif command == "quit program":
                mode = "disabled"
            elif command == "what time is it":
                self.get_time()
            elif command == "what's the weather like":
                self.weather_classify()
        return mode

    def navigate_to_object(self, instruction, rotation_degrees, distance):
        if instruction == "stop":
            self.speak(instruction)
        elif instruction == "straight":
            self.speak(instruction + "      at " + str(round(distance / 1000, 1)) + " meters")
        else:
            self.speak(instruction + "      at " + str(rotation_degrees) + " degrees. And" +
                       str(round(distance / 1000, 1)) + " meters")

    def inform_obstacle_location(self, direction, size, obstacle_class, prob):
        self.speak(f"{size} obstacle on {direction}")
        if obstacle_class and prob:
            self.speak(f"Probably {obstacle_class}") #  with confidence {int(prob)} percent

    def close(self):
        self.audio.terminate()
        self.engine.stop()
        self.engine.runAndWait()
        print("Virtual Assistant closed")

