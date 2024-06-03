import pyttsx3
import pyaudio
from vosk import Model, KaldiRecognizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
from thefuzz import process
import datetime
from tools.classifier import Classifier
import threading
import queue
from subprocess import call, Popen
from multiprocessing import Process
nltk.data.path.append("./../nltk_data")
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from huggingface_hub import hf_hub_download
import time
import cv2

def download_nltk_resources():
    if not os.path.exists("./../nltk_data/corpora/stopwords"):
        print("Downloading stopwords")
        nltk.download('stopwords', download_dir="./../nltk_data")
    if not os.path.exists("./../nltk_data/tokenizers/punkt"):
        print("Downloading punkt")
        nltk.download('punkt', download_dir="./../nltk_data")
    if not os.path.exists("./../nltk_data/taggers/averaged_perceptron_tagger"): 
        print("Downloading averaged_perceptron_tagger")   
        nltk.download('averaged_perceptron_tagger')


def extract_nouns(sentence):
    words = word_tokenize(sentence)
    tagged_words = pos_tag(words)
    nouns = [word for word, tag in tagged_words if tag in ('NN', 'NNS')]
    return nouns


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


class VirtualAssistant:
    def __init__(self, recognizer_model_path, rs_camera, words_per_minute=150, volume=0.9):
        self.recognizer_model_path = recognizer_model_path
        self.rs_camera = rs_camera
        self.audio = pyaudio.PyAudio()
        self.recognizer = KaldiRecognizer(Model(self.recognizer_model_path), 16000)

        # VA Voice
        self.engine = pyttsx3.init()
        self.engine_is_running = False
        self.engine.setProperty('rate', words_per_minute)
        self.engine.setProperty('volume', volume)
        self.callback_event = threading.Event()

        #queue
        self.task_queue = queue.PriorityQueue()
        self.current_process = None
        self.lock = threading.Lock()
        self.processing_thread = threading.Thread(target=self.process_tasks)
        self.processing_thread.start()
        download_nltk_resources()

        # Load from the file
        with open("o365.txt", "r") as file:
            self.o365 = file.read().splitlines()
        
    def on_speech_start(self, name, location):
        self.callback_event.clear()

    def on_speech_finish(self, name, completed):
        if completed:
            self.callback_event.set()

    def run_on_separate_thread(self, text):

        self.engine.say(text, 'over')

        # Start the event loop to process the speaking command and fire callbacks
        self.engine.startLoop()

    def process_tasks(self):
        while True:
            priority, timestamp, text = self.task_queue.get()
            with self.lock:
                if self.current_process is not None:
                    # Terminate the current process if it's a low priority task and a high priority task comes in
                    self.current_process.terminate()
                    self.current_process = None
                self.current_process = Popen(["python", "tools/speak.py", text])
                self.current_process.wait()
                self.current_process = None
            self.task_queue.task_done()

    def remove_oldest_item(self):
        temp_list = []
        while not self.task_queue.empty():
            temp_list.append(self.task_queue.get())
        temp_list.sort()
        if temp_list:
            removed_item = temp_list.pop()
            print(f"Removing oldest item from queue: {removed_item[2]}")
        for item in temp_list:
            self.task_queue.put(item)
        
    def speak_subprocess(self, text, priority=1):
        time.sleep(0.00001)
        if self.task_queue.qsize() >= 3:
                self.remove_oldest_item()
        timestamp = -time.time()
        self.task_queue.put((priority, timestamp, text))

    
    def speak_threading(self, text):
        thread = threading.Thread(target=self.run_on_separate_thread, args=(text,))
        thread.start()

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def set_rs_camera(self, rs_camera):
        self.rs_camera = rs_camera
        


    def receive_object(self):
        object_to_find = None
        previous_text = None
        self.speak("What object do you want to find?")
        stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=2048)
        stream.start_stream()
        print("Listening...")
        while not object_to_find:
            data = stream.read(1024, exception_on_overflow=False)
            if self.recognizer.AcceptWaveform(data):
                text = self.recognizer.Result()
                
                text = text[14:-3].lower().strip()
                print("raw text:", text)
                text = remove_stopwords(text)
                print("removed stop word:", text)
                # remove duplicates
                text = ' '.join(dict.fromkeys(text.split()))
                print("removed duplicates:", text)
                # Make sure text contain object name, this is NER task, remove every word that is not an object name
                # TODO
                text = ' '.join(extract_nouns(text))
                print("extracted nouns:", text)
                if text:    
                    if text in ["ok", "k", "okay"]:
                        # print("loop 1:", '\ntext: ', text, '\nprev: ', previous_text, '\ncmd: ', command)
                        if not previous_text:
                            # self.speak("Please provide a command first")
                            print("Please provide a command first")
                        else:
                            object_to_find = previous_text
                            break
                    elif text in ["exit", "quit", "stop", "cancel"]:
                        return None, None
                    else:
                        # print("loop 2:", '\ntext: ', text, '\nprev: ', previous_text, '\ncmd: ', command)
                        if not previous_text:
                            print("first time")
                            choice = process.extractOne(text, self.o365)
                            confidence = choice[1]
                            # self.speak(f"You want to {confirm_command} {text}! Say 'ok' to confirm")
                            if confidence > 55:
                                previous_text = choice[0]
                                self.speak(f"You want to find {previous_text}? Say 'ok' to confirm")
                                print(f"---> You want to find {previous_text}?")
                                print(confidence)
                        else:
                            previous_text = text
                            self.speak(f"You want to find {previous_text}? Say 'ok' to confirm")
                            print(f"---> You want to find {previous_text}?")

                        
                else:
                    self.speak("Say it again")
                    print("Say it again")

        stream.stop_stream()
        stream.close()
        conf_threshold = 0.25 if object_to_find.lower() in [item.lower() for item in self.o365] else 0.01        
        return object_to_find, conf_threshold

    def recognize_command(self, command_prompt="None", confirm_command="None"):
        choices = ["change mode to finding", "change mode to walking", "take note", "listen note",
                   "quit program", "what time is it", "what's the weather like", "what's the traffic sign", "what's this food", "change setting", "disabled all modes"]
        command = None
        previous_text = None
        special_cases = {
            "tom" : "time",
            "thomas" : "time",
            "top" : "time",
            "whether" : "weather",
            "fighting" : "finding",
            "quick" : "quit",   

        }
        stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
        stream.start_stream()
        print("Listening...")
        while not command:
            data = stream.read(2048, exception_on_overflow=False)
            if self.recognizer.AcceptWaveform(data):
                text = self.recognizer.Result()
                text = text[14:-3].lower().strip()
                text = remove_stopwords(text)
                # if special cases exist, replace them
                text = ' '.join([special_cases[word] if word in special_cases else word for word in text.split()])
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
                    else:
                        # print("loop 2:", '\ntext: ', text, '\nprev: ', previous_text, '\ncmd: ', command)
                        choice = process.extractOne(text, choices)
                        confidence = choice[1]
                        # self.speak(f"You want to {confirm_command} {text}! Say 'ok' to confirm")
                        if confidence > 55:
                            previous_text = choice[0]
                            if previous_text == "what time is it":
                                print("You want to know the current time! Say 'ok' to confirm")
                                self.speak("You want to know the current time! Say 'ok' to confirm")
                            elif previous_text == "what's the weather like":
                                print("You want to know the current weather! Say 'ok' to confirm")
                                self.speak("You want to know the current weather! Say 'ok' to confirm")
                            elif previous_text == "what's the traffic sign":
                                print("You want to know the traffic sign! Say 'ok' to confirm")
                                self.speak("You want to know the traffic sign! Say 'ok' to confirm")
                            elif previous_text == "what's this food":
                                print("You want to know what food this is! Say 'ok' to confirm")
                                self.speak("You want to know what food this is! Say 'ok' to confirm")
                            else:
                                print(f"You want to {previous_text}! Say 'ok' to confirm")
                                self.speak(f"You want to {previous_text}! Say 'ok' to confirm")
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
        self.speak("Please start dictating your note.")
        stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=2048)
        stream.start_stream()
        with open("note.txt", "a") as file:
            while True:
                data = stream.read(1024, exception_on_overflow=False)
                if self.recognizer.AcceptWaveform(data):
                    text = self.recognizer.Result()
                    text = text[14:-3].strip()
                    text = remove_stopwords(text)

                    if text.lower() in {"stop", "exit", "quit"}:
                        self.speak("Note taking stopped.")
                        break
                    file.write(text + "\n")
                    self.speak("Note added.")
        stream.stop_stream()
        stream.close()

    def listen_note(self):
        self.speak("Listening to notes.")
        with open("note.txt", "r") as file:
            lines = file.readlines()
            for line in lines:
                self.speak(line.strip())

    def play_music(self):
        pass

    def change_setting(self):
        pass


    def weather_classify(self):
        model_path = "models/vit-base-patch16-224-in21k-weather-images-classification"
        if not os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
            # download the model using huggingface cli
            hf_hub_download(repo_id="DunnBC22/vit-base-patch16-224-in21k-weather-images-classification", filename="pytorch_model.bin", local_dir=model_path)
        weather_classifier = Classifier(model_path)
        ret, color_frame, _, _ = self.rs_camera.get_frame_stream()
        if not ret:
            print("Error: Could not read frame.")
            return
        # show image color
        cv2.imshow("Color", color_frame)
        cv2.waitKey(0)
        class_label, prob = weather_classifier.predict(color_frame)
        print(class_label, prob)
        if prob > 0.5:
            self.speak(f"The weather is {class_label} with {int(prob * 100)} percent confidence.")
            print(f"The weather is {class_label} with {int(prob * 100)} percent confidence.")
        else:
            self.speak("I could not detect the weather.")
            print("I could not detect the weather.")    
        del weather_classifier

    
    def traffic_sign_classify(self):
        model_path = "models/traffic-sign-classifier"
        if not os.path.exists(os.path.join(model_path, "model.safetensors")):
            # download the model using huggingface cli
            hf_hub_download(repo_id="dima806/traffic_sign_detection", filename="model.safetensors", local_dir=model_path)
        traffic_sign_classifier = Classifier(model_path, use_safetensor=True)
        ret, color_frame, _, _ = self.rs_camera.get_frame_stream()
        if not ret:
            print("Error: Could not read frame.")
            return
        class_label, prob = traffic_sign_classifier.predict(color_frame)
        if prob > 0.5:
            self.speak(f"I see a {class_label} with {int(prob * 100)} percent confidence.")
            print(f"I see a {class_label} with {int(prob * 100)} percent confidence.")
        else:
            self.speak("I could not detect any traffic sign.")
            print("I could not detect any traffic sign.")
        del traffic_sign_classifier
        


    def food_classify(self):
        model_path = "models/food-classifier"
        if not os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
            # download the model using huggingface cli
            hf_hub_download(repo_id="nateraw/food", filename="pytorch_model.bin", local_dir=model_path)
        food_classifier = Classifier(model_path)
        ret, color_frame, _, _ = self.rs_camera.get_frame_stream()
        if not ret:
            print("Error: Could not read frame.")
            return
        class_label, prob = food_classifier.predict(color_frame)
        if prob > 0.5:
            self.speak(f"I see a {class_label} with {int(prob * 100)} percent confidence.")
            print(f"I see a {class_label} with {int(prob * 100)} percent confidence.")
        else:
            self.speak("I could not detect any food.")
            print("I could not detect any food.")
        del food_classifier


    def hey_virtual_assistant(self, first_run):
        mode = "assistant"
        while mode == "assistant":
            if first_run:
                self.speak("Hello, I am your virtual assistant. How can I help you today?")
            else:
                self.speak("Welcome back, how can I help you today?")
            command = self.recognize_command()
            if command == "change mode to finding":
                mode = "finding"
            elif command == "change mode to walking":
                mode = "walking"
            elif command == "disabled all modes":
                mode = "disabled"
            elif command == "quit program":
                mode = "off"
            elif command == "what time is it":
                self.get_time()
            elif command == "what's the weather like":
                self.weather_classify()
            elif command == "what's the traffic sign":
                self.traffic_sign_classify()
            elif command == "what's this food":
                self.food_classify()
            elif command == "take note":
                self.take_note()
            elif command == "listen note":
                self.listen_note()
        return mode

    def navigate_to_object(self, instruction, rotation_degrees, distance, withRotate=False):
        if instruction == "stop":
            self.speak_subprocess(instruction)
        elif instruction == "straight":
            self.speak_subprocess(instruction + "      at " + str(round(distance / 1000, 1)) + " meters")
        else:
            if withRotate:
                self.speak_subprocess(instruction + "      at " + str(rotation_degrees) + " degrees and" +
                       str(round(distance / 1000, 1)) + " meters")
            else:
                self.speak_subprocess(instruction + "      at " + str(round(distance / 1000, 1)) + " meters")


    def inform_obstacle_location(self, direction, size, obstacle_class, prob):
        self.speak_subprocess(f"{size} obstacle on {direction}")
        if obstacle_class and prob > 0.7:
            self.speak_subprocess(f"Probably {obstacle_class}") #  with confidence {int(prob)} percent

    def close(self):
        self.audio.terminate()
        print("Virtual Assistant closed")


if __name__ == "__main__":
    pass

