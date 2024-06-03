# count = 0
# for i in range(10000):
#     if 2**(i - 1) % 7 == 0:
#         count+=1
#         print(i)
# print(count)

import json
from tools.realsense_camera import RealsenseCamera
import cv2
import numpy as np
from tools.virtual_assistant import VirtualAssistant

# rs_camera = RealsenseCamera(width=640, height=480) # This is max allowed
# print("Starting RealSense camera. Press 'q' to quit.")
# while True:
#     ret, color_frame, depth_frame, infrared_frame, frame_number = rs_camera.get_frame_stream()
#     if not ret:
#         print("Error: Could not read frame.")
#         break
#
#     # Display the resulting frame
#     cv2.imshow('Color Image', color_frame)
#     # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)
#     # cv2.imshow('Disparity Map', depth_colormap)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         # open json file to save color frame
#         with open('infrared_frame_values.json', 'w') as file:
#             json.dump(infrared_frame.tolist(), file)
#         break


# # Open json file to read depth_frame as numpy array
# with open('depth_frame_values.json', 'r') as file:
#     depth_frame = np.array(json.load(file))
#
# # Viualize depth_frame values with disparity map
# depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)
# while True:
#     cv2.imshow('Disparity Map', depth_colormap)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# import nltk
# import os
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
#
# def download_nltk_resources():
#     nltk.data.path.append("./nltk_data")
#     if not os.path.exists("./nltk_data/corpora/stopwords"):
#         nltk.download('stopwords', download_dir="./nltk_data")
#     if not os.path.exists("./nltk_data/tokenizers/punkt"):
#         nltk.download('punkt', download_dir="./nltk_data")
#
# def remove_stopwords(text):
#     words = word_tokenize(text)
#     stop_words = set(stopwords.words('english'))
#     filtered_words = [word for word in words if word.lower() not in stop_words]
#     filtered_text = ' '.join(filtered_words)
#     return filtered_text
#
# # Example text
# text = "ok okay k"
#
# # Download NLTK resources
# download_nltk_resources()
#
# # Remove stopwords from the example text
# filtered_text = remove_stopwords(text)
#
# # Print the filtered text
# print("Original Text:", text)
# print("Filtered Text:", filtered_text)
#
# from ultralytics import YOLO
#
# # Load a model
# model = YOLO('yolov8x-cls.pt')  # load an official model
#
# # Predict with the model
# results = model('pictures/1.png', conf=0.1)  # predict on an image
# for result in results:
#     print(result)


# import numpy as np
# import cv2
#
# def detect_covering(color_frame, depth_frame, visualize=False):
#     # Extract ROI
#     middle_x = 1280 // 2
#     middle_y = 720 // 2
#     square_size = middle_x // 2
#     roi_depth_frame = depth_frame[middle_y - square_size:middle_y + square_size,
#                       middle_x - square_size:middle_x + square_size]
#
#     covered_pixels = np.count_nonzero(roi_depth_frame < 30)
#     # print(roi_depth_frame < 100)
#
#     # Calculate the total number of pixels in the ROI
#     total_pixels = roi_depth_frame.size
#     # Calculate the percentage of covered pixels
#     coverage_percentage = covered_pixels / total_pixels
#     print(coverage_percentage)
#     # Check if the coverage percentage exceeds the threshold
#     if visualize:
#         cv2.rectangle(color_frame, (middle_x - square_size, middle_y - square_size),
#                       (middle_x + square_size, middle_y + square_size), (0, 0, 0), 2)
#     if coverage_percentage > 0.9:
#         return True
#     else:
#         return False
#
# # Assuming you have already initialized the RealSenseCamera instance
# rs_camera = RealsenseCamera(width=1280, height=720)
#
# # Main loop
# while True:
#     ret, color_frame, depth_frame, frame_number = rs_camera.get_frame_stream()
#     if ret:
#         # Detect covering
#         if detect_covering(color_frame, depth_frame, visualize=True):
#             print("Something is covering your camera lens!")
#
#         # Display frames (optional)
#         cv2.imshow("Color Frame", color_frame)
#         cv2.imshow("Depth Frame", depth_frame)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
# # Release the camera
# rs_camera.release()
# cv2.destroyAllWindows()

# from math import pow
#
# def calculate_series(max_terms):
#     series_sum = 0
#     for n in range(3, max_terms + 1):
#         term = n * pow(2/3, n-3) * (1/27)
#         series_sum += term
#     return series_sum
#
# # Example usage
# max_terms = 200  # Calculate the sum up to 20 terms
# result = calculate_series(max_terms)
# print(f"The sum of the series up to {max_terms} terms is: {result:.6f}")

# from thefuzz import process
# from thefuzz import fuzz

# # Define the two strings
# string1 = "change finding mode to finding"
# string2 = "chain moat to fighting"
# string3 = "change mode to walking"

# # Calculate the similarity score using token sort ratio
# similarity_score = fuzz.partial_token_sort_ratio(string1, string2)
# similarity_score2 = fuzz.partial_token_sort_ratio(string1, string3)

# # Print the similarity score
# print("Similarity Score:", similarity_score)
# print("Similarity Score2:", similarity_score2)


from tools.virtual_assistant import VirtualAssistant
import time
import pyttsx3
# # Create a VirtualAssistant instance


virtual_assistant = VirtualAssistant("tools/vosk-model-en-us-0.22-lgraph", None,
                                         words_per_minute=290, volume=0.9)

# virtual_assistant.speak_threading("hello one over")
# virtual_assistant.speak_threading("hello two over")
# virtual_assistant.run_on_separate_thread("hello three")


virtual_assistant.speak_subprocess(f"Hello, I am your virtual assistant one. How can I help you today?", 1)
print(f"Hello, I am your virtual assistant one. How can I help you today?")


virtual_assistant.speak_subprocess(f"Hello, I am your virtual assistant two. How can I help you today?", 0)
print(f"Hello, I am your virtual assistant two. How can I help you today?")


virtual_assistant.speak_subprocess("Hello, I am your virtual assistant three. How can I help you today?", 1)
print("Hello, I am your virtual assistant three. How can I help you today?")


virtual_assistant.speak_subprocess("Hello, I am your virtual assistant four. How can I help you today?", 0)
print("Hello, I am your virtual assistant four. How can I help you today?")


virtual_assistant.speak_subprocess("Hello, I am your virtual assistant five. How can I help you today?", 0)
print("Hello, I am your virtual assistant five. How can I help you today?")


virtual_assistant.speak_subprocess("Hello, I am your virtual assistant six. How can I help you today?", 3)
print("Hello, I am your virtual assistant six. How can I help you today?")


virtual_assistant.speak_subprocess("Hello, I am your virtual assistant seven. How can I help you today?", 0)
print("Hello, I am your virtual assistant seven. How can I help you today?")


virtual_assistant.speak_subprocess("Hello, I am your virtual assistant eight. How can I help you today?", -1)
print("Hello, I am your virtual assistant eight. How can I help you today?")



# for i in range(10):
#     print(i)

# engine = pyttsx3.init()
# def onStart(name):
#    print('starting', name)
# def onWord(name, location, length):
#    print('word', name, location, length)
# def onEnd(name, completed):
#    print('finishing', name, completed)
#    if name == 'fox':
#       engine.say('What a lazy dog!', 'dog')
#    elif name == 'dog':
#       engine.endLoop()
# engine = pyttsx3.init()
# engine.connect('started-utterance', onStart)
# engine.connect('started-word', onWord)
# engine.connect('finished-utterance', onEnd)
# engine.say('The quick brown fox jumped over the lazy dog.', 'fox')
# engine.startLoop()

# engine = pyttsx3.init()
# engine.say('The quick brown fox jumped over the lazy dog.', 'fox')
# engine.startLoop(False)
# # engine.iterate() must be called inside externalLoop()
# externalLoop()
# engine.endLoop()

# import pyttsx3
# import time

# def external_loop(engine):
#     while engine.isBusy():
#         time.sleep(0.1)  # This is a simple way to keep the loop running

# # Initialize the engine
# engine = pyttsx3.init()

# # Queue the text to be spoken
# print("This will print before the text is spoken.")
# # engine.say("The quick brown fox jumped over the lazy dog.")


# print("This will print after the text is spoken.")

import pyttsx3
import time

# Define the callback functions
# def onStart(name):
#     print(f"starting {name}")

# def onWord(name, location, length):
#     print(f"word {name} {location} {length}")

# def onEnd(name, completed):
#     print(f"finishing {name}, completed: {completed}")
#     if name == 'fox':
#         engine.say('What a lazy dog!', 'dog')
#     elif name == 'dog':
#         engine.endLoop()

# # Initialize the TTS engine
# engine = pyttsx3.init()

# # Connect the callbacks to the engine
# # engine.connect('started-utterance', onStart)
# # engine.connect('started-word', onWord)
# engine.connect('finished-utterance', onEnd)

# # Queue commands to speak text
# engine.say('The quick brown fox jumped over the lazy dog.', 'dog')

# # Start the event loop to process the speaking command and fire callbacks
# engine.startLoop()




# from subprocess import call
# call(["", "Hello, I am your virtual assistant. How can I help you today?"])

# from gtts import gTTS
# tts = gTTS("Hello, I am your personal virtual assistant. How can I help you today?", lang='en')
# tts.save('hello.mp3')
# play the mp3 file in a threading
# import os
# os.system("mpg321 hello.mp3")
# import pygame
# import threading
# import time

# def play_mp3(file):
#     pygame.mixer.init()
#     pygame.mixer.music.load(file)
#     pygame.mixer.music.play()

# def main():
#     count = 3
#     while count > 0:
#         mp3_file = "hello.mp3"  # Replace this with the path to your MP3 file

#         # Create a thread for playing the MP3 file
#         mp3_thread = threading.Thread(target=play_mp3, args=(mp3_file,))
#         mp3_thread.start()
#         print("Playing MP3 file...")
#         count -= 1


# if __name__ == "__main__":
#     main()

# # Import necessary libraries
# import spacy
# from transformers import BertTokenizer, BertModel
# import torch
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# # Load pre-trained models and tokenizers
# nlp = spacy.load('en_core_web_trf')
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

# # Function to get BERT embeddings
# def get_bert_embeddings(text):
#     inputs = tokenizer(text, return_tensors='pt')
#     outputs = model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# # Function to calculate cosine similarity
# def cosine_sim(vec1, vec2):
#     return cosine_similarity(vec1, vec2)[0][0]

# # Example object names
# object_name1 = "dinning table"
# object_name2 = "table"

# # Get embeddings
# embedding1 = get_bert_embeddings(object_name1)
# embedding2 = get_bert_embeddings(object_name2)
# # Calculate similarity
# similarity = cosine_sim(embedding1, embedding2)
# print(f"Cosine Similarity: {similarity}")


# import gensim.downloader as api
# from sklearn.metrics.pairwise import cosine_similarity

# # Load the GloVe model
# glove_model = api.load("glove-wiki-gigaword-100")  # You can choose other versions, e.g., 50, 200

# # Function to get word embeddings
# def get_word_embedding(word, model):
#     return model[word]

# # Function to calculate cosine similarity
# def cosine_sim(vec1, vec2):
#     return cosine_similarity([vec1], [vec2])[0][0]

# # Example object names
# object_name1 = "man"
# object_name2 = "woman"

# # Get embeddings
# embedding1 = get_word_embedding(object_name1, glove_model)
# embedding2 = get_word_embedding(object_name2, glove_model)

# # Calculate similarity
# similarity = cosine_sim(embedding1, embedding2)
# print(f"Cosine Similarity: {similarity}")

# import nltk
# nltk.download('averaged_perceptron_tagger')
# from nltk import ngrams
# from nltk.tokenize import word_tokenize
# from nltk import pos_tag

# # Function to extract n-grams
# def extract_ngrams(text, num):
#     # Tokenize the text
#     tokens = word_tokenize(text)
#     # Generate n-grams
#     n_grams = list(ngrams(tokens, num))
#     return n_grams

# # Function to identify noun phrases
# def identify_noun_phrases(ngrams):
#     noun_phrases = []
#     for gram in ngrams:
#         # POS tag the n-gram
#         tags = pos_tag(gram)
#         # Check if the n-gram is a noun phrase (e.g., all words are nouns or adjectives followed by nouns)
#         if all(tag in ('NN', 'NNS', 'NNP', 'NNPS', 'JJ') for word, tag in tags):
#             noun_phrases.append(' '.join(gram))
#     return noun_phrases

# # Example text
# text = "Bottle bot up to me and I will give you a bottle of water."

# # Extract n-grams and identify noun phrases
# for n in range(2, 4):  # Using 2-grams and 3-grams for this example
#     ngrams_list = extract_ngrams(text, n)
#     noun_phrases = identify_noun_phrases(ngrams_list)
#     print(f"{n}-grams: {ngrams_list}")
#     print(f"Noun phrases from {n}-grams: {noun_phrases}")



