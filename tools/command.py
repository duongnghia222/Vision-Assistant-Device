import pyaudio
import vosk
import subprocess  # For playing audio
from vosk import Model, KaldiRecognizer
# Initialize PyAudio
audio = pyaudio.PyAudio()

# Initialize Vosk recognizer
model = Model(r"../tools/vosk-model-en-us-0.22-lgraph")
recognizer = KaldiRecognizer(model, 16000)


# Function to play audio
def speak(text):
    # Use espeak command to speak the text
    subprocess.call(['espeak', text])


# Main loop
while True:
    # Initialize variables
    object_to_find = None
    previous_text = None

    # Ask for command
    speak("What do you want to find?")

    # Open audio stream
    stream = audio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=8192)

    # Process audio until command is received
    while not object_to_find:
        data = stream.read(4096, exception_on_overflow=False)
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            text = result["text"]
            print("You said:", text)

            # Check if confirmation is received
            if text.lower() in ["ok", "k", "okay"]:
                if previous_text:
                    object_to_find = previous_text
                    break
                else:
                    speak("Please provide a command first.")
            else:
                previous_text = text
                speak("You want to find {}! Say 'ok' to confirm".format(text))

    # Close audio stream
    stream.stop_stream()
    stream.close()

    # Terminate PyAudio
    audio.terminate()

    # Print the confirmed command
    print("Object to find:", object_to_find)
