import pyttsx3


def speak(text):
    try:
        # Initialize the text-to-speech engine
        engine = pyttsx3.init()

        # Set properties (optional)
        engine.setProperty('rate', 150)  # Speed of speech
        engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

        # Say the provided text
        engine.say(text)

        # Wait for the speech to finish
        engine.runAndWait()

    except Exception as e:
        print(f"An error occurred: {e}")

