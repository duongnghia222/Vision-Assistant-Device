from tools.voice_navigator import TextToSpeech
speaker = TextToSpeech()
from obj_finding import ObjectFindingProcessor
from vosk import Model, KaldiRecognizer
import pyaudio
from pyaudio import PyAudio
from tools.realsense_camera import *
from tools.custom_segmentation import segment_object

micro = PyAudio()
vosk_model = Model(r"tools/vosk-model-en-us-0.22-lgraph")
recognizer = KaldiRecognizer(vosk_model, 16000)

def get_obj_from_voice():
    """Performs voice recognition to get the target object.

    Returns:
        str: The target object obtained from voice recognition.

    """
    speaker.speak("Please wait for system to start")

    stream = micro.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
    speaker.speak("What?")
    stream.start_stream()

    prev_text = None
    target_obj = None
    while not target_obj:
        data = stream.read(4096, exception_on_overflow=False)
        if recognizer.AcceptWaveform(data):
            text = recognizer.Result()
            text = text[14:-3]
            print(text)
            
            if text == "ok" or text == "k" or text == "okay":
                target_obj = prev_text
                break
            
            stream.stop_stream()
            speaker.speak("You want to find {} !Say ok to confirm".format(text))
            stream.start_stream()
            # time.sleep(2)
            prev_text = text


    stream.stop_stream()
    stream.close()
    micro.terminate()
    return target_obj

    
if __name__ == "__main__":
    mode = 'finding'
    
    
    if mode == 'avoiding':
        pass
    
    elif mode == 'finding':
        target_obj = get_obj_from_voice()
        processor = ObjectFindingProcessor(0, [target_obj], 'yolov8l-worldv2.pt', [1280, 720])
        processor.process_camera()
        
    else: # mode == 'normal'
        pass