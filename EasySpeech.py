import os
import pyttsx3
import speech_recognition as sr

from dotenv import load_dotenv
from elevenlabs import play
from elevenlabs.client import ElevenLabs

class EasySpeech:
    def __init__(self):
        load_dotenv()

        self.recognizer = sr.Recognizer()

        self.local_synthetizer = pyttsx3.init()
        self.local_synthetizer.setProperty('rate', 150)
        self.local_synthetizer.setProperty('volume', 1.0)
        self.local_synthetizer.setProperty('voice', self.local_synthetizer.getProperty('voices')[0].id)

        self.natural_synthetizer = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
        self.natural_voice = "Rachel"
        
    def ListenToHuman(self):
        with sr.Microphone() as source:
            print("Listening...")
            audio = self.recognizer.listen(source)
        try:
            print("Recognizing...")
            text = self.recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text.lower()

        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return ""

        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return ""

    def Listen(self):
        # Todo: Implement this
        input("Listening...")

    def TextToSpeechLocal(self, text):
        self.local_synthetizer.say(text)
        self.local_synthetizer.runAndWait()

    def TextToSpeechNatural(self, text):
        audio = self.natural_synthetizer.generate(
            text=text,
            voice=self.natural_voice,
            model="eleven_multilingual_v2"
        )
        play(audio)

easySpeech = EasySpeech()

if __name__ == "__main__":
    phrase = easySpeech.ListenToHuman()