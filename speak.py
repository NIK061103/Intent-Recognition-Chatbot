from os import system


def start_speaking():
    print("A.I : Hi. I am your personal voice-chat assistant.")
    speak("Hi. I am your personal voice-chat assistant.")
    print("A.I : How can i help you?")
    speak("How can i help you?")


def speak(text):
    system("say {}".format(text))

# speak("Hi. My name is A.I.")
