import speech_recognition as sr


def AI_listening():
    print("A.I : I'm listening...")


def listen():
    ear = sr.Recognizer()

    with sr.Microphone() as source:
        ear.pause_threshold = 1
        audio = ear.listen(source, phrase_time_limit=5)
        statement_heard = ear.recognize_google(audio, language="en-in")
        print(f"YOU : {statement_heard}")
    return statement_heard
