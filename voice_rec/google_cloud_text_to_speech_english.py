import os
from google.cloud import speech

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'client_service_key.json'
from google.cloud import texttospeech

client = texttospeech.TextToSpeechClient()

sample_text = 'good morning everyone'

input_text = texttospeech.SynthesisInput(ssml=sample_text)

voice = texttospeech.VoiceSelectionParams(
    language_code="en-US",
    name="en-US-Standard-C",
    ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
)

audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3
)

response = client.synthesize_speech(
    input=input_text, voice=voice, audio_config=audio_config
)

with open("voice_outputs/output.mp3", "wb") as out:
    out.write(response.audio_content)
    print('Audio content written to file "output.mp3"')
