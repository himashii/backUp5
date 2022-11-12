import os
from google.cloud import speech

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'client_service_key.json'
speech_client = speech.SpeechClient()

media_file_name_mp3 = 'audios/test4.mp3'

with open(media_file_name_mp3, 'rb') as f1:
    byte_data_mp3 = f1.read()
audio_mp3 = speech.RecognitionAudio(content=byte_data_mp3)

config_mp3 = speech.RecognitionConfig(
    sample_rate_hertz=48000,
    enable_automatic_punctuation=True,
    language_code='si-LK'
)

response_standard_mp3 = speech_client.recognize(
    config=config_mp3,
    audio=audio_mp3
)



print(response_standard_mp3)

operation = speech_client.long_running_recognize(config=config_mp3, audio=audio_mp3)

print("Waiting for operation to complete...")
response = operation.result(timeout=90)

for result in response.results:
    # The first alternative is the most likely one for this portion.
    print(u"Transcript: {}".format(result.alternatives[0].transcript))
    print("Confidence: {}".format(result.alternatives[0].confidence))
