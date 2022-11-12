# import os
# from google.cloud import speech
#
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'client_service_key.json'
# speech_client = speech.SpeechClient()
#
#
# def get_name_from_voice(path):
#     # media_file_name_mp3 = 'audios/path'
#
#     with open(media_file_name_mp3, 'rb') as f1:
#         byte_data_mp3 = f1.read()
#     audio_mp3 = speech.RecognitionAudio(content=byte_data_mp3)
#     config_mp3 = speech.RecognitionConfig(
#         sample_rate_hertz=48000,
#         enable_automatic_punctuation=True,
#         language_code='en-US'
#     )
#
#     response_standard_mp3 = speech_client.recognize(
#         config=config_mp3,
#         audio=audio_mp3
#     )
#
#     print(response_standard_mp3)
#
#     operation = speech_client.long_running_recognize(config=config_mp3, audio=audio_mp3)
#
#     print("Waiting for operation to complete...")
#     response = operation.result(timeout=90)
#
#     re_str = ''
#
#     for result in response.results:
#         # The first alternative is the most likely one for this portion.
#         re_str += result.alternatives[0].transcript + ' '
#         print(u"Transcript: {}".format(result.alternatives[0].transcript))
#         print("Confidence: {}".format(result.alternatives[0].confidence))
#
#     return re_str
