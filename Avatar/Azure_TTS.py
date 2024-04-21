import os
import azure.cognitiveservices.speech as speechsdk
import configparser

def config_read(config_path, section='DingTalkAPP_chatGLM', option1='Client_ID', option2=None):
    """
    option2 = None 时,仅输出第一个option1的值; 否则输出section下的option1与option2两个值
    """
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')
    option1_value = config.get(section=section, option=option1)
    if option2 is not None:
        option2_value = config.get(section=section, option=option2)
        return option1_value, option2_value
    else:
        return option1_value




config_path = r"L:\Python_WorkSpace\config\Azure_Resources.ini"
# Creates an instance of a speech config with specified subscription key and service region.
key, region = config_read(config_path=config_path, section='Azure_TTS',option1='key', option2='region')

speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
# Note: the voice setting will not overwrite the voice element in input SSML.
speech_config.speech_synthesis_voice_name = "zh-CN-XiaoxiaoNeural" #"en-US-AvaMultilingualNeural"

text = "what a wonderful day神奇的一天"
ssml_string = """
<speak version="1.0" xmlns="https://www.w3.org/2001/10/synthesis" xml:lang="en-US">
  <voice name="zh-CN-XiaoxiaoNeural">
    离离原上草,一岁一枯荣,野火烧不尽,春风吹又生
  </voice>
</speak>
"""

# use the default speaker as audio output.
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

# result = speech_synthesizer.speak_text_async(text).get()
result = speech_synthesizer.speak_ssml_async(ssml_string).get()
# Check result
if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
    print("Speech synthesized for text [{}]".format(text))
elif result.reason == speechsdk.ResultReason.Canceled:
    cancellation_details = result.cancellation_details
    print("Speech synthesis canceled: {}".format(cancellation_details.reason))
    if cancellation_details.reason == speechsdk.CancellationReason.Error:
        print("Error details: {}".format(cancellation_details.error_details))




