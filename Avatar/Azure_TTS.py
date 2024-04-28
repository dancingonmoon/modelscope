import os
import azure.cognitiveservices.speech as speechsdk
import configparser


def config_read(
    config_path, section="DingTalkAPP_chatGLM", option1="Client_ID", option2=None
):
    """
    option2 = None 时,仅输出第一个option1的值; 否则输出section下的option1与option2两个值
    """
    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")
    option1_value = config.get(section=section, option=option1)
    if option2 is not None:
        option2_value = config.get(section=section, option=option2)
        return option1_value, option2_value
    else:
        return option1_value


def azure_TTS(key, region, text):
    """
    使用azure TTS 实现TTS,输出格式内存对象(流形式),或者缺省扬声器,音频格式为Ogg16Khz16BitMonoOpus
    :return:
    """
    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Ogg16Khz16BitMonoOpus
    )
    speech_config.speech_synthesis_voice_name = (
        "zh-CN-XiaoxiaoNeural"  # "en-US-AvaMultilingualNeural"
    )
    speech_synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=None
    )
    result = speech_synthesizer.speak_text_async(text).get()
    stream = speechsdk.AudioDataStream(result)
    # stream.save_to_wav_file("path/to/write/file.wav")

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        # print("Speech synthesized for text [{}]".format(text))
        return stream
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        reason =  "Speech synthesis canceled: {}".format(cancellation_details.reason)
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            reason = "Error details: {}".format(cancellation_details.error_details)




config_path = r"e:\Python_WorkSpace\config\Azure_Resources.ini"
# Creates an instance of a speech config with specified subscription key and service region.
key, region = config_read(
    config_path=config_path, section="Azure_TTS", option1="key", option2="region"
)

speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
# Note: the voice setting will not overwrite the voice element in input SSML.
speech_config.speech_synthesis_voice_name = (
    "zh-CN-XiaoxiaoNeural"  # "en-US-AvaMultilingualNeural"
)

text = "what a wonderful day神奇的一天"
ssml_string = """
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="zh-CN">
  <voice name="zh-CN-XiaoxiaoNeural">
        女儿看见父亲走了进来，问道：
        <mstts:express-as  style="cheerful">
            “您来的挺快的，怎么过来的？”
        </mstts:express-as>
        父亲放下手提包，说：
        <mstts:express-as role="OlderAdultMale" style="fearful">
            “刚打车过来的，路上还挺顺畅。”
        </mstts:express-as>
        <prosody pitch="high" volume="+80.00%">特别情况</prosody>
        <prosody contour="(10%,+80%)(20%,+50%)(30%,+10%)(40%,-20%)(50%,-30%)(60%,-60%) (70%,+50%)(80%,+60%)(100%,+80%)" >
            房间里面有人吗?
        </prosody>
        <prosody rate="+50%" volume="+80%">糟糕了,糟糕了,钱包丢了</prosody>
        <emphasis level="strong">有人</emphasis>
  </voice>
</speak>
"""

# use the default speaker as audio output.
speech_synthesizer = speechsdk.SpeechSynthesizer(
    speech_config=speech_config, audio_config=audio_config
)

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
