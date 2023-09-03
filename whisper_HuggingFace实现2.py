# Use a pipeline as a high-level helper
from transformers import pipeline
import accelerate

# path = r"H:\music\让我们荡起双桨 - 黑鸭子.mp3"
path = r"H:\music\Music\许巍\04.像风一样自由.mp3"
# path = r"H:\music\环球音乐极品典藏集 Lesson 13\04 - Lie.mp3"
# path = r"H:\music\Music\Fading like flower.mp3"

### Pipe方法:
# pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v2") # 此处带有task="automatic-speech-recognition",导致下载失败,去除task成功,但是下载6.17G文件太大了.
# pipe = pipeline( model="openai/whisper-large-v2") # 下载6.17G:  Downloading model.safetensors:6.17G 太大取消;
# pipe = pipeline(model="openai/whisper-small")
# pipe = pipeline(model="openai/whisper-small", device_map='auto',chunk_length_s=30)



# result = pipe(path)
# result = pipe("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
# print(result)

#### Load model directly
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# processor = AutoProcessor.from_pretrained("openai/whisper-large-v2")
# model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v2")  # 下载6.17G:  Downloading model.safetensors:6.17G 太大取消;
### transformers.Whisper API方法:

from transformers import WhisperProcessor, WhisperForConditionalGeneration
# from datasets import Audio, load_dataset

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
forced_decoder_ids = processor.get_decoder_prompt_ids(language="Chinese", task="transcribe")

# load streaming dataset and read first audio sample
# ds = load_dataset("common_voice", "fr", split="test", streaming=True)
# ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
# input_speech = next(iter(ds))["audio"]


# input_features = processor(input_speech["array"], sampling_rate=input_speech["sampling_rate"], return_tensors="pt").input_features
input_features = processor(path, return_tensors="pt").input_features

# generate token ids
predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
# decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
