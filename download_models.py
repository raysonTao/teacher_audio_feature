
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import os

ASR_MODEL_NAME = "openai/whisper-small"
EMOTION_MODEL_NAME = "Dpngtm/wav2vec2-emotion-recognition"
ASR_TARGET_DIR = "models/whisper-small"
EMOTION_TARGET_DIR = "models/wav2vec2-emotion"

os.makedirs(ASR_TARGET_DIR, exist_ok=True)
os.makedirs(EMOTION_TARGET_DIR, exist_ok=True)

def download_asr():
    print(f"下载 Whisper 模型到: {ASR_TARGET_DIR}")
    AutoProcessor.from_pretrained(ASR_MODEL_NAME).save_pretrained(ASR_TARGET_DIR)
    AutoModelForSpeechSeq2Seq.from_pretrained(ASR_MODEL_NAME).save_pretrained(ASR_TARGET_DIR)

def download_emotion():
    print(f"下载情绪识别模型到: {EMOTION_TARGET_DIR}")
    Wav2Vec2Processor.from_pretrained(EMOTION_MODEL_NAME).save_pretrained(EMOTION_TARGET_DIR)
    Wav2Vec2ForSequenceClassification.from_pretrained(EMOTION_MODEL_NAME).save_pretrained(EMOTION_TARGET_DIR)

if __name__ == "__main__":
    print("开始模型下载...")
    download_asr()
    download_emotion()
    print("✅ 所有模型已成功保存到本地 models/ 目录")
