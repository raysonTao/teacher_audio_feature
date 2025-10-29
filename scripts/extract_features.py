# 教师风格音频特征提取与情绪识别代码包
# 环境要求：Python 3.9+, PyTorch 2.0+, CUDA 11.7+, ffmpeg 已安装

# =============================
# 环境搭建指南
# =============================
# 建议在虚拟环境中运行（如conda）
# 创建环境：
# conda create -n teacher-audio python=3.9 -y
# conda activate teacher-audio
# 安装依赖：
# pip install torch torchaudio transformers librosa numpy jieba ffmpeg-python
# pip install git+https://github.com/openai/whisper.git

import os
import torch
import torchaudio
import librosa
import numpy as np
import jieba
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from collections import Counter
import soundfile as sf
import subprocess
import json

# -----------------------------
# 参数配置
# -----------------------------
ASR_MODEL = "openai/whisper-small"
EMOTION_MODEL = "Dpngtm/wav2vec2-emotion-recognition"  # 可替换为中文情感模型
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLING_RATE = 16000

# -----------------------------
# 工具函数
# -----------------------------
def preprocess_audio_ffmpeg(input_path, output_path):
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", str(SAMPLING_RATE), "-ac", "1", output_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def load_audio(audio_path):
    tmp_path = "temp_clean.wav"
    preprocess_audio_ffmpeg(audio_path, tmp_path)
    waveform, sr = torchaudio.load(tmp_path)
    os.remove(tmp_path)
    return waveform.to(torch.float32), sr

def transcribe_audio(waveform):
    print("加载 Whisper 模型并强制中文识别...")
    processor = WhisperProcessor.from_pretrained(ASR_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL).to(DEVICE)

    inputs = processor(
        waveform.squeeze().numpy(),
        sampling_rate=SAMPLING_RATE,
        return_tensors="pt",
        language="zh",
        task="transcribe"
    ).input_features.to(DEVICE)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")
    predicted_ids = model.generate(inputs, forced_decoder_ids=forced_decoder_ids)
    result = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return result

def extract_keywords(text, topk=5):
    words = [w for w in jieba.lcut(text) if len(w.strip()) > 1]
    freq = Counter(words)
    return [w for w, _ in freq.most_common(topk)]

def get_question_ratio(text):
    questions = text.count("？") + text.count("?")
    total = max(1, text.count("。") + text.count("？") + text.count("?"))
    return questions / total

def get_speech_rate(text, duration_sec):
    chars = len(text.replace(" ", ""))
    words = len([w for w in jieba.lcut(text) if len(w.strip()) > 1])
    return chars / (duration_sec / 60), words / (duration_sec / 60)

def get_pitch_energy(waveform, sr):
    samples = waveform.squeeze().numpy()
    pitches, magnitudes = librosa.piptrack(y=samples, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    pitch_values = pitch_values[pitch_values > 0]

    avg_pitch = float(np.mean(pitch_values)) if len(pitch_values) else 0
    std_pitch = float(np.std(pitch_values)) if len(pitch_values) else 0

    energy = float(np.sqrt(np.mean(samples**2)))
    energy_curve = librosa.feature.rms(y=samples).flatten()
    energy_var = float(np.std(energy_curve))

    return avg_pitch, std_pitch, energy, energy_var

def predict_emotion(waveform):
    model = Wav2Vec2ForSequenceClassification.from_pretrained(EMOTION_MODEL).to(DEVICE)
    processor = Wav2Vec2Processor.from_pretrained(EMOTION_MODEL)
    inputs = processor(waveform.squeeze().numpy(), sampling_rate=SAMPLING_RATE, return_tensors="pt")
    with torch.no_grad():
        logits = model(**{k: v.to(DEVICE) for k, v in inputs.items()}).logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    pred_id = int(np.argmax(probs))
    emotion_label = model.config.id2label[pred_id]
    return emotion_label, probs, model.config.id2label

# -----------------------------
# 主流程入口
# -----------------------------
def analyze_audio(audio_path):
    print(f"加载音频: {audio_path}")
    waveform, sr = load_audio(audio_path)
    duration = waveform.shape[1] / sr

    print("执行语音转写...")
    text = transcribe_audio(waveform)
    print("转写内容:", text)

    print("提取文本特征...")
    keywords = extract_keywords(text)
    q_ratio = get_question_ratio(text)

    print("计算语速...")
    cpm, wpm = get_speech_rate(text, duration)

    print("提取音高与能量特征...")
    avg_pitch, std_pitch, energy, energy_var = get_pitch_energy(waveform, sr)

    print("识别语音情绪...")
    emotion_label, emotion_probs, id2label = predict_emotion(waveform)

    features = {
        "speech_rate_chars_per_min": round(cpm, 2),
        "speech_rate_words_per_min": round(wpm, 2),
        "avg_pitch_hz": round(avg_pitch, 2),
        "pitch_std_dev": round(std_pitch, 2),
        "avg_energy": round(energy, 5),
        "energy_std_dev": round(energy_var, 5),
        "dominant_emotion": emotion_label,
        "emotion_probs": emotion_probs.tolist(),
        "emotion_id2label": id2label,
        "keywords": keywords,
        "question_ratio": round(q_ratio, 3),
        "transcript": text
    }

    print("\n提取的特征结果:")
    for k, v in features.items():
        print(f"{k}: {v}")

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", os.path.splitext(os.path.basename(audio_path))[0] + "_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)
    print(f"✅ 结果已保存到 {out_path}")

    return features

# -----------------------------
# 运行示例
# -----------------------------
if __name__ == "__main__":
    print("脚本启动成功，准备加载音频...")
    test_audio = "data/example.mp3"  # 替换为你的课堂录音路径
    analyze_audio(test_audio)
