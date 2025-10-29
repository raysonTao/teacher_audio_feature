
# 教师风格音频特征提取系统（适配 RTX 4090）

## ✅ 环境要求

- Python 3.9+
- CUDA 12.1 驱动（默认已内置于 RTX 4090 平台）
- PyTorch 2.1+
- ffmpeg（用于 MP3 音频解码）

## 📦 安装依赖（推荐使用 conda）

```bash
conda create -n teacher-audio python=3.9 -y
conda activate teacher-audio

# 安装 PyTorch 2.1.2 + CUDA 12.1
pip install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其余依赖
pip install transformers==4.35.2 librosa jieba ffmpeg-python
pip install git+https://github.com/openai/whisper.git
```

## 🔧 下载模型

```bash
python download_models.py
```

模型将下载到 `models/` 子目录中。

## 🚀 执行音频特征提取

将课堂音频（MP3格式）放入 `data/` 目录，然后执行：

```bash
python scripts/extract_features.py
```

控制台将显示关键词、语速、音调、能量、情绪等信息。

## 📁 目录结构

```
teacher_audio_feature_extractor_rtx4090/
├── README.md
├── download_models.py
├── data/
│   └── example.mp3 （你自己的音频文件）
├── models/ （自动生成）
├── scripts/
│   └── extract_features.py
```
