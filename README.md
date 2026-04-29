# 🎙️ Qwen3-TTS WebUI (Colab Edition)


一款基于阿里巴巴 **Qwen3-TTS-12Hz** 模型开发的非官方全能型语音合成界面。本项目针对 Google Colab 进行了深度优化，支持长文本合成、自动克隆及多种格式字幕生成。

---

### 🔗 快速开始 / Quick Start

点击下方按钮在 Google Colab 中直接运行，无需本地配置：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10KBH_YE7gCDjng6QWu9YVCshyjP5q82M?usp=sharing)
[![YouTube](https://img.shields.io/badge/YouTube-FM19.2-red)](https://www.youtube.com/@fm19.2?sub_confirmation=1)
---

### 🌟 项目亮点 / Key Features

#### **[CN] 中文特性描述**
* **声音设计 (Voice Design)**：只需输入自然语言描述（如“一个充满恐慌的低沉男声”），即可创造出独特的虚拟人声。
* **零样本声音克隆 (Voice Clone)**：上传一段几秒钟的参考音频，系统即可快速捕捉并模仿目标人物的音色与语调。
* **标准 TTS (CustomVoice)**：内置 9 位高质量预设说话人（如 Ryan, Serena 等），支持通过指令微调情感与风格。
* **长文本自动处理**：内置自动分段处理机制，轻松应对数千字的超长文章合成。
* **全自动字幕流**：集成 Whisper 模型，在音频生成的同时自动导出 SRT、易读版、逐词对齐及短视频专用四种格式字幕。

#### **[EN] English Key Capabilities**
* **Voice Design**: Create unique, high-quality voices from scratch using natural language descriptions (e.g., "A deep, magnetic male voice with a hint of panic")。
* **Zero-shot Voice Cloning**: Replicate any voice with high fidelity by simply uploading a short reference audio clip—no fine-tuning required。
* **Multilingual Standard TTS**: Features professional preset speakers with robust support for 10 languages including Chinese, English, Japanese, and Korean。
* **Advanced Text Processing**: Optimized auto-chunking logic allows for the seamless synthesis of extremely long articles。
* **Automated Subtitle Pipeline**: Automatically generates four subtitle formats (SRT, Readable, Word-level, and Shorts/Reels optimized) alongside your audio。

---

### 🛠️ 技术架构 / Technical Stack
* **核心模型 (Core Model)**: Qwen3-TTS-12Hz (0.6B / 1.7B)
* **前端界面 (Interface)**: Gradio
* **音频处理 (Audio Engine)**: Pydub & Soundfile
