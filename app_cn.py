# %cd /content/Qwen3-TTS-Colab
from subtitle import subtitle_maker
from process_text import text_chunk
from qwen_tts import Qwen3TTSModel
import subprocess
import os
import gradio as gr
import numpy as np
import torch
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import split_on_silence
from huggingface_hub import snapshot_download
from hf_downloader import download_model
import gc 
from huggingface_hub import login

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
  HF_TOKEN=None

# 全局模型持有者
loaded_models = {}
MODEL_SIZES = ["0.6B", "1.7B"]

# 说话人和语言选项
SPEAKERS = [
    "Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"
]
# 界面显示中文，但传给后端的参数保持英文/标准格式
LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish", "Portuguese", "Russian"]

# --- 辅助函数 ---

def get_model_path(model_type: str, model_size: str) -> str:
    """根据类型和大小获取模型路径。"""
    try:
      return snapshot_download(f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}")
    except Exception as e:
      return download_model(f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}", download_folder="./qwen_tts_model", redownload= False)

def clear_other_models(keep_key=None):
    """删除除当前模型外的所有已加载模型。"""
    global loaded_models
    keys_to_delete = [k for k in loaded_models if k != keep_key]
    for k in keys_to_delete:
        try:
            del loaded_models[k]
        except Exception:
            pass
    for k in keys_to_delete:
        loaded_models.pop(k, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_model(model_type: str, model_size: str):
    """加载模型并清理其他模型以避免 Colab OOM。"""
    global loaded_models
    key = (model_type, model_size)
    if key in loaded_models:
        return loaded_models[key]
    
    clear_other_models(keep_key=key)
    model_path = get_model_path(model_type, model_size)
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map="cuda",
        dtype=torch.bfloat16,
    )
    loaded_models[key] = model
    return model

def _normalize_audio(wav, eps=1e-12, clip=True):
    """归一化音频。"""
    x = np.asarray(wav)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"不支持的数据类型: {x.dtype}")
    if clip:
        y = np.clip(y, -1.0, 1.0)
    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)
    return y

def _audio_to_tuple(audio):
    """转换 Gradio 音频输入。"""
    if audio is None: return None
    if isinstance(audio, str):
        try:
            wav, sr = sf.read(audio)
            wav = _normalize_audio(wav)
            return wav, int(sr)
        except Exception as e:
            print(f"读取音频文件出错: {e}")
            return None
    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = _normalize_audio(wav)
        return wav, int(sr)
    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr
    return None

def transcribe_reference(audio_path, mode_input, language="English"):
    """使用 subtitle_maker 提取参考音频文本。"""
    should_run = False
    if isinstance(mode_input, bool): should_run = mode_input
    elif isinstance(mode_input, str) and ("高质量" in mode_input or "High-Quality" in mode_input): should_run = True

    if not audio_path or not should_run: return gr.update()
    
    print(f"开始转录: {audio_path}")
    src_lang = language if language != "Auto" else "English"
    try:
        results = subtitle_maker(audio_path, src_lang)
        transcript = results[7]
        return transcript if transcript else "未能识别语音。"
    except Exception as e:
        print(f"转录错误: {e}")
        return f"转录过程中出错: {str(e)}"

# --- 音频处理工具 ---

def remove_silence_function(file_path, minimum_silence=100):
    """移除音频静音。"""
    try:
        output_path = file_path.replace(".wav", "_no_silence.wav")
        sound = AudioSegment.from_wav(file_path)
        audio_chunks = split_on_silence(sound,
                                        min_silence_len=minimum_silence,
                                        silence_thresh=-45,
                                        keep_silence=50)
        combined = AudioSegment.empty()
        for chunk in audio_chunks:
            combined += chunk
        combined.export(output_path, format="wav")
        return output_path
    except Exception as e:
        print(f"移除静音出错: {e}")
        return file_path

def process_audio_output(audio_path, make_subtitle, remove_silence, language="Auto"):
    """处理静音移除和字幕生成。"""
    final_audio_path = audio_path
    if remove_silence:
        final_audio_path = remove_silence_function(audio_path)
    
    default_srt, custom_srt, word_srt, shorts_srt = None, None, None, None
    if make_subtitle:
        try:
            results = subtitle_maker(final_audio_path, language)
            default_srt = results[0]
            custom_srt = results[1]
            word_srt = results[2]
            shorts_srt = results[3]
        except Exception as e:
            print(f"字幕生成错误: {e}")

    return final_audio_path, default_srt, custom_srt, word_srt, shorts_srt

def stitch_chunk_files(chunk_files,output_filename):
    """拼接音频分片并清理临时文件。"""
    if not chunk_files:
        return None

    combined_audio = AudioSegment.empty()
    print(f"正在拼接 {len(chunk_files)} 个音频文件...")
    for f in chunk_files:
        try:
            segment = AudioSegment.from_wav(f)
            combined_audio += segment
        except Exception as e:
            print(f"拼接分片 {f} 出错: {e}")

    combined_audio.export(output_filename, format="wav")
    
    for f in chunk_files:
        try:
            if os.path.exists(f):
                os.remove(f)
        except Exception as e:
            print(f"警告: 无法删除临时文件 {f}: {e}")
            
    return output_filename

# --- 生成逻辑 ---

def generate_voice_design(text, language, voice_description, remove_silence, make_subs):
    if not text or not text.strip(): return None, "错误: 文本不能为空。", None, None, None, None
    
    try:
        text_chunks, tts_filename = text_chunk(text, language, char_limit=280)
        chunk_files = []
        tts = get_model("VoiceDesign", "1.7B")

        for i, chunk in enumerate(text_chunks):
            wavs, sr = tts.generate_voice_design(
                text=chunk.strip(),
                language=language,
                instruct=voice_description.strip(),
                non_streaming_mode=True,
                max_new_tokens=2048,
            )
            temp_filename = f"temp_chunk_{i}_{os.getpid()}.wav"
            sf.write(temp_filename, wavs[0], sr)
            chunk_files.append(temp_filename)
            del wavs
            torch.cuda.empty_cache()
            gc.collect()
        
        stitched_file = stitch_chunk_files(chunk_files,tts_filename)
        final_audio, srt1, srt2, srt3, srt4 = process_audio_output(stitched_file, make_subs, remove_silence, language)
        return final_audio, "生成成功！", srt1, srt2, srt3, srt4
    except Exception as e:
        return None, f"错误: {e}", None, None, None, None

def generate_custom_voice(text, language, speaker, instruct, model_size, remove_silence, make_subs):
    if not text or not text.strip(): return None, "错误: 文本不能为空。", None, None, None, None
    
    try:
        text_chunks, tts_filename = text_chunk(text, language, char_limit=280)
        chunk_files = []
        tts = get_model("CustomVoice", model_size)
        formatted_speaker = speaker.lower().replace(" ", "_")

        for i, chunk in enumerate(text_chunks):
            wavs, sr = tts.generate_custom_voice(
                text=chunk.strip(),
                language=language,
                speaker=formatted_speaker,
                instruct=instruct.strip() if instruct else None,
                non_streaming_mode=True,
                max_new_tokens=2048,
            )
            temp_filename = f"temp_custom_{i}_{os.getpid()}.wav"
            sf.write(temp_filename, wavs[0], sr)
            chunk_files.append(temp_filename)
            del wavs
            torch.cuda.empty_cache()
            gc.collect()
            
        stitched_file = stitch_chunk_files(chunk_files,tts_filename)
        final_audio, srt1, srt2, srt3, srt4 = process_audio_output(stitched_file, make_subs, remove_silence, language)
        return final_audio, "生成成功！", srt1, srt2, srt3, srt4
    except Exception as e:
        return None, f"错误: {e}", None, None, None, None

def smart_generate_clone(ref_audio, ref_text, target_text, language, mode, model_size, remove_silence, make_subs):
    if not target_text or not target_text.strip(): return None, "错误: 目标文本不能为空。", None, None, None, None
    if not ref_audio: return None, "错误: 缺少参考音频。", None, None, None, None

    use_xvector_only = ("快速" in mode or "Fast" in mode)
    final_ref_text = ref_text
    audio_tuple = _audio_to_tuple(ref_audio)

    if not use_xvector_only:
        if not final_ref_text or not final_ref_text.strip():
            print("自动转录参考音频...")
            try:
                final_ref_text = transcribe_reference(ref_audio, True, language)
                if not final_ref_text or "错误" in final_ref_text or "Error" in final_ref_text:
                     return None, f"转录失败: {final_ref_text}", None, None, None, None
            except Exception as e:
                return None, f"转录错误: {e}", None, None, None, None
    else:
        final_ref_text = None

    try:
        text_chunks, tts_filename = text_chunk(target_text, language, char_limit=280)
        chunk_files = []
        tts = get_model("Base", model_size)

        for i, chunk in enumerate(text_chunks):
            wavs, sr = tts.generate_voice_clone(
                text=chunk.strip(),
                language=language,
                ref_audio=audio_tuple,
                ref_text=final_ref_text.strip() if final_ref_text else None,
                x_vector_only_mode=use_xvector_only,
                max_new_tokens=2048,
            )
            temp_filename = f"temp_clone_{i}_{os.getpid()}.wav"
            sf.write(temp_filename, wavs[0], sr)
            chunk_files.append(temp_filename)
            del wavs
            torch.cuda.empty_cache()
            gc.collect()

        stitched_file = stitch_chunk_files(chunk_files,tts_filename)
        final_audio, srt1, srt2, srt3, srt4 = process_audio_output(stitched_file, make_subs, remove_silence, language)
        return final_audio, f"成功！模式: {mode}", srt1, srt2, srt3, srt4
    except Exception as e:
        return None, f"错误: {e}", None, None, None, None


# --- 界面构建 ---

def on_mode_change(mode):
    return gr.update(visible=("高质量" in mode or "High-Quality" in mode))

def build_ui():
    theme = gr.themes.Soft(font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"])
    css = ".gradio-container {max-width: none !important;} .tab-content {padding: 20px;}"

    with gr.Blocks(theme=theme, css=css, title="Qwen3-TTS 演示") as demo:
        gr.HTML("""
        <div style="text-align: center; margin: 20px auto; max-width: 800px;">
            <h1 style="font-size: 2.5em; margin-bottom: 5px;">🎙️ Qwen3-TTS </h1>
            <a href="https://www.youtube.com/@fm19.2?sub_confirmation=1" target="_blank" style="display: inline-block; padding: 10px 20px; background-color: #4285F4; color: white; border-radius: 6px; text-decoration: none; font-size: 1em;">🥳 更多 Colab 脚本</a>
        </div>""")

        with gr.Tabs():
            # --- Tab 1: 声音设计 ---
            with gr.Tab("声音设计 (Voice Design)"):
                with gr.Row():
                    with gr.Column(scale=2):
                        design_text = gr.Textbox(label="待合成文本", lines=4, value="在最上面的抽屉里... 等等，是空的？怎么可能！我确定我把它放进去了！",
                                                 placeholder="请输入您想转换成语音的文字...")
                        design_language = gr.Dropdown(label="语言", choices=LANGUAGES, value="Auto")
                        design_instruct = gr.Textbox(label="声音特质描述", lines=3,  placeholder="描述您想要的声音特征，例如：语气、情感...",
                            value="用一种怀疑的语气说话，声音中带着一点开始蔓延的恐慌感。")
                        design_btn = gr.Button("设计并生成语音", variant="primary")
                        with gr.Accordion("更多选项", open=False):
                            with gr.Row():
                              design_rem_silence = gr.Checkbox(label="自动移除静音", value=False)
                              design_make_subs = gr.Checkbox(label="生成配套字幕", value=False)

                    with gr.Column(scale=2):
                        design_audio_out = gr.Audio(label="生成音频预览", type="filepath")
                        design_status = gr.Textbox(label="状态信息", interactive=False)
                        
                        with gr.Accordion("📝 字幕文件", open=False):
                            with gr.Row():
                                d_srt1 = gr.File(label="原始 (Whisper)")
                                d_srt2 = gr.File(label="易读格式")
                            with gr.Row():
                                d_srt3 = gr.File(label="逐词对齐")
                                d_srt4 = gr.File(label="短视频/Reels格式")

                design_btn.click(
                    generate_voice_design, 
                    inputs=[design_text, design_language, design_instruct, design_rem_silence, design_make_subs], 
                    outputs=[design_audio_out, design_status, d_srt1, d_srt2, d_srt3, d_srt4]
                )

            # --- Tab 2: 声音克隆 ---
            with gr.Tab("声音克隆 (Voice Clone)"):
                with gr.Row():
                    with gr.Column(scale=2):
                        clone_target_text = gr.Textbox(label="目标文本", lines=3, placeholder="输入您想让克隆声音说的话...")
                        clone_ref_audio = gr.Audio(label="参考音频 (上传您想要克隆的人声样本)", type="filepath")
                        
                        with gr.Row():
                            clone_language = gr.Dropdown(label="语言", choices=LANGUAGES, value="Auto",scale=1)
                            clone_model_size = gr.Dropdown(label="模型规模", choices=MODEL_SIZES, value="1.7B",scale=1)
                            clone_mode = gr.Dropdown(
                                label="克隆模式",
                                choices=["高质量 (音频 + 文本转录)", "快速 (仅需音频)"],
                                value="高质量 (音频 + 文本转录)",
                                interactive=True,
                                scale=2
                            )
                        
                        clone_ref_text = gr.Textbox(label="参考文本 (转录内容)", lines=2, visible=True)
                        clone_btn = gr.Button("开始克隆并生成", variant="primary")
                        with gr.Accordion("更多选项", open=False):
                            with gr.Row():
                              clone_rem_silence = gr.Checkbox(label="自动移除静音", value=False)
                              clone_make_subs = gr.Checkbox(label="生成配套字幕", value=False)

                    with gr.Column(scale=2):
                        clone_audio_out = gr.Audio(label="生成音频预览", type="filepath")
                        clone_status = gr.Textbox(label="状态信息", interactive=False)
                        
                        with gr.Accordion("📝 字幕文件", open=False):
                            with gr.Row():
                                c_srt1 = gr.File(label="原始字幕")
                                c_srt2 = gr.File(label="易读字幕")
                            with gr.Row():
                                c_srt3 = gr.File(label="逐词字幕")
                                c_srt4 = gr.File(label="短视频字幕")

                clone_mode.change(on_mode_change, inputs=[clone_mode], outputs=[clone_ref_text])
                clone_ref_audio.change(transcribe_reference, inputs=[clone_ref_audio, clone_mode, clone_language], outputs=[clone_ref_text])
                
                clone_btn.click(
                    smart_generate_clone,
                    inputs=[clone_ref_audio, clone_ref_text, clone_target_text, clone_language, clone_mode, clone_model_size, clone_rem_silence, clone_make_subs],
                    outputs=[clone_audio_out, clone_status, c_srt1, c_srt2, c_srt3, c_srt4]
                )

            # --- Tab 3: 标准 TTS ---
            with gr.Tab("标准 TTS (CustomVoice)"):
                with gr.Row():
                    with gr.Column(scale=2):
                        tts_text = gr.Textbox(label="文本内容", lines=4,   placeholder="输入您想转换成语音的文字...",
                            value="您好！欢迎使用文本转语音系统。这是 Qwen3-TTS 的功能演示。")
                        with gr.Row():
                            tts_language = gr.Dropdown(label="语言", choices=LANGUAGES, value="Chinese")
                            tts_speaker = gr.Dropdown(label="角色/说话人", choices=SPEAKERS, value="Ryan")
                        with gr.Row():
                            tts_instruct = gr.Textbox(label="风格指令 (可选)", lines=2,placeholder="例如：用轻快、充满活力的语气说话")
                            tts_model_size = gr.Dropdown(label="模型规模", choices=MODEL_SIZES, value="1.7B")
                        tts_btn = gr.Button("生成语音", variant="primary")
                        with gr.Accordion("更多选项", open=False):
                            with gr.Row():
                              tts_rem_silence = gr.Checkbox(label="自动移除静音", value=False)
                              tts_make_subs = gr.Checkbox(label="生成配套字幕", value=False)

                    with gr.Column(scale=2):
                        tts_audio_out = gr.Audio(label="生成音频预览", type="filepath")
                        tts_status = gr.Textbox(label="状态信息", interactive=False)
                        
                        with gr.Accordion("📝 字幕文件", open=False):
                            with gr.Row():
                                t_srt1 = gr.File(label="原始字幕")
                                t_srt2 = gr.File(label="易读字幕")
                            with gr.Row():
                                t_srt3 = gr.File(label="逐词字幕")
                                t_srt4 = gr.File(label="短视频字幕")

                tts_btn.click(
                    generate_custom_voice, 
                    inputs=[tts_text, tts_language, tts_speaker, tts_instruct, tts_model_size, tts_rem_silence, tts_make_subs], 
                    outputs=[tts_audio_out, tts_status, t_srt1, t_srt2, t_srt3, t_srt4]
                )

            # --- Tab 4: 关于 ---
            with gr.Tab("关于项目"):
                gr.Markdown("""
                # Qwen3-TTS 
                一个集成了三种强大模式的统一文本转语音演示界面：
                - **声音设计 (Voice Design)**：通过自然语言描述创建自定义声音。
                - **声音克隆 (Voice Clone)**：仅需一段参考音频即可克隆任何声音。
                - **标准 TTS (CustomVoice)**：使用预设的说话人角色，支持可选的风格指令。

                基于阿里巴巴 Qwen 团队的 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 项目构建。
                """)

                gr.HTML("""
                <hr>
                <p style="color: red; font-weight: bold; font-size: 16px;">
                ⚠️ 注意
                </p>
                <p>
                此 Gradio 界面非 Qwen3-TTS 官方项目，而是基于官方 Demo 改进的 Colab 版本：<br>
                <a href="https://huggingface.co/spaces/Qwen/Qwen3-TTS" target="_blank">
                https://huggingface.co/spaces/Qwen/Qwen3-TTS
                </a>
                </p>

                <p><b>增强功能：</b></p>
                <ul>
                  <li>支持通过 whisper-large-v3-turbo 自动转录参考音频文本。</li>
                  <li>支持超长文本分段处理。</li>
                  <li>集成字幕生成功能（SRT、逐词对齐、短视频格式）。</li>
                </ul>
                """)

    return demo

import click
@click.command()
@click.option("--debug", is_flag=True, default=False, help="开启调试模式。")
@click.option("--share", is_flag=True, default=False, help="开启公网分享。")
def main(share,debug):
    demo = build_ui()
    demo.queue().launch(share=share,debug=debug)

if __name__ == "__main__":
    main()