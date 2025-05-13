import os
import threading
import requests
import dashscope
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor
from ali_key import KEY

class MultiThreadASR:
    def __init__(self, texts, save_dir, voice='Serena'):
        self.texts = texts
        self.save_dir = save_dir
        self.voice = voice
        os.makedirs(save_dir, exist_ok=True)

    def _synthesize(self, text, index):
        try:
            response = dashscope.audio.qwen_tts.SpeechSynthesizer.call(
                model="qwen-tts",
                api_key=KEY,
                text=text,
                voice=self.voice,
            )
            audio_url = response.output.audio["url"]
            r = requests.get(audio_url)
            r.raise_for_status()
            save_path = os.path.join(self.save_dir, f"audio_{index:03d}.wav")
            with open(save_path, 'wb') as f:
                f.write(r.content)
            print(f"[Ali-TTS] 保存：{save_path}")
        except Exception as e:
            print(f"[Ali-TTS] 线程 {index} 合成失败：{e}")

    def run(self):
        threads = []
        for i, text in enumerate(self.texts):
            t = threading.Thread(target=self._synthesize, args=(text, i))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        print("[Ali-TTS] 所有线程完成")

class MultiProcessVoiceFeature:
    def __init__(self, audio_paths):
        self.audio_paths = audio_paths

    def _extract_features(self, path):
        try:
            y, sr = librosa.load(path, sr=None)
            pitch = librosa.yin(y, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C7'))
            rms = librosa.feature.rms(y=y).flatten()
            onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
            duration = len(y) / sr
            wps = len(onsets) / duration
            return {
                "file": os.path.basename(path),
                "mean_pitch": float(np.mean(pitch)),
                "mean_rms": float(np.mean(rms)),
                "wps": wps,
                "duration": duration
            }
        except Exception as e:
            print(f"[Feature] 失败：{path}，错误：{e}")
            return None

    def run(self):
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(self._extract_features, self.audio_paths))
        features = [r for r in results if r]
        for f in features:
            print(f)
        return features



try:
    from kokoro import KPipeline, KModel

    class KokoroBatchTTS:
        def __init__(self, voice='zf_001', device='cpu'):
            self.voice = voice
            self.device = device
            self.model = KModel(repo_id='hexgrad/Kokoro-82M-v1.1-zh').to(device).eval()
            self.pipeline = KPipeline(lang_code='z', repo_id='hexgrad/Kokoro-82M-v1.1-zh', model=self.model)

        def synthesize_batch(self, texts, save_dir):
            os.makedirs(save_dir, exist_ok=True)
            for i, text in enumerate(texts):
                generator = self.pipeline(text, voice=self.voice)
                for j, (_, _, audio) in enumerate(generator):
                    path = os.path.join(save_dir, f'kokoro_{i:03d}_{j:02d}.wav')
                    sf.write(path, audio, 24000)
                    print(f"[Kokoro-TTS] 保存：{path}")
            print("[Kokoro-TTS] 合成完成")
except ImportError:
    print("[Kokoro-TTS] 未检测到 kokoro 安装，如需本地合成请先安装。")


def main():
    
    # 示例文字数据
    texts = [
        "大家好，欢迎来到我的播客，本集将与大家分享《人类群星闪耀时》。",
        "一个人生命中最大的幸运，莫过于在年富力强时发现了自己的使命。",
        "一个伟大的人或一部伟大的作品通常是很难被同时代的人一眼识出的。"
    ]

    # ① 使用阿里 TTS 多线程合成音频
    ali_output_dir = 'audios_qwen'
    print("[Main] 开始多线程 Ali-TTS 合成")
    asr = MultiThreadASR(texts, ali_output_dir)
    asr.run()

    # ② 多进程提取声音特征
    audio_files = [os.path.join(ali_output_dir, f) for f in os.listdir(ali_output_dir) if f.endswith('.wav')]
    print("[Main] 开始多进程声音特征提取")
    vf = MultiProcessVoiceFeature(audio_files)
    features = vf.run()
 
    # ③ 本地 Kokoro 模型合成
    use_kokoro = True
    if use_kokoro:
        kokoro_output_dir = r"D:\未完成作业\计算机\python-\week12\audios_kokoro"
        print("[Main] 开始使用 Kokoro 本地合成")
        kokoro = KokoroBatchTTS()
        kokoro.synthesize_batch(texts, kokoro_output_dir)


if __name__ == '__main__':
    main()
