import queue
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import os

# cudnn の dll を明示的にロードする
# これをしないと cudnn のエラーが発生する
os.add_dll_directory(
    "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\bin"
)


model = WhisperModel(
    model_size_or_path="deepdml/faster-whisper-large-v3-turbo-ct2",
    device="cuda",
    compute_type="float32",
)

fs = 16000
selected_device_index = 1  # 環境に合わせて変更
audio_queue = queue.Queue()


def callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())


# ストリーム開始
stream = sd.InputStream(
    samplerate=fs, channels=1, callback=callback, device=selected_device_index
)
stream.start()

print("Recording... Press Ctrl+C to stop.")

try:
    buffer_size = fs * 3  # 2秒分のはず
    audio_buffer = np.array([], dtype=np.float32)

    while True:
        # queue からフレームを取り出しバッファに追加
        chunk = audio_queue.get()
        chunk = chunk.flatten().astype(np.float32)
        audio_buffer = np.concatenate((audio_buffer, chunk))

        # バッファが十分溜まったら transcribe
        if len(audio_buffer) >= buffer_size:
            segments, info = model.transcribe(
                audio_buffer,
                beam_size=15,
                language="ja",
                vad_filter=True,
                temperature=0.1,
            )

            for segment in segments:
                print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

            # 一度使ったバッファは空にする
            audio_buffer = np.array([], dtype=np.float32)

except KeyboardInterrupt:
    print("Stopped.")
finally:
    stream.stop()
    stream.close()
