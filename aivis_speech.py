import io
import os
import logging
import httpx
import sounddevice as sd
import soundfile as sf
import numpy as np
from urllib.parse import urlencode

# ロギング設定
logging.basicConfig(level=logging.INFO)

# 出力ファイル
output_pcm = os.path.abspath("output.pcm")  # PCMファイルの保存先
output_wav = os.path.abspath("output.wav")  # 最終的なWAVファイルの保存先

# サンプルレートをグローバル変数で記録
global_sample_rate = 44100  # デフォルト値


async def create_query(text: str, speaker: str) -> dict:
    """
    音声合成用のクエリデータを生成する関数。
    """
    logging.info(f"create_query start, text: {text}")
    base_url = "http://127.0.0.1:10101/audio_query"
    query_params = {"text": text, "speaker": speaker}
    headers = {"accept": "application/json"}
    timeout = 30

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            url = f"{base_url}?{urlencode(query_params)}"
            response = await client.post(url, headers=headers)
            response.raise_for_status()
            logging.info(f"create_query succeeded, status code: {response.status_code}")
            return response.json()
    except httpx.HTTPError as e:
        logging.error(f"create_query failed with error: {e}")
        return {}


async def save_pcm_and_play(text: str, speaker: str, output_pcm: str):
    """
    音声合成APIからデータを取得し、PCM形式で保存しつつ再生する関数。
    """
    global global_sample_rate  # グローバル変数でサンプルレートを記録
    logging.info(f"save_pcm_and_play start, text: {text}")
    base_url = "http://127.0.0.1:10101/synthesis"
    headers = {"accept": "audio/wav", "Content-Type": "application/json"}
    timeout = 60

    # クエリデータを生成
    query_data = await create_query(text, speaker)
    if not query_data:
        logging.error("Failed to create query data.")
        return {"status": "error", "message": "Failed to create query data"}

    try:
        # 音声合成APIを呼び出し
        async with httpx.AsyncClient(timeout=timeout) as client:
            url = f"{base_url}?{urlencode({'speaker': speaker})}"
            response = await client.post(url, headers=headers, json=query_data)
            response.raise_for_status()
            logging.info(
                f"Synthesis API call succeeded, status code: {response.status_code}"
            )

            # バッファリングして音声データをメモリ上に保持
            audio_buffer = io.BytesIO(response.content)
            audio_data, sample_rate = sf.read(
                audio_buffer, dtype="int16"
            )  # PCM形式で読み込む

            # サンプルレートを記録
            if global_sample_rate is None:
                global_sample_rate = sample_rate
                logging.info(f"Sample rate set to {global_sample_rate} Hz")
            elif global_sample_rate != sample_rate:
                raise ValueError("Inconsistent sample rate detected!")

            # PCMデータを追記保存
            with open(output_pcm, "ab") as file:
                file.write(audio_data.tobytes())
                logging.info(f"PCM data appended to {output_pcm}")

            # 音声再生
            logging.info("Starting audio playback...")
            sd.play(audio_data, samplerate=sample_rate)
            sd.wait()  # 再生が終わるまで待機
            logging.info("Audio playback complete")
            return {"status": "ok"}
    except httpx.HTTPError as e:
        logging.error(f"Failed to call synthesis API: {e}")
        return {"status": "error", "message": "API call failed"}
    except Exception as e:
        logging.error(f"Error during playback: {e}")
        return {"status": "error", "message": "Playback failed"}


def convert_pcm_to_wav(pcm_file: str, output_wav: str, channels: int = 1):
    """
    PCMデータをWAV形式に変換する関数。
    """
    global global_sample_rate
    if global_sample_rate is None:
        raise ValueError("Sample rate not set. Ensure audio data has been processed.")

    with open(pcm_file, "rb") as f:
        # PCMデータを読み込み
        pcm_data = np.frombuffer(f.read(), dtype=np.int16)

    # モノラルの場合は2次元配列に整形
    if channels == 1:
        pcm_data = pcm_data.reshape(-1, 1)

    # WAVファイルとして保存
    sf.write(output_wav, pcm_data, samplerate=global_sample_rate)
    logging.info(f"Converted {pcm_file} to {output_wav}")


if __name__ == "__main__":
    import asyncio

    # texts = [
    #     "こんにちは、リアルタイム音声合成のテストです。",
    #     "この音声はPCM形式で保存されています。",
    #     "最後にWAV形式に変換します。",
    # ]
    # speaker = "888753762"

    # # 各テキストを処理し、PCMファイルに追記
    # for text in texts:
    #     asyncio.run(save_pcm_and_play(text, speaker, output_pcm))

    # PCMファイルをWAV形式に変換
    convert_pcm_to_wav("output.pcm", "output.wav", channels=1)
    logging.info("Processing completed.")
