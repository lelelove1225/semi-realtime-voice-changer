import io
import logging
import httpx
import sounddevice as sd
import soundfile as sf
from urllib.parse import urlencode

# ロギング設定
logging.basicConfig(level=logging.INFO)


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


async def get_wav_and_play(text: str, speaker: str):
    """
    音声合成APIからデータを取得し、リアルタイムで再生する関数。
    """
    logging.info(f"get_wav_and_play start, text: {text}")
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
            audio_data, sample_rate = sf.read(audio_buffer)  # 音声データを読み込む

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


# テスト用の関数
if __name__ == "__main__":
    import asyncio

    logging.info("Starting aivis_speech test...")
    text = "こんにちは、リアルタイム音声合成のテストです。"
    speaker = "888753762"

    asyncio.run(get_wav_and_play(text, speaker))
    # logging.info("Test completed.")
