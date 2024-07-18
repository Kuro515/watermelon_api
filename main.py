from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import librosa
import base64
import uuid
import os

app = FastAPI()

scores = {}


# def mp3_to_base64(file_path):
#     with open(file_path, "rb") as mp3_file:
#         base64_bytes = base64.b64encode(mp3_file.read())
#         base64_string = base64_bytes.decode("utf-8")
#     print(base64_string)


# def base64_to_mp3(base64_string, output_file_path):
#     with open(output_file_path, "wb") as mp3_file:
#         mp3_file.write(base64.b64decode(base64_string))

#     judge_suica(output_file_path, 1)


def urlsafe_base64_to_mp3(base64_string, output_file_path):
    try:
        mp3_bytes = base64.urlsafe_b64decode(base64_string)
        with open(output_file_path, "wb") as mp3_file:
            mp3_file.write(mp3_bytes)
        return judge_suica(output_file_path, 1)
        # return True
    except Exception as e:
        print(f"Error decoding Base64: {e}")
        return e


def judge_suica(filename, number):
    y, sr = librosa.load(filename)
    # 短時間エネルギーの計算
    frame_length = 2048
    hop_length = 512
    energy = np.array(
        [sum(abs(y[i : i + frame_length] ** 2)) for i in range(0, len(y), hop_length)]
    )

    # エネルギーの正規化
    max_energy = np.max(energy)
    normalized_energy = energy / max_energy if max_energy != 0 else energy

    # 音の弾みを判別するためのしきい値を設定
    threshold = (np.mean(normalized_energy) + 2 * np.std(normalized_energy)) / 7

    # 弾みの判定
    bounce = normalized_energy > threshold

    # 弾みの強度の平均値を計算
    bounce_strengths = normalized_energy[bounce]
    average_bounce_strength = (
        np.mean(bounce_strengths) if bounce_strengths.size > 0 else 0
    )

    # envelope = np.abs(librosa.onset.onset_strength(y=y, sr=sr))
    # duration = np.sum(envelope > 0.01) / sr  # 持続時間を計算
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    avg_magnitude = np.mean(S, axis=1)
    dominant_freq = round(freqs[np.argmax(avg_magnitude)], 3)

    bounce_score = round((average_bounce_strength * 100), 3)

    # print(f"No.{number}", f"Bounce: {bounce_score}Points", f"{dominant_freq}Hz")

    score = round(bounce_score + dominant_freq / 10, 3)
    # print(f"Score:{score}")

    # scores[number] = score
    return score


class Sound(BaseModel):
    b6: str


# @app.get("/sounds/")
# async def read_item(b6: str):
#     if b6 == None:
#         return {"Score": "Error"}
#     else:
#         return {"Score": urlsafe_base64_to_mp3(b6, "tmp.mp3")}


@app.post("/sounds/")
async def read_item(sound: Sound):
    tmp_file_path = os.path.join("/tmp", f"{uuid.uuid4()}.mp3")
    score = urlsafe_base64_to_mp3(sound.b6, tmp_file_path)
    if score is None:
        return {"Score": "Error decoding Base64"}
    return {"Score": score}
