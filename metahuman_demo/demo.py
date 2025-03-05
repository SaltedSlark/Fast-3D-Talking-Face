from numba.cuda import args
import numpy as np
import time
import socket
from data2json import datas2jsonframes
import requests
import librosa
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess the VOCA dataset and generate ARKit blendshapes"
    )
    parser.add_argument(
        "--audio2face_url",
        type=str,
        default="http://0.0.0.0:3392",
    )
    parser.add_argument(
        "--wav_path",
        type=str,
        default= "../test/wav/speech_long.wav",
    )
    parser.add_argument(
        "--mood",
        type=str,
        default= "0",
    )
    parser.add_argument(
        "--livelink_host",
        type=str,
        default="192.168.51.119",
    )
    parser.add_argument(
        "--livelink_port",
        type=str,
        default="1234",
    )
    args = parser.parse_args()
    audio_file = librosa.load(args.wav_path, sr=16000)[0]
    response = requests.post(args.audio2face_url, json={'file': audio_file.tolist(),'mood':int(args.mood)}).json()
    topic = 'iPhoneBlack'
    datas = np.array(response["bs"])

    frames = datas2jsonframes(topic, datas)
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.connect((args.livelink_host, int(args.livelink_port)))
    frame_duration = 1/30
    for item in frames:
        client.send(item.encode())
        time.sleep(frame_duration)
    client.close()
if __name__ == '__main__':
    main()
