# Borrowed from https://github.com/EvelynFan/FaceFormer/blob/main/data_loader.py
import os
import torch
from collections import defaultdict
from torch.utils import data
import numpy as np
import pickle
from tqdm import tqdm
from transformers import Wav2Vec2Processor
import librosa
from concurrent.futures import ProcessPoolExecutor, as_completed


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, data_type="train"):
        self.data = data
        self.len = len(self.data)
        self.data_type = data_type

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        bs = self.data[index]["bs"]
        template = self.data[index]["template"]
        emotion_label = self.data[index]["emo"]
        return torch.FloatTensor(audio), torch.FloatTensor(bs), torch.FloatTensor(template), file_name,torch.tensor(emotion_label[0])

    def __len__(self):
        return self.len

def process_file(args, processor, audio_path, bs_path, f):
    data = {}
    if f.endswith("wav"):
        wav_path = os.path.join(audio_path, f)
        speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
        input_values = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
        key = f.replace("wav", "npy")
        data["audio"] = input_values
        temp = np.zeros((1, args.bs_dim))
        data["name"] = f
        data["template"] = temp.reshape((-1))
        cur_bs_path = os.path.join(bs_path, f.replace("wav", "npy"))
        if not os.path.exists(cur_bs_path):
            return None
        else:
            data["bs"] = np.load(cur_bs_path, allow_pickle=True)
        if "joy" in key:
            data["emo"] = np.array([1])
        elif "anger" in key:
            data["emo"] = np.array([2])
        elif "disgust" in key:
            data["emo"] = np.array([3])
        elif "pain" in key:
            data["emo"] = np.array([4])
        else:
            data['emo'] = np.array([0])
        return key, data
    return None

def read_data(args):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    audio_path = os.path.join(args.dataset, args.wav_path)
    bs_path = os.path.join(args.dataset, args.bs_path)
    processor = Wav2Vec2Processor.from_pretrained("./distil-wav2vec2/")

    # Use a process pool to parallelize file processing
    with ProcessPoolExecutor() as executor:
        futures = []
        for r, ds, fs in os.walk(audio_path):
            for f in fs:
                futures.append(executor.submit(process_file, args, processor, r, bs_path, f))
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                key, value = result
                data[key] = value

    count = 0
    for _, v in data.items():
        count += 1
        ratio = count / len(data)
        if ratio <= 0.8:
            train_data.append(v)
        elif ratio <=0.9:
            valid_data.append(v)
        else:
            test_data.append(v)

    print('Loaded data: Train-{}, Val-{}, Test-{}'.format(len(train_data), len(valid_data), len(test_data)))
    return train_data, valid_data, test_data

def get_dataloaders(args):
    dataset = {}
    train_data, valid_data, test_data = read_data(args)
    train_data = Dataset(train_data,  "train")
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    valid_data = Dataset(valid_data,"val")
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False)
    test_data = Dataset(test_data, "test")
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    return dataset


if __name__ == "__main__":
    get_dataloaders()
