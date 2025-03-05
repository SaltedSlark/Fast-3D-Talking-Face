import numpy as np
import os, argparse
from SelfTalk_student import SelfTalk
from transformers import Wav2Vec2Processor
import torch
import time
from fastapi import FastAPI
import uvicorn
app = FastAPI()

os.environ['PYOPENGL_PLATFORM'] = 'egl'  # egl

parser = argparse.ArgumentParser(
    description='SelfTalk: A Self-Supervised Commutative Training Diagram to Comprehend 3D Talking Faces')
parser.add_argument("--model_name", type=str, default="save_10_11_16_13/5_model")
parser.add_argument("--dataset", type=str, default="nv_a2f_data")
parser.add_argument("--feature_dim", type=int, default=128, help='128 for lite')
parser.add_argument("--bs_dim", type=int, default=52 ,
                    help='number of blendshape')
parser.add_argument("--num_emotions", type=int, default=5,
                    help='number of emotions')
parser.add_argument("--emotion_emb_dim", type=int, default=10,
                    help='number of emotion embedding')
parser.add_argument("--device", type=str, default="cuda", help='cuda or cpu')
args = parser.parse_args()

# build model
model = SelfTalk(args)
model.load_state_dict(torch.load(os.path.join(args.dataset, '{}.pth'.format(args.model_name)),
                                 map_location='cpu'),strict=False)
model = model.half().cuda()
model.eval()
processor = Wav2Vec2Processor.from_pretrained("./distil-wav2vec2/")
torch.cuda.empty_cache()

def replace_columns_with_mean(array, column_pairs):
    """
    用于将指定列对替换为它们的均值。

    参数：
    - array: 要操作的 NumPy 数组。
    - column_pairs: 一个包含列索引对的列表，每个对表示需要计算均值的列。

    返回：
    - 修改后的数组。
    """
    for col1, col2 in column_pairs:
        mean_values = array[:, [col1, col2]].mean(axis=1)
        array[:, col1] = mean_values
        array[:, col2] = mean_values
    return array

from pydantic import BaseModel

class Item(BaseModel):
    file:list
    mood:int

@app.post('/')
@torch.no_grad()
def test_model(item:Item):
    start = time.time()
    speech_array = np.array(item.file)
    audio_feature = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
    audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
    audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)
    mood = torch.tensor([item.mood]).to(device=args.device)
    print("input process time: ",time.time() - start)
    start = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            prediction = model.predict(audio_feature.half(),mood)
    prediction_cpu = prediction.squeeze().detach().cpu().numpy().reshape(-1,52)
    del prediction
    # 嘴部左右对称
    column_pairs = [(23, 24), (25, 26), (27, 28), (29, 30), (31, 32), (33, 34), (35, 36), (37, 38), (39, 40)]
    prediction_cpu = replace_columns_with_mean(prediction_cpu, column_pairs)
    # 眨眼
    blink_weights = [0.1, 0.7, 0.7, 0.3, 0.1]

    # 遍历帧数，每隔 45 帧眨眼一次
    for start_frame in range(0, prediction_cpu.shape[0], 45):
        # 确保不会超出数组的边界
        for i, weight in enumerate(blink_weights):
            frame_index = start_frame + i
            if frame_index < prediction_cpu.shape[0]:
                prediction_cpu[frame_index, 0] = weight  # 修改左眼眨眼权重
                prediction_cpu[frame_index, 7] = weight  # 修改右眼眨眼权重

    torch.cuda.empty_cache()
    end = time.time()
    print("Model predict time: ", end - start, prediction_cpu.shape)
    result = {"blendshape":prediction_cpu.tolist()}
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3393)
