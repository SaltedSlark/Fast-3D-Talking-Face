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
        prediction = model.predict(audio_feature.half(),mood)
    prediction_cpu = prediction.squeeze().detach().cpu().numpy()*0.8
    del prediction
    torch.cuda.empty_cache()
    end = time.time()
    print("Model predict time: ", end - start, prediction_cpu.shape)
    result = {"blendshape":prediction_cpu.tolist()}
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3393)
