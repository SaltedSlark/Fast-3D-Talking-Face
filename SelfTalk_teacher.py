import torch.nn as nn
from wav2vec import Wav2Vec2Model, Wav2Vec2ForCTC, linear_interpolation
import numpy as np


class SelfTalk(nn.Module):
    def __init__(self, args):
        super(SelfTalk, self).__init__()
        self.audio_encoder = Wav2Vec2Model.from_pretrained("./wav2vec2-large-xlsr-53-english")
        self.text_encoder = Wav2Vec2ForCTC.from_pretrained("./wav2vec2-large-xlsr-53-english")
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.audio_encoder.feature_extractor._freeze_parameters()

        # 情绪嵌入层
        num_emotions = args.num_emotions  # 情绪种类数量，需要根据您的数据集设置
        emotion_emb_dim = args.emotion_emb_dim  # 情绪嵌入维度，可自行设定
        self.emotion_embedding = nn.Embedding(num_emotions, emotion_emb_dim)
        self.emotion_proj = nn.Linear(emotion_emb_dim, 512)

        self.lip_mask = np.array([i for i in range(32)])
        self.lip_map = nn.Linear(args.bs_dim, 1024)

        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=4,
                                                   dim_feedforward=2 * 512, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.audio_feature_map = nn.Linear(1024, 512)
        self.transformer = nn.Transformer(d_model=1024, batch_first=True)
        self.bs_map_r = nn.Linear(512, args.bs_dim)
        self.device = args.device
        self.dropout = nn.Dropout(p=0.0, inplace=False)
        self.lm_head = nn.Linear(1024, 33)

        nn.init.constant_(self.bs_map_r.weight, 0)
        nn.init.constant_(self.bs_map_r.bias, 0)

    def forward(self, audio, bs, emotion_labels):
        frame_num = bs.shape[1]

        hidden_states = self.audio_encoder(audio, frame_num=frame_num).last_hidden_state
        bs_input = self.audio_feature_map(hidden_states)
         # 获取情绪嵌入并投影到特征维度
        emotion_emb = self.emotion_embedding(emotion_labels)  # (batch_size, emotion_emb_dim)
        emotion_emb_proj = self.emotion_proj(emotion_emb)     # (batch_size, feature_dim)
        emotion_emb_proj = emotion_emb_proj.unsqueeze(1).expand(-1, bs_input.size(1), -1)  # (batch_size, seq_length, feature_dim)
        bs_input = bs_input + emotion_emb_proj
        bs_out = self.transformer_decoder(bs_input, bs_input)
        bs_out = self.bs_map_r(bs_out)
        audio_model = self.text_encoder(audio)
        text_hidden_states = audio_model.hidden_states
        text_logits = audio_model.logits
        frame_num = text_hidden_states.shape[1]
        lip_out = bs_out
        lip_offset = self.lip_map(lip_out)
        lip_offset = linear_interpolation(lip_offset, 25, 50, output_len=frame_num)
        lip_features = self.transformer(lip_offset, lip_offset)
        logits = self.lm_head(self.dropout(lip_features))

        return bs_out, lip_features, text_hidden_states, logits, text_logits

    def predict(self, audio, emotion_labels):
        hidden_states = self.audio_encoder(audio).last_hidden_state
        bs_input = self.audio_feature_map(hidden_states)
        emotion_emb = self.emotion_embedding(emotion_labels)
        emotion_emb_proj = self.emotion_proj(emotion_emb)
        emotion_emb_proj = emotion_emb_proj.unsqueeze(1).expand(-1, bs_input.size(1), -1)

        # 将情绪信息融入bs_input
        bs_input = bs_input + emotion_emb_proj
        bs_out = self.transformer_decoder(bs_input, bs_input)
        bs_out = self.bs_map_r(bs_out)
        return bs_out
