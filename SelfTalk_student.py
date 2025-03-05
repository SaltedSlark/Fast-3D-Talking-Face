import torch.nn as nn
from wav2vec import Wav2Vec2Model


class SelfTalk(nn.Module):
    def __init__(self, args):
        super(SelfTalk, self).__init__()
        self.audio_encoder = Wav2Vec2Model.from_pretrained("./distil-wav2vec2/")
        self.audio_encoder.feature_extractor._freeze_parameters()

        # 情绪嵌入层
        num_emotions = args.num_emotions  # 情绪种类数量，需要根据您的数据集设置
        emotion_emb_dim = args.emotion_emb_dim  # 情绪嵌入维度，可自行设定
        self.emotion_embedding = nn.Embedding(num_emotions, emotion_emb_dim)
        self.emotion_proj = nn.Linear(emotion_emb_dim, 128)


        decoder_layer = nn.TransformerDecoderLayer(d_model=128, nhead=2,
                                                   dim_feedforward=2 * 128, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.audio_feature_map = nn.Linear(768, 128)
        self.bs_map_r = nn.Linear(128, args.bs_dim)
        self.device = args.device

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
        return bs_out

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
