# Fast-3D-Talking-Face: Blendshape-based Audio-Driven 3D-Talking-Face with Transformer

这是Fast-3D-Talking-Face 的轻量级版本同时新增了情绪嵌入。通过在原始模型的基础上（audio-encoder基础wav2vec2-large-xlsr)进行蒸馏（
audio-encoder替换为./distil-wav2vec2，同时简化网络结构），大大降低模型大小，且几乎没有性能损失。

[wav2vec2-large-xlsr](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english)和[distil-wav2vec2](https://huggingface.co/OthmaneJ/distil-wav2vec2)这两个模型未上传至仓库，可以在hugging face上下载。
