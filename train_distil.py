import numpy as np
import argparse
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import time
from data_loader import get_dataloaders
from SelfTalk_teacher import SelfTalk as Teacher_SelfTalk
from SelfTalk_student import SelfTalk

def compute_distillation_loss(student_logits, teacher_logits, targets, temperature=5, alpha=0.5, criterion=nn.MSELoss()):
    """
    计算蒸馏损失
    :param student_logits: 学生模型的输出
    :param teacher_logits: 教师模型的输出
    :param targets: 真实标签
    :param temperature: 温度参数
    :param alpha: 蒸馏损失与真实标签损失的权重
    :param criterion: 交叉熵损失函数
    """
    soft_targets = torch.softmax(teacher_logits / temperature, dim=1)
    distillation_loss = nn.KLDivLoss(reduction='batchmean')(
        torch.log_softmax(student_logits / temperature, dim=1),
        soft_targets
    ) * (temperature ** 2)

    student_loss = criterion(student_logits, targets)
    return alpha * distillation_loss + (1. - alpha) * student_loss

def trainer(args, train_loader, dev_loader, teacher_model,model, optimizer, criterion, epoch, last_train):
    save_path = os.path.join(args.dataset, args.save_path)
    save_path = save_path + '_'  + str(time.strftime("%m_%d_%H_%M", time.localtime()))
    os.makedirs(save_path, exist_ok=True)
    if last_train != 0:
        model.load_state_dict(torch.load(os.path.join(args.load_path, '{}_model.pth'.format(last_train)),
                                         map_location=torch.device('cpu')))
        model = model.to(args.device)
    iteration = 0
    for e in range(epoch + 1):
        loss_log = []
        # train
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        optimizer.zero_grad()

        for i, (audio, bs, template, file_name,emotion_label) in pbar:
            iteration += 1
            # to gpu
            audio, bs, template,emotion_label = audio.to(args.device), bs.to(args.device), template.to(args.device),emotion_label.to(args.device)
            bs_out= model(audio,bs,emotion_label)
            teacher_out,_,_,_,_ = teacher_model(audio,bs,emotion_label)
            loss = compute_distillation_loss(bs_out,teacher_out,bs)
            loss.backward()
            loss_log.append(loss.item())
            if i % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            pbar.set_description(
                    "(Epoch {}, iteration {}) TRAIN LOSS:{:.7f}".format(
                    e, iteration, loss.item()))
        # scheduler.step()
        if e % 10 == 0:
            print("current lr:",optimizer.param_groups[0]['lr'])
        valid_loss_log = []
        model.eval()
        with torch.no_grad():

            for audio, bs, template, file_name, emotion_label in dev_loader:
                # to gpu
                audio, bs, template, emotion_label = audio.to(args.device), bs.to(args.device), template.to(args.device),emotion_label.to(args.device)
                bs_out= model(audio,bs,emotion_label)
                loss = criterion(bs_out, bs)
                valid_loss_log.append(loss.item())

        current_loss = np.mean(valid_loss_log)

        if (e > 0 and e % 5 == 0) or e == args.max_epoch:
            torch.save(model.state_dict(), os.path.join(save_path, '{}_model.pth'.format(e)))

        print("epcoh: {}, current loss:{:.7f}".format(e + 1, current_loss))
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # BlendMICA
    parser = argparse.ArgumentParser(
        description='SelfTalk: A Self-Supervised Commutative Training Diagram to Comprehend 3D Talking Faces')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="BlendMICA", help='blendshape of MICA')
    parser.add_argument("--bs_dim", type=int, default=52,
                        help='blendshape num of BlendMICA')
    parser.add_argument("--num_emotions", type=int, default=5,
                        help='blendshape num of BlendMICA')
    parser.add_argument("--emotion_emb_dim", type=int, default=10,
                        help='blendshape num of BlendMICA')
    parser.add_argument("--wav_path", type=str, default="audio", help='path of the audio signals')
    parser.add_argument("--bs_path", type=str, default="bs_npy", help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=20, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda", help='cuda or cpu')
    parser.add_argument("--last_train", type=int, default=0, help='last train')
    parser.add_argument("--load_path", type=str, default=None, help='path to the trained models')
    parser.add_argument("--save_path", type=str, default="save", help='path of the trained models')
    args = parser.parse_args()
    teacher_model = Teacher_SelfTalk(args)
    teacher_model.load_state_dict(torch.load("./teacher.pth"))
    teacher_model = teacher_model.to(torch.device(args.device))
    teacher_model.eval()

    model = SelfTalk(args)
    print("model parameters: ", count_parameters(model))

    # to cuda
    assert torch.cuda.is_available()
    model = model.to(args.device)

    # load data
    dataset = get_dataloaders(args)
    # loss
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=args.lr)
    trainer(args, dataset["train"], dataset["valid"],teacher_model, model, optimizer, criterion, epoch=args.max_epoch,
            last_train=args.last_train)


if __name__ == "__main__":
    main()
