import numpy as np
import argparse
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import time
from data_loader import get_dataloaders
from SelfTalk_teacher import SelfTalk


def trainer(args, train_loader, dev_loader, model, optimizer, criterion, epoch, last_train):
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
            bs_out, lip_features, text_hidden_states, logits, text_logits = model(audio,bs,emotion_label)
            loss1 = criterion(bs_out, bs)
            gt_vel = bs[:, 1:, :] - bs[:, :-1, :]
            pred_vel = bs_out[:, 1:, :] - bs_out[:, :-1, :]
            loss2 = criterion(pred_vel, gt_vel)
            loss3 = criterion(lip_features, text_hidden_states)
            loss = torch.mean(1000 * loss1 + 1000 * loss2 + 0.001 * loss3)
            loss.backward()
            loss_log.append(loss.item())
            if i % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            pbar.set_description(
                    "(Epoch {}, iteration {}) TRAIN LOSS:{:.7f}, loss1:{:.7f}, loss2:{:.7f}, loss3:{:.7f}".format(
                    e, iteration, loss.item(), loss1.item(), loss2.item(), loss3.item()))
        # scheduler.step()
        if e % 10 == 0:
            print("current lr:",optimizer.param_groups[0]['lr'])
        valid_loss_log = []
        model.eval()
        with torch.no_grad():

            for audio, bs, template, file_name, emotion_label in dev_loader:
                # to gpu
                audio, bs, template, emotion_label = audio.to(args.device), bs.to(args.device), template.to(args.device),emotion_label.to(args.device)
                train_subject = "_".join(file_name[0].split("_")[:-1])
                bs_out, lip_features, text_hidden_states, logits, text_logits = model(audio,bs,emotion_label)
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
    parser.add_argument("--period", type=int, default=30, help='period in PPE - 30 ')
    parser.add_argument("--wav_path", type=str, default="audio", help='path of the audio signals')
    parser.add_argument("--bs_path", type=str, default="bs_npy", help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=100, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda", help='cuda or cpu')
    parser.add_argument("--last_train", type=int, default=0, help='last train')
    parser.add_argument("--load_path", type=str, default=None, help='path to the trained models')
    parser.add_argument("--save_path", type=str, default="save", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="result", help='path to the predictions')
    args = parser.parse_args()

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
    trainer(args, dataset["train"], dataset["valid"], model, optimizer, criterion, epoch=args.max_epoch,
            last_train=args.last_train)


if __name__ == "__main__":
    main()
