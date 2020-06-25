import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn 
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import FCN8s
from data import SoundSegmentationDataset
from utils import scores

import os 

torch.manual_seed(1234)
np.random.seed(1234)


def train(args): 
    loader = SoundSegmentationDataset(data_path, split="train", task=task, spatial_type=spatial_type, mic_num=mic_num, angular_resolution=angular_resolution, input_dim=input_dim)
    trainloader = DataLoader(loader, batch_size=args.batch_size, num_workers=4, shuffle=True)

    model = FCN8s(n_classes=n_classes, input_dim=input_dim)
    print(model)

    model.cuda()

    loss_function = nn.MSELoss()
    lr = args.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)

    print("Training start")
    for epoch in range(args.n_epoch):
        for i, (images, labels) in tqdm(enumerate(trainloader)):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            print(" Epoch: {}/{} Iteration: {}/{} Loss: {} lr:{}".format(epoch+1, args.n_epoch, i, len(trainloader), loss.item(), lr))

        if epoch % 5 == 0:
            lr = lr * args.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr 
        model.save()

def val(args):
    loader = SoundSegmentationDataset(data_path, split="val", task=task, spatial_type=spatial_type, mic_num=mic_num, angular_resolution=angular_resolution, input_dim=input_dim)
    valloader = DataLoader(loader, batch_size=args.batch_size, num_workers=4)

    model = FCN8s(n_classes=n_classes, input_dim=input_dim)
    model.load(args.model_path)

    #model.cuda()

    model.eval()

    n_classes = n_classes
    gts, preds = [], []
    for i, (images, labels) in tqdm(enumerate(valloader)):
        #images = Variable(images.cuda())
        #labels = Variable(labels.cuda())
        outputs = model(images)
        pred = outputs.data.max(1)[1].cpu().numpy()             # .item()に変更？
        gt = labels.data.cpu().numpy()                          # .item()に変更？ 

        for gt_, pred_ in zip(gt, pred):
            gts.append(gt_)
            preds.append(pred_)

    score, class_iou = scores(gts, preds, n_class=n_classes)

    for k, v in score.items():
        print(k, v)
    for i in range(n_classes):
        print(i, class_iou[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FCN Hyperparams')
    
    # params for train phase
    parser.add_argument('--n_epoch', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--momentum', type=float, default=0.95)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    
    # params for test phase
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--out_path', type=str, default='./results/')

    

    data_path = '/home/yui-sudo/document/dataset/sound_segmentation/datasets/multi_segdata75_256_no_sound_random_sep_72/'

    n_classes = 75#10
    task = "segmentation"

    spatial_type = None
    mic_num = 1
    angular_resolution = 1
    if mic_num == 1:
        input_dim = 1
    elif spatial_type == "ipd":
        input_dim = mic_num * 2 - 1
    elif spatial_type == "complex":
        input_dim = mic_num * 3
    else:
        raise ValueError("This type of spatial feature is not supported!")

    args = parser.parse_args()
    
    train(args)
    val(args)
    