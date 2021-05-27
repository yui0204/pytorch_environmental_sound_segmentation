import os 
import time
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch import nn 
from torch import optim
from torch.utils.data import DataLoader
from torch.backends import cudnn

from model import read_model, FCN8s, UNet, CRNN, Deeplabv3plus
from data import SoundSegmentationDataset
from utils import scores, rmse, save_score_array
from utils import plot_loss, plot_mixture_stft, plot_class_stft, CustomMSE
from utils import restore

from sklearn.metrics import f1_score

torch.manual_seed(1234)
np.random.seed(1234)


def train(): 
    train_dataset = SoundSegmentationDataset(dataset_dir, split="train", task=task, n_classes=n_classes, spatial_type=spatial_type, mic_num=mic_num, angular_resolution=angular_resolution, input_dim=input_dim)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    val_dataset = SoundSegmentationDataset(dataset_dir, split="val", task=task, n_classes=n_classes, spatial_type=spatial_type, mic_num=mic_num, angular_resolution=angular_resolution, input_dim=input_dim)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    model = read_model(model_name, n_classes=n_classes, angular_resolution=angular_resolution, input_dim=input_dim)

    #model.load(os.path.join('results', dataset_name, "2020_0703" + "_"  + task + "_" + model_name, model_name + ".pth"))    

    #model = torch.nn.DataParallel(model) # make parallel
    #cudnn.deterministic = True
    #cudnn.benchmark = True

    criterion = nn.BCELoss()
    #criterion = nn.MSELoss()
    #criterion = CustomMSE()
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
    
    losses, val_losses = [], []
    loss_temp, val_loss_temp = 0, 0
    best_val_loss = 99999

    print("Training start")
    for epoch in range(epochs):
        # Training
        model.train()
        for i, (images, labels, phase) in tqdm(enumerate(train_loader)):
            images = images.cuda()
            labels = labels.cuda()
            
            outputs = model(images)
            #print(torch.sum(outputs, dim=2, keepdim=True).size(), (torch.sum(labels, dim=2, keepdim=True) > 0.2).size())

            #torch.sum(labels, dim=2, keepdim=True) > 0.2).to(torch.float32)

            #weak_labels = (labels > 0.0).to(torch.float32)
            #weak_labels = torch.mul(images[:, 0, :, :].unsqueeze(1), weak_labels)

            #loss = criterion(torch.sum(outputs, dim=2, keepdim=True), (torch.sum(labels, dim=2, keepdim=True) > 0.2).to(torch.float32))
            loss = criterion(outputs, labels)

            loss_temp += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_temp = loss_temp / len(train_loader)
        losses.append(loss_temp)
        
        print("Train Epoch: {}/{}  Loss: {:.6f} lr: {:.6f}".format(epoch+1, epochs, loss_temp, lr_scheduler.get_lr()[0]))

        # Validation per 1 epoch
        model.eval()
        with torch.no_grad():
            for i, (images, labels, phase) in tqdm(enumerate(val_loader)):
                images = images.cuda()
                labels = labels.cuda()
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_temp += loss.item()
            
        val_loss_temp = val_loss_temp / len(val_loader)
        val_losses.append(loss.item())

        print("Validation Epoch: {}/{}  Loss: {:.6f}".format(epoch+1, epochs, val_loss_temp))

        lr_scheduler.step()
        
        if val_loss_temp < best_val_loss:
            print("Best loss, model saved")
            best_val_loss = val_loss_temp
            model.save(save_dir=save_dir)
        loss_temp, val_loss_temp = 0, 0
    
        plot_loss(losses, val_losses, save_dir)

        shutil.copy("main.py", save_dir)
        #shutil.copy("Unet.py", save_dir)
        #shutil.copy("Deeplab.py", save_dir)
        #shutil.copy("CNN.py", save_dir)    
        if os.path.exists(os.getcwd() + "/nohup.out"):
            shutil.copy("nohup.out", save_dir)


def val():
    val_dataset = SoundSegmentationDataset(dataset_dir, split="val", task=task, n_classes=n_classes, spatial_type=spatial_type, mic_num=mic_num, angular_resolution=angular_resolution, input_dim=input_dim)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    model = read_model(model_name, n_classes=n_classes, angular_resolution=angular_resolution, input_dim=input_dim)
    model.load(os.path.join(save_dir, model_name + ".pth"))

    model.cuda()
    print("Evaluation start")
    model.eval()
    X_ins = np.zeros((1, input_dim, 256, 256))
    phases = np.zeros((1, 512, 256))
    #gts, preds = np.zeros((1, n_classes * angular_resolution, 256, 256)), np.zeros((1, n_classes * angular_resolution, 256, 256))
    gts, preds = np.zeros((1, n_classes * angular_resolution, 1, 256)), np.zeros((1, n_classes * angular_resolution, 1, 256))
    with torch.no_grad():
        for i, (images, labels, phase) in tqdm(enumerate(val_loader)):
            images = images.cuda()
            labels = labels.cuda()
            
            outputs = model(images)
            
            X_in = images.data.cpu().numpy()
            pred = outputs.data.cpu().numpy()             
            gt = labels.data.cpu().numpy()
            
            X_ins = np.concatenate((X_ins, X_in), axis=0)
            phases = np.concatenate((phases, phase), axis=0)
            preds = np.concatenate((preds, pred[:,:,np.newaxis,:]), axis=0)
            gts = np.concatenate((gts, gt[:,:,np.newaxis,:]), axis=0)

            if task == "cube" and n_classes == 75 and i > 10:
                break
            
    if task == "sed" or task == "ssl" or task == "seld":
        preds = (preds > 0.5) * 1
        f1 = f1_score(gts[1:]ravel(), preds[1:].ravel())
        print("F_score", f1)
        with open(save_dir + "f1_" + str(f1) + ".txt","w") as f:
            f.write(str(f1))   

    elif task == "segmentation" or task == "ssls" or task == "cube":
        scores_array = rmse(gts[1:], preds[1:], classes=n_classes)
        save_score_array(scores_array, save_dir)

    for n in range(len(preds)):
        if n < 10:
            plot_mixture_stft(X_ins[1:], no=n, save_dir=save_dir)
            if task == "sed" or task == "ssl" or task == "seld":
                plot_event(gts[1:], preds[1:], no=n, save_dir=save_dir, classes=n_classes, ang_reso=angular_resolution, label=label_csv)

            elif task == "segmentation" or task == "ssls" or task == "cube":
                plot_class_stft(gts[1:], preds[1:], no=n, save_dir=save_dir, classes=n_classes, ang_reso=angular_resolution, label=label_csv)
                restore(gts[1:], preds[1:], phases[1:], no=n, save_dir=save_dir, classes=n_classes, ang_reso=angular_resolution, label=label_csv, dataset_dir=dataset_dir)
            

if __name__ == '__main__':
    # params for train phase
    epochs = 10#100
    batch_size = 64
    lr = 0.001
    lr_decay = 0.95
    momentum = 0.95
    weight_decay = 5e-4
    
    # dataset
    n_classes = 75
    root = "/misc/export3/sudou/sound_data/datasets/"
    dataset_name = "multi_segdata75_256_-20dB_random_sep_72/"
    dataset_dir = root + dataset_name
    
    label_csv = pd.read_csv(filepath_or_buffer=os.path.join(dataset_dir, "label.csv"), sep=",", index_col=0)

    task = "sed" # "sed", "segmentation", "ssl", "ssls", "cube"
    model_name = "CRNN_SED" #"Deeplabv3plus"

    # make save_directory
    date = time.strftime('%Y_%m%d')
    dirname = date + "_"  + task + "_" + model_name
    save_dir = os.path.join('results', dataset_name, dirname)    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # sptatial feature type (None, IPD, complex)
    spatial_type = "ipd"
    mic_num = 8
    angular_resolution = 1#8
    if mic_num == 1:
        input_dim = 1
    elif spatial_type == "ipd":
        input_dim = mic_num * 2 - 1
    elif spatial_type == "complex":
        input_dim = mic_num * 3
    else:
        raise ValueError("This type of spatial feature is not supported!")

    train()
    val()
    