'''
-----------------------------------------------
File Name: main5.py$
Description: 
  5-fold cross validation of semi-supervised deep learning for regression based on Pytroch
  target: Head circumference prediction
Author: Jing
Date: 07/07/2023
-----------------------------------------------
'''

import os
import numpy as np
import statistics
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm
import random # Generate pseudo-random numbers
import timm
import torchvision
import torchvision.transforms as transforms # data augmentation
import torch
import torch.nn as nn
import torch.nn.functional as F # activation, loss, pool, etc.
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
import torch.optim as optim
from lion_pytorch import Lion
from torchmetrics import R2Score
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score

# hyper parameters
summary_dir = './logs'
torch.backends.cudnn.benchmark = True
print('cuda',torch.cuda.is_available())
print('gpu number',torch.cuda.device_count())
for i in range(torch.cuda.device_count()): print(torch.cuda.get_device_name(i))
summaryWriter = SummaryWriter(summary_dir)
baseline = 'vgg11'
n_label = 50 # 50, 100, 200, 300
tau = 11/24 # 11/24, 5/12 1/3, 1/4
bs = 2 # 2, 4, 8, 10, batch size
alpha2 = 1/12 #  1/12 1/6 1/3 1/2 alpha2 ps loss
alpha1 = 1-alpha2 # alpha1 cons
img_width = 224 # ori 800
img_height = 224 # ori 540
num_epochs = 200
criterion = nn.HuberLoss()
mae = MeanAbsoluteError()
mape = MeanAbsolutePercentageError()
r2score = R2Score()

best_model_path = '/home/jing/python_code/DeepRT/07semi-super/fix-match-reg/models/'
root = "/home/jing/python_code/DeepRT/07semi-super/HC/version1"
img_path = "/home/jing/python_code/DeepRT/07semi-super/HC/version1/HC"
csv_file = os.path.join(root, "HC.csv")

def data_split_crossval(file, fold= 0, num_label=300, aff_info = True):
# 5-fold cross validation (1000+300=1300)
# train: labeled:50,100,200,300; unlabeled:600
# valid: 200
# test:  200

    dataframe = pd.read_csv(file)

    fold0 = dataframe[dataframe['fold']==0]
    fold1 = dataframe[dataframe['fold']==1]
    fold2 = dataframe[dataframe['fold']==2]
    fold3 = dataframe[dataframe['fold']==3]
    fold4 = dataframe[dataframe['fold']==4]
    fold5 = dataframe[dataframe['fold']==5] # images with no labels
    img_ngt = fold5['filename'].tolist()

    fold0_image = fold0['filename'].tolist()
    fold0_label = fold0['head circumference (mm)'].tolist()
    fold1_image = fold1['filename'].tolist()
    fold1_label = fold1['head circumference (mm)'].tolist()
    fold2_image = fold2['filename'].tolist()
    fold2_label = fold2['head circumference (mm)'].tolist()
    fold3_image = fold3['filename'].tolist()
    fold3_label = fold3['head circumference (mm)'].tolist()
    fold4_image = fold4['filename'].tolist()
    fold4_label = fold4['head circumference (mm)'].tolist()

    if fold == 0:
        train_image = fold2_image + fold3_image + fold4_image 
        train_label = fold2_label + fold3_label + fold4_label
        train_labeled = train_image[:num_label]
        train_gt = train_label[:num_label]
        train_unlabeled = train_image[-300:] + img_ngt
        val_image = fold1_image
        val_label = fold1_label
        test_image =  fold0_image
        test_label = fold0_label

    if fold == 1:
        train_image = fold0_image + fold3_image + fold4_image
        train_label = fold0_label + fold3_label + fold4_label
        train_labeled = train_image[:num_label]
        train_gt = train_label[:num_label]
        train_unlabeled = train_image[-300:] + img_ngt
        val_image = fold2_image
        val_label = fold2_label
        test_image = fold1_image
        test_label = fold1_label

    if fold == 2:
        train_image = fold1_image + fold0_image + fold4_image
        train_label = fold1_label + fold0_label + fold4_label
        train_gt = train_label[:num_label]        
        train_labeled = train_image[:num_label]
        train_unlabeled = train_image[-300:] + img_ngt
        val_image = fold3_image
        val_label = fold3_label
        test_image =  fold2_image
        test_label = fold2_label

    if fold == 3:
        train_image = fold1_image + fold2_image + fold0_image
        train_label = fold1_label + fold2_label + fold0_label
        train_labeled = train_image[:num_label]
        train_gt = train_label[:num_label]
        train_unlabeled = train_image[-300:] + img_ngt
        val_image = fold4_image
        val_label = fold4_label
        test_image =  fold3_image
        test_label = fold3_label

    if fold == 4:
        train_image = fold1_image + fold2_image + fold3_image
        train_label = fold1_label + fold2_label + fold3_label
        train_labeled = train_image[:num_label]
        train_gt = train_label[:num_label]
        train_unlabeled = train_image[-300:] + img_ngt
        val_image = fold0_image
        val_label = fold0_label
        test_image = fold4_image
        test_label = fold4_label
        
    if aff_info == True:
        print("Total # images: {},  train_labeled: {}, train_unlabeled: {},  val: {}, test: {}".\
        format(len(train_labeled+train_unlabeled+ val_image+ test_image), len(train_labeled),len(train_unlabeled), len(val_image), len(test_image)))
   
    return train_labeled, train_gt, train_unlabeled, val_image,val_label,test_image,test_label

# weak augmentation, weak augmentation list, strong augmentation
channel_stats = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2470,  0.2435,  0.2616])
totensor = transforms.ToTensor()
normalize = transforms.Normalize(**channel_stats)
# augmentation types
color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5)
hflip = transforms.RandomHorizontalFlip()
noise = transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5))
rotat = transforms.RandomRotation(5)
persp = transforms.RandomPerspective(distortion_scale=0.5)

transform_weak = [hflip, rotat, totensor,normalize]
transform_weak_list = []
for _ in range(10):
    transform_w = transforms.Compose(transform_weak)
    transform_w.transforms[0].size = (32 + random.randint(-2, 2), 32 + random.randint(-2, 2)) # random crop
    transform_w.transforms[1].p = 0.5 + random.uniform(-0.1, 0.1) # probability
    transform_weak_list.append(transform_w)

weak_aug = transforms.Compose([hflip, rotat, totensor, normalize])
strong_aug = transforms.Compose([hflip, rotat, color_jitter, noise, persp, totensor,normalize])
transform_norm = transforms.Compose([totensor, normalize]) # for valid and test

class HC_Data(Dataset):
    
    def __init__(self, data_path, image_list = '', label_list = '', mode = 'train', supervised = True):
        self.data_path = data_path
        self.image_list = image_list
        self.label_list = label_list
        self.mode = mode
        self.supervised = supervised
        
    def __len__(self): return len(self.image_list)

    def __getitem__(self, idx):
        
        image_name = self.image_list[idx]
        image_path = os.path.join(self.data_path, image_name)
        image = Image.open(image_path).convert("RGB").resize((img_width,img_height))
        
        if self.mode == 'train':
            if self.supervised == True: 
                x1 = weak_aug(image)
                label = self.label_list[idx]
                return x1, np.float32(label)
            else: 
                x2_w = weak_aug(image)
                x2_list = []
                for weak_augm in transform_weak_list: x2_list.append(weak_augm(image))
                x2_s = strong_aug(image)
                return x2_w, x2_list, x2_s
        
        if self.mode == 'test':
            image = transform_norm(image)
            label = self.label_list[idx]
            return image, np.float32(label), image_name

        if self.mode  == "valid":
            image = transform_norm(image)
            label = self.label_list[idx]
            return image, np.float32(label)

all_r2 = []
all_mae = []
all_mape = []
all_ps = []
# 5-fold cross validation
for folds in range(0,5):
    print(f'================fold {folds}===============================')
    model = timm.create_model(baseline, pretrained=True, num_classes=1, in_chans=3)
    model.cuda()
    optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-4)
    model = torch.compile(model) # in torch2.0
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    train_labeled, train_gt, train_unlabeled, valid_image, valid_gt, test_image, test_gt = data_split_crossval(csv_file, fold= folds, num_label=n_label)
    labeled_set = HC_Data(data_path=img_path,image_list=train_labeled,label_list=train_gt,mode='train', supervised = True)
    unlabeled_set = HC_Data(data_path=img_path,image_list=train_unlabeled, mode='train', supervised = False)
    valid_set = HC_Data(data_path=img_path,image_list=valid_image,label_list=valid_gt,mode='valid')
    test_set = HC_Data(data_path=img_path,image_list=test_image,label_list=test_gt,mode='test')
    multiple = int(len(unlabeled_set)/len(labeled_set))
    print('supervised:', len(labeled_set), 'unsupervised:', len(unlabeled_set),'valid', len(valid_set), 'test:', len(test_set), 'multiple:', multiple)

    labeled_loader = DataLoader(dataset=labeled_set, batch_size=bs, shuffle=True, num_workers=6)
    unlabeled_loader = DataLoader(dataset=unlabeled_set, batch_size=bs*multiple, shuffle=True, num_workers=6)
    valid_loader = DataLoader(dataset=valid_set, batch_size=bs, shuffle=False, num_workers=6)
    test_loader = DataLoader(dataset=test_set, batch_size=bs, shuffle=False, num_workers=6)

    print(f'labeled loader {len(labeled_loader)}, unlabeled_loader {len(unlabeled_loader)}, valid_loader {len(valid_loader)}, test_loader {len(test_loader)} ')
    all_num_ps = 0 # number of pseudo labels in all epochs
    best_r2=-10

    for epoch in range(num_epochs):
        train_losses = []
        num_ps = 0
        model.train()

        for labeled_data, unlabeled_data in zip(labeled_loader, unlabeled_loader):
            x1, gt = labeled_data
            wx2, wx2_list, sx2 = unlabeled_data
            x1, gt = x1.cuda(), gt.cuda()
            wx2, sx2 = wx2.cuda(), sx2.cuda()
            for item in range(len(wx2_list)): wx2_list[item] = wx2_list[item].cuda()
            with torch.enable_grad():
                
                model.zero_grad()
                p1 = model(x1)
                p1 = torch.squeeze(p1,dim=1)
                l1 = criterion(gt,p1)

                wp2 = model(wx2)
                sp2 = model(sx2)
                wp2 = torch.squeeze(wp2,dim=1)
                sp2 = torch.squeeze(sp2,dim=1)
                l2 = F.l1_loss(wp2, sp2)

                # compute pseudo labels
                weak_stacked = torch.stack(wx2_list,dim=0) #[10, bs, 3, 224, 224]
                list_wx2 = torch.reshape(weak_stacked, (len(wx2_list)*weak_stacked.shape[1],3, img_height, img_width))
                wp2_list = model(list_wx2)
                wp2_list_array = torch.chunk(wp2_list, bs*multiple, dim=0)
                sp2_array = torch.chunk(sp2, bs*multiple, dim=0)
                l3 = 0
                for each in range(len(sp2_array)):
                    wp2_list_temp = wp2_list_array[each]
                    norm_temp = (wp2_list_temp - wp2_list_temp.min()) / (wp2_list_temp.max() - wp2_list_temp.min())
                    if torch.std(norm_temp) < tau:
                        num_ps += 1
                        pseudo_label = torch.mean(wp2_list_array[each])
                        pseudo_label = torch.unsqueeze(pseudo_label,dim=0)
                        l3 += criterion(pseudo_label, sp2_array[each])
                    else: l3 = 0

                total_loss = l1 + alpha1 * l2 + alpha2 * l3 / bs / multiple # total loss
                total_loss.backward() 
                optimizer.step()
                train_losses.append(total_loss.item())
            
        avg_loss = np.array(train_losses).mean()
        print("[TRAIN] epoch={}/{} train_loss={:.3f}".format(epoch+1, num_epochs, avg_loss))
        summaryWriter.add_scalars('loss', {"train": (avg_loss)}, epoch)
        all_num_ps +=num_ps
        
        torch.cuda.empty_cache()
        model.eval()
        val_labels_list = []
        val_logits_list = []

        with torch.no_grad():
            for x, y in valid_loader:

                x = x.cuda()
                logits = model(x)
                logits = torch.squeeze(logits,dim=1)
                val_logits_list.extend(logits.detach().cpu())
                val_labels_list.extend(y)
            
        r2 = r2score(torch.tensor(val_logits_list),torch.tensor(val_labels_list)).item()
        avg_mae = mae(torch.tensor(val_logits_list),torch.tensor(val_labels_list)).item()
        avg_mape = mape(torch.tensor(val_logits_list),torch.tensor(val_labels_list)).item()
        print("[EVAL] epoch={}/{}  val_r2={:.3f} val_mae={:.3f} val_mape={:.3f}".format(epoch+1, num_epochs, r2, avg_mae, avg_mape))   
        summaryWriter.add_scalars('r2', {"val": r2}, epoch)
        summaryWriter.add_scalars('mae', {"val": avg_mae}, epoch)
        scheduler.step()
        if r2 >= best_r2:
            print(f'best r2 {r2:.3f} at epoch {epoch+1}')
            best_r2 = r2
            torch.save(model.state_dict(), best_model_path+str(folds)+'_model.pth') 
    # end of training

    ps_epoch = all_num_ps/num_epochs
    all_ps.append(ps_epoch)
    print('################# TRAIN FINISH, START TEST #################')
    model.load_state_dict(torch.load(best_model_path+str(folds)+'_model.pth'))
    model.eval()
    test_loader = DataLoader(dataset=test_set,batch_size=10,shuffle=False,num_workers=6,pin_memory=True)
    test_logits_list = []
    test_labels_list = []
    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.cuda()
            logits = model(x)
            logits = torch.squeeze(logits,dim=1)
            test_logits_list.extend(logits.detach().cpu())
            test_labels_list.extend(y)
                
    r2 = r2score(torch.tensor(test_logits_list),torch.tensor(test_labels_list)).item()
    mean_mae = mae(torch.tensor(test_logits_list),torch.tensor(test_labels_list)).item()
    mean_mape = mape(torch.tensor(test_logits_list),torch.tensor(test_labels_list)).item()
    print("[TEST] test_r2={:.3f} test_mae={:.3f} test_mape={:.3f}".format(r2, mean_mae, mean_mape))
    all_r2.append(r2)
    all_mae.append(mean_mae)
    all_mape.append(mean_mape)

avg_r2 = statistics.mean(all_r2)
std_r2 = statistics.stdev(all_r2)
avg_mae = statistics.mean(all_mae)
std_mae = statistics.stdev(all_mae)
avg_mape = statistics.mean(all_mape)
std_mape = statistics.stdev(all_mape)

print('**************5-Fold results*************************************************')
print(f'average number of pseudo labels {int(statistics.mean(all_ps))}')
print(f'avg r2 {avg_r2:.3f}, std {std_r2:.3f},avg mae {avg_mae:.3f}, std {std_mae:.3f},avg mape {avg_mape:.3f}, std {std_mape:.3f}')