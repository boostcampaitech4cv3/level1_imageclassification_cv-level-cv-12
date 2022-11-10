#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch, gc
gc.collect()
torch.cuda.empty_cache()


# In[2]:


import GPUtil
GPUtil.showUtilization()


# In[3]:


import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch_optimizer as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import os
from torchmetrics import F1Score


# In[4]:


random_seed = 12
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


# In[5]:


train_dir_path = '../input/data/train/'
train_image_path = '../input/data/train/images/'

dt_train = pd.read_csv(train_dir_path+'train.csv')
dt_train


# In[6]:


def get_age_range(age):
    if age < 30:
        return 0
    elif 30 <= age < 60:
        return 1
    else:
        return 2


# In[7]:


dt_train['age_range'] = dt_train['age'].apply(lambda x : get_age_range(x))


# In[8]:


dt_train


# In[9]:


train_idx, valid_idx = train_test_split(np.arange(len(dt_train)),
                                       test_size=0.2,
                                       shuffle=True,
                                       stratify=dt_train['age_range'])


# In[10]:


train_image = []
train_label = []

for idx in train_idx:
    path = dt_train.iloc[idx]['path']
    for file_name in [i for i in os.listdir(train_image_path+path) if i[0] != '.']:
        _, file_extension = os.path.splitext(file_name)
        if file_extension not in ['.jpg', '.jpeg', '.png']:
            continue
        train_image.append(train_image_path+path+'/'+file_name)
        train_label.append((path.split('_')[1], path.split('_')[3], file_name.split('.')[0]))                                 


# In[11]:


valid_image = []
valid_label = []

for idx in valid_idx:
    path = dt_train.iloc[idx]['path']
    for file_name in [i for i in os.listdir(train_image_path+path) if i[0] != '.']:
        _, file_extension = os.path.splitext(file_name)
        if file_extension not in ['.jpg', '.jpeg', '.png']:
            continue
        valid_image.append(train_image_path+path+'/'+file_name)
        valid_label.append((path.split('_')[1], path.split('_')[3], file_name.split('.')[0]))                                 


# In[12]:


def onehot_enc(x):
    def gender(i):
        if i == 'male':
            return 0
        elif i == 'female':
            return 3
    def age(j):
        j = int(j)
        if j < 30:
            return 0
        elif j >= 30 and j < 60:
            return 1
        elif j >= 60:
            return 2
    def mask(k):
        if k == 'normal':
            return 12
        elif 'incorrect' in k:
            return 6
        else:
            return 0
    return gender(x[0]) + age(x[1]) + mask(x[2])


# In[13]:


# sr_data = pd.Series(whole_image_path)
# sr_label = pd.Series(whole_target_label)


# In[14]:


train_data = pd.Series(train_image)
train_label = pd.Series(train_label)

valid_data = pd.Series(valid_image)
valid_label = pd.Series(valid_label)


# In[15]:


class Dataset_Mask(Dataset):
    def __init__(self, data, label, encoding=True, midcrop=True, transform=None):
        self.encoding = encoding
        self.midcrop = midcrop
        self.data = data
        self.label = label
        self.transform = transform

        if encoding:
            self.label = self.label.apply(onehot_enc)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        X = Image.open(self.data[idx])
        X = X.crop((17, 70, 367, 420))
        X = self.transform(X)
        y = self.label[idx]
        return X, y


# In[16]:


mask_train_set = Dataset_Mask(data=train_data, label=train_label, transform = transforms.Compose([
                                #transforms.RandomResizedCrop((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ]))


# In[17]:


mask_val_set = Dataset_Mask(data=valid_data, label=valid_label, transform = transforms.Compose([
                                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ]))


# In[18]:


#t_image = [mask_train_set[i][1] for i in range(len(mask_train_set))]
#v_image = [mask_val_set[i][1] for i in range(len(mask_val_set))]


# In[19]:


#t_df = pd.DataFrame(t_image, columns=['counts'])
#v_df = pd.DataFrame(v_image, columns=['counts'])


# In[20]:


#import seaborn as sns

#fig, axes = plt.subplots(1, 2, figsize=(15, 5))

#sns.countplot(x='counts', data=t_df, ax=axes[0])
#axes[0].set_xlabel("train set labels")
#sns.countplot(x='counts', data=v_df, ax=axes[1])
#axes[1].set_xlabel("valid set labels")


# In[21]:


#print(f'training data size : {len(mask_train_set)}')
#print(f'validation data size : {len(mask_val_set)}')


# In[22]:


batch_size = 2048

train_dataloader_mask = DataLoader(dataset = mask_train_set, batch_size=batch_size, num_workers=2)
val_dataloader_mask = DataLoader(dataset = mask_val_set, batch_size=batch_size, num_workers=2)


# In[23]:

basemodel_convnext_l = torchvision.models.convnext_large(torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
print('필요 입력 채널  개수', basemodel_convnext_l.features[0][0].weight.shape[1])
print('네트워크 출력 채널 개수', basemodel_convnext_l.classifier[2].weight.shape[0])
print(basemodel_convnext_l)

# In[24]:


import math
class_num = 18
"""
basemodel_convnext_l.classifier[2] = nn.Linear(in_features=1536, out_features=class_num, bias=True)
nn.init.xavier_uniform_(basemodel_convnext_l.classifier[2].weight)
stdv = 1. / math.sqrt(basemodel_convnext_l.classifier[2].weight.size(1))
basemodel_convnext_l.classifier[2].bias.data.uniform_(-stdv, stdv)
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using {device}")

if device != "cpu":
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    if torch.cuda.device_count() > 1:
        basemodel_convnext_l = torch.nn.DataParallel(basemodel_convnext_l)

# In[26]:

print(">>>>>>>>>>")
basemodel_convnext_l = torch.load('../checkpoint/basemodel_convnext_l-2/checkpoint_best.pth')
#if "state_dict" in pretrained_dict.keys():
#    pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
#else:
#    pretrained_dict = remove_prefix(pretrained_dict, 'module.')
#basemodel_convnext_l.load_state_dict(pretrained_dict, strict=False)

basemodel_convnext_l.to(device)

LEARNING_RATE = 0.001
NUM_EPOCH = 301

criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(basemodel_convnext_l.parameters(), lr=LEARNING_RATE)
#optimizer = torch.optim.AdamW(basemodel_convnext_l.parameters(), lr=LEARNING_RATE, eps=1e-8)

optimizer = optim.AdamP(basemodel_convnext_l.parameters(), lr=LEARNING_RATE, weight_decay=1e-5, nesterov=True)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# In[27]:


np.set_printoptions(precision=3)
n_param = 0
for p_idx, (param_name, param) in enumerate(basemodel_convnext_l.named_parameters()):
    if param.requires_grad:
        if param_name.startswith('module.classifier') or param_name.startswith('classifier'):
            param.requires_grad = True  # Train
        else:
            param.requires_grad = False # Freeze

        param_numpy = param.detach().cpu().numpy()
        n_param += len(param_numpy.reshape(-1))
        print ("[%d] name:[%s] shape:[%s]."%(p_idx,param_name,param_numpy.shape))
        print (f"    val:{(param_numpy.reshape(-1)[:5])} -- {param.requires_grad}")
print ("Total number of parameters:[%s]."%(format(n_param,',d')))


# In[ ]:


best_val_acc = 0
best_val_loss = np.inf
patience = 10
cur_count = 0

f1 = F1Score(num_classes=class_num, average='macro').to(device)
best_f1_score = 0

for epoch in range(NUM_EPOCH):
    basemodel_convnext_l.train()
    loss_value = 0
    matches = 0
    for train_batch in tqdm(train_dataloader_mask):
        inputs, labels = train_batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outs = basemodel_convnext_l(inputs)
        preds = torch.argmax(outs, dim=-1)
        loss = criterion(outs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()
        
        if epoch % 10 == 0:
            torch.save(basemodel_convnext_l, '../checkpoint/basemodel_convnext_l-detection/checkpoint_ep_%d.pth'% epoch)
        
        loss_value += loss.item()
        matches += (preds == labels).sum().item()
        
        train_loss = loss_value / batch_size
        train_acc = matches / batch_size
        
        loss_value = 0
        matches = 0
    print(f"epoch[{epoch}/{NUM_EPOCH}] training loss {train_loss:.3f}, training accuracy {train_acc:.3f}")
        
    with torch.no_grad():
        basemodel_convnext_l.eval()
        val_loss_items = []
        val_acc_items = []
        for val_batch in tqdm(val_dataloader_mask):
            inputs, labels = val_batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outs = basemodel_convnext_l(inputs)
            preds = torch.argmax(outs, dim=-1)
            
            loss_item = criterion(outs, labels).item()
            acc_item = (labels==preds).sum().item()
            val_loss_items.append(loss_item)
            val_acc_items.append(acc_item)
            
        val_loss = np.sum(val_loss_items) / len(val_dataloader_mask)
        val_acc = np.sum(val_acc_items) / len(mask_val_set)

        f1_score = f1(outs, labels)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
        if f1_score > best_f1_score:
            best_f1_score = f1_score
#             cur_count = 0
            torch.save(basemodel_convnext_l, '../checkpoint/basemodel_convnext_l-detection/checkpoint_best.pth')
#         else:
#             cur_count += 1
#             if cur_count >= patience:
#                 print("Early Stopping!")
#                 break
            
            
        print(f"[val] acc : {val_acc:.3f}, loss : {val_loss:.3f}, f1 score: {f1_score:.3f}")
        print(f"best acc : {best_val_acc:.3f}, best loss : {best_val_loss:.3f}, best f1 : {best_f1_score:.3f}")


# In[ ]:


print(f'Best f1 score:{best_f1_score}')


# In[ ]:


# meta 데이터와 이미지 경로를 불러옵니다.
test_dir_path = '../input/data/eval/'
test_image_path = '../input/data/eval/images/'

basemodel_convnext_l = torch.load('../checkpoint/basemodel_convnext_l-detection/checkpoint_best.pth')
submission = pd.read_csv(test_dir_path+'info.csv')
submission.head()


# In[ ]:


image_paths = [os.path.join(test_image_path, img_id) for img_id in submission.ImageID]
test_image = pd.Series(image_paths)


# In[ ]:


class Test_Dataset(Dataset):
    def __init__(self, midcrop=True, transform=None):
        self.midcrop = midcrop
        self.data = test_image
        self.transform = transform
        
    def __len__(self):
        return len(test_image)
    
    def __getitem__(self, idx):
        img = Image.open(self.data[idx])
        img = img.crop((17, 70, 367, 420))
        img = self.transform(img)
        
        return img


# In[ ]:


dataset = Test_Dataset(transform = transforms.Compose([
                                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ]))

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)

# 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)
device = torch.device('cuda')
model = basemodel_convnext_l.to(device)
model.eval()

# 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
all_predictions = []
for images in loader:
    with torch.no_grad():
        images = images.to(device)
        pred = model(images)
        pred = pred.argmax(dim=-1)
        all_predictions.extend(pred.cpu().numpy())
submission['ans'] = all_predictions

# 제출할 파일을 저장합니다.
submission.to_csv(os.path.join(test_dir_path, 'submission_convnext_l.csv'), index=False)
print('test inference is done!')

