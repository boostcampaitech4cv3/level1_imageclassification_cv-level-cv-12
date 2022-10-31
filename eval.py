import os
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

from align_faces import warp_and_crop_face, get_reference_facial_points

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torchvision.transforms as T
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, CenterCrop, ColorJitter
from PIL import Image

### RetinaFace
sys.path.append('RetinaFace')
from data import cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm

### Data configure
eval_dir_path = '../input/data/eval/'
eval_csv_path = eval_dir_path + 'info.csv' 

cropped_img_path = '../input/data/eval_crop/'
try: 
    os.mkdir(cropped_img_path, mode = 0o755)
except OSError as error:
    pass

### Configure
cfg = cfg_re50
confidence_threshold = 0.015
top_k = 5000
nms_threshold = 0.4
keep_top_k = 1
vis_thres = 0.6

### Definition
# Position
BBoxX1 = 0 ## Left
BBoxY1 = 1 ## Top
BBoxX2 = 2 ## Right
BBoxY2 = 3 ## Bottom
# Score
SCORE = 4
# Landmark
LE_X = 5
LE_Y = 6
RE_X = 7
RE_Y = 8
N_X = 9
N_Y = 10
LM_X = 11
LM_Y = 12
RM_X = 13
RM_Y = 14


def get_files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield os.path.join(path, file)


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

# Fix SEED
random_seed = 12
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


### net and model
device = torch.device("cuda")

detector = RetinaFace(cfg=cfg, phase='test')
detector = load_model(detector, './RetinaFace/weights/Resnet50_Final.pth', False)
detector.eval()
detector = detector.to(device)

transform = T.Compose([
            T.ToTensor()
            #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


basemodel_resnet34 = torch.load('../checkpoint/resnet34_with_detect/checkpoint_ep_30.pth')
basemodel_resnet34.eval()
basemodel_resnet34 = basemodel_resnet34.to(device)

base_transform = transforms.Compose([
                                    ToTensor(),
                                    #Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                            ])


default_square = True
inner_padding_factor = 0.05
outer_padding = (0, 0)
output_size = (224, 224)

all_predictions = []

with torch.no_grad():
    resize = 1

    # Convert and Detect -- Eval (Test only)
    eval_csv = pd.read_csv(eval_csv_path)
    det_idx = 0
    for i in tqdm(range(len(eval_csv))):
        ## Load and detection
        filepath = eval_dir_path + f"images/{eval_csv.iloc[i]['ImageID']}"
        file_name, file_extension = os.path.splitext(filepath)
        if os.path.isfile(file_name + '.npy') == False:
            img = cv2.cvtColor(cv2.imread(filepath, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            np.save(file_name + '.npy', img)
        
        img_raw_rgb = np.load(file_name + '.npy')
        img_raw_bgr = cv2.cvtColor(img_raw_rgb, cv2.COLOR_BGR2RGB)

        img = np.float32(img_raw_bgr)
        img -= (104, 117, 123)
    
        img_height, img_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img = transform(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        loc, conf, landms = detector(img)  # Forward
        
        priorbox = PriorBox(cfg, image_size=(img_height, img_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, nms_threshold,force_cpu=False)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)

        # Not Found Face
        if len(dets) != 1:
            print(filepath)
            #b = [0, 384,  64, 448, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            b = list(map(int, dets[0]))

        # show image
        #text = "{:.4f}".format(b[SCORE])
        #img_output = img_raw.copy()
        #cv2.rectangle(img_output, (b[BBoxX1], b[BBoxY1]), (b[BBoxX2], b[BBoxY2]), (0, 0, 255), 2)
        #cx = b[BBoxX1]
        #cy = b[BBoxY1] + 12
        #cv2.putText(img_output, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        #cv2.circle(img_output, (b[LE_X], b[LE_Y]), 1, (0, 0, 255), 4)
        #cv2.circle(img_output, (b[RE_X], b[RE_Y]), 1, (0, 255, 255), 4)
        #cv2.circle(img_output, (b[ N_X], b[ N_Y]), 1, (255, 0, 255), 4)
        #cv2.circle(img_output, (b[LM_X], b[LM_Y]), 1, (0, 255, 0), 4)
        #cv2.circle(img_output, (b[RM_X], b[RM_Y]), 1, (255, 0, 0), 4)

        # save image
        #cv2.imwrite(f"{cropped_img_path}{det_idx}.jpg", img_output)

        # affine
        #crop_with = b[BBoxX2] - b[BBoxX1]
        #crop_height = b[BBoxY2] - b[BBoxY1]
        #center_pos = (int(crop_with / 2) + b[BBoxX1], int(crop_height / 2) + b[BBoxY1])
        #crop_offset = max(int(crop_with / 2), int(crop_height / 2))

        #cropped_img = img_affine[max(0, center_pos[1] - crop_offset):min(img_height, center_pos[1] + crop_offset),
        #                        max(0, center_pos[0] - crop_offset):min(img_height, center_pos[0] + crop_offset)]
        #cropped_img = cv2.resize(cropped_img, (256, 256), interpolation=cv2.INTER_CUBIC)

        
        facial5points = [[b[LE_X], b[LE_Y]],
                         [b[RE_X], b[RE_Y]],
                         [b[ N_X], b[ N_Y]],
                         [b[LM_X], b[LM_Y]],
                         [b[RM_X], b[RM_Y]]
                        ]
        reference_5pts = get_reference_facial_points(output_size, inner_padding_factor, outer_padding, default_square)
        align_crop_img = warp_and_crop_face(img_raw_bgr, facial5points, reference_pts=reference_5pts, crop_size=output_size)

        cv2.imwrite(f"{cropped_img_path}{det_idx}-cropped.jpg", align_crop_img)
        det_idx += 1

        ## Classification
        align_crop_img = base_transform(align_crop_img).unsqueeze(0)
        align_crop_img = align_crop_img.to(device)

        pred = basemodel_resnet34(align_crop_img)
        pred = pred.argmax(dim=-1)
        all_predictions.extend(pred.cpu().numpy())

eval_csv['ans'] = all_predictions
eval_csv.to_csv(os.path.join(eval_dir_path, 'submission_detect.csv'), index=False)
print('test inference is done!')