import torch
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2,MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import io
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import seaborn as sns
from skimage.segmentation import clear_border
from skimage.segmentation import expand_labels



#def get_model():
#    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,\
#                                     box_detections_per_img = 200) #pretrained=True
#    in_features = model.roi_heads.box_predictor.cls_score.in_features
#    model.roi_heads.box_predictor = FastRCNNPredictor(
#            in_features, num_classes=1+4) 
#    return model
def get_model(num_classes=5, num_domains=6):
    model = CustomMaskRCNN(num_classes=num_classes, num_domains=num_domains)
    return model



def load_checkpoint(checkpoint, model, optimizer, lr):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def predict_multi_frame(model, frame, IMAGE_SIZE, score=0.7, msk_prob = 0.7, DEVICE = "cuda" if torch.cuda.is_available() else "cpu"):
    images = cv2.resize(frame, IMAGE_SIZE, cv2.INTER_LINEAR)/255
    images = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0)
    images = images.swapaxes(1, 3).swapaxes(2, 3)
    images = list(image.to(DEVICE) for image in images)
    
    with torch.no_grad():
      pred = model(images)

    # color information 
    cell_gray = [0,64,128,192,255]
    cell_name = ['stuff','vessel','ray','fiber','parenchyma']

    im = images[0].swapaxes(0, 2).swapaxes(0, 1).detach().cpu().numpy().astype(np.float32)
    im3 = np.zeros_like(im[:,:,0]).astype(np.float32)
    
    for i in range(len(pred[0]['masks'])):
        msk = pred[0]['masks'][i,0].detach().cpu().numpy()
        scr = pred[0]['scores'][i].detach().cpu().numpy()
        cell_type = pred[0]['labels'][i].detach().cpu().numpy()
        
        if scr > score:  #0.78  ############
            #im3[:, :][msk > msk_prob] = cell_gray[cell_type]
            im3[msk > msk_prob] = cell_gray[cell_type]

    im4 = expand_labels(im3.astype(np.uint8), distance=1)  ## 
            
    return (im3)

def predict_multi_frame_0(frame):
    images = cv2.resize(frame, IMAGE_SIZE, cv2.INTER_LINEAR)/255
    images = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0)
    images = images.swapaxes(1, 3).swapaxes(2, 3)
    images = list(image.to(DEVICE) for image in images)
    
    with torch.no_grad():
      pred = model(images)

    # color information 
    cell_color = [(0,0,0),(1,159/255,155/255),(144/255,229/255,161/255),\
                  (161/255,201/255,244/255),(1,180/255,130/255)]
    cell_gray = [0,64,128,192,255]
  
    cell_name = ['stuff','vessel','ray','fiber','parenchyma']
    im = images[0].swapaxes(0, 2).swapaxes(0, 1).detach().cpu().numpy().astype(np.float32)
    im2 = np.zeros_like(im).astype(np.float32)
    im3 = np.zeros_like(im[:,:,0]).astype(np.float32)
    
    msk=pred[0]['masks'].detach().cpu().numpy()
    scr=pred[0]['scores'].detach().cpu().numpy()
    box=pred[0]['boxes'].detach().cpu().numpy()
    cell_ids=pred[0]['labels'].detach().cpu().numpy()
    
    dfx = pd.DataFrame()
    dfx['labels'] = cell_ids
    dfx['scores'] = scr
    dfx['width'] = [x2-x1 for x1,y1,x2,y2 in list(box)]
    dfx['height'] = [y2-y1 for x1,y1,x2,y2 in list(box)]
    area_w = [np.unique(k.flatten()>0.5, return_counts=True) for k in msk]
    area = [j[1][1] for j in area_w]
    dfx['area'] = area
    cell_types = ['boundary', 'vessel','ray','fiber', 'parenchyma']
    cell_name = [cell_types[n] for n in cell_ids]
    dfx['cell_name']= cell_name
    
    for i in range(len(pred[0]['masks'])):
        msk=pred[0]['masks'][i,0].detach().cpu().numpy()
        scr=pred[0]['scores'][i].detach().cpu().numpy()
        box=pred[0]['boxes'][i].detach().cpu().numpy()
        cell_type=pred[0]['labels'][i].detach().cpu().numpy()
        
        if scr>0.6 :  #0.9  ############
            im2[:, :, :][msk > 0.87] = cell_color[cell_type]   ################
            im3[:, :][msk > 0.87] = cell_gray[cell_type] 
    return (cv2.addWeighted(im, 0, im2, 1, 0)*255).astype(np.uint8),im3.astype(np.uint8),dfx, pred



def predict_single_frame(frame):
    images = cv2.resize(frame, IMAGE_SIZE, cv2.INTER_LINEAR)/255
    images = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0)
    images = images.swapaxes(1, 3).swapaxes(2, 3)
    images = list(image.to(DEVICE) for image in images)
    
    with torch.no_grad():
      pred = model(images)

    # color information 
    #colors = [(0,0,0),(1,0,0),(0,1,0),(1,1,1),(0,0,1)]
    cell_color = [(0,0,0),(1,159/255,155/255),(144/255,229/255,161/255),\
                  (161/255,201/255,244/255),(1,180/255,130/255)]
    
    cell_name = ['stuff','vessel','ray','fiber','parenchyma']
    im = images[0].swapaxes(0, 2).swapaxes(0, 1).detach().cpu().numpy().astype(np.float32)
    im2 = np.zeros_like(im).astype(np.float32)
    
    msk=pred[0]['masks'].detach().cpu().numpy()
    scr=pred[0]['scores'].detach().cpu().numpy()
    box=pred[0]['boxes'].detach().cpu().numpy()
    cell_ids=pred[0]['labels'].detach().cpu().numpy()
    
    dfx = pd.DataFrame()
    dfx['labels'] = cell_ids
    dfx['scores'] = scr
    dfx['width'] = [x2-x1 for x1,y1,x2,y2 in list(box)]
    dfx['height'] = [y2-y1 for x1,y1,x2,y2 in list(box)]
    area_w = [np.unique(k.flatten()>0.5, return_counts=True) for k in msk]
    area = [j[1][1] for j in area_w]
    dfx['area'] = area
    cell_types = ['boundary', 'vessel','ray','fiber', 'parenchyma']
    cell_name = [cell_types[n] for n in cell_ids]
    dfx['cell_name']= cell_name
    
    for i in range(len(pred[0]['masks'])):
        msk=pred[0]['masks'][i,0].detach().cpu().numpy()
        scr=pred[0]['scores'][i].detach().cpu().numpy()
        box=pred[0]['boxes'][i].detach().cpu().numpy()
        cell_type=pred[0]['labels'][i].detach().cpu().numpy()
        
        if scr>0.6 :  #0.9  ############
            #cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), colors[cell_type], 2)
            cv2.putText(im, str(cell_name[cell_type])+"_"+"{0:.2f}%".format(scr*100), (int(box[0]+5), int(box[1])+15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1, cv2.LINE_AA)
            im2[:, :, :][msk > 0.67] = cell_color[cell_type]   ################

    return (cv2.addWeighted(im, 0.6, im2, 0.4, 0)*255).astype(np.uint8),dfx, pred

def circle_plot(df):
    category_totals = df.groupby('cell_name')['area'].sum().reset_index()
    
    labels = category_totals['cell_name']
    sizes = category_totals['area']
    u, counts = np.unique(df['cell_name'], return_counts=True)
    colors = sns.color_palette('pastel')[0:len(labels)]
    
    items = [labels.values, labels]
    amount = [sizes.values, counts]
    comment = ['Occupied area of each cell','Number of each cell']
    
    fig, ax = plt.subplots(2,1,figsize=(4,4))
    for j in range(2):
        ax[j].pie(amount[j], labels=items[j], colors=colors, autopct='%1.1f%%', startangle=140)
        ax[j].axis('equal') 
        ax[j].set_title(comment[j])
    plt.tight_layout()
    plt.show()

def image_circle_plot(result,df, save_data=False):
    category_totals = df.groupby('cell_name')['area'].sum().reset_index()
    
    labels = category_totals['cell_name']
    sizes = category_totals['area']
    u, counts = np.unique(df['cell_name'], return_counts=True)
    colors = sns.color_palette('pastel')[0:len(labels)]
    
    items = [labels.values, labels]
    amount = [sizes.values, counts]
    comment = ['Occupied area of each cell','Number of each cell']

    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.imshow(result)
    plt.axis('off')
    plt.subplot(222)
    plt.pie(amount[0], labels=items[0], colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal') 
    plt.title(comment[0],fontsize=10)
    plt.subplot(224)
    plt.pie(amount[1], labels=items[1], colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal') 
    plt.title(comment[1],fontsize=10)
    plt.tight_layout()
    if save_data == True:
        buf = io.BytesIO() # prepare buffer
        plt.savefig(buf, format='png') # store in the buffer
        enc = np.frombuffer(buf.getvalue(), dtype=np.uint8) # read from buffer
        dst = cv2.imdecode(enc, 1) # decoding
        dst = dst[:,:,::-1] # BGR->RGB
        plt.close()
        return dst
    else:
        plt.show()

def image_sequence(result,save_data=False):
    # category_totals = df.groupby('cell_name')['area'].sum().reset_index()
    
    # labels = category_totals['cell_name']
    # sizes = category_totals['area']
    # u, counts = np.unique(df['cell_name'], return_counts=True)
    # colors = sns.color_palette('pastel')[0:len(labels)]
    
    # items = [labels.values, labels]
    # amount = [sizes.values, counts]
    # comment = ['Occupied area of each cell','Number of each cell']

    plt.figure(figsize=(5,5))
    plt.subplot(111)
    plt.imshow(result)
    plt.axis('off')
    if save_data == True:
        buf = io.BytesIO() # prepare buffer
        plt.savefig(buf, format='png') # store in the buffer
        enc = np.frombuffer(buf.getvalue(), dtype=np.uint8) # read from buffer
        dst = cv2.imdecode(enc, 1) # decoding
        dst = dst[:,:,::-1] # BGR->RGB
        plt.close()
        return dst
    else:
        plt.show()