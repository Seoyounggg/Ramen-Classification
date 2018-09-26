# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import RandomCrop, Resize, Compose, ToTensor
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
import time

import sys
sys.path.insert(0, '/home/yangho/철기연과제/피킹/picker_net/pyrealsense/')
import pyrealsense2 as rs
import cv2


imagesize = (224,224,3)
labeldict = {'0' : '오징어짬뽕', '1' : '비빔면', '2' : '짜왕', '3' : '무파마'}


def predict(model, imgarr, device):

    inimage = cv2.cvtColor(imgarr, cv2.COLOR_BGR2RGB)
    inimage = cv2.resize(inimage, (imagesize[0], imagesize[1]))
    inimage = np.transpose(inimage, (2, 0, 1)) / 255
    inimage = np.expand_dims(inimage, axis=0)
    inimage = torch.tensor(inimage.astype('float32'))
    inimage = inimage.to(device)
    netout = model(torch.tensor(inimage))
    label = str(netout.argmax().item())
    return labeldict[label]



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('runing device : {}'.format(device))



model = torch.load('./best#111.pb')
model.to(device)
model.eval()


pipeline = rs.pipeline()
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

cropxmin, cropxmax, cropymin, cropymax = 100, 580, 0, 480

while True:
    # Get frameset of color and depth
    frames = pipeline.wait_for_frames()
    # Align the depth frame to color frame
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    rec_c = color_image[cropymin:cropymax, cropxmin:cropxmax, :]
    cv2.imwrite('img1.png', rec_c)
    img = cv2.imread('./img1.png')
    #rec_c = cv2.cvtColor(rec_c, cv2.COLOR)
    print(rec_c.shape)
    sol = predict(model, img, device)
    print('prediction : {}'.format(sol))

    cv2.imshow('result', img)

    cv2.waitKey(40)


