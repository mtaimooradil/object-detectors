# -----------------------------------------------------
# Written by Muhammad Taimoor Adil on 13/07/2023
# 
# This file contains the loss class for YOLOv3
# -----------------------------------------------------

import torch
import torch.nn as nn

from utils import intersection_over_union as iou


class YOLOv3Loss(nn.Module):

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Loss weighting constants
        self.lambda_object = 1
        self.lambda_no_object = 1
        self.lambda_class = 1
        self.lambda_box = 1


    def forward(self, predictions, targets, anchors):

        # Object and No Object Masks
        obj = targets[..., 0] == 1  # first element of last dimension of targets contain objectness score (0 or 1)
        noobj = targets[..., 0] == 0

        #---------- No Object Loss ----------#
        loss_no_object = self.lambda_no_object * self.bce(predictions[..., 0:1][noobj], targets[..., 0:1][noobj]) # 0:1 instead of 0 to keep the dimension, we don't want it to squeeze it

        #---------- Object Loss ----------#
        anchors = anchors.reshape(1, 3, 1, 1, 2) # from original shape of (3, 2) for broadcasting
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5] * anchors)], dim = -1)
        ious = iou(box_preds[obj], targets[..., 1:5][obj]).detach()
        
        loss_object = self.lambda_object * self.bce(predictions[..., 0:1][obj], ious)

        #---------- Class Loss ----------#
        loss_class = self.lambda_class * self.entropy(predictions[..., 5:][obj], targets[..., 5][obj].long()) # long() converts it to int64

        #---------- Box Prediciton Loss ----------#
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3]) # x, y to be between [0, 1]
        # Here, according to the formulas, we would have to take exp of network outputs to match target format but taking log of target  
        # This allows for better gradient flow than exponential function
        targets[..., 3:5] = torch.log(1e-6 + targets[..., 3:5] / anchors)

        loss_box = self.lambda_box * self.mse(predictions[..., 1:5][obj], targets[..., 1:5][obj]) # 1e-6 for numerical stability

        #---------- Total Loss ----------#
        loss = loss_no_object + loss_object + loss_class + loss_box

        return loss #, tuple(loss_no_object, loss_object, loss_class, loss_box)


