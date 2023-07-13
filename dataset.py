# ---------------------------------------------------------------------------------
# Written by Muhammad Taimoor Adil on 13/07/2023
# 
# This file contains the dataset class for YOLOv3 to import PASCAL VOC and MS COCO
# ---------------------------------------------------------------------------------

import config
import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLOv3Dataset(Dataset):

    def __init__(
            self,
            annotations_file,
            labels_dir,
            images_dir,
            anchors,
            image_size=416,
            S = [13, 25, 52], # grid size
            C = 20, # number of classes
            transforms = None
    ):
        super().__init__()
        self.annotations = pd.read_csv(annotations_file)
        self.labels_dir = labels_dir
        self.images_dir = images_dir
        self.transforms = transforms
        self.S = S
        self.C = C
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        
        labels_path = os.path.join(self.labels_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=labels_path, delimiter=" ", ndmin=2), 4, axis=1).tolist() # [class, x, y, w, h] -> [x, y, w, h, class]
        images_path = os.path.join(self.images_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(images_path).convert('RGB'))

        if self.transforms:
            augmentations = self.transforms(image = image, bboxes = bboxes)
            image = augmentations['image']
            bboxes = augmentations['bboxes']

        # Below assumes same number of anchors per scale
        targets = [torch.zeros((self.num_anchors_per_scale, S, S, 6)) for S in self.S] # 6 values for # [p_o, x, y, w, h, class]

        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors) # Here we want to know how much of an area an anchor covers with the box
            anchor_idices = iou_anchors.argsort(descending = True, dim = 0)
            x, y, width, height, class_label = box
            has_anchors = [False] * len(self.S) # This is to track whether a scale for this specific bounding box has already been assigned an anchor 

            for anchor_idx in anchor_idices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx] # get the grid size for particular scale
                i, j = int(S * y), int(S * x) # get which cell the box is in
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                if not anchor_taken and not has_anchors[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1 # # Now we have taken this anchor, set the probability of objectness to 1
                    # Now we need to calculate x, y, width and height with respect to the grid cells
                    # It is the decimal number which was discarded when we did int(S * y) and int(S * x)
                    x_cell, y_cell = S * x - j, S * y - i # both between [0,1]
                    width_cell, height_cell = width * S, height * S # can be greater than 1 since it's relative to cell

                    # box  coordinates relative to cell
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                    # Set up the targets (that will be returned) 
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = class_label

                    has_anchors[scale_idx] = True # The present scale has been assigned an anchor

                # If this scale already has an anchor and there is another candidate then check iou threshold and ignore
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1 # Ignore the prediction

        return image, tuple(targets)
    


#------ I did not write the following test function ------#


def test():
    anchors = config.ANCHORS

    transform = config.test_transforms

    dataset = YOLOv3Dataset(
        "D:\Image Datasets\PASCAL_VOC/train.csv",
        "D:\Image Datasets\PASCAL_VOC/labels/",
        "D:\Image Datasets\PASCAL_VOC/images/",
        S=[13, 26, 52],
        anchors=anchors,
        transforms=transform,
    )
    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for x, y in loader:
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            print(anchor.shape)
            print(y[i].shape)
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]
        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        print(boxes)
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)


if __name__ == "__main__":
    test()