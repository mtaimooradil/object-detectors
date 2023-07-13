# -----------------------------------------------------
# Written by Muhammad Taimoor Adil on 13/07/2023
# 
# This file contains following utility functions:
#
# iou_width_height
# intersection_over_union
# non_max_suppression
# mean_average_precision
# get_data_loaders
# cells_to_bboxes
# plot_image
# seed_everything
# -----------------------------------------------------

import config
import torch
from torch.utils.data import DataLoader
from collections import Counter

from dataset import YOLODataset

def iou_width_height(box1, box2):

    # box[..., 0] -> width
    # box[..., 1] -> height

    intersection = torch.min(box1[..., 0], box2[..., 0]) * torch.min(box1[..., 0], box2[..., 0])

    union = box1[..., 0] * box1[..., 1] +  box2[..., 0] * box2[..., 1] - intersection

    return intersection / union


def intersection_over_union(box1, box2, box_format = 'midpoint'):

    # box shape -> (batch_size, 4)
    # box_format = corners -> (x1, y1, x2, y2)
    # box_format = midpoint -> (x, y, w, h)

    if box_format == "corners":
        box1_x1 = box1[..., 0:1]
        box1_y1 = box1[..., 1:2]
        box1_x2 = box1[..., 2:3]
        box1_y2 = box1[..., 3:4]
        box2_x1 = box2[..., 0:1]
        box2_y1 = box2[..., 1:2]
        box2_x2 = box2[..., 2:3]
        box2_y2 = box2[..., 3:4]

    elif box_format == "midpoints":
        box1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
        box1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
        box1_x2 = box1[..., 2:3] + box1[..., 2:3] / 2
        box1_y2 = box1[..., 3:4] + box1[..., 3:4] / 2
        box2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
        box2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
        box2_x2 = box2[..., 2:3] + box2[..., 2:3] / 2
        box2_y2 = box2[..., 3:4] + box2[..., 3:4] / 2

    intersection_x1 = torch.max(box1_x1, box2_x1)
    intersection_y1 = torch.max(box1_y1, box2_y1)
    intersection_x2 = torch.min(box1_x2, box2_x2)
    intersection_y2 = torch.min(box1_y2, box2_y2)

    # clamp(0) makes sure that intersection is zero when there is no intersection
    intersection = (intersection_x2 - intersection_x1).clamp(0) * (intersection_y2 - intersection_y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    union = box1_area + box2_area - intersection

    return intersection / union

def non_max_suppression(bboxes, iou_threshold, probability_threshold, box_format = "corners"):

    ## bboxes (list) -> bounding boxes [class_pred, prob_score, x1, y1, x2, y2]

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if bboxes[1] > probability_threshold] # Choose the box with highest probability
    bboxes = sorted(bboxes, key = lambda x: x[1], reverse = True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box 
            for box in bboxes
            if box[0] != chosen_box[0] # Don't want to compare boxes of different classes, if the boxes are of different classes then keep it
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format = box_format
            ) < iou_threshold # keep the box if intersection_over_union is less than threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def mean_average_precision(pred_boxes, true_boxes, iou_threshold = 0.5, box_format = "corners", num_classes = 20):
    
    # pred_boxes (list) : [[train_idx, class, probability, x1, y1, x2, y2], ...]

    average_precisions = []
    epsilon = 1e-6 # for nuumerical stability

    for c in range(num_classes):

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        detections = [detection for detection in pred_boxes if detection[1] == c]
        ground_truths = [ground_truth for ground_truth in true_boxes if ground_truth[1] == c]

        # img 0 has 3 bboxes
        # img 1 has 5 bboxes
        # amount_bboxes = {0: 3, 1: 5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We need to keep track of target bounding boxes that we have covered so far and 
        # we can't have multiple prediction bounding boxes for one target bounding box 
        # and count them all as correct. Only the first one that covers a target bounding box
        # is correct and others would be false positives. This is for keeping track of target 
        # bounding boxes for particular image that we have covered so far.

        # amount_bboxes = {0: torch.tensor([0,0,0]), 1: torch.tensor([0,0,0,0,0])}
        for key, value in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(value)

        # sort by box probabilities which is index 2
        detections.sort(key = lambda x: x[2], reverse = True)

        true_positives = torch.zeros(len(detections))
        false_positives = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):

            # We need comparison only for same training image
            ground_truth_img = [bbox for bbox in ground_truths if ground_truths[0] == detection_idx]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(ground_truth_img[3:]),
                    box_format = box_format
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0: # If we have not covered this ground truth box before
                    # true positive and add this bounding box to seen
                    true_positives[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] == 1
                else:
                    false_positives[detection_idx] = 0

            # if IOU is lower then the detection is a false positive
            else:
                false_positives[detection_idx] = 1

        # [1, 1, 0, 1, 0] -> [1, 2, 2, 3, 3]
        TP_cumsum = torch.cumsum(true_positives, dim = 0)
        FP_cumsum = torch.cumsum(false_positives, dim = 0)

        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat(torch.tensor([1]), precisions) # We need a point for (0,1) to correctly perform numerical integration
        recalls = torch.cat(torch.tensor([0]), recalls) # We need a point for (0,1) to correctly perform numerical integration

        average_precisions.append(torch.trapezoid(precisions, recalls))

    return sum(average_precisions) / len (average_precisions) 



#------------ Following functions are not written by me, I will write them myself later -----------#


def get_loaders(train_csv_path, test_csv_path):

    IMAGE_SIZE = config.IMAGE_SIZE
    
    train_dataset = YOLODataset(
        train_csv_path,
        transform=config.train_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    test_dataset = YOLODataset(
        test_csv_path,
        transform=config.test_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )
    train_eval_dataset = YOLODataset(
        train_csv_path,
        transform=config.test_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    train_eval_loader = DataLoader(
        dataset=train_eval_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, test_loader, train_eval_loader

def cells_to_bboxes():
    pass

def plot_image():
    pass

def seed_everything():
    pass