# -----------------------------------------------------
# Written by Muhammad Taimoor Adil on 13/07/2023
# 
# This file contains training code for YOLOv3
# -----------------------------------------------------

import config
import torch
import torch.optim as optim
from tqdm import tqdm

from model import YOLOv3
from loss import YOLOv3Loss
from utils import (
    get_data_loaders,
    save_checkpoint,
    check_class_accuracy,
    get_evaluation_bboxes,
    mean_average_precision
)

import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True


def training_function(dataloader, model, loss_function, optimizer, scaler, scaled_anchors):
    loop = tqdm(dataloader, leave=True) # This is for progress bar for loops
    losses = []

    for batch, (x, y) in enumerate(tqdm(dataloader)):
        x = x.to(config.DEVICE)
        x = x.float()

        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE)
        )

        with torch.cuda.amp.autocast():
            out = model(x)

            # Total loss is the sum of losses for each scale prediction
            loss = (
                loss_function(out[0], y0, scaled_anchors[0]) +
                loss_function(out[1], y1, scaled_anchors[1]) +
                loss_function(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)



def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    loss_function = YOLOv3Loss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr = config.LEARNING_RATE, 
        weight_decay = config.WEIGHT_DECAY
        )

    # https://pytorch.org/docs/stable/amp.html#gradient-scaling
    scaler = torch.cuda.amp.GradScaler()
    
    train_loader, test_loader, train_eval_loader = get_data_loaders(
        train_csv_path = "./data/" + config.DATASET + "//8examples.csv",
        test_csv_path = "./data/" + config.DATASET + "//8examples.csv"
    )

    # Scale anchors to each prediction scale (originally anchors are w.r.t original image size, we need them relative to grid cell size)
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    for epoch in range(config.NUM_EPOCHS):
        training_function(test_loader, model, loss_function, optimizer, scaler, scaled_anchors)

        if config.SAVE_MODEL:
            save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")

        if epoch % 10 == 0 and epoch > 0:
            print("On Test loader:")
            check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
            # Run model on test set and convert outputs to bounding boxes relative to image
            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            # Compute mean average precision 
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")
            model.train()

if __name__ == "__main__":
    main()
