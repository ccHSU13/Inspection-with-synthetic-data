import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageDraw
import cv2

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T


class DefectDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "OriginalImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "Masks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "OriginalImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "Masks", self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        mask = cv2.imread(mask_path)
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret,binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

        contours,_  = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        labels = []

        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            if w < 10 or h < 10:
              continue
            
            b = int(mask[y+ h//2, x+w//2][0])
            g = int(mask[y+ h//2, x+w//2][1])
            r = int(mask[y+ h//2, x+w//2][2])

            boxes.append([x,y,x+w,y+h])
            
            # red -> crack
            if r-g > 70 and r-b > 70:
                labels.append(1)

            # green -> efflorescence
            elif g-r > 70 and g-b > 70:
                labels.append(2)
                
            # white -> spalling
            elif g-r < 10 and r-b < 10:
                labels.append(3)
                
            # Blue -> exposed rebar
            elif b -g >70 and b-r > 70:
                labels.append(3)
            
            # Yellow -> crack & efflorescence
            elif g-b > 70 and r-b > 70:
                labels.append(2)
                boxes.append([x,y,x+w,y+h])
                labels.append(1)
            
            # Cyan -> exposed rebar & spalling
            elif b-r > 70 and g-r > 70:
                labels.append(4)
                boxes.append([x,y,x+w,y+h])
                labels.append(3)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there are four classes: Spalling/Exposed rebar/Efflorescence/Crack
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_faster_rcnn_model(num_classes):
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.8))
      
    return T.Compose(transforms)


# use our dataset and defined transformations
root = "/home/chunching/myenv/src/BridgeInspection/OutputWithoutColorramp"

dataset = DefectDataset(root, get_transform(train=True))
dataset_test = DefectDataset(root, get_transform(train=False))

print("...Complete loading dataset")


# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 5

# get the model using our helper function
model2 = get_faster_rcnn_model(num_classes)
# move model to the right device
model2.to(device)

# construct an optimizer
params = [p for p in model2.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

print("...Start training")

num_epochs = 30

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model2, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model2, data_loader_test, device=device)
    

print("...Finish training")

# pick one image from the test set
img, targets = dataset_test[0]

# put the model in evaluation mode
model2.eval()
with torch.no_grad():
    prediction = model2([img.to(device)])
img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())

bbs= prediction[0]['boxes'].byte().cpu().numpy()
for bb in bbs:
    draw = ImageDraw.Draw(img)
    draw.rectangle([bb[0], bb[1], bb[2], bb[3]],outline='yellow',width=2)

ground_truth = targets['boxes']
for box in ground_truth:
    draw = ImageDraw.Draw(img)
    draw.rectangle([box[0], box[1], box[2], box[3]],outline='red',width=2)

img = img.save('./output1.jpg')