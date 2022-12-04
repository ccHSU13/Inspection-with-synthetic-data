import torch
import torch.utils.data as data
from pycocotools.coco import COCO
import os
from PIL import Image
import cv2


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        ann_id = self.ids[index]
        # print(coco.anns[ann_id])

        labels = [coco.anns[ann_id]['category_id']]
        x, y, w, h = coco.anns[ann_id]['bbox']
        boxes = [[x, y, x+w+1, y+h+1]]
        area = [coco.anns[ann_id]['area']]
        img_id = coco.anns[ann_id]['image_id']
        iscrowd = torch.zeros(1, dtype=torch.int64)

        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root, path)).convert('RGB')

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] =  torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([img_id])
        target["area"] = torch.as_tensor(area, dtype=torch.float32)
        target["iscrowd"] = iscrowd

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.ids)

class DefectDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(
            sorted(os.listdir(os.path.join(root, "OriginalImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "Masks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "OriginalImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "Masks", self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        mask = cv2.imread(mask_path)
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        labels = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 30 or h < 30:
                continue

            b = int(mask[y + h//2, x+w//2][0])
            g = int(mask[y + h//2, x+w//2][1])
            r = int(mask[y + h//2, x+w//2][2])

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
                labels.append(4)

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
    
