import os
import torch
import torch.utils.data

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T
from dataset_loader import CocoDataset, DefectDataset

import matplotlib.pyplot as plt

cur_dir = os.path.abspath(os.path.dirname(__file__))


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


def load_real_data():
    root = "root"
    json = f"{root}/coco.json"

    dataset = CocoDataset(root, json, get_transform(train=False))
    dataset_test = CocoDataset(root, json, get_transform(train=False))

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-1000])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-1000:])

    return dataset, dataset_test


def load_synthetic_data(path, test_path="", split_it=True):
    dataset = DefectDataset(path, get_transform(train=True))
    dataset_test = DefectDataset(path, get_transform(train=False))

    if test_path:
        dataset_test = DefectDataset(test_path, get_transform(train=False))

    if split_it:
        # split the dataset in train and test set
        torch.manual_seed(1)
        indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:-300])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[-300:])
    else:
        dataset = torch.utils.data.Subset(dataset, range(len(dataset)))
        dataset_test = torch.utils.data.Subset(dataset_test, range(len(dataset_test)))

    return dataset, dataset_test


def save_loss_plot(OUT_DIR, train_loss):
    figure_1, train_ax = plt.subplots()
    train_ax.plot(train_loss, color="tab:blue")
    train_ax.set_xlabel("iterations")
    train_ax.set_ylabel("train loss")
    figure_1.savefig(f"{OUT_DIR}/train_loss.png")
    print("SAVING PLOTS COMPLETE...")
    plt.close("all")


def train_model(
    dataset,
    dataset_test,
    num_classes,
    num_epochs,
    save_it=True,
    output_path=f"{cur_dir}/../models/model.pth",
):
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # get the model using our helper function
    model = get_faster_rcnn_model(num_classes)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    print("...Start training")

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_loss = train_one_epoch(
            model, optimizer, data_loader, device, epoch, print_freq=50
        )
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        eval_result = evaluate(model, data_loader_test, device=device)
        save_loss_plot("plots", train_loss)

    print("...Finish training")

    if save_it:
        # save the parameter of models
        torch.save(model.state_dict(), output_path)
        print("...Model was saved")


def predict_result(test_path, model_path="../models/model.pth"):
    dataset_test = DefectDataset(test_path, get_transform(train=False))
    dataset_test = torch.utils.data.Subset(dataset_test, range(len(dataset_test)))
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_faster_rcnn_model(4)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    evaluate(model, data_loader_test, device=device)


root2 = f"{cur_dir}/BridgeInspection/combined"
test_root2 = f"{cur_dir}/BridgeInspection/testing/output_test"

# dataset, dataset_test = load_synthetic_data(root2) # load_real_data()
# print("...Complete loading dataset")

# train_model(dataset, dataset_test, 4, 10, True)
# predict_result(test_root2)

### WIP: store the model here to test evaluation

# for i in range(len(dataset_test)):
#     # pick one image from the test set
#     img, targets = dataset_test[i]

#     # put the model in evaluation mode
#     model.eval()
#     with torch.no_grad():
#         prediction = model([img.to(device)])
#     img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())

#     # red 1/ green 2/ white 3/ blue 4
#     bbs= prediction[0]['boxes'].cpu().detach().numpy()
#     lab = prediction[0]['labels']
#     # for bb in bbs:
#     #     draw = ImageDraw.Draw(img)
#     #     draw.rectangle([bb[0], bb[1], bb[2], bb[3]],outline='yellow',width=2)

#     for i in range(len(bbs)):
#         color = 'red'

#         bb = bbs[i]
#         draw = ImageDraw.Draw(img)
#         draw.rectangle([bb[0], bb[1], bb[2], bb[3]],outline=color,width=2)

#     ground_truth = targets['boxes']
#     for box in ground_truth:
#         draw = ImageDraw.Draw(img)
#         draw.rectangle([box[0], box[1], box[2], box[3]],outline='red',width=2)

#     img = img.save('./predict_result/spalling/600_seperatedTest{}.jpg'.format(str(i)))
