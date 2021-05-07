import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets


class TrainFolder(Dataset):
    def __init__(self, classes, images, encoder, transform):
        self.classes = classes
        self.images = images
        self.encoder = encoder
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        image_path = self.images[idx]
        image = Image.open(image_path)
        image = np.asarray(image)
        image = torch.from_numpy(image.copy())
        if self.transform:
            image = self.transform(image)
        label = self.classes[idx]
        return image, label


class SiameseDataset(Dataset):
    def __init__(self, train_loader, same_classes_dict) -> None:
        self.train_loader = train_loader
        self.same_classes_dict = same_classes_dict

    def __len__(self):
        return len(self.train_loader)

    def __getitem__(self, idx):
        image_1, label_1 = next(iter(self.train_loader))
        same_class_tag = random.randint(0, 1)
        if same_class_tag:
            same_class_id = np.random.choice(self.same_classes_dict[label_1])
            image_2, label_2 = self.train_loader.__getitem__(same_class_id)
        else:
            different_class_id = np.random.choice(
                [
                    f
                    for f in range(self.__len__())
                    if f not in self.same_classes_dict[label_1]
                ]
            )
            image_2, label_2 = self.train_loader.__getitem__(different_class_id)
        return (
            image_1,
            image_2,
            torch.from_numpy(
                np.array([int(label_1) != int(label_2)], dtype=np.float32)
            ),
        )


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 5),
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2)
            + (label)
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss_contrastive


def main():
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((100, 100)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_path = os.path.join("data", "training")
    classes = []
    images = []
    encoder = {}
    same_classes_dict = {}
    current_ind = 0
    for path in os.listdir(train_path):
        if not path.startswith("s"):
            continue
        encoder[path] = current_ind
        current_images = os.listdir(os.path.join(train_path, path))
        current_images = [f for f in current_images if f.endswith(".pgm")]
        current_images = [os.path.join(train_path, path, f) for f in current_images]
        images += current_images
        same_classes_dict[current_ind] = [
            len(classes) + f for f in range(0, len(current_images))
        ]
        classes += [current_ind] * len(current_images)

        current_ind += 1
    assert len(images) == len(classes)

    train_folder = TrainFolder(classes, images, encoder, transform)
    # # data_loader = DataLoader(train_folder, batch_size=)
    # print(next(iter(train_folder))[1])
    siamese_loader = SiameseDataset(train_folder, same_classes_dict)
    dataloader = DataLoader(siamese_loader, batch_size=8, shuffle=True)
    net = SiameseNetwork()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)

    counter = []
    loss_history = []
    iteration_number = 0
    device = "cpu"
    for epoch in range(0, 20):
        for i, data in enumerate(dataloader):
            img0, img1, label = data
            img0 = img0.to(device)
            img1 = img1.to(device)
            label = label.to(device)

            img0 = img0.view(-1, 1, 100, 100)
            img1 = img1.view(-1, 1, 100, 100)
            optimizer.zero_grad()
            output1, output2 = net(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            if i % 10 == 0:
                print(
                    "Epoch number {}\n Current loss {}\n".format(
                        epoch, loss_contrastive.item()
                    )
                )
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())


if __name__ == "__main__":
    main()
