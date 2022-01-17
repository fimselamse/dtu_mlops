"""
LFW dataloading
"""
import argparse
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import glob
import matplotlib.pyplot as plt


class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        self.imgs_path = path_to_folder
        file_list = glob.glob(self.imgs_path + "*")
        # print(file_list)
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("\\")[-1]
            for img_path in glob.glob(class_path + "\\*.jpg"):
                self.data.append([img_path, class_name])
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        entry = self.data[index]
        image = Image.open(entry[0])
        label = entry[1]
        
        return self.transform(image), label

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='lfw/', type=str)
    parser.add_argument('-batch_size', default=1028, type=int)
    parser.add_argument('-num_workers', default=0, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', action='store_true')
    parser.add_argument('-batches_to_check', default=5, type=int)
    
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.ToTensor()
    ])
    
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
        
    # Define dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    
    if args.visualize_batch:
        # TODO: visualize a batch of images
        figure = plt.figure(figsize=(14, 8))
        cols, rows = int(len(dataloader)/2), 2
        batch = next(iter(dataloader))
        images = batch[0]
        labels = batch[1]
        for i in range(1, cols * rows + 1):
            img, label = images[i - 1], labels[i - 1]
            figure.add_subplot(rows, cols, i)
            plt.title(label)
            plt.axis("off")
            plt.imshow(img.permute(1,2,0), cmap="gray")
        plt.savefig("visualization.jpg")
                
    if args.get_timing:
        # lets do some repetitions
        res = [ ]
        for _ in range(5):
            start = time.time()
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > args.batches_to_check:
                    break
            end = time.time()

            res.append(end - start)
            
        res = np.array(res)
        print(f'Timing: {np.mean(res)}+-{np.std(res)}')
