import os, sys
import numpy as np

import torch
from torch.utils.data import Dataset

class CT2DDatasetTrain(Dataset):
    def __init__(self, args=None, config=None):
        super(CT2DDatasetTrain, self).__init__()

        self.image_dir = args.train_file

        with open(self.image_dir, 'r') as f:
            self.image_path_list = f.readlines()
        self.image_path_list = [file.strip() for file in self.image_path_list]

    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        label_path = image_path.replace('img', 'label')
        img_arr = np.load(image_path)
        label_arr = np.load(label_path)
        image_tensor = torch.FloatTensor(img_arr).unsqueeze(0)
        label_tensor = torch.FloatTensor(label_arr)
        return image_tensor, label_tensor


    def __len__(self):
        return len(self.image_path_list)

# the test code
if __name__ == '__main__':
    class TempArgs:
        def __init__(self, train_file):
            self.train_file = train_file
    data_train = CT2DDatasetTrain(args=TempArgs('/home/zli17/data/nifti/split2d_train/img_slice_list.txt'))
    from torch.utils.data import DataLoader

    train_loader = DataLoader(data_train
                              , batch_size=2
                              , shuffle=False
                              , num_workers=0
                              )
    for i_batch, sample_batched in enumerate(train_loader):
        image_array, label_array = sample_batched
        print(image_array.shape)
        print(label_array.shape)
        break

