import os, sys
import numpy as np

import torch
import SimpleITK as sitk
from torch.utils.data import Dataset

class CT2DDatasetTest(Dataset):
    def __init__(self, args=None, config=None):
        '''
        :param root:
        :param mode:
        '''
        super(CT2DDatasetTest, self).__init__()

        self.image_dir = args.test_file

        self.prefix_img = config.get('data').get('prefix_img')
        self.suffix_img = config.get('data').get('suffix_img')
        self.prefix_label = config.get('data').get('prefix_label')
        self.suffix_label = config.get('data').get('suffix_label')

        self.new_spacing = config.get('data').get('spacing')
        self.cropping = config.get('data').get('cropping')

        with open(self.image_dir, 'r') as f:
            self.image_path_list = f.readlines()
        self.image_path_list = [file.strip() for file in self.image_path_list]

    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        self.image_id = image_path
        image_path = os.path.join(self.prefix_img, self.image_id, self.suffix_img)
        label_path = os.path.join(self.prefix_label, self.image_id, self.suffix_label)

        image = sitk.ReadImage(image_path, sitk.sitkInt16)
        label = sitk.ReadImage(label_path, sitk.sitkUInt8)

        image = self.resample(image, target_spacing=self.new_spacing)
        label = self.resample(label, target_spacing=self.new_spacing)

        label_array = sitk.GetArrayFromImage(label)
        image_array = sitk.GetArrayFromImage(image)

        image_array = self.clip_intensity(image_array)

        image_array, label_array = self.centricCrop(image_array, label_array, crop_size=self.cropping)
        image_array = torch.FloatTensor(image_array).unsqueeze(0)
        label_array = torch.FloatTensor(label_array)
        return image_array.permute(1,0,2,3), label_array, self.image_id

    def __len__(self):
        return len(self.image_path_list)

    def clip_intensity(self, ct_array, intensity_range=(-250, 0)):
        ct_array[ct_array > intensity_range[1]] = intensity_range[1]
        ct_array[ct_array < intensity_range[0]] = intensity_range[0]
        return ct_array

    def centricCrop(self, input_image, input_label, crop_size=(96, 96, 32)):

        assert isinstance(input_image, np.ndarray), 'the input_image should be np.ndarray'

        new_x = int(np.ceil((input_image.shape[0] - crop_size[0]) / 2))
        end_x = new_x + crop_size[0] - 1
        new_y = int(np.ceil((input_image.shape[1] - crop_size[1]) / 2))
        end_y = new_y + crop_size[1] - 1
        new_z = int(np.ceil((input_image.shape[2] - crop_size[2]) / 2))
        end_z = new_z + crop_size[2] - 1
        #
        image_array = input_image[new_x:end_x + 1, new_y:end_y + 1, new_z:end_z + 1]
        label_array = input_label[new_x:end_x + 1, new_y:end_y + 1, new_z:end_z + 1]

        return image_array, label_array

    def resample(self, input_image, target_spacing=(0.5, 0.5, 0.5)):
        '''
        resample the CT image
        :parm input_image:
        :param target_spacing:
        '''
        assert isinstance(input_image, sitk.Image), 'the input_image should be the object of SimpleITK.SimpleITK.Image'
        origin_spacing = input_image.GetSpacing()
        origin_size = input_image.GetSize()
        scale = [target_spacing[index] / origin_spacing[index] for index in range(len(origin_size))]
        new_size = [int(origin_size[index] / scale[index]) for index in range(len(origin_size))]
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
        resample.SetDefaultPixelValue(0)
        resample.SetOutputSpacing(target_spacing)
        resample.SetOutputOrigin(input_image.GetOrigin())
        resample.SetOutputDirection(input_image.GetDirection())
        resample.SetSize(new_size)
        new_image = resample.Execute(input_image)
        return new_image

# the test code
if __name__ == '__main__':
    from config.config import load_config
    args, config = load_config()

    data_test = CT2DDatasetTest(args=args, config=config)
    from torch.utils.data import DataLoader

    test_loader = DataLoader(data_test
                              , batch_size=1
                              , shuffle=False
                              , num_workers=4
                              )

    for i, sample_batched in enumerate(test_loader):
        image_array, label_array, image_path = sample_batched
        print(image_array.shape)
        print(label_array.shape)
        print(image_path)
        break