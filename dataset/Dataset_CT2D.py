import os, sys
import numpy as np
import random
import SimpleITK as sitk
from torch.utils.data import Dataset

lower = -250

class CT2DDataset(Dataset):
    def __init__(self, mode='train', args=None, config=None):
        super(CT2DDataset, self).__init__()
        self.mode = mode
        if self.mode == 'train':
            self.image_dir = args.train_file

        elif self.mode == 'val':
            self.image_dir = args.val_file

        self.prefix_img = config.get('data').get('prefix_img')
        self.suffix_img = config.get('data').get('suffix_img')
        self.prefix_label = config.get('data').get('prefix_label')
        self.suffix_label = config.get('data').get('suffix_label')

        f = open(self.image_dir, 'r')
        self.image_path_list = f.readlines()
        self.image_path_list = [file.strip() for file in self.image_path_list]

    def __getitem__(self, index):

        image_path = self.image_path_list[index]

        self.image_id = image_path
        image_path = os.path.join(self.prefix_img, self.image_id, self.suffix_img)
        label_path = os.path.join(self.prefix_label, self.image_id, self.suffix_label)

        image = sitk.ReadImage(image_path, sitk.sitkInt16)
        label = sitk.ReadImage(label_path, sitk.sitkUInt8)
        if config.fake_mask:
            label.SetSpacing(image.GetSpacing())

        new_spacing = config.new_spacing
        image = self.resample(image, target_spacing=new_spacing)
        label = self.resample(label, target_spacing=new_spacing)

        label_array = sitk.GetArrayFromImage(label)  # [384, 240, 80] ndarray in 0,1,2
        # label_array[label_array != 2] = 0 #Get the Tumor label

        image_array = sitk.GetArrayFromImage(image)  # [384, 240, 80] ndarray in range [-250, 250]

        image_array = self.clip_intensity(image_array)

        cube_glob = image_array
        label_glob = label_array
        _, _, _, _, z_min, z_max = self.getBoundbox(label_glob)
        if self.slice_number is not None:
            # sample the tensor only in axis Z, output (x,y,slice_number)
            start_slice = random.randint(max(z_min - 16, 0), min(max(z_max - 8, z_min), label_glob.shape[
                2] - self.slice_number))  # (0, image_array.shape[-1] -self.slice_number)
            end_slice = start_slice + self.slice_number - 1
            # print(start_slice, end_slice, label_glob.shape[2])
            if end_slice >= label_glob.shape[2]:
                print('no!!!!')
                exit()
            image_array = image_array[:, :, start_slice:end_slice + 1]
            label_array = label_array[:, :, start_slice:end_slice + 1]
        else:
            start_slice = 0
            end_slice = 0

        # array to tensor
        # print(f'original size: {image_array.shape}')
        image_array, label_array = self.centricCrop(image_array, label_array, crop_size=config.centric_cropping)
        image_array = torch.FloatTensor(image_array).unsqueeze(0)  # [1, 384, 240, 80]
        # image_array = image_array.permute(0,3,1,2) # [1, 80, 384, 240]
        label_array = torch.FloatTensor(label_array)  # [384, 240, 80]
        if self.glob_flag:
            return image_path, image_array, label_array, image_array, start_slice, end_slice, label_array
        else:
            return image_array, label_array

    def __len__(self):
        return len(self.image_path_list)

    def clip_intensity(self, ct_array, intensity_range=(-250, 0)):
        ct_array[ct_array > intensity_range[1]] = intensity_range[1]
        ct_array[ct_array < intensity_range[0]] = intensity_range[0]
        return ct_array

    def zoom(self, ct_array, seg_array, patch_size):

        shape = ct_array.shape  # [384, 240, 80]
        length_hight = int(shape[0] * patch_size)
        length_width = int(shape[1] * patch_size)

        length = int(256 * patch_size)

        x1 = int(random.uniform(0, shape[0] - length_hight))
        y1 = int(random.uniform(0, shape[1] - length_width))

        x2 = x1 + length_hight
        y2 = y1 + length_width

        ct_array = ct_array[x1:x2 + 1, y1:y2 + 1, :]
        seg_array = seg_array[x1:x2 + 1, y1:y2 + 1, :]

        with torch.no_grad():
            ct_array = torch.FloatTensor(ct_array).unsqueeze(dim=0).unsqueeze(dim=0)
            ct_array = ct_array
            ct_array = F.interpolate(ct_array, (shape[0], shape[1], shape[2]), mode='trilinear',
                                     align_corners=True).squeeze().detach().numpy()

            seg_array = torch.FloatTensor(seg_array).unsqueeze(dim=0).unsqueeze(dim=0)
            seg_array = seg_array
            seg_array = F.interpolate(seg_array, (shape[0], shape[1], shape[2])).squeeze().detach().numpy()

            return ct_array, seg_array

    def randomCrop(self, input_image, input_label, crop_size=(96, 96, 32)):
        '''
        random crop the cubic in object region
        :param input_image:
        :param crop_size:
        :return:
        '''
        assert input_label.shape == input_image.shape, 'the shape of mask and input_image should be same'
        assert isinstance(input_image, np.ndarray), 'the input_image should be np.ndarray'

        # randm crop the cubic
        new_x = random.randint(0, input_image.shape[0] - crop_size[0])
        end_x = new_x + crop_size[0] - 1
        new_y = random.randint(0, input_image.shape[1] - crop_size[1])
        end_y = new_y + crop_size[1] - 1
        new_z = random.randint(0, input_image.shape[2] - crop_size[2])
        end_z = new_z + crop_size[2] - 1
        #
        image_array = input_image[new_x:end_x + 1, new_y:end_y + 1, new_z:end_z]
        label_array = input_label[new_x:end_x + 1, new_y:end_y + 1, new_z:end_z]

        return image_array, label_array

    def centricCrop(self, input_image, input_label, crop_size=(96, 96, 32)):
        '''
        crop the cubic in the center of object region
        :param input_image:
        :param crop_size:
        :return:
        '''
        # print(self.image_id)
        # print(f'image shape: {input_image.shape}, label shape: {input_label.shape}')
        # assert input_label.shape == input_image.shape,'the shape of mask and input_image should be same'
        assert isinstance(input_image, np.ndarray), 'the input_image should be np.ndarray'

        # centric crop the cubic
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
        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetDefaultPixelValue(0)
        resample.SetOutputSpacing(target_spacing)
        resample.SetOutputOrigin(input_image.GetOrigin())
        resample.SetOutputDirection(input_image.GetDirection())
        resample.SetSize(new_size)
        new_image = resample.Execute(input_image)
        return new_image

    def getBoundbox(self, input_array):
        '''
        get the bouding box for input_array (the non-zero range is our object)
        '''
        assert isinstance(input_array, np.ndarray)
        x, y, z = input_array.nonzero()
        return [x.min(), x.max(), y.min(), y.max(), z.min(), z.max()]


# the test code
if __name__ == '__main__':
    data_train = CTDataLoader(mode='train'
                              , use_weight=False
                              # , args={'train_file': '/home/zli17/work/projects/VNet/dataset/train_new.txt'
                              #        , 'val_file': '/home/zli17/work/projects/VNet/dataset/val_new.txt'}
                              )
    test_image, test_label = data_train.__getitem__(0)
    numbers = data_train.__len__()
    print(test_image.shape)
    print(test_label.shape)
    print('the number of cases in dataset: ', data_train.__len__())

    image_path = data_train.image_path_list[0]
    print(f'file path: {image_path}')
    label_path = image_path.replace('nifti.nii', 'lungMask_edit.nii')
    # label_path = image_path.replace('nifti.nii', 'lungMask_edit_random.nii')
    image = sitk.ReadImage(image_path, sitk.sitkInt16)
    label = sitk.ReadImage(label_path, sitk.sitkInt16)
    # fake_label = sitk.ReadImage(label_path, sitk.sitkInt16)
    print('----------test resample----------')
    print(f'image size: {image.GetSize()}')
    print(f'image Spacing: {image.GetSpacing()}')
    print(f'label size: {label.GetSize()}')
    print(f'label Spacing: {label.GetSpacing()}')
    # sitk.WriteImage(image, './image.nii')
    # sitk.WriteImage(label, './label.nii')

    new_spacing = (1, 1, 1)

    image_new = data_train.resample(image, target_spacing=new_spacing)
    label_new = data_train.resample(label, target_spacing=new_spacing)
    print(f'image_new size: {image_new.GetSize()}')
    print(f'image_new Spacing: {image_new.GetSpacing()}')
    print(f'label_new size: {label_new.GetSize()}')
    print(f'label_new Spacing: {label_new.GetSpacing()}')
    sys.exit()
    sitk.WriteImage(image_new, './image-new.nii')
    sitk.WriteImage(label_new, './label-new.nii')

    print('----------test centricCrop---------')
    image_new_array = sitk.GetArrayFromImage(image_new)
    label_new_array = sitk.GetArrayFromImage(label_new)
    print(f'image_new_array shape:{image_new_array.shape}')
    print(f'image_new_array shape:{image_new_array.shape}')
    # image_new_array = data_train.clip_intensity(image_new_array)
    image_array, label_array = data_train.centricCrop(image_new_array, label_new_array, crop_size=(256, 400, 400))
    print(f'image_array shape:{image_array.shape}')
    print(f'label_array shape:{label_array.shape}')
    image_crop = sitk.GetImageFromArray(image_array)
    image_crop.SetSpacing(new_spacing)
    label_crop = sitk.GetImageFromArray(label_array)
    label_crop.SetSpacing(new_spacing)
    sitk.WriteImage(image_crop, './image_crop.nii')
    sitk.WriteImage(label_crop, './label_crop.nii')

    print('----------test zoom---------------')
    print(f'image shape before zoom:{image_array.shape}')
    print(f'label shape before zoom:{label_array.shape}')
    ct_array, seg_array = data_train.zoom(image_array, label_array, 1)
    print(f'image shape after zoom:{ct_array.shape}')
    print(f'label shape after zoom:{seg_array.shape}')

