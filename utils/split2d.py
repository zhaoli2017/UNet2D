import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import SimpleITK as sitk
from tqdm import tqdm

prefix_img = '/home/zli17/data/nifti/nifti/nifti'
suffix_img = 'nifti.nii'
prefix_label = '/home/zli17/data/nifti/nifti/reader1'
suffix_label = 'lungMask_edit.nii.gz'
centric_cropping = (128, 256, 256)
new_spacing = (1.7, 1.7, 1.7)

def clip_intensity(ct_array, intensity_range=(-250, 0)):
    ct_array[ct_array > intensity_range[1]] = intensity_range[1]
    ct_array[ct_array < intensity_range[0]] = intensity_range[0]
    return ct_array

def centricCrop(input_image, input_label, crop_size=(96, 96, 32)):
    '''
    crop the cubic in the center of object region
    :param input_image:
    :param crop_size:
    :return:
    '''
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

def resample(input_image, target_spacing=(0.5, 0.5, 0.5)):
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

def split(image_dir, output_dir):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    img_slice_p_l = []
    label_slice_p_l = []

    with open(image_dir, 'r') as f:
        image_path_list = f.readlines()
    image_path_list = [file.strip() for file in image_path_list]
    for img_i in tqdm(range(len(image_path_list))):
        image_id = image_path_list[img_i]
        image_path = os.path.join(prefix_img, image_id, suffix_img)
        label_path = os.path.join(prefix_label, image_id, suffix_label)

        image = sitk.ReadImage(image_path, sitk.sitkInt16)
        label = sitk.ReadImage(label_path, sitk.sitkUInt8)

        image = resample(image, target_spacing=new_spacing)
        label = resample(label, target_spacing=new_spacing)

        label_array = sitk.GetArrayFromImage(label)
        image_array = sitk.GetArrayFromImage(image)
        image_array = clip_intensity(image_array)

        image_array, label_array = centricCrop(image_array, label_array, crop_size=centric_cropping)

        #print(f'image_array shape: {image_array.shape}')
        #print(f'label_array shape: {label_array.shape}')
        if image_array.shape != label_array.shape or image_array.shape != centric_cropping:
            print(f'{image_id}: dimension unmatched, ignore!')
            continue

        img_root_dir = os.path.join(output_dir, image_id)
        if not os.path.exists(img_root_dir):
            os.mkdir(img_root_dir)

        for i in range(image_array.shape[0]):
            image_slice = image_array[i]
            label_slice = label_array[i]

            img_slice_p = os.path.join(img_root_dir, 'img_' + str(i) + '.npy')
            label_slice_p = os.path.join(img_root_dir, 'label_' + str(i) + '.npy')
            np.save(img_slice_p, image_slice)
            np.save(label_slice_p, label_slice)

            img_slice_p_l.append(img_slice_p)
            label_slice_p_l.append(label_slice_p)


    img_df = pd.DataFrame(img_slice_p_l)
    img_df.to_csv(os.path.join(output_dir, 'img_slice_list.txt'), index=False, header=False)


if __name__ == '__main__':
    split('/home/zli17/work/projects/VNet/dataset/train_new.txt', '/data/zli17/nifti/split2d_train')
