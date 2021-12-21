import os
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import random

# Utils
def subdirs(folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True) -> List[str]:
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isdir(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


def subfiles(folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True) -> List[str]:
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


def nifti_files(folder: str, join: bool = True, sort: bool = True) -> List[str]:
    return subfiles(folder, join=join, sort=sort, suffix='.nii.gz')


def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)


def split_path(path: str) -> List[str]:
    """
    splits at each separator. This is different from os.path.split which only splits at last separator
    """
    return path.split(os.sep)


## size check
import numpy as np
import os

def img_size(file_path: str = 'DB/validation/A4C/'):

    cur_dir = os.getcwd()

    DB_dir = file_path
    a2c_dir = os.path.join(cur_dir, DB_dir)

    a2c_imgs = subfiles(a2c_dir, suffix='png')
    a2c_labels = subfiles(a2c_dir, suffix='npy')

    shape_list = []

    if len(a2c_imgs) == len(a2c_labels):
        for i in range(len(a2c_imgs)):
            pngTemp = plt.imread(a2c_imgs[i])
            npyTemp = np.load(a2c_labels[i])

            shap = npyTemp.shape

            shape_list.append(shap)

            if pngTemp[:,:,0].shape == npyTemp.shape:
                pass
            else:
                print('png image file and npy label file is not correct.')
        shape_list.sort()
        shape_list_unq = np.unique(shape_list, axis=0)
        print(file_path, ', Unique Size : ')
        print(shape_list_unq)
    else:
        print('The order pair of png image file and npy label file is not correct.')
    return shape_list_unq

img_size('DB/validation/A4C/')



##
import os
import nibabel as nib


def png2nifti(training_folder: str = 'DB/validation/A4C', test_folder: str = '/mnt/dataset/test_set',
              save_folder: str = 'DB_nifti/validation/A4C'):

    # training_folder : training file path '/mnt/dataset/train-valid_set'
    # test_folder : test file path '/mnt/dataset/test_set'
    # save_folder : [imagesTr, imagesTs] Save Folder path


    maybe_mkdir_p(os.path.join(save_folder, 'imagesTr'))
    maybe_mkdir_p(os.path.join(save_folder, 'imagesTs'))
    maybe_mkdir_p(os.path.join(save_folder, 'labelsTr'))

    # training_files = os.listdir(training_folder)

    v2_list = img_size(training_folder) # training_folder

    # file group setting
    v2_file_group = []
    for i in v2_list:
        a, b = i
        file_name = 't2fg_{}'.format(a)
        v2_file_group.append(file_name)

        globals()['image_{}'.format(a)] = np.zeros([a, b, 0], dtype=np.single)
        globals()['label_{}'.format(a)] = np.zeros([a, b, 0], dtype=np.uint8)
        print(file_name)


    validation_npys = subfiles(training_folder, suffix='.npy', join=False) # training_folder


    for v in validation_npys:

        vn = v[:-4]

        v_img = os.path.join(training_folder, vn + '.png')
        v_label = os.path.join(training_folder, vn + '.npy')
        pngTemp = plt.imread(v_img)
        npyTemp = np.load(v_label)

        x_, _, _= pngTemp.shape

        pngTemp = np.expand_dims(pngTemp[:,:,0], axis=2)
        npyTemp = np.expand_dims(npyTemp, axis=2)

        globals()['image_{}'.format(x_)] = np.append(globals()['image_{}'.format(x_)], pngTemp, axis=2)
        globals()['label_{}'.format(x_)] = np.append(globals()['label_{}'.format(x_)], npyTemp, axis=2)


    for sv in v2_file_group:

        sv_NUM = sv[-3:]

        niim = nib.Nifti1Image(globals()['image_{}'.format(sv_NUM)], affine=np.eye(4))
        nib.save(niim, os.path.join(save_folder, 'imagesTr/{}.nii.gz'.format(sv)))
        print(globals()['image_{}'.format(sv_NUM)].shape)


        nila = nib.Nifti1Image(globals()['label_{}'.format(sv_NUM)], affine=np.eye(4))
        nib.save(nila, os.path.join(save_folder, 'labelsTr/{}.nii.gz'.format(sv)))
        print(globals()['label_{}'.format(sv_NUM)].shape)



    print('"{}" Image & Label Completed !!'.format(os.path.basename(os.path.normpath(save_folder))))
#     print('Image Patient : {}'.format(len(os.listdir(input_dir))))

png2nifti()


## creating JSON
from collections import OrderedDict
import json


def json_mk(save_dir: str = 'DB_nifti/validation/A4C'):
    # Path
    imagesTr = os.path.join(save_dir, 'imagesTr')
    imagesTs = os.path.join(save_dir, 'imagesTs')
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(imagesTs)

    overwrite_json_file = True
    json_file_exist = False

    if os.path.exists(os.path.join(save_dir, 'dataset.json')):
        print('dataset.json already exist!')
        json_file_exist = True

    if json_file_exist == False or overwrite_json_file:

        json_dict = OrderedDict()
        json_dict['name'] = "SoNo"
        json_dict['description'] = "Medical Image AI Challenge 2021"
        json_dict['tensorImageSize'] = "3D"
        json_dict['reference'] = "https://maic.or.kr/competitions/"
        json_dict['licence'] = "SNUH"
        json_dict['release'] = "18/10/2021"

        json_dict['modality'] = {
            "0": "sono"
        }
        json_dict['labels'] = {
            "0": "background",
            "1": "A2C"
        }

        train_ids = sorted(os.listdir(imagesTr))
        test_ids = sorted(os.listdir(imagesTs))
        json_dict['numTraining'] = len(train_ids)
        json_dict['numTest'] = len(test_ids)

        json_dict['training'] = [{'image': "./imagesTr/%s" % i, "label": "./labelsTr/%s" % i} for i in train_ids]

        json_dict['test'] = ["./imagesTs/%s" % i for i in test_ids] #(i[:i.find("_0000")])

        with open(os.path.join(save_dir, "dataset.json"), 'w') as f:
            json.dump(json_dict, f, indent=4, sort_keys=False)

        if os.path.exists(os.path.join(save_dir, 'dataset.json')):
            if json_file_exist == False:
                print('dataset.json created!')
            else:
                print('dataset.json overwritten!')

json_mk()