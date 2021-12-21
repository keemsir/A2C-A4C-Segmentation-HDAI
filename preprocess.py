import os
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import nibabel as nib
from collections import OrderedDict
import json

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

def img_size(file_path: str):

    cur_dir = os.getcwd()

    DB_dir = file_path
    a2c_dir = os.path.join(cur_dir, DB_dir)

    a2c_imgs = subfiles(a2c_dir, suffix='png')

    shape_list = []


    for i in range(len(a2c_imgs)):
        pngTemp = plt.imread(a2c_imgs[i])

        shap = pngTemp.shape

        shape_list.append(shap)

    shape_list.sort()
    shape_list_unq = np.unique(shape_list, axis=0)
    print(file_path, ', Unique Size : ')
    print(shape_list_unq)

    return shape_list_unq

# img_size('DB/test/A2C/')
# img_size('DB/test/A4C/')


##





def png2nifti(test_folder: str, save_folder: str):

    # test_folder : test file path 'DB/test/'
    # save_folder : [imagesTr, imagesTs] Save Folder path
    task_name = test_folder[-3:]

    maybe_mkdir_p(os.path.join(save_folder, 'imagesTr'))
    maybe_mkdir_p(os.path.join(save_folder, 'imagesTs'))
    maybe_mkdir_p(os.path.join(save_folder, 'labelsTr'))


    v2_list = img_size(test_folder) # Shape list

    # file group setting
    v2_file_group = []


    for i in v2_list:
        a, b, _ = i
        file_name = 'tfg_{}'.format(a)
        v2_file_group.append(file_name)

        globals()['image_{}'.format(a)] = np.zeros([a, b, 0], dtype=np.single)
        globals()['label_{}'.format(a)] = np.zeros([a, b, 0], dtype=np.uint8)
        print(file_name)


    # png to nii
    v2_file_original = {}

    test_pngs = subfiles(test_folder, suffix='.png', join=False)

    for tp in test_pngs:

        v_img = os.path.join(test_folder, tp)

        pngTemp = plt.imread(v_img)

        IMG_NAME, _ = os.path.splitext(tp)

        x_, _, _= pngTemp.shape

        pngTemp = np.expand_dims(pngTemp[:,:,0], axis=2)

        # File name Dictionary write
        if '{}'.format(x_) in v2_file_original:
            v2_file_original['{}'.format(x_)].append('{}'.format(IMG_NAME))
        else:
            v2_file_original['{}'.format(x_)] = [IMG_NAME]

        globals()['image_{}'.format(x_)] = np.append(globals()['image_{}'.format(x_)], pngTemp, axis=2)


    for sv in v2_file_group:

        sv_NUM = sv[-3:]

        niim = nib.Nifti1Image(globals()['image_{}'.format(sv_NUM)], affine=np.eye(4))
        nib.save(niim, os.path.join(save_folder, 'imagesTs/{}.nii.gz'.format(sv)))
        print(globals()['image_{}'.format(sv_NUM)].shape)


    pickle.dump(v2_file_original, open('filename_{}.pkl'.format(task_name), 'wb'))
    # pickle.dump(v2_list, open('fileshape_{}.pkl'.format(task_name), 'wb'))

    print('"{}" Image & Label Completed !!'.format(os.path.basename(os.path.normpath(save_folder))))



## creating JSON


def json_mk(save_dir: str):
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
        json_dict['description'] = "Heart Disease AI Datathon 2021"
        json_dict['tensorImageSize'] = "3D"
        json_dict['reference'] = "https://github.com/DatathonInfo/H.D.A.I.2021"
        json_dict['licence'] = "NIA"
        json_dict['release'] = "26/11/2021"

        json_dict['modality'] = {
            "0": "sono"
        }
        json_dict['labels'] = {
            "0": "background",
            "1": "AC"
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
                print('{} : dataset.json created!'.format(save_dir))
            else:
                print('{} : dataset.json overwritten!'.format(save_dir))


##

png2nifti(test_folder='DB/test/A2C', save_folder='DB_nifti/test/Task02_A2C')
png2nifti(test_folder='DB/test/A4C', save_folder='DB_nifti/test/Task04_A4C')

json_mk(save_dir='DB_nifti/test/Task02_A2C')
json_mk(save_dir='DB_nifti/test/Task04_A4C')


