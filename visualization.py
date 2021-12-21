import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

nii_group = 't2fg_434'

images_arr = np.array(nib.load('media/ncc/nnUNet_raw_data_base/nnUNet_raw_data/Task224_A4C/imagesTr/{}_0000.nii.gz'.format(nii_group)).dataobj)
acc_arr = np.array(nib.load('media/ncc/nnUNet_raw_data_base/nnUNet_raw_data/Task224_A4C/labelsTr/{}.nii.gz'.format(nii_group)).dataobj)

pred_arr = np.array(nib.load('Output/2d_A4C_1000/{}.nii.gz'.format(nii_group)).dataobj)

print(images_arr.shape)
print(acc_arr.shape)
print('len arrange :', images_arr.shape)

def visual_image(NUM: int, take1: str = images_arr, take2: str = acc_arr, take3: str = pred_arr):

    # image
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(take1[:,:,NUM])

    plt.axis('off')

    # acc_arr
    plt.subplot(1, 3, 2)
    plt.imshow(take2[:,:,NUM])
    plt.axis('off')

    # pred
    plt.subplot(1, 3, 3)
    plt.imshow(take3[:,:,NUM])

    plt.axis('off')

    plt.show()
    # plt.savefig('')

## Evaluation Code for A2C (Dice Score, Jaccard Index)

import os
import numpy as np
import nibabel as nib
from typing import List

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


def compute(test=None, reference=None):

    if test is None or reference is None:
        raise ValueError("'test' and 'reference' must both be set to compute confusion matrix.")

    assert test.shape == reference.shape, "Shape mismatch: {} and {}".format(test.shape, reference.shape)

    tp = int(((test != 0) * (reference != 0)).sum())
    fp = int(((test != 0) * (reference == 0)).sum())
    tn = int(((test == 0) * (reference == 0)).sum())
    fn = int(((test == 0) * (reference != 0)).sum())

    return tp, fp, tn, fn


def dice_score(test=None, reference=None):
    """2TP / (2TP + FP + FN)"""

    # arr1 : *.nii.gz [n, n, (s)]
    # arr2 : *.nii.gz [n, n, (s)]

    tp_, fp_, tn_, fn_ = compute(test, reference)

    return float(2. * tp_ / (2 * tp_ + fp_ + fn_))


def jaccard_score(test=None, reference=None):
    """TP / (TP + FP + FN)"""

    # arr1 : *.nii.gz [n, n, (s)]
    # arr2 : *.nii.gz [n, n, (s)]

    tp, fp, tn, fn = compute(test, reference)


    return float(tp / (tp + fp + fn))


nii_group_list = subfiles('DB_ref/A2C/labelsTs', join=False)

def score_average(pred_path = nii_group_list):
    # pred_path : pred directory

    # OUTPUT_path = 'output/2d_A2C_1000'
    # OUTPUT_path = 'output/2d_DTK10_A2C_1000'
    OUTPUT_path = 'output/staple/A2C'

    count = 0
    dice_total = 0
    jac_total = 0
    pred_path_len = len(pred_path)

    for i in pred_path:

        test = np.array(nib.load('{}/{}'.format(OUTPUT_path, i)).dataobj)
        reference = np.array(nib.load('media/ncc/nnUNet_raw_data_base/nnUNet_raw_data/Task542_A2C/labelsTs/{}'.format(i)).dataobj)


        if reference.shape == test.shape:
            _,_,z = reference.shape
            count += z
        else:
            print('Shape mismatch: {} and {}'.format(test, reference))

        dice_total += z * dice_score(test, reference)
        jac_total += z * jaccard_score(test, reference)

    return print(OUTPUT_path), print('Dice Score :', dice_total/count), print('Jaccard Scroe :', jac_total/count)

score_average()

## Evaluation Code for A4C (Dice Score, Jaccard Index)

import os
import numpy as np
import nibabel as nib
from typing import List

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


def compute(test=None, reference=None):

    if test is None or reference is None:
        raise ValueError("'test' and 'reference' must both be set to compute confusion matrix.")

    assert test.shape == reference.shape, "Shape mismatch: {} and {}".format(test.shape, reference.shape)

    tp = int(((test != 0) * (reference != 0)).sum())
    fp = int(((test != 0) * (reference == 0)).sum())
    tn = int(((test == 0) * (reference == 0)).sum())
    fn = int(((test == 0) * (reference != 0)).sum())

    return tp, fp, tn, fn


def dice_score(test=None, reference=None):
    """2TP / (2TP + FP + FN)"""

    # arr1 : *.nii.gz [n, n, (s)]
    # arr2 : *.nii.gz [n, n, (s)]

    tp_, fp_, tn_, fn_ = compute(test, reference)

    return float(2. * tp_ / (2 * tp_ + fp_ + fn_))


def jaccard_score(test=None, reference=None):
    """TP / (TP + FP + FN)"""

    # arr1 : *.nii.gz [n, n, (s)]
    # arr2 : *.nii.gz [n, n, (s)]

    tp, fp, tn, fn = compute(test, reference)


    return float(tp / (tp + fp + fn))


nii_group_list = subfiles('DB_ref/A4C/labelsTs', join=False)

def score_average(pred_path = nii_group_list):
    # pred_path : pred directory

    # OUTPUT_path = 'output/2d_A4C_1000'
    # OUTPUT_path = 'output/2d_DTK10_A4C_1000'
    OUTPUT_path = 'output/staple/A4C'

    count = 0
    dice_total = 0
    jac_total = 0
    pred_path_len = len(pred_path)

    for i in pred_path:

        test = np.array(nib.load('{}/{}'.format(OUTPUT_path, i)).dataobj)
        reference = np.array(nib.load('media/ncc/nnUNet_raw_data_base/nnUNet_raw_data/Task544_A4C/labelsTs/{}'.format(i)).dataobj)


        if reference.shape == test.shape:
            _,_,z = reference.shape
            count += z
        else:
            print('Shape mismatch: {} and {}'.format(test, reference))

        dice_total += z * dice_score(test, reference)
        jac_total += z * jaccard_score(test, reference)

    return print(OUTPUT_path), print('Dice Score :', dice_total/count), print('Jaccard Scroe :', jac_total/count)

score_average()


## Count
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

nii_group = 'tfg_600'

images_arr = np.array(nib.load('DB_nifti/test/Task02_A2C/imagesTs/{}.nii.gz'.format(nii_group)).dataobj)

print(images_arr.shape)

print('len arrange :', images_arr.shape)