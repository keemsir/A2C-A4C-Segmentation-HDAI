import os
import numpy as np
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


def split_path(path: str) -> List[str]:
    """
    splits at each separator. This is different from os.path.split which only splits at last separator
    """
    return path.split(os.sep)


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


def score_average(test_path: str, reference_path: str):

    # test_path : prediction directory
    # reference_path : reference directory


    count = 0
    dice_total = 0
    jac_total = 0


    test_file_list = subfiles(test_path, suffix='npy')
    # reference_file_list = subfiles(reference_path, suffix='npy')


    for tfl in test_file_list:

        SP = split_path(tfl)[-1]

        test = np.load(tfl)

        reference = np.load(os.path.join(reference_path, SP))


        if reference.shape == test.shape:
            count += 1
        else:
            print('Shape mismatch: {} and {}'.format(test.shape, reference.shape))

        dice_total += dice_score(test, reference)
        jac_total += jaccard_score(test, reference)

    return print(test_path), print('Dice Score :', dice_total/count), print('Jaccard Scroe :', jac_total/count)

##
score_average(test_path='output/staple/npy/A2C', reference_path='DB/validation/A2C')
score_average(test_path='output/staple/npy/A4C', reference_path='DB/validation/A4C')
