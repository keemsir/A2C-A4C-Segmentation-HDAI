import os
from typing import List
import nibabel as nib
import numpy as np
import pickle

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

##


def nifti2npy(output_path: str, save_path: str):

    maybe_mkdir_p(save_path)

    task_name = output_path[-3:]

    v2_file_original = pickle.load(open('filename_{}.pkl'.format(task_name), 'rb'))
    # v2_list = pickle.load(open('fileshape_{}.pkl'.format(task_name), 'rb'))

    nifti_file_list = nifti_files(output_path)

    for nfl in nifti_file_list:

        nfl_spl = split_path(nfl)

        file_ = np.array(nib.load(nfl).dataobj)
        x_, y_, z_ = file_.shape

        for i in range(z_):
            npy_name = v2_file_original[nfl_spl[-1][5:8]][i]
            file_array = file_[:,:,i]

            np.save(os.path.join(save_path, '{}'.format(npy_name)), file_array)

    print('save path :', save_path)


##
nifti2npy('output/staple/A2C', 'output/staple/npy/2d_DTK10_A2C')
nifti2npy('output/staple/A4C', 'output/staple/npy/2d_DTK10_A4C')

