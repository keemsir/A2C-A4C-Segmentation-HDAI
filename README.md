# A2C_A4C_Segmentation 사용법

## Inference code

## Training & Inference 환경

1. OS(운영체제) : Ubuntu 20.04.2 LTS
2. CPU : Intel Xeon Silver 4216 CPU @ 2.10GHz processor : 64
3. GPU : Quadro GV100 (Memory : 32GB)
4. RAM : 93GB

## 경로 설명


    SoNoSeg/
    ├── DB
    │   └── Test
    │       ├── A2C (*.npy, *.png)
    │       └── A4C (*.npy, *.png)
    ├── DB_nifti
    │   └── Test
    │       ├── Task02_A2C (*.npy, *.png)
    │       │   ├── imagesTr
    │       │   ├── imagesTs (*.nii.gz)
    │       │   ├── labelsTr
    │       │   └── dataset.json
    │       └── Task04_A4C (*.npy, *.png)
    │           ├── imagesTr
    │           ├── imagesTs (*.nii.gz)
    │           ├── labelsTr
    │           └── dataset.json
    ├── media
    │   └── ncc (User Name)
    │       ├── fold_0
    │       ├── nnUNet_raw_data_base
    │       │   └── nnUNet_raw_data
    │       └── nnunet_trained_models
    │           └── nnUNet
    │               └── 2d
    │                   ├── Task532_A2C
    │                   └── Task534_A4C
    ├── nnUNet (Library)
    │           ├── fold_1
    │           ├── fold_2
    │           ├── fold_3
    │           └── fold_4
    └── output
        ├── A2C
        └── A4C
            ├── A2C (*.npy, *.png)
            └── A4C (*.npy, *.png)


## 0. 기본설정



File path

```bash
nnUNet_trainer 2d 
```


