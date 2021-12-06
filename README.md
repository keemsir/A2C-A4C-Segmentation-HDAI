# A2C_A4C_Segmentation 사용법

## Inference code

## Training & Inference 환경

1. OS(운영체제) : Ubuntu 20.04.2 LTS
2. CPU : Intel Xeon Silver 4216 CPU @ 2.10GHz processor : 64
3. GPU : Quadro GV100 (Memory : 32GB)
4. RAM : 93GB
5. Version : Python 3. 이상
             Pytorch 1.6 이상

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


## 0. 기본환경설정 (Ubuntu)

Ctrl + Alt + t 를 눌러 Command 창 실행



#### Ubuntu 편집기 Nano 설치

```bash
sudo apt-get install nano
```

#### nnUNet 설치

첨부된 SoNoSeg 압축파일 해제 후 해당 폴더로 이동
예시) `cd PycharmProject/SoNoSeg`

해당 폴더 이동 후 다음의 Command 입력

```bash
cd nnUNet
pip install -e .
```


#### Model 및 Inference 등 경로 환경 설정

`touch /home/(user_name)/.bashrc`
로 bash 파일 생성 (user_name : 사용자 이름으로 되어있는 경로)

`nano /home/(user_name)/.bashrc`

로 파일을 열어서 가장 하단에 다음을 복사하여 붙여넣기
```bash
export nnUNet_raw_data_base="media/ncc/nnUNet_raw_data_base"
export nnUNet_preprocessed="media/ncc/nnUNet_preprocessed"
export RESULTS_FOLDER="media/ncc/nnunet_trained_models"
```
입력 후 Ctrl + x, y(Yes) 를 통해 나가기

`source /home/(user_name)/.bashrc`로 실행



## 1. Test(Input) 데이터 전처리

첨부파일의 상위 경로로 이동

예시) `cd PycharmProject/SoNoSeg`

위의 예시와같이 압축 풀었던 최상위 폴더로 이동

```bash
nnUNet_convert 
```




```bash
nnUNet_trainer 2d nnUNetTrainerV2 
```
