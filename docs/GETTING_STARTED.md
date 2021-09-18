# Getting Started

## Dataset Preparation

Please place the data (RefCOCO, RefCOCO+, RefCOCOg, ReferItGame) or the soft link of dataset folder under ./ln_data/. We follow dataset structure DMS. To accomplish this, the [download_data.sh](../ln_data/download_data.sh) bash script from DMS can be used.

```
cd ./ln_data
bash download_data.sh --path .
```

Please download data indices from [[Gdrive]](https://drive.google.com/file/d/1fVwdDvXNbH8uuq_pHD_o5HI7yqeuz0yS/view?usp=sharing) or [[One Drive]](https://uofr-my.sharepoint.com/:f:/g/personal/zyang39_ur_rochester_edu/???), and place them as the ./data folder.

```
rm -r ./data
tar -xvf data.tar
```

## Pretrained Checkpoints Preparation

Please download the pretrained checkpoints from [[Gdrive]](https://drive.google.com/drive/folders/1SOHPCCR6yElQmVp96LGJhfTP46RxVwzF?usp=sharing), and put these checkpoints into ./checkpoints.

Note that when pre-train these checkpoints on MSCOCO, the overlapping images of val/test set of corresponding datasets are excluded.