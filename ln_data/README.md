# Data Folder
* RefCOCO, RefCOCO+, RefCOCOg, ReferItGame Dataset: place the data or the soft link of dataset folder under ``./ln_data/``. We follow dataset structure [DMS](https://github.com/BCV-Uniandes/DMS). To accomplish this, the ``download_dataset.sh`` [bash script](https://github.com/BCV-Uniandes/DMS/blob/master/download_data.sh) from DMS can be used.
    ```bash
    bash download_data --path .
    ```

* Data index: download the generated index files and place them as the ``./data`` folder. Availble at [[Gdrive]](https://drive.google.com/file/d/1fVwdDvXNbH8uuq_pHD_o5HI7yqeuz0yS/view?usp=sharing), [[One Drive]](https://uofr-my.sharepoint.com/:f:/g/personal/zyang39_ur_rochester_edu/???).
    ```
    cd ..
    rm -r data
    tar xf data.tar
    ```