
# Classification of fire and non-fire aerial images Using Transfer Learning on FLAMES Data set

### Dataset
* The datset used for this project is called 'FLAMES' dataset.
* This dataset is available on IEEE dataport and you can download 7th and 8th repository from [here](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs), the 7th repository contains training data nd 8th repository has test data. After downloading, unzip them.
* Training/Validation dataset: This dataset has 39,375 frames that are resized to 254x254 and the image format is JPEG. This data is in a directory called training, which contains 2 sub-directories(Fire, Non-fire), one per class.
* Test dataset : Test datset has 8,617 frames that are labeled.This data is in a directory called test, which contains 2 sub-directories(Fire, Non-fire), one per class.

* This table shows the directory structure of training data:
```bash
/Training
        ├── Fire/*.jpg
        ├── No_Fire/*.jpg
```
* The test direcotry looks like this:
```bash
/Test
    ├── Fire/*.jpg
    ├── No_Fire/*.jpg
```

### Model
- Two mdoels are created for this task. One is a simple CNN and other is a Finetuned Xception model.
* Below is the architecture of simple CNN model:
* The CNN model has 3 convolutional layers followed by a max-pooling layers.
* A dropout layer is added after 3rd maxpool operation to avoid overfitting.

![BaseModel:Simple CNN](https://github.com/Jhansi-27/Forest_Fires_CNN/blob/main/Baseline_new.png?raw=true). 

* Finetuned  Xception Model:
* The architecture of the model is given below:

![BaseModel:Simple CNN](https://github.com/Jhansi-27/Forest_Fires_CNN/blob/main/Baseline_new.png?raw=true). 

## Requirements
* Keras 
* Tensorflow
* Scikitlearn
* Matplotlib.pyplot
* Seaborn
* Numpy
* Pandas

## Code
This code is run and tested on Python 3.6 on Windows 10  machine with no issues.
Download the [Forest_fires.ipynb](https://github.com/Jhansi-27/Forest_Fires_CNN/blob/main/ForestFires.ipynb) file.
This is the main IPython Notebook, run it using Jupyter notebook in your local system or it can be run using [Google Collab](https://colab.research.google.com).

## Results
* The following are the classification accuracy and Confusion Matrix of the baseline model:
### Accuracy
![Accuracy](https://github.com/Jhansi-27/Forest_Fires_CNN/blob/main/accuracy.PNG?raw=true)
### Confusion marix
![Confusion matrix](https://github.com/Jhansi-27/Forest_Fires_CNN/blob/main/cm.PNG?raw=true)
