
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

![FineTunedModel:Xception](https://github.com/Jhansi-27/CE888/blob/main/FinalProject/Results/FIneTuned_Xception_Results/model%20(1).png). 

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
The code for simple CNN can be found here [CNN.ipynb](https://github.com/Jhansi-27/CE888/blob/main/FinalProject/CNN.ipynb)
The code for finetuned Xception model can be found here [Xception_finetuned.ipynb](https://github.com/Jhansi-27/CE888/blob/main/FinalProject/xception-finetuned.ipynb)
This is the main IPython Notebook, run it using Jupyter notebook in your local system or it can be run using [Google Collab](https://colab.research.google.com).

## Results
* The following are the classification accuracy and Confusion Matrix of the simple CNN and transfer learning model(Xception finetuned):
### Simple CNN:
* Accuracy
![Accuracy](https://github.com/Jhansi-27/CE888/blob/main/FinalProject/Results/SimpleCNN%20Results/CNN_results.png)
* Confusion marix and Classification matrix

![Classification Report](https://github.com/Jhansi-27/CE888/blob/main/FinalProject/Results/SimpleCNN%20Results/Capture.JPG)

### Transfer Learning with Finetuned Xception model:
* Accuracy
![Accuracy](https://github.com/Jhansi-27/CE888/blob/main/FinalProject/Results/FIneTuned_Xception_Results/finetuned_model_results_2%20(1).png)
* Confusion marix and Classification matrix
![Classification Report](https://github.com/Jhansi-27/CE888/blob/main/FinalProject/Results/FIneTuned_Xception_Results/classification_report.JPG)

## Comparing performance of simple CNN and Xception finetuned model:
### Simple CNN                   and         Finetuned Xception
![Loss and Accuracy](https://github.com/Jhansi-27/CE888/blob/main/FinalProject/Results/SimpleCNN%20Results/loss_accuracy_cnn.JPG)![](https://github.com/Jhansi-27/CE888/blob/main/FinalProject/Results/FIneTuned_Xception_Results/Dataframe_loss_acc.JPG)
![Confusion Matrix](https://github.com/Jhansi-27/CE888/blob/main/FinalProject/Results/SimpleCNN%20Results/confusion_simpleCNN.png)![](https://github.com/Jhansi-27/CE888/blob/main/FinalProject/Results/FIneTuned_Xception_Results/cfm.png)

