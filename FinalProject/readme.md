
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
* Two models are created for this task. One is a simple CNN and other is a Finetuned Xception model.
#### Architecture of simple CNN model:
![BaseModel:Simple CNN](https://github.com/Jhansi-27/Forest_Fires_CNN/blob/main/Baseline_new.png?raw=true). 

#### Architecture of Finetuned Xception Model:
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
The code for finetuned Xception model can be found here [Xception_finetuned.ipynb](https://github.com/Jhansi-27/CE888/blob/main/FinalProject/xception-finetuned.ipynb).This notebook contains all the details of model creation, model compiling and fitting along with all the visaulization. The light version of same model can be found here [Xception_finetunedd.py](https://github.com/Jhansi-27/CE888/blob/main/FinalProject/Xception_finetunedd.py).

## Instructions to run the model and to evaluate the model on test data

#### The model can be run this file, [Xception_finetunedd.py](https://github.com/Jhansi-27/CE888/blob/main/FinalProject/Xception_finetunedd.py).
* In this file, just change the following directory paths to directory path in which your train and test data are located.
* change paths for training data and test data
```
train_dir = '../input/firedata/Training/Training' # change to your training directory path 
```
```
test_dir = '../input/firedata/Test/Test' # change to your test directory path
```
* Then run Xception_finetunedd.py
```
python Xception_finetunedd.py
```
#### Evaluating model on test data:
* To evaluate model, use this file [Evaluate_Model.py](https://github.com/Jhansi-27/CE888/blob/main/FinalProject/Evaluate_BestModel.py) and run the following command. 
```
python Evaluate_BestModel(XceptionFinetuned)
```
* Before running the code, provide the path to XceptionFinetuned model. This is the link to [best model](https://www.kaggle.com/ravieamani/bestmodel).
* Download this model and in Main function provide the name of the downloaded file 'Xception_finetuned.h5'.
* Upon running the code, the code asks user for test directory path and batch size.
* After providing these, the model gets evaluated on test data and performance metric such as accuracy, confusion matrix and classification report is printed.

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
### Loss and Acuuracy:  Simple CNN(Left)                  and                                     Finetuned Xception(Right)
![Loss and Accuracy](https://github.com/Jhansi-27/CE888/blob/main/FinalProject/Results/SimpleCNN%20Results/loss_accuracy_cnn.JPG)![](https://github.com/Jhansi-27/CE888/blob/main/FinalProject/Results/FIneTuned_Xception_Results/Dataframe_loss_acc.JPG)

### Confusion matrix: Simple CNN (Left)                  and                                     Finetuned Xception(Right)
![Confusion Matrix](https://github.com/Jhansi-27/CE888/blob/main/FinalProject/Results/SimpleCNN%20Results/confusion_simpleCNN.png)![](https://github.com/Jhansi-27/CE888/blob/main/FinalProject/Results/FIneTuned_Xception_Results/cfm.png)



