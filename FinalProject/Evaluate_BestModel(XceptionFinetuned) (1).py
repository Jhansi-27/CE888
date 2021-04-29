#!/usr/bin/env python
# coding: utf-8

# In[12]:


# Import Libraries
import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix,classification_report
from tensorflow.keras.models import Sequential,load_model
import seaborn as sns


# In[21]:


if __name__ == "__main__":
    best_model = 'Xception_finetuned.h5'
    model = load_model(best_model)
    
    print("model is loading")
    print("Model Summary: \n",model.summary())
    
    print("Please provide full path to your test data directory\n")
    print("Make sure that Test directory is parent directory and fire and no-fire directories are subdirectories inside the test directory\n")
    test_path = input("Enter test data directory path:(eg: CE888_ForestFires/Test/Test) ")
    batch_size = int(input("Enter the batch_size: "))
    
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    test_data = test_datagen.flow_from_directory(
                                                test_path,
                                                target_size=(254, 254),
                                                shuffle = False,
                                                color_mode='rgb',
                                                class_mode='binary',
                                                batch_size=batch)
    
    print("Evaluating the model on test data \n")
    evaluation = model.evaluate(test_data)
    loss,accuracy = evaluation[0],evaluation[1]
    
    print(f"\nThe accuracy on test data is : {accuracy}")
    


# In[ ]:


# Prediction on test data
print("Please wait the model is making predictions on test data, this takes few minutes\n")
predictions = model.predict(test_data)


# In[ ]:



# Converting predictions into labels 0,1
pred_list = list(predictions)
predicted_classes = [1 if pred > 0.5 else 0 for pred in pred_list]

# Extracting original class labels from test data
true_classes = test_data.classes

# Plotting Confusion Matrix
cmf = confusion_matrix(true_classes,predicted_classes)
print("Confusion Matrix \n")
sns.heatmap(cm,annot=True,cmap='Blues_r',
        fmt='g',xticklabels=['Fire', 'No_Fire'], 
        yticklabels=['Fire', 'No_Fire']) 

# 0 is fire, 1 is No-fire

print('Classification Report')
print(classification_report(test_data.classes, predicted_classes,target_names=['fire','no_fire']))

