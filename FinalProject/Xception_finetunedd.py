#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Activation,Conv2D,Dense,MaxPool2D,Dropout,Flatten,Input,GlobalAveragePooling2D,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import Xception
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread
from IPython.display import Image, display
from sklearn.metrics import confusion_matrix,plot_confusion_matrix,classification_report
get_ipython().run_line_magic('matplotlib', 'inline')
from plot_keras_history import plot_history
import seaborn as sns

# paths for training data and test data
train_dir = '../input/firedata/Training/Training'
test_dir = '../input/firedata/Test/Test'

# Augmenting the training data and splitting 20% data for validation using ImageDataGenerartor
train_images = ImageDataGenerator(   rescale=1.0/255,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     horizontal_flip=True,
                                     vertical_flip=True,
                                     validation_split=0.2 )

# training data (80%)
training_data = train_images.flow_from_directory(   train_dir,
                                                    target_size=(254,254), 
                                                    color_mode='rgb', 
                                                    class_mode='binary',
                                                    shuffle=True,
                                                    seed=120,
                                                    batch_size=64,
                                                    subset='training')
# validation data (20%)
validation_data = train_images.flow_from_directory(     train_dir, 
                                                        target_size=(254,254),
                                                        color_mode='rgb',
                                                        class_mode='binary',
                                                        batch_size=64,
                                                        shuffle=True,
                                                        seed=120,
                                                        subset='validation')


# Loading the test data
test_images = ImageDataGenerator(rescale=1.0/255)

test_data = test_images.flow_from_directory(test_dir,
                                            target_size=(254,254), 
                                            color_mode='rgb', 
                                            class_mode='binary',
                                            shuffle=False,
                                            batch_size=1)


# # **Transfer Learning Using Xception Model**

# Using Xception model for transfer learning
# workflow
# 1. create model
# 2. compile the model
# 3. Fit the model on training data
# 4. Evaluate the finetuned model on test data

# Instantiating a base model with pre-trained weights
base_model = Xception(weights='imagenet',
                      input_shape=(254,254,3), # since include_top = false
                      include_top=False)

# Freezing the base model
base_model.trainable = False

# Create a new model on top
inputs = Input(shape=(254,254,3))

x = base_model(inputs,training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)  
outputs = Dense(1)(x)

model = Model(inputs,outputs)
model.summary()

# for i,layer in enumerate(base_model.layers):
#     print(i,layer.name,layer.trainable)

plot_model(model, to_file='model.png',show_shapes=True,dpi=96)

# compile the model created
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Defining the early stopping criteria
early_stop = EarlyStopping( monitor="val_loss",
                            min_delta=1e-5,
                            patience=10,
                            verbose=1,
                            mode="min",
                            restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor="val_loss",
                              factor=0.1,
                              patience=10,
                              verbose=1,
                              mode="min",# In 'min' mode, the learning rate will be reduced when the quantity monitored has stopped decreasing
                              min_delta=0.0001)

check_point = ModelCheckpoint(  filepath=" weights.{epoch:02d}-{val_loss:.2f}.h5",
                                monitor="val_loss",
                                verbose=1,
                                save_best_only=True,
                                save_weights_only=False,
                                mode="min")

tb_logs = TensorBoard(log_dir="logs")
:


# train the model on new training data
my_callbacks = [early_stop, reduce_lr, check_point, tb_logs]
batch_size = 64
epochs = 20

history = model.fit(
                    training_data,
                    steps_per_epoch=training_data.samples // batch_size,
                    validation_data = validation_data, 
                    validation_steps = validation_data.samples // batch_size,
                    callbacks=my_callbacks,
                    epochs=epochs,
                    verbose=1
    )
model.save('./Xception_pretrained.h5')

# Plotting loss and accruracy on training data and validation data before finetuning
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(14,9))
plt.subplot(211)
plt.plot(epochs, acc, 'r-o', label='Training accuracy')
plt.plot(epochs, val_acc, 'b--*', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.subplot(212)
plt.plot(epochs, loss, 'r-o', label='Training Loss')
plt.plot(epochs, val_loss, 'b--*', label='Validation Loss')
plt.title('Training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.savefig('base_model_result.png')

plot_history(history, path="base_model_results_2.png")


# # **Fine tuning**

# Once the model has converged on the new data, we can unfreeze all or part of the base model
# and retrain the whole model end-to-end with a very low learning rate.

# Unfreezing the whole base model
base_model.trainable = True


# It's important to recompile the model after making any changes to the `trainable` attribute 
# of any inner layer, so that the changes are take into account

model.compile(optimizer=keras.optimizers.Adam(1e-5),  # Very low learning rate
              loss='binary_crossentropy',
              metrics=['acc'])


# Training end-to-end. 
history = model.fit(training_data,
                    steps_per_epoch=training_data.samples  // batch_size,
                    epochs=20,
                    validation_data = validation_data, 
                    validation_steps = validation_data.samples // batch_size,
                    callbacks=my_callbacks,
                    verbose=1
    )
model.save('./Xception_finetuned.h5')

# evaluating the fine tuned model
test = model.evaluate(test_data)
test_loss , test_acc = test[0],test[1]

valid = model.evaluate(validation_data)
valid_loss,valid_acc =  valid[0],valid[1]

train = model.evaluate(training_data)
train_loss,train_acc = train[0],train[1]

# creating dataframe with loss and accuracy on train, validation and test sets
pd.DataFrame({'test':{'loss':test_loss,'accuracy':test_acc},
              'validation':{'loss':valid_loss,'accuracy':valid_acc},
              'train':{'loss':train_loss,'accuracy':train_acc}})

# Plotting loss and accruracy on training data and validation data after finetuning
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(9,12))
plt.subplot(211)
plt.plot(epochs, acc, 'r-o', label='Training accuracy')
plt.plot(epochs, val_acc, 'b--*', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.subplot(212)
plt.plot(epochs, loss, 'r-o', label='Training Loss')
plt.plot(epochs, val_loss, 'b--*', label='Validation Loss')
plt.title('Training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.savefig('finetuned_model_result.png')

plot_history(history, path="finetuned_model_results_2.png")

# making predictions on test data
predictions = model.predict(test_data)


# Converting predictions into model [0, 1]
pred_list = list(predictions)
predicted_classes = [1 if pred > 0.5 else 0 for pred in pred_list]

# Getting true classes,[0, 1] from test data
true_classes = test_data.classes

# Plotting Confusion Matrix
cm=confusion_matrix(true_classes,predicted_classes)
plot_cm = sns.heatmap(cm,annot=True,
                      fmt='g',cmap='Blues_r',
                      xticklabels=['Fire', 'No_Fire'],
                      yticklabels=['Fire', 'No_Fire'])
#Fire is 0 and No_Fire is 1

plot_cm.figure.savefig("cfm.png")


print('Classification Report')
print(classification_report(test_data.classes, predicted_classes,target_names=['fire','no_fire']))

report = classification_report(test_data.classes,predicted_classes,target_names=['fire','no_fire'],output_dict=True)
df = pd.DataFrame(report).transpose()
df

