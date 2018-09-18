# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 18:44:32 2018

@author: 0x
"""

import model_detection as md
import os
import glob
import numpy as np


################# Custom Convolution Model #############################

load=False

if load:
    model = md.convolution_network_model()
    model = md.cargar_weights(model)
    print(model.summary())  
else:
    history = md.LossHistory()
    model = md.train_model(epoch=5)
    # Save loss history to file
    model.save('models/my_model3.h5')
    model.save_weights('models/model_weights3.h5')
    loss_history_path = os.path.join("loss_history3.log")
    myFile = open(loss_history_path, 'w+')
    myFile.write(md.history.losses)
    myFile.close()



# The model class indices are: {'neg': 0, 'pos': 1}
images = "dataset/single_prediction/*.p"
test_pos = "dataset/test_set/pos/*"
test_neg = "dataset/test_set/neg/*"

pred_neg = []
for filename in glob.glob(test_neg):
    pred_neg.append(md.predict_image(model, filename))


pred_pos = []
for filename in glob.glob(test_pos):
    pred_pos.append(md.predict_image(model, filename))


neg = np.zeros(len(pred_neg))
pos = np.ones(len(pred_pos))

y_test = np.concatenate((neg,pos))

y_pred = np.array(pred_neg+pred_pos).squeeze() 
y_pred = np.where(y_pred > 0.5, 1, 0)
    
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, y_pred)
print(cm2)  



################# VGG Model #############################


vgg_model = md.vgg16_train_model(epoch=5, load=True)


test_pos = "dataset/test_set/pos/*"
test_neg = "dataset/test_set/neg/*"

pred_neg = []
for filename in glob.glob(test_neg): #assuming gif
    pred_neg.append(md.predict_image(vgg_model, filename))
pred_pos = []
for filename in glob.glob(test_pos): #assuming gif
    pred_pos.append(md.predict_image(vgg_model, filename))


neg = np.zeros(len(pred_neg))
pos = np.ones(len(pred_pos))

y_test = np.concatenate((neg,pos))

y_pred = np.array(pred_neg+pred_pos).squeeze() 
y_pred = np.where(y_pred > 0.5, 1, 0)
    
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)


    
    
    
