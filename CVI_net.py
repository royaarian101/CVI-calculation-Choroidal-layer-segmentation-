# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 07:32:18 2021
Choroid segmentation
@author: Roya Arian, royaarian101@gmail.com
"""

#import  
import os
import cv2
from skimage.transform import resize
from pathlib import Path
from keras.preprocessing.image import  img_to_array
import re
import numpy as np # linear algebra
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from sklearn.model_selection import train_test_split
from keras.layers import Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# Set some parameters

im_size      = ... #please enter the size of input of the model(like 128)
num_image    = ... #please enter the number of all bscans
number_class = 3
im_height    = ... #please enter the height of original bscan
im_width     = ... #please enter the width of original bscan

#### Input Path  ####
path_img = ... #please enter the adress of bscans folder
path_mask =... #please enter the adress of grandtruths folder


#### spliting test and train data ####

testarray = ... #please enter the array of the number of all test sampels among all sampels

n_test= ... #please enter the number of test sampels

# train data
x_train = np.zeros(((num_image-n_test), im_size, im_size,1), dtype=np.float64)
labels_train = np.full(((num_image-n_test),im_size, im_size,number_class),0)
# test data
x_test = np.zeros((n_test, im_size, im_size,1), dtype=np.float64)

#### using for niblack
x_test_nondenoised = np.zeros((n_test, im_height, im_width), dtype=np.float64)
x_test_denoised = np.zeros((n_test, im_height, im_width), dtype=np.float64)
x_test_niblack = np.zeros((n_test, im_height, im_width), dtype=np.float64)
####
       
labels_test = np.full((n_test,im_size, im_size,number_class),0)
label = np.zeros((n_test, im_size, im_size), dtype=np.float64)

numm=np.zeros(n_test)

test=0
train=0
p = '[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+'  # finding the number of thr id in the folder
id_bscan = next(os.walk(path_img))[2] # list of names all bscans in the given path
id_mask = next(os.walk(path_mask))[2] # list of names all masks in the given path

for n in range(num_image):
    column = []
    img = cv2.imread(path_img+id_bscan[n])
    #### using for niblack
    nondenoised = img
    denoised = cv2.fastNlMeansDenoising(img,None,10)
    ####
    
    mask = cv2.imread(path_mask+id_mask[n])
    
    i = id_bscan[n]
    if re.search(p, i) is not None:  
        for catch in re.finditer(p, i):
            num=int(catch[0])   # finding the number of thr id in the folder   
    x_img = img_to_array(img[:,:,1])
    x_img_niblack = img_to_array(denoised[:,:,1]) # using denoised image to calculate CVI
    x_mask = np.round(mask[:,:,1]/255.0) 

    # cutting zeros colunms of masks and images consequently
    for i in range (x_mask.shape[1]):
        if (np.all((x_mask[:,i] == 0))):
            column.append(i)
    column = np.array(column)
    x_mask = np.delete(x_mask, column , axis=1)
    x_img = np.delete(x_img, column , axis=1)

    x_img_niblack = np.delete(x_img_niblack, column , axis=1)
    x_img_nondenoised = np.delete(nondenoised, column , axis=1)
    x_img_denoised = np.delete(denoised, column , axis=1)
    
    shape = x_img.shape

    x_img_niblack = x_img_niblack[:,:,0]/255.0
    x_img_nondenoised = x_img_nondenoised[:,:,0]
    x_img_denoised = x_img_denoised[:,:,0]    
    
    # resizing masks
    mask2 = resize(x_mask, (im_size, im_size), mode = 'constant', preserve_range = True)
    mask = np.round(mask2)
    # making 3 classes masks from 2 classes ones (one hot encoding)
    for i in range (mask.shape[1]):
        for j in range (mask.shape[0]-1):
            if (mask[j,i] == 1 and mask[j+1,i] == 0 ):
                mask[j+1:,i] = 2  
    
    # resizing images(bscans)
    x_img = resize(x_img, (im_size, im_size), mode = 'constant', preserve_range = True)
    
    # spliting test and train data
    if (num in testarray):
        x_test[test,:,:] = x_img/255.0
        
        # resizing niblack images
        x_img_nondenoised = np.round(resize(x_img_nondenoised, (im_height,im_width), mode = 'constant', preserve_range = True))
        x_img_denoised = np.round(resize(x_img_denoised, (im_height,im_width), mode = 'constant', preserve_range = True))
        x_img_niblack = np.round(resize(x_img_niblack, (im_height,im_width), mode = 'constant', preserve_range = True))

        
        x_test_nondenoised[test,:,:] = x_img_nondenoised
        x_test_denoised[test,:,:] = x_img_denoised
        x_test_niblack[test,:,:] = x_img_niblack
        

        
        for i in range (number_class):
            labels_test[test,:,:,i] = np.where(mask ==i,1,0)
        label[test,:,:]= mask
        numm[test]=num 
        test+=1
  
    else:
        x_train[train,:,:] = x_img/255.0
        for i in range (number_class):
            labels_train[train,:,:,i] = np.where(mask ==i,1,0)
        train+=1

######---------------------------------

#### Unet function ####

from keras.models import Model
from keras.layers import  BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def get_unet(input_img,number_class, n_filters = 32, dropout = 0.1, batchnorm = True):
    """Function to define the UNET Model"""
    # Contracting Path    
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    # Bottel neck (Middle)
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)

    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(number_class, (1, 1), activation='sigmoid')(c9)
    #1 is number of class
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

###################


input_img = Input((im_size, im_size,1))
model = get_unet(input_img, number_class,n_filters=32, dropout=0.05, batchnorm=True)
  
#### spliting train and validation data #### 
x_train, x_valid, y_train, y_valid = train_test_split (x_train, labels_train, test_size=0.2, random_state=42)

####################################################################
# Choosing weights for WCCE
####################################################################

# please choose one of following weigths

# parametes
k=500
q=20000
alpha=2
beta=0.8


w1=np.zeros((number_class))
w2=np.zeros((number_class))
w3=np.zeros((number_class))
w4=np.zeros((number_class))
w5=np.zeros((number_class))

# w1: assign w1=1 for background and w1=2 for choroid layer  
for j in range (number_class):
    if ((j==0) or (j==number_class-1)):
        w1[j]= 1
    else:
        w1[j]= 2
fij=0        
for j in range (number_class):
    for i in range (y_train.shape[0]):
        fij = fij+ np.sum(y_train[i,:,:,j])/(np.size(y_train,1)*np.size(y_train,2))
    fj = fij/(np.size(y_train,0))
    w2[j] = k/fj
    w3[j] = np.log(q/fj)    
    w4[j] = (1-np.power(beta,fj))/(1-beta)

    
i=0    
def weights(i):
    switcher={
        0:np.ones((number_class)),   # first version of Balanced Cross Entropy
        1:w1,                        # Balanced Cross Entropy
        2:w2,                        # Inverse class freq. linear (k=500)
        3:w3,                        # Inverse class freq. logarithmic (q=20000)
        4:w4,                        # Effective Number of Object Class (beta=0.8)
      }
    return switcher.get(i,"weights")

weights=weights(4)

####################################################################
# Choosing loss function
####################################################################

# define custom loss and metric functions 

#### loss functions ####
from keras import backend as K
import os
os.environ['KERAS_BACKEND'] = 'theano'

def combined_loss(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    def dice_coef(y_true, y_pred, smooth=1):
        """
        Dice = (2*|X & Y|)/ (|X|+ |Y|)
             =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
        ref: https://arxiv.org/pdf/1606.04797v1.pdf
        """
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
    
    def dice_coef_loss(y_true, y_pred):
        return 1-dice_coef(y_true, y_pred)
           
        
    def total_variation_loss(y_pred):
        img_width=np.size(y_pred,1)
        img_height=np.size(y_pred,1)
        a = K.square(
            y_pred[:, :img_height - 1, :img_width - 1, :] -
            y_pred[:, 1:, :img_width - 1, :])
        b = K.square(
            y_pred[:, :img_height - 1, :img_width - 1, :] -
            y_pred[:, :img_height - 1, 1:, :])
        return K.sum(K.pow(a + b, 1.25))/(255*im_size*im_size)  
    
    
    def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-10):
        y_true = K.flatten(y_true) 
        y_pred = K.flatten(y_pred) 
        truepos = K.sum(y_true * y_pred) 
        fp_and_fn = alpha * K.sum(y_pred * (1 - y_true)) + beta * K.sum((1 - y_pred) * y_true) 
        answer = (truepos + smooth) / ((truepos + smooth) + fp_and_fn) 
        return -answer
    
    def WCCE_dice_loss(y_true, y_pred):
        return loss(y_true, y_pred)+dice_coef_loss(y_true, y_pred)
    
    
    def WCCE_dice_tv_loss(y_true, y_pred):
        return loss(y_true, y_pred)+dice_coef_loss(y_true, y_pred)+total_variation_loss( y_pred)
    
    
    def WCCE_dice_tversky_loss(y_true, y_pred):
        return loss(y_true, y_pred)+dice_coef_loss(y_true, y_pred)+tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-10)
    
    
    return WCCE_dice_loss, WCCE_dice_tv_loss, WCCE_dice_tversky_loss, dice_coef

#####################
# choose one of the loss functions
   
WCCE_dice_loss, WCCE_dice_tv_loss ,WCCE_dice_tversky_loss, dice_coef = combined_loss(weights)

model.compile(optimizer=Adam(),  loss=[...], metrics=[dice_coef,'accuracy']) #please choose one of the loss functions and fill in the blanck

callbacks = [EarlyStopping(patience=20 , verbose=1), ReduceLROnPlateau(factor=0.1, patience=10, min_lr=0.000000001, verbose=1),
    ModelCheckpoint('model-oct.h5', verbose=1, save_best_only=True, save_weights_only=True)
]



results = model.fit(x_train, y_train, batch_size=8, epochs=200, callbacks=callbacks,\
                    validation_data=(x_valid, y_valid))
plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_loss"], label="val_loss")
plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend()


################################################################
# Evaluation
################################################################

# load the best model
model.load_weights('model-oct.h5')
# Evaluate on validation set (this must be equals to the best log_loss)
model.evaluate(x_valid, y_valid, verbose=1)
model.evaluate(x_test, labels_test, verbose=1)


# Predict on train, val and test

preds_train = model.predict(x_train, verbose=1)
preds_val = model.predict(x_valid, verbose=1)
preds_test= model.predict(x_test, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t= (preds_test > 0.5).astype(np.uint8)
pred = preds_test_t.copy()
########################################################
# ploting predicted layers
########################################################

from skimage import measure
from skimage.transform import resize

kernel = np.ones((5,5),np.uint8)

# One Hot Decoding
y_test_dc = np.argmax(labels_test, axis = 3)

Largest_area = np.zeros_like((preds_test_t)) # removing small white balls
        
for i in range (preds_test_t.shape[0]):
    for l in range(number_class):
        # removing small white balls
        labels_mask = measure.label(preds_test_t[i,:,:,l])                       
        regions = measure.regionprops(labels_mask)
        regions.sort(key=lambda x: x.area, reverse=True)
        if len(regions) > 1:
            for rg in regions[1:]:
                labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
        labels_mask[labels_mask!=0] = 1
        if (l!=1):
            # inverting the image
            # removing small black balls
            img_not = 1 - np.asarray(labels_mask)
            # finding the largest area
            labels_mask = measure.label(img_not)                       
            regions = measure.regionprops(labels_mask)
            regions.sort(key=lambda x: x.area, reverse=True)
            if len(regions) > 1:
                for rg in regions[1:]:
                    labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
            labels_mask[labels_mask!=0] = 1
            Largest_area_not = labels_mask
            
            # inverting the image again
            Largest_area[i,:,:,l] = 1 - np.asarray(Largest_area_not)
        else:
            Largest_area[i,:,:,l] = np.asarray(labels_mask)
        
# One Hot Decoding
Largest_area_dc = np.argmax(Largest_area, axis = 3)  
        
x_test = x_test[:,:,:,0]

#####################################
# finging boundaries 
#####################################

#### layer2boundary_bunch_Morphology function 

import skimage.morphology

# layer2boundary_bunch_Makeboundry function
width = 200
def layer2boundary_bunch(layer, layer_thresh):
    image_size=im_size-1
    segmented_lines_quick = np.zeros((layer.shape[0], number_class-1 , layer.shape[2]))
    loc_image = layer.copy()
    loc=np.where(loc_image==0,-1,1)

    for sampel in range(layer.shape[0]):
        boundries = np.zeros((width,image_size+1))
        last_boundries = np.zeros((image_size+1))
        last_boundries=np.where(last_boundries==0,np.nan,last_boundries) 
        for i in range (layer.shape[2]):
            if (len(np.where(np.diff(np.sign(loc[sampel,:,i])))[0])!=0):
                b = np.where(np.diff(np.sign(loc[sampel,:,i])))[0] 
                boundries [0:len(b),i] = b 
            if (len(boundries[1])>image_size+1):
                boundries [:,i] = boundries [:,0:i-1]
            boundries=np.where(boundries==0,np.nan,boundries) 
            last_boundries[i] = boundries[0,i]
            segmented_lines_quick[sampel, 0 , :] = last_boundries
 
# Finding second boundary 
    loc_image = layer.copy()
    loc=np.where(loc_image==1,1,-1)
    image_size=im_size-1
    
    for sampel in range(layer.shape[0]):
        boundries = np.zeros((width,image_size+1))
        last_boundries = np.zeros((image_size+1))
        last_boundries=np.where(last_boundries==0,np.nan,last_boundries) 
        for i in range (layer.shape[2]):
            if (len(np.where(np.diff(np.sign(loc[sampel,:,i])))[0])!=0):
                b = np.where(np.diff(np.sign(loc[sampel,:,i])))[0] 
                boundries [0:len(b),i] = b 
            if (len(boundries[1])>image_size+1):
                boundries [:,i] = boundries [:,0:i-1]
            boundries=np.where(boundries==0,np.nan,boundries) 
            last_boundries[i] = boundries[0,i]
            segmented_lines_quick[sampel, 1 , :] = last_boundries

    return segmented_lines_quick,segmented_lines_quick


def predict_two_boundaries(predictions, th):

    # Normalize masks
    predictions_normal = np.zeros_like(predictions)
    predictions_thresh = np.zeros_like(predictions)

    prediction_local = predictions
    prediction_local = (prediction_local - np.min(prediction_local))/(np.max(prediction_local) - np.min(prediction_local))
    rem = skimage.morphology.remove_small_holes(prediction_local > 0.3, connectivity=8, in_place=False)
    predictions_normal = prediction_local
    predictions_thresh = rem
    segmented_lines, segmented_lines_u = layer2boundary_bunch(predictions_normal, predictions_thresh)
    return segmented_lines
segmented_lines_quick = predict_two_boundaries(Largest_area_dc,64)
true_lines_quick = predict_two_boundaries(y_test_dc,64)


for j in range (preds_test_t.shape[0]):
    for num_seglayer in range(np.size(segmented_lines_quick,1)):
        pred_layer = segmented_lines_quick[j,num_seglayer,:]
        true_layer=true_lines_quick[j,num_seglayer,:]
        plt.plot(pred_layer,'b')
        plt.plot(true_layer,'r')
        plt.imshow(np.power(x_test[j,:,:],0.5),cmap='gray')
    plt.show()

############################################
# calculating signed and unsigned error
############################################
where_are_NaNs = np.isnan(segmented_lines_quick)
segmented_lines_quick[where_are_NaNs] = 0
where_are_NaNs = np.isnan(true_lines_quick)
true_lines_quick[where_are_NaNs] = 0

signed_error=np.zeros((np.size(true_lines_quick,1),np.size(true_lines_quick,0)))
unsigned_error=np.zeros((np.size(true_lines_quick,1),np.size(true_lines_quick,0)))

mean_signed_error=np.zeros((np.size(true_lines_quick,1)))
mean_unsigned_error=np.zeros((np.size(true_lines_quick,1)))


for i in range (np.size(true_lines_quick,1)):
    for j in range (np.size(true_lines_quick,0)):
        signed_error[i,j] = np.sum(np.subtract(true_lines_quick[j,i,:], segmented_lines_quick [j,i,:]))/(np.size(true_lines_quick,2))
        unsigned_error[i,j] = abs (np.sum(np.subtract(true_lines_quick[j,i,:], segmented_lines_quick [j,i,:]))/(np.size(true_lines_quick,2)))
        
    mean_signed_error[i]= np.sum( signed_error[i,:] )/(np.size(true_lines_quick,0))
    mean_unsigned_error[i]= np.sum( unsigned_error[i,:] )/(np.size(true_lines_quick,0))

    print("mean_unsigned_error_overall Layer ", i+1 , " = ",mean_unsigned_error[i])
    print("mean_signed_error_overall Layer ", i+1 , " = ",mean_signed_error[i])

######################################################################################

############################################
# niblack
############################################
    
from skimage.filters import (threshold_niblack,
                           threshold_sauvola)


pred_layers = pred[:,:,:,1]
cvi = np.zeros((n_test))

for n in range(n_test):
    
    x_test_nondenoised_v = x_test_nondenoised[n,:,:] #vector
    x_test_denoised_v = x_test_denoised[n,:,:] #vector
    x_test_niblack_v = x_test_niblack[n,:,:] #vector

    # resizing pred Layers       
    choroidlayer = np.round(resize(pred_layers[n,:,:], (x_test_niblack_v.shape[0],x_test_niblack_v.shape[1]), mode = 'constant', preserve_range = True))
    
    window_size = 15    
    thresh_niblack = threshold_niblack(x_test_niblack_v, window_size=window_size, k=0.001)
    thresh_sauvola = threshold_sauvola(x_test_niblack_v, window_size=window_size)
    
    binary_niblack = x_test_niblack_v > (thresh_niblack*1.03)
    binary_sauvola = x_test_niblack_v > thresh_sauvola
    # removing small balls that are not in the willing layer
    niblack = binary_niblack*choroidlayer
    sauvola = binary_sauvola*choroidlayer
    
    pred_not = np.ones(choroidlayer.shape)-choroidlayer
    niblack_zero = pred_not + niblack
    niblack_not = np.ones(niblack_zero.shape)-niblack_zero
    
    cvi[n] = (np.sum(niblack_not))/(np.sum(choroidlayer))
    
    
    # ploting
    # choroid layer
    choroid = x_test_nondenoised_v*choroidlayer
            
            
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(x_test_nondenoised_v, cmap=plt.cm.gray)
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.title('denoised image')
    plt.imshow(x_test_denoised_v, cmap=plt.cm.gray)
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(choroid, cmap=plt.cm.gray)
    plt.title('denoised choroid layer')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(niblack, cmap=plt.cm.gray)
    plt.title('Niblack Threshold')
    plt.axis('off')
      
    plt.show()
        

