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
import numpy as np # linear algebra
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from keras.layers import Input
import matplotlib.pyplot as plt

# Set some parameters

im_size      = 256
num_img      = ... #please enter the number of all bscans
number_class = 3
im_height    = ... #please enter the height of original bscan
im_width     = ... #please enter the width of original bscan

#### Input Path  ####
path_img = ... #please enter the adress of bscans folder

id_bscan = next(os.walk(path_img))[2] # list of names all images in the given path

x_test = np.zeros((num_img, im_size, im_size,1), dtype=np.float64)
#### using for niblack
x_test_nondenoised = np.zeros((num_img, im_height, im_width), dtype=np.float64)
x_test_denoised = np.zeros((num_img, im_height, im_width), dtype=np.float64)
x_test_niblack = np.zeros((num_img, im_height, im_width), dtype=np.float64)

for n in range(num_img):
    img_test = cv2.imread(path_img+id_bscan[n])
    x_img_nondenoised = img_test
    x_img_denoised = cv2.fastNlMeansDenoising(img_test,None,10)
    img_test = img_to_array(img_test[:,:,1])
    
    # cropping background
    for i in range (np.size(img_test,0)):
        for j in range (np.size(img_test,1)):
            if (img_test[i,j,:] - 255 == 0):
                img_test[i,j,:]=0
                
    x_img_test = resize(img_test, (im_size, im_size), mode = 'constant', preserve_range = True)
    x_test[n,:,:] = x_img_test/255.0

    x_img_niblack = img_to_array(x_img_denoised[:,:,1]) # using denoised image to calculate CVI

    x_img_niblack = x_img_niblack[:,:,0]/255.0
    x_img_nondenoised = x_img_nondenoised[:,:,0]
    x_img_denoised = x_img_denoised[:,:,0]    
    
        
    x_test_nondenoised[n,:,:] = x_img_nondenoised
    x_test_denoised[n,:,:] = x_img_denoised
    x_test_niblack[n,:,:] = x_img_niblack
######---------------------------------

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
  

################################################################
# Evaluation
################################################################

# load the best model
model.load_weights('...') # please choose raster.h5 for raster and choose EDI.h5 for EDI test data

# Predict on train, val and test

preds_test= model.predict(x_test, verbose=1)

# Threshold predictions
preds_test_t= (preds_test > 0.5).astype(np.uint8)
pred = preds_test_t.copy()
########################################################
# ploting predicted layers
########################################################

from skimage import measure
from skimage.transform import resize

kernel = np.ones((5,5),np.uint8)

# One Hot Decoding

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


for j in range (preds_test_t.shape[0]):
    for num_seglayer in range(np.size(segmented_lines_quick,1)):
        pred_layer = segmented_lines_quick[j,num_seglayer,:]
        plt.plot(pred_layer,'b')
        plt.imshow(np.power(x_test[j,:,:],0.5),cmap='gray')
        plt.title('X-test number: ' + str(j))
    plt.show()

######################################################################################

############################################
# niblack
############################################
    
from skimage.filters import (threshold_niblack,
                           threshold_sauvola)


pred_layers = pred[:,:,:,1]
cvi = np.zeros((num_img))

for n in range(num_img):
    
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
        

