import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
from keras.preprocessing import image
import cv2
import dlib
import matplotlib.pyplot as plt
import SVM_preprocessing as l2

#basedir = './dataset'
#images_dir = os.path.join(basedir,'celeba')
#labels_filename = 'labels.csv'

#Specify the folder containing the attribute list file and the folder with the pictures
basedir = './assignment'


#Specigy the image folder
images_dir = os.path.join(basedir,'dataset')


#Specify the attrbite list file
labels_filename = 'attribute_list.csv'


#I am using dlib´s frontal face detector function
detector = dlib.get_frontal_face_detector()




#This list contains the paths for all the pictures
image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]



target_size = None

#Makes the csv file readable
labels_file = open(os.path.join(basedir, labels_filename), 'r')
lines = labels_file.readlines()




#This function returns the number of good images, the images as a numpy array, the names of the good images, and the labels read from the .csv file
def select_images(lbl):


    #This will store the names of the good images
    good_images_names = []

    #This will store the good images
    good_images = []

    #This will contain the labels
    all_labels = []


    # This will read the labels from a column in the .csv file. lbl specifies which column:
    # lbl == 1 for hair colour detection, lbl == 2 for glasses detection
    # lbl == 3 for emotion detection, lbl == 4 for age detection
    # lbl == 5 for human detection
    label = {line.split(',')[0] : int(line.split(',')[lbl]) for line in lines[2:]}

    #This will count the number of good images
    num=0
    
    if os.path.isdir(images_dir):
       
       
    #Every image is considered, so this for covers all the image paths.   
        for img_path in image_paths:

            #This function assigns to file_name the name of the image, without its path and without .png
            file_name= img_path.split('\\') [2].split('.')[0]
            #Loads the images and transforms them into an array
            img = image.img_to_array(image.load_img(img_path, target_size=target_size, interpolation='bicubic'))

            #This will inspect whether the picture depicts a human or not or whether it is too noisy
            features, _ = l2.run_dlib_shape(img)
            
            if features is not None:
                #If the classification is for hair detection, the algorithm sorts the images labeled with a -1 out.
                #Otherwise, it adds the image name, the imag and the label to their corresponding lists.
                if lbl==1:
                    if label[file_name]!=-1:
                        good_images_names.append(int(file_name))
                        good_images.append(img)
                        all_labels.append(label[file_name])
                        num=num+1
                else:
                    good_images_names.append(int(file_name))
                    good_images.append(img)
                    all_labels.append(label[file_name])
                    num=num+1

        # The images are in 256 X 256 X 3 format, but for the CNN we need a num X 256 X 256 X 3 variable, where num is the number of images.
        # x is also a numpy array
        x=np.zeros((num,256,256,3))
        i=0
        for j in good_images:
            x[i]= good_images[i]
            i=i+1

        x=x/255
        # If the classification is not regarding the hair colour, the labels are -1 and 1. This converts them to 0 and 1. Otherwise, it leaves them as they are.
        if(lbl!=1):
            label = (np.array(all_labels) + 1)/2
        else:
            label = np.array(all_labels)
            
        return num, x,  good_images_names, label

  
    
    

#This function takes the information about the proper pictures, and it divide it into training and testing sets
def division(i):

    num, x, good_image_names, y = select_images(i)
    training_number=int(70/100*num)
    x_tr = x[: training_number]
    x_te = x[training_number :]
    img_tr = good_image_names[: training_number]
    img_te = good_image_names[training_number :]
    y_tr = y[: training_number]
    y_te = y[training_number :]
    return num, x_tr, x_te, y_tr, y_te, img_tr, img_te

