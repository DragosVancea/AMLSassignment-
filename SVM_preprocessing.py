import os
import numpy as np
from keras.preprocessing import image
import cv2
import dlib



global basedir, image_paths, target_size


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

#The predictor is used to detect the facial landmarks 
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# The following code is concerned with detecting 68 face landmarks (20 for the mouth, 5 for the left eyebrow, 5 for the right eyebrow, 6 for the left eye, 6 for the right eye
# 8 for the nose and 18 for the jaw). These will be used as inputs for the SVM classification, in order to train the computer to detect a smile, a pair of glasses, whether the
# person is young or old or whether the picture is a cartoon or a real human. The code was created using dlib's implementation of the paper:
# One Millisecond Face Alignment with an Ensemble of Regression Trees by Vahid Kazemi and Josephine Sullivan, CVPR 2014
# and was trained on the iBUG 300-W face landmark dataset (see https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):
# C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic.
# 300 faces In-the-wild challenge: Database and results. Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.


def shape_to_np(shape, dtype="int"):
    # This will contain the (x,y) coordinates.
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # The for covers all facial landmarks and converts them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

def rect_to_bb(rect):
    # This takes a bounding predicted by dlib and then transforms it
    # to the (x, y, w, h) format frequently used with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)

# This function reads the image and detects the facial landmarks and then, it returns them.
def run_dlib_shape(image):
    
    # These lines read the input image. After that, they resize it, and they convert it to grayscale.
    
    resized_image = image.astype('uint8')
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # This uses the dlib detector the detect face features from the grayscaled image.
    rects = detector(gray, 1)
    num_faces = len(rects)

    # If the features were not discovered, this will return a zero. This will be useful when sorting out the noisy images.
    if num_faces == 0:
        return None, resized_image

    # These will contain facial landmarks detected with the shape predictor.
    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # This covers all the face detections
    
    for (i, rect) in enumerate(rects):
        
        # This will detects the facial landmarks and the covert the temp_shape to a numpy array.
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)



        # This coverts the dlib rectangle to to the (x, y, w, h) format frequently used with OpenCV 
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # This finds the largest face and saves it.
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image




#This list contains the paths for all the pictures
image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]


target_size = None


#Makes the csv file readable
labels_file = open(os.path.join(basedir, labels_filename), 'r')
lines = labels_file.readlines()

#The elements of the flag will be 0 if their corresponding image is not noisy or 1 if it is noisy.
flag=np.zeros((1,5000))

#This function returns the features array for the good images, the good image names and the number of good images
def extract_features_labels():
  
  
    #This will store the features of the good images.
    all_features = []

    #This will store the names of the good images.
    good_images = []
  
    
    if os.path.isdir(images_dir):
       
        #This will count the number of good images
        num=1
        
        #Every image is considered, so this for covers all the image paths.
        for img_path in image_paths:



            #This function assigns to file_name the name of the image, without its path and without .png
            file_name= img_path.split('\\') [2].split('.')[0]
            
            #Loads the images and transforms them into an array
            img = image.img_to_array(image.load_img(img_path,target_size=target_size, interpolation='bicubic'))


            #This will inspect whether the picture depicts a human or not or whether it is too noisy
            features, _ = run_dlib_shape(img)


            #If features were found, this adds the image name and the features to their corresponding lists.
            if features is not None:
                all_features.append(features)
                good_images.append(int(file_name))
                num=num+1
            #Otherwise it marks the flag element as 1.                
            else:
                flag[0, int(file_name)-1]=1
                   
            


            
    #Transforms landmark_features into a numpy array.   
    landmark_features = np.array(all_features)
    return landmark_features, good_images, num


#This will return the labels read from column i of the .csv file.        
def extract_labels(i):

    #This will contain the labels that correspond to the good images.
    all_labels = []
    #Reads the all the labels from column i. i == 2 for glasses detection
    # i == 3 for emotion detection, li == 4 for age detection
    # i == 5 for human detection
    label = {line.split(',')[0] : int(line.split(',')[i]) for line in lines[2:]}
    if os.path.isdir(images_dir):
        
        
        
        #Every image is considered, so this for covers all the image paths.
        for img_path in image_paths:
            #This function assigns to file_name the name of the image, without its path and without .png
            file_name= img_path.split('\\') [2].split('.')[0]
            if(flag[0,int(file_name)-1]==0):
                #If features were found, this adds the image label to its corresponding list.
                all_labels.append(label[file_name])
                
            
        #The converts the labels from -1 and 1 to 0 and 1.
        label = (np.array(all_labels) + 1)/2
        
    return label

 

    
