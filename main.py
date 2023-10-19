
# coding: utf-8

# Face Recognition with OpenCV

# ### Import Required Modules

# Before starting the actual coding we need to import the required modules for coding. So let's import them first. 
# 
# - **cv2:** is _OpenCV_ module for Python which we will use for face detection and face recognition.
# - **os:** We will use this Python module to read our training directories and file names.
# - **numpy:** We will use this module to convert Python lists to numpy arrays as OpenCV face recognizers accept numpy arrays.

# In[1]:

#import OpenCV module
import cv2
#import os module for reading training data directories and paths
import os
#import numpy to convert python lists to numpy arrays as 
#it is needed by OpenCV face recognizers
import numpy as np

import tkinter as tk
from tkinter import simpledialog

import math
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
#from intersect import intersection


# ### Training Data

# In[2]:

#manage input
genuines = 0
impostors = 0

total_num = 0
for dir in os.listdir("./dataset"):
    total_num+=1

#Creating User Input Dialog With Tkinter
ROOT = tk.Tk()
ROOT.withdraw()

#Parameters asked as user inputs
impostors = simpledialog.askinteger(title="Impostors",
                                  prompt="Number of Impostors [out of "+str(total_num)+"]")
while impostors >= total_num:
    impostors = simpledialog.askinteger(title="Impostors",
                                  prompt="BEWARE: Number of Impostors [out of "+str(total_num)+"]")
genuines = total_num - impostors
print("GENUINES: "+str(genuines))
print("IMPOSTORS: "+str(impostors))

percentage_test = simpledialog.askinteger(title="Percentage",
                                  prompt="Percentage [%] of Genuine probes for Testing (complementary to Probes for Testing)")
while percentage_test<=0 and percentage_test>=100:
    percentage_test = simpledialog.askinteger(title="Percentage",
                                  prompt="Percentage [%] of Genuine probes for Testing (complementary to Probes for Testing)")

input_threshold = simpledialog.askinteger(title="Input Threshold",
                                  prompt="Input Acceptance Threshold [0-255] (multiple of 5)")
while (input_threshold<0) or(input_threshold>255) or (input_threshold%5 is not 0):
    input_threshold = simpledialog.askinteger(title="Input Threshold",
                                  prompt="Input Acceptance Threshold (multiple of 5)")

input_resolution = simpledialog.askstring(title="Input Resolution",
                                  prompt="Do you want to reduce the images resolution? [Y/N]")
while input_resolution not in ('y','Y','n','N'):
    input_resolution = simpledialog.askstring(title="Input Resolution",
                                  prompt="Do you want to reduce the images resolution? [Y/N]")
resolution_reduced = 0
if input_resolution in ('y','Y'):
    resolution_reduced = 1

#create subjects-in-the-gallery array
#there is no label 0 in our training data so subject name for index/label 0 is empty
subjects = [""]
for dir in os.listdir("./dataset"):
    subjects.append(str(dir))

#total genuine probes for testing
global genuineForTesting
genuineForTesting = 0
global DBSize
DBSize = 0


# ### Prepare training data

# You may be wondering why data preparation, right? Well, OpenCV face recognizer accepts data in a specific format. It accepts two vectors, one vector is of faces of all the persons and the second vector is of integer labels for each face so that when processing a face the face recognizer knows which person that particular face belongs too. 
 
# Preparing data step can be further divided into following sub-steps.
# 
# 1. Read all the folder names of subjects/persons provided in training data folder. So for example, in this tutorial we have folder names: `s1, s2`. 
# 2. For each subject, extract label number. **Do you remember that our folders have a special naming convention?** Folder names follow the format `sLabel` where `Label` is an integer representing the label we have assigned to that subject. So for example, folder name `s1` means that the subject has label 1, s2 means subject label is 2 and so on. The label extracted in this step is assigned to each face detected in the next step. 
# 3. Read all the images of the subject, detect face from each image.
# 4. Add each face to faces vector with corresponding subject label (extracted in above step) added to labels vector. 
#

# In[3]:

#function to detect face using OpenCV
def detect_face(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #load OpenCV face detector: LBP could be used and it would be faster
    #but we chose to employ Haar classifier (more accurate but slower)
    face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')

    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
    
    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0]
    
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]


# In[4]:

#this function will read all persons' training images, detect face from each image
#and will return two lists of exactly same size, one list 
# of faces and another list of labels for each face
def prepare_training_data(data_folder_path):
    
    #------STEP-1--------
    #get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
    
    #list to hold all subject faces
    faces = []
    #list to hold labels for all subjects
    labels = []
    
    current_label = 1
    current_genuine = 0
    current_impostors = 0
    #let's go through each directory and read images within it
    for dir_name in dirs:
        img_count = len(os.listdir(data_folder_path + "/" + dir_name))
        globals()["DBSize"]+=img_count
        if current_genuine >= genuines:
            #finish assigning impostors left in the database
            print(str(dir_name)+" IMPOSTOR")
            subjects[current_label]="" #impostor
            current_impostors+=1
            current_label+=1
            if current_genuine > (len(dirs)-current_impostors):
                break
            continue
        
        #check if subject-in-gallery
        ##############################################################
        #check if there is at least one sample per subject for the Training process
        if math.ceil(img_count*percentage_test/100)==img_count:
            print(str(dir_name)+" IMPOSTOR")
            subjects[current_label]="" #impostor
            current_label+=1
            current_impostors+=1
            if current_impostors >= impostors:
                   print("Too many impostors: VERY SAD!!! Try again next time") 
                   exit()
            continue 

        current_genuine+=1

        ##########TRAIN WITH SELECTED SUBJECTS

        #------STEP-2--------
        #define label number of subject
        label = current_label
        current_label+=1

        #build path of directory containin images for current subject subject
        #sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name
        
        #get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)
        
        
        #------STEP-3--------
        #go through each image name, read image, 
        #detect face and add face to list of faces

        #TRAINING [and visit only images devoted to Training --> only (100%-PercentageForTesting)]
        for index in range(math.ceil(img_count*percentage_test/100),img_count):

            image_name = subject_images_names[index]

            #ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;
            
            #build image path
            #sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            #read image
            image = cv2.imread(image_path)
            
            #display an image window to show the image 
            # (with normal or reduced resolution according to what defined as input parameter)
            if resolution_reduced == 0:
                cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            else:
                image_dest = np.array(image)
                #factor chosen is 4.3 as sampling ratio
                image_dest = cv2.resize(image, (0,0) , fx = 4.3, fy = 4.3, interpolation = cv2.INTER_NEAREST)
                cv2.imshow("Training on reduced-resolution image...", image_dest)
                image = image_dest
            cv2.waitKey(100)
            
            #detect face
            face, rect = detect_face(image)
            
            #------STEP-4--------
            #for the purpose of this tutorial
            #we will ignore faces that are not detected
            if face is not None:
                #add face to list of faces
                faces.append(face)
                #add label for this face
                labels.append(label)
        for index in range(0,math.ceil(img_count*percentage_test/100)):
            globals()["genuineForTesting"]+=1

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    print("tutto ok")
    return faces, labels


# The following function takes the path, where training subjects' folders are stored, as parameter, and follows the same four prepare data substeps mentioned above. 
# 
# 1) On "line 8" the method `os.listdir` is used to read names of all folders stored on path passed to function as parameter, and "line 10-13" labels and faces vectors are defined. 
# 
# 2) After traversing through all subjects' folder names and from each subject's folder name on "line 27" the label information is extracted.
# 
# 3) On "line 34", all the images names of the current subject being traversed are read and analysed
#
# 4) On "line 62-66", the detected face and label are added to their respective vectors.

# In[5]:

#let's first prepare our training data
print("Preparing data...")
faces, labels = prepare_training_data("dataset")
print("Data prepared")

# ### Train Face Recognizer

# OpenCV comes equipped with three face recognizers.
# 
# 1. EigenFace Recognizer: This can be created with `cv2.face.createEigenFaceRecognizer()`
# 2. FisherFace Recognizer: This can be created with `cv2.face.createFisherFaceRecognizer()`
# 3. Local Binary Patterns Histogram (LBPH): This can be created with `cv2.face.LBPHFisherFaceRecognizer()`
# 
# The LBPH face recognizer is the one used here but any face recognizer can be chosen (the code will still work). 

# In[6]:

#create the LBPH face recognizer 
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#or use EigenFaceRecognizer by replacing above line with 
#face_recognizer = cv2.face.EigenFaceRecognizer_create()

#or use FisherFaceRecognizer by replacing above line with 
#face_recognizer = cv2.face.FisherFaceRecognizer_create()


# Now that we have initialized our face recognizer and we also have prepared our training data, it's time to train the face recognizer. We will do that by calling the `train(faces-vector, labels-vector)` method of face recognizer. 

# In[7]:

#train our face recognizer of our training faces
face_recognizer.train(faces, np.array(labels))

# ### Prediction
# In[8]:

#function to draw rectangle on image 
#according to given (x, y) coordinates and 
#given width and heigh
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
#function to draw text on give image starting from
#passed (x, y) coordinates. 
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x-1, y-1), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 2)

# Now that we have the drawing functions, we just need to call the face recognizer's `predict(face)` method to test our face recognizer on test images.

# In[9]:

#this function recognizes the person in image passed
#and draws a rectangle around detected face with name of the 
#subject
def predict(test_img):
    #make a copy of the image as we don't want to change the original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face(img)
    if face is None:
        return (None,None)

    #predict the image using our face recognizer 
    label, confidence = face_recognizer.predict(face)
    #get name of respective label returned by face recognizer
    label_text = subjects[label]
    
    #draw a rectangle around face detected
    draw_rectangle(img, rect)
    #draw name of predicted person
    draw_text(img, label_text, rect[0], rect[1]-5)
    
    return (label,img)

# Now that we have the prediction function well defined, next step is to actually call this function on our test images and display those test images to see if our face recognizer correctly recognized them. So let's do it. This is what we have been waiting for. 

# In[10]:

dir_array = []
far_array = []
roc_array = []
threshold_array = []

print("Predicting images...")
current_threshold = 0
global input_point
input_point = (0,0)
while current_threshold<=255:

    #changing threshold
    print("Current analysed THRESHOLD is "+str(current_threshold))
    face_recognizer.setThreshold(current_threshold)

    #load test images
    predicted_img = []
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0

    index_genuine = []
    for index in range(1,len(subjects)):
        if subjects[index]=="":
            continue
        index_genuine.append(index)

    counter = 1
    totalForTesting = 0
    for subfolder in os.listdir("./dataset"):  
        if subfolder.startswith("."):
                continue;
        current_subfolder="./dataset/"+subfolder
        subfolder_size = len(os.listdir(current_subfolder))
        #loop over the probes to test
        for index in range(0,math.ceil(subfolder_size*percentage_test/100)):
            totalForTesting+=1
            path = os.listdir(current_subfolder)[index]

            test_img = cv2.imread(current_subfolder+"/"+str(path))
            #perform a prediction
            (label_pred,predicted) = predict(test_img)
            if subjects[counter] == "": #is an IMPOSTOR
                check = 0
                for index in index_genuine:
                    if label_pred == index: #because concatenating both to "s" would give the same result
                        check = 1
                        break
                if check == 1:
                    #print("FALSE POSITIVE")
                    false_pos+=1
                else:
                    #print("FALSE NEGATIVE")
                    false_neg+=1
                continue
                
            if predicted is not None:
                predicted_img.append((label_pred,predicted))
                if label_pred is counter: #because concatenating both to "s" would give the same result
                    #print("TRUE POSITIVE")
                    true_pos+=1
                #else:
                    #here we have that the subject is correctly accepted but with the wrong identity
                    #print("WRONG IDENTITY")
            else:
                #print("FALSE NEGATIVE")
                false_neg+=1
            
        counter+=1

    #compute impostor probes for testing (total-genuine)
    impostorForTesting = totalForTesting-genuineForTesting

    #check performance
    DIR = true_pos / genuineForTesting #Detection and Identification Rate, at rank 1
    FAR = false_pos / impostorForTesting #False Acceptance Rate = #FA / impostors
    FRR = 1 - DIR #False Rejection Rate = 1-DIR(t,1) with threshold t
    EER = 0 #Equal Error Rate = point where FRR==FAR
    dir_array.append(DIR)
    far_array.append(FAR)
    threshold_array.append(current_threshold)
    roc_array.append((FAR,DIR))
   
    if current_threshold == input_threshold:
        #display all predicted images (accepted samples)
        for (label_pred,predicted) in predicted_img:
            cv2.imshow(subjects[label_pred], cv2.resize(predicted, (400, 500)))
        globals()["input_point"] = (FAR,DIR)
        print("For INPUT THRESHOLD of "+str(input_threshold)+" we have FAR = "+str(FAR)+", FRR = "+str(FRR)+", and DIR = "+str(DIR))
    #increment threshold by default
    current_threshold+=5


# Plotting the ROC first and then the FAR-FRR curves (EER)

#########first figure
x = np.array(far_array)
y = np.array(dir_array)
plt.plot(x, y)
plt.axline((0, 1), slope=-1, linestyle = "--")

#first EER
dx = np.array([0,1])
dy = np.array([1,0])
x1, y1 = intersection(x, y, dx, dy)
plt.plot(x1, y1, "ro")
print("ERR raw curve: found for FAR = "+str(x1[0])+" and DIR = "+str(y1[0])+" (so FRR = "+str(float(1)-float(y1[0]))+")")

#let's sanitize the two arrays
newX=[]
newY=[]
e=0
while e < len(x):
    if x[e] not in newX:
        newX.append(x[e])
        newY.append(y[e])
    e+=1
xPlot = np.array(newX)
yPlot = np.array(newY)
#let's use the monotone cubic spline
X_Y_Spline = PchipInterpolator(xPlot, yPlot)
X_ = np.linspace(xPlot.min(),xPlot.max(), 500)
Y_ = X_Y_Spline(X_)
plt.plot(X_, Y_, color = "green")

#second EER
x2, y2 = intersection(X_, Y_, dx, dy)
plt.plot(x2, y2, "ko")
print("ERR smooth curve: found for FAR = "+str(x2[0])+" and DIR = "+str(y2[0])+" (so FRR = "+str(float(1)-float(y2[0]))+")")
plt.plot(input_point[0],input_point[1], "mo")

#draw first figure
plt.title("Plot ROC Curve (marking EER point)")
plt.xlabel("FAR")
plt.ylabel("DIR")
plt.show()

#########second figure
#plot FAR with threshold
xPlot = np.array(threshold_array)
yPlot = np.array(far_array)
#let's use the monotone cubic spline
X_Y_Spline = PchipInterpolator(xPlot, yPlot)
X_ = np.linspace(xPlot.min(),xPlot.max(), 500)
Y_ = X_Y_Spline(X_)
plt.plot(X_, Y_, color = "green")

#plot FRR with threshold
frr_array = []
for i in range(0, len(dir_array)):
    frr_array.append(float(1)-float(dir_array[i]))
xPlot = np.array(threshold_array)
yPlot = np.array(frr_array)
#let's use the monotone cubic spline
X_Y_Spline = PchipInterpolator(xPlot, yPlot)
X1_ = np.linspace(xPlot.min(),xPlot.max(), 500)
Y1_ = X_Y_Spline(X1_)
plt.plot(X1_, Y1_, color = "blue")

#third EER
x3, y3 = intersection(X1_, Y1_, X_, Y_)
plt.plot(x3, y3, "ro")
print("Final ERR: found for THRESHOLD t = "+str(x3[0])+" and FRR(t) = FAR(t) = "+str(y3[0]))
plt.axvline(x3[0], color = 'r', linestyle = "--")

#draw second figure
plt.title("Plot FAR and FRR intersection in the EER")
plt.xlabel("Threshold")
plt.ylabel("Error rate")
plt.show()



cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
