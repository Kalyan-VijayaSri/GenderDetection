import numpy as np
import pandas as pd
import cv2

'''data = pd.read_csv("wiki5.csv")
y = data.iloc[:,1:2].values
print(data.shape[0])
for i in range(data.shape[0]):
    data_img = (data.iloc[:,3:].values[i])
    data_img = np.array(data_img)
    img = np.reshape(data_img, (100,100))
    if(y[i] == 1):
        cv2.imwrite("genderDetect/Male/frame%d.jpg" % i, img)
    elif(y[i] == 0):
        cv2.imwrite("genderDetect/Female/frame%d.jpg" % i, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('genderDetection/wiki12_train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('genderDetection/wiki12_test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 2000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)

from keras.models import load_model

classifier.save('/home/vijayasri/svn_wc/cr24_k.vijayasri/Deep_Learning_A_Z/GenderDetection/genderDetection/prediction/my_model1.h5') 
classifier = load_model('/home/vijayasri/svn_wc/cr24_k.vijayasri/Deep_Learning_A_Z/GenderDetection/genderDetection/prediction/my_model1.h5')

import cv2
import numpy as np
from keras.preprocessing import image
face_cascade = cv2.CascadeClassifier('/home/vijayasri/Downloads/opencv-4.0.1/data/haarcascades_cuda/haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture('/home/vijayasri/Downloads/opencv-4.0.1/samples/data/Megamind.avi')
check, frame = cap.read()
check = True
while(check == True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x + w, y + h),(255, 0, 0), 2)
        detected_face = img[int(y):int(y+h), int(x):int(x+w)] 
        #detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) 
        detected_face = cv2.resize(detected_face, (64, 64)) 
        test_image = image.img_to_array(detected_face)
        test_image = np.expand_dims(test_image, axis = 0)
        test_image /= 255
        result = classifier.predict(test_image)
        if result[0][0] == 1:
            prediction = 'male'
        else:
            prediction = 'female'
        cv2.putText(img, prediction, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)

    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


'''from keras.preprocessing import image
test_image = image.load_img('genderDetection/prediction/index1.jpeg', target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'male'
else:
    prediction = 'female'''
    
