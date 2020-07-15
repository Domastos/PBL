import numpy as np
import tensorflow as tf
from copy import deepcopy
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

def loss():
    return 0


def deepFoolMultiClass(imageName, modelName, numOfLabels):

    #load_image
    img = load_img(imageName, target_size=(32, 32))
    img = img_to_array(img)
    img = img.reshape(1, 32, 32, 3)
    img = img.astype('float32')/ 255.0


    I = (np.array(img)).flatten().argsort()[::-1]
    I = I[0:numOfLabels]

    #f()
    model = load_model(modelName)

    #f(X0)
    initLable = model.predict_classes(img)[0]
          
    #f(x1)
    Label = initLable 

    #when ki = k
    # while Label == initLable:

        #for classes that are not image                   
        # for i in range(1, numOfLabels):

    return img

