import numpy as np
import tensorflow as tf
from copy import deepcopy


def deepFoolMultiClass(image, classifier):

    shape=(28, 28, 1)

    image_array = np.array(image)
    image_normalized = np.reshape(tf.cast(image_array / 255.0 - 0.5, tf.float32),\
                                   shape)
    image_f = classifier(image_normalized[tf.newaxis, ...]).flatten()


    
    return 0


