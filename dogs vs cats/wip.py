from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.applications import VGG16
import PIL
import os
import numpy as np
import random

conv_base = VGG16(weights = 'imagenet',
                  include_top = False,
                  input_shape = (150, 150, 3))


train_dir = r'C:\Users\hydro\Downloads\dogs-vs-cats-small\train\cats'

imgs = [os.path.join(train_dir, i) for i in os.listdir(train_dir)]

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=90)

for i in range(20):
    val = random.choice(imgs)
    orig = image.load_img(val, target_size = (150, 150))
    orig_to_array = image.img_to_array(orig)
    orig_to_array_reshape = np.reshape(orig_to_array, (1,)+orig_to_array.shape)
    print(i)
    for x in train_datagen.flow(orig_to_array_reshape, batch_size = 1):
        plt.figure(i)
        x1 = plt.imshow(image.array_to_img(x[0]))
        print(x.shape)
        break

plt.show()
    
    

#for d in train_datagen.flow(data, batch_size = 20):
    




'''j = 0

for i in imgs:
    plt.figure(j)
    j += 1
    x = image.load_img(i, target_size=(150, 150))
    plt.imshow(x)
    if j >= 20:
        break

plt.show()'''

'''x1 = image.img_to_array(x)

x1 = x1.reshape((1, ) + x1.shape)

x2 = image.array_to_img(x1)'''




