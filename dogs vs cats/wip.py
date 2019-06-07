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

train_datagen = ImageDataGenerator(rescale=1./255)

orig_to_array = np.zeros((20, 50, 50, 3), dtype = 'float32')

for i in range(20):
    val = imgs[:20]
    orig = image.load_img(val[i], target_size = (50, 50))
    orig_to_array[i, :, :, :] = image.img_to_array(orig)
    #orig_to_array_reshape = np.reshape(orig_to_array, (20,)+orig_to_array.shape)

save_dir = r'C:\Users\hydro\Downloads\New folder'
i = 0    
for x in train_datagen.flow(orig_to_array, batch_size = , save_to_dir = save_dir):
    i += 1
    if i == 20:
        break
    # i is the images it will take from the directory, independent, based on orig_to_array
    # batch_size is the number of ops done on each
    # so 20 images with 10 ops each
    #I THINK THATS IT...but well !

    
    

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




