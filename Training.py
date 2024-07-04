from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import tensorflow as tf
print(tf.__version__)


# import the libraries as shown below

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
#import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt



# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'Datasets/train'
valid_path = 'Datasets/test'


vgg16 = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

print("Stage 1")

for layer in vgg16.layers:
    layer.trainable = False
 # useful for getting number of output classes
folders = glob('Datasets/train/*')

print("Stage 2")

x = Flatten()(vgg16.output)
print("x")
print (x)

prediction = Dense(len(folders), activation='softmax')(x)

print("prediction")
print (prediction)

print("Stage 3")

model = Model(inputs=vgg16.input, outputs=prediction)
model.summary()


print("Stage 4")

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

print("Stage 5")
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


print("test_datagen")
print (test_datagen)


print("Stage 6")
training_set = train_datagen.flow_from_directory('Datasets/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


print("training_set")
print (training_set)


print("Stage 7")
test_set = test_datagen.flow_from_directory('Datasets/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


print("test_set")
#print (test_set)
print(len(training_set))
print(len(test_set))


print("Stage 7")

FE_r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=20,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)


print("FE_r")
print (FE_r)


print("Stage 8")

plt.plot(FE_r.history['loss'], label='train loss')
plt.plot(FE_r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

print("Stage 9")

# plot the accuracy
plt.plot(FE_r.history['accuracy'], label='train acc')
plt.plot(FE_r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('accuracy')

print("Stage 10")

model.save('full_model.h5')

y_pred = model.predict(test_set)
y_pred = np.argmax(y_pred, axis=1)

print("Stage 11")

