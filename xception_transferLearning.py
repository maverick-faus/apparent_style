from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications import xception
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime, os
import pickle

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

IMAGE_SIZE = [128, 128]

from google.colab import drive
drive.mount('/content/drive')

path_pickles="/content/drive/My Drive/A20/Tesis/Experimentos/Style14/dataset/pickles_Style14_128"
path_logs="/content/drive/My Drive/A20/Tesis/Experimentos/Style14"

!ls "/content/drive/My Drive/A20/Tesis/Experimentos/Style14/dataset/pickles_Style14_128"

"""# Cargamos Dataset

Recuperamos pickles
"""

x_train=np.empty(0)
y_train=np.empty(0)

for files in tqdm(os.listdir(path_pickles)):
  if "test" in files:
    test=np.array(pickle.load(open(path_pickles+"/"+files,"rb")))
  else:
    train=np.array(pickle.load(open(path_pickles+"/"+files,"rb")))
    x_train  = np.append(x_train, train[:,0],0)
    y_train  = np.append(y_train, train[:,1],0)

x_test = test[:,0]
y_test =test[:,1]

y_test=np.vstack(y_test)
x_test=np.vstack(x_test)
x_test=x_test.reshape(-1,128,128,3)

y_train=np.vstack(y_train)
x_train=np.vstack(x_train)
x_train=x_train.reshape(-1,128,128,3)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print()

batch_size = 32

num_train = x_train.shape[0]
num_validation =x_test.shape[0]
num_classes = 7

plt.imshow(x_test[0])
plt.show()
y_test[0]

y_train_cat = np.empty(0)
y_test_cat = np.empty(0)

for i in tqdm(y_train):
  y_train_cat = np.append(y_train_cat,np.argmax(i)+1).astype(int)
for j in tqdm(y_test):
  y_test_cat = np.append(y_test_cat,np.argmax(j)+1).astype(int)

"""## Preprocesameinto"""

datagen = ImageDataGenerator(rescale=1/255.,validation_split=0.2)

training_generator = datagen.flow(x_train, y_train, batch_size=286,seed=7)
testing_generator = datagen.flow(x_test, y_test, batch_size=286,seed=7)

plt.figure(figsize=(10,5))
for i in range(6):
    plt.subplot(2,3,i+1)
    for x,y in testing_generator:
        plt.imshow(x[0])
        plt.title('y={}'.format(y[0]))
        plt.axis('off')
        break
plt.tight_layout()
plt.show()

"""# Modelo"""

xception = Xception(include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(128,128,3),
    pooling=None,
    classes=7,
    classifier_activation="softmax",
)

xception.input

for layer in xception.layers:
  layer.trainable = False

x = Flatten()(xception.output)
prediction = Dense(7, activation='softmax')(x)
model = Model(inputs=xception.input, outputs=prediction)
model.summary()

from keras import optimizers

adam = optimizers.Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

from datetime import datetime
from keras.callbacks import ModelCheckpoint
start = datetime.now()
chkpath= "/content/drive/MyDrive/A20/Tesis/Experimentos/Transfer/Checkpoints/XCEPTION"+'/XCEPTION_'+str(start)+'.h5'
checkpoint = ModelCheckpoint(filepath=chkpath, 
                               verbose=2, save_best_only=True, monitor='accuracy')

callbacks = [checkpoint]



model_history=model.fit_generator(
  training_generator,
  validation_data=testing_generator,
  epochs=400,
  steps_per_epoch=5,
  validation_steps=30,
  callbacks=callbacks ,verbose=0)


duration = datetime.now() - start
print("Training completed in time: ", duration)

plt.plot(model_history.history['accuracy'])
plt.title('XCEPTION Model Testing Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

plt.plot(model_history.history['loss'])
plt.title('XCEPTION Model Testing Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

