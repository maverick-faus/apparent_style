# -*- coding: utf-8 -*-
"""LIP_tf2_Revisado.ipynb
<center><img src="https://raw.githubusercontent.com/maverick-faus/Files/master/monogram.png" width="100">  </center>

<div align="center">
    <b>José Antonio  Faustinos</b><br>
CIC IPN - MCC - B190403  <br>
</div>

---
<div align="center">
    <i>Proyecto de Tesis - Dr Carlos Duchanoy & Dr. Marco Moreno Armendáriz </i><br>
   Arquitectura LIP<br>
   
</div>

En este notebook se presenta la implementación de la arquitectura del experimento LIP, basado en un autoencoder convolucional utilizando el preprocesamiento LIP128 del dataset, el cual implica un padding y un resize hasta obtener imágnes de 128*128.

La arquitectura implementadase especifica a continuación:
"""

report={}
report["expName"]= "Style14"
report["dataset"] = "Style14_128"
report["datasetObs"] = "None"

# Commented out IPython magic to ensure Python compatibility.
# Load the TensorBoard notebook extension
# %load_ext tensorboard

!pip install -U git+http://github.com/bufordtaylor/python-texttable
! pip install PyPDF2
! pip install reportlab

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x

# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
import datetime, os
import numpy as np
import pickle
import io
import json
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PyPDF2 import PdfFileWriter, PdfFileReader
from PIL import Image 
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from texttable import Texttable
from tqdm import tqdm
from google.colab import auth



from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.client import GoogleCredentials

# %matplotlib inline

"""Montamos Drive"""

from google.colab import drive
drive.mount('/content/drive/')
auth.authenticate_user()

path_pickles="/content/drive/My Drive/A20/Tesis/Experimentos/LIP/pickles_LIP128"
path_logs="/content/drive/My Drive/A20/Tesis/Experimentos/LIP"

# !ls "/content/drive/My Drive/A20/Tesis/Experimentos/LIP/pickles_LIP128"

listaPickles=[]
for files in tqdm(os.listdir(path_pickles)):
  listaPickles.append(path_pickles+"/"+files)

"""# Cargamos Dataset

Recuperamos pickles
"""

def recuperarPickle(nombre):
  pick=np.array(pickle.load(open(nombre,"rb")))
  np.random.shuffle(pick)
  return pick

x_test=np.empty([1,128,128,3],dtype="uint8")

for i in tqdm(range(10)):
  pick = recuperarPickle(listaPickles[i])
  x_test  =  np.concatenate((x_test, pick))
x_test = np.delete(x_test, 0,0)

x_train=np.empty([1,128,128,3],dtype="uint8")

for i in tqdm(range(10,len(listaPickles))):
  pick = recuperarPickle(listaPickles[i])
  x_train  =  np.concatenate((x_train, pick))
x_train = np.delete(x_train, 0,0)

print(str(x_train.shape))
print(str(x_test.shape))

plt.imshow(x_test[0])
plt.show()
plt.imshow(x_train[0])
plt.show()

"""# Declarando la arquitectura

Forward from here
"""

class DNN_model(tf.Module):
  def __init__(self):
    self.C1LW = tf.Variable(np.random.rand(5,5,3,8)*0.0001,name="hl1weigths",dtype="float32")
    self.C1LB = tf.Variable(np.random.rand(8)*0.0001,name="hl1bias",dtype="float32")
    self.C2LW = tf.Variable(np.random.rand(5,5,8,16)*0.0001,name="hl2weigths",dtype="float32")
    self.C2LB = tf.Variable(np.random.rand(16)*0.0001,name="hl2bias",dtype="float32")
    self.C3LW = tf.Variable(np.random.rand(5,5,16,32)*0.0001,name="hl3weigths",dtype="float32")
    self.C3LB = tf.Variable(np.random.rand(32)*0.0001,name="hl3bias",dtype="float32")
#------------------------------------------------------------------------------------------------------------------
    self.D0LW = tf.Variable(np.random.rand(5,5,16,32)*0.0001,name="hl0weigths",dtype="float32")
    self.D0LB = tf.Variable(np.random.rand(16)*0.0001,name="hl0bias",dtype="float32")
    self.D1LW = tf.Variable(np.random.rand(5,5,8,16)*0.0001,name="hl1weigths",dtype="float32")
    self.D1LB = tf.Variable(np.random.rand(8)*0.0001,name="hl1bias",dtype="float32")
    self.D2LW = tf.Variable(np.random.rand(5,5,3,8)*0.0001,name="hl2weigths",dtype="float32")
    self.D2LB = tf.Variable(np.random.rand(3)*0.0001,name="hl2bias",dtype="float32")
    self.trainableVar =[self.C1LW,self.C1LB,self.C2LW,self.C2LB,self.C3LW,self.C3LB,
                        self.D0LW,self.D0LB,self.D1LW,self.D1LB,self.D2LW,self.D2LB]   
         
  def __call__(self,x,Batch): 

      # Declarando la arquitectura
      x  = tf.cast(x, tf.float32)
      img = tf.reshape(x, shape=[-1, 128, 128, 3])
      #128*128*3, 49152

      C1= tf.nn.conv2d(img,self.C1LW, strides=[1,2,2,1], padding='SAME')  
      C1 = tf.add(C1, self.C1LB)
      C1 = tf.nn.relu(C1)
      #64*64*8, 32768

      C2= tf.nn.conv2d(C1,self.C2LW, strides=[1,2,2,1], padding='SAME')  
      C2 = tf.add(C2, self.C2LB)
      C2 = tf.nn.relu(C2)
      #32*32*16, 16384

      C3= tf.nn.conv2d(C2,self.C3LW, strides=[1,2,2,1], padding='SAME')  
      C3 = tf.add(C3, self.C3LB)
      C3 = tf.nn.relu(C3)
      #16*16*32, 8192

      D0= tf.nn.conv2d_transpose(C3,self.D0LW, tf.constant([Batch,32,32,16]),strides=[1,2,2,1],padding='SAME')  
      D0 = tf.add(D0, self.D0LB)
      D0 = tf.nn.relu(D0)
      #imagen resultante de #32*32*16
      
      D1= tf.nn.conv2d_transpose(C2,self.D1LW, tf.constant([Batch,64,64,8]),strides=[1,2,2,1],padding='SAME')  
      D1 = tf.add(D1, self.D1LB)
      D1 = tf.nn.relu(D1)
      #imagen resultante de 64*64*8

      D2= tf.nn.conv2d_transpose(D1,self.D2LW, tf.constant([Batch,128,128,3]),strides=[1,2,2,1],padding='SAME')  
      D2 = tf.add(D2, self.D2LB)
      D2 = tf.nn.relu(D2)
      #imagen resultante de 128*128*3

      
      return D2

DNN = DNN_model()

#Just 2 verify
DNN.trainableVar.trainable_variables[0][0][0][0]

optimizador  =tf.keras.optimizers.Adam(0.0001)

train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')

@tf.function
def train_step(model,tdata, labels):
  with tf.GradientTape() as tape:
    predictions = model(tdata,len(tdata))
    #calculo de funcion de error 
    x = tf.cast(labels, tf.float32)
    img = tf.reshape(x, shape=[-1, 128, 128, 3])
    loss = tf.reduce_mean(tf.math.squared_difference(img, predictions))
   
  gradients = tape.gradient(loss, model.trainableVar.trainable_variables)
  
  capped_grads_and_vars = [(grad,model.trainableVar.trainable_variables[index]) for index, grad in enumerate(gradients) if grad is not None]
#capped_grads_and_vars = [(grad,model.trainableVar.trainable_variables[index]) for index, grad in enumerate(gradients)]
  optimizador.apply_gradients(capped_grads_and_vars)
  train_loss(loss)

@tf.function
def test_step(model,tdata, labels):
  predictions = model(tdata,len(tdata))
  x = tf.cast(labels, tf.float32)
  img = tf.reshape(x, shape=[-1, 128, 128, 3])
  t_loss = tf.reduce_mean(tf.math.squared_difference(img, predictions))
  test_loss(t_loss)
  return predictions

"""Bitácoras"""

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
report["expNumber"]=current_time
train_log_dir = '/content/drive/My Drive/A20/Tesis/Experimentos/LIP/logs/' + current_time + '/train'
test_log_dir = '/content/drive/My Drive/A20/Tesis/Experimentos/LIP/logs/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

"""Checkpoint"""

path_checkpoints = "/content/drive/My Drive/A20/Tesis/Experimentos/LIP/checkpoints_LIP/"+current_time
checkpoint_prefix = os.path.join(path_checkpoints, "ckpt")
checkpoint= tf.train.Checkpoint(optmizer=optimizador, model = DNN)

current_time

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir "drive/My Drive/A20/Tesis/Experimentos/LIP/logs/"$current_time

def fitting(model,train_x,train_y,test_x,test_y,EPOCHS,N_batch,batch_size,N_batch_test,batch_size_test):
  

  table = Texttable()
  table.set_deco(Texttable.HEADER)
  table.set_cols_width([8,20,20])
  table.set_cols_align(["c", "r", "r"])
  table.set_cols_dtype(['i','e','e'])
  table.add_rows([["Epoch","Perdida", "Perdida_test"]])
  print(table.draw())

  for epoch in range(EPOCHS):
    
    #Train Step------------------------------------------------------------------------------------
    i=0
    while i+batch_size < len(train_x) or i+batch_size<batch_size*N_batch:
      start = i
      end = i+batch_size
      batch_x = train_x[start:end]
      batch_y = train_y[start:end]
      train_step(model,batch_x,batch_y)
      i+=batch_size
    checkpoint.save(file_prefix=checkpoint_prefix)

    with train_summary_writer.as_default():
      tf.summary.scalar('Train Loss', train_loss.result(), step=epoch)
    
    #Test Step------------------------------------------------------------------------------------
    j=0
    while j+batch_size_test < len(test_x) or j+batch_size_test<batch_size_test*N_batch_test:
      start = j
      end = j+batch_size_test
      batch_x_test = test_x[start:end]
      batch_y_test = test_y[start:end]
      pr = np.reshape(batch_x_test[0], (128, 128, 3)).astype(np.uint8)
      preds = test_step(model,batch_x_test,batch_y_test)
      aux_pred = preds.numpy()
      j+=batch_size_test


    with test_summary_writer.as_default():
      tf.summary.scalar('Test loss', test_loss.result(), step=epoch)
      images = np.array([pr, np.reshape(aux_pred[0], (128, 128, 3))])
      # tf.summary.image("Input vs Decoded", np.reshape(images, (-1, 128, 128, 3)) , max_outputs=2, step=epoch)
      tf.summary.image("Input", np.reshape(pr, (-1, 128, 128, 3)), max_outputs=1, step=epoch)
      tf.summary.image("Decoded", np.reshape(np.reshape(aux_pred[0], (128, 128, 3)), (-1, 128, 128, 3)), max_outputs=1, step=epoch)
      
      

      
    table.set_deco(Texttable.HLINES)
    table.set_cols_align(["c", "r", "r"])
    table.add_rows([[int(epoch+1),train_loss.result().numpy(),  test_loss.result().numpy()]])
    print(table.draw())


    train_loss.reset_states()
    test_loss.reset_states()

"""## 500 epochs"""

#model,train_x,train_x,test_x,test_x,EPOCHS,N_batch,batch_size,N_batch_test,batch_size_test
init = datetime.datetime.now()# Ignorar esto, es para medir cuanto tiempo dura la ejecución

#fitting(DNN,x_train,x_train,x_test,x_test,5000,10,100,10,100)
fitting(DNN,x_train,x_train,x_test,x_test,500,10,100,10,100)

end=datetime.datetime.now()

print(str(round((end - init).total_seconds()/60,2))+" min.")

"""Imágenes de validación para propagar"""

#https://raw.githubusercontent.com/maverick-faus/LIP/main/prep_1.jpg
#https://raw.githubusercontent.com/maverick-faus/LIP/main/prep_2.jpg
#https://raw.githubusercontent.com/maverick-faus/LIP/main/prep_3.jpg
#https://raw.githubusercontent.com/maverick-faus/LIP/main/prep_4.jpg
#https://raw.githubusercontent.com/maverick-faus/LIP/main/prep_5.jpg

import urllib.request
urllib.request.urlretrieve("https://raw.githubusercontent.com/maverick-faus/LIP/main/prep_1.jpg", "file2Propagate.jpg") 

np_arr_prop = np.array(Image.open("file2Propagate.jpg"))
np_arr_prop=np_arr_prop.reshape(1,128,128,3)
np_arr_prop.shape

plt.imshow(np_arr_prop[0])
plt.show()
print(np_arr_prop[0].max())
#128*128 = 16,384 * 3 = 49,152
#Imagen de 128 x 85 = 10,880 * 3 = 32,140 
#Error final 11,145/3 = 3,715

prediccion= DNN(np_arr_prop,len(np_arr_prop))
print(prediccion)
print(type(prediccion))
out1 = prediccion.numpy()
print(type(out1))
print(out1[0].shape)
plt.imshow(out1[0].astype(int))
plt.show()

out1

