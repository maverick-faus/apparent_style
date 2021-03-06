# -*- coding: utf-8 -*-
"""ClassEmbedding2048.ipynb
<center><img src="https://raw.githubusercontent.com/maverick-faus/Files/master/monogram.png" width="100">  </center>

<div align="center">
    <b>José Antonio  Faustinos</b><br>
CIC IPN - MCC - B190403  <br>
</div>

---
<div align="center">
    <i>Proyecto de Tesis - Dr Carlos Duchanoy & Dr. Marco Moreno Armendáriz </i><br>
   Arquitectura <b><i>ClassEmbedding2048</i></b><br>
   
</div>
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

expN = "20210117-060319"

path_projector = "/content/drive/My Drive/A20/Tesis/Experimentos/LIP/projector/"+expN

aux_train =  pickle.load(open(path_projector+"/projector_"+expN+"_train.pickle","rb"))
aux_test =  pickle.load(open(path_projector+"/projector_"+expN+"_test.pickle","rb"))

aux_train[0].shape

aux_train[2].shape

x_train = aux_train[0].astype(np.float32)
y_train = aux_train[2].astype(np.float32)

x_test = aux_test[0].astype(np.float32)
y_test = aux_test[2].astype(np.float32)

"""### Declarando la arquitectura

Generando función 
"""

class DNN_model(object):
  def __init__(self,
               n_nodes_hl1=500,
               n_nodes_hl2=500,
               n_nodes_hl3=500,
               n_classes=7):
    self.h1LW = tf.Variable(np.random.rand(2048, n_nodes_hl1)*0.01,name="hl1weigths",dtype="float32")
    self.h1LB = tf.Variable(np.random.rand(n_nodes_hl1)*0.01,name="hl1bias",dtype="float32")
    self.h2LW = tf.Variable(np.random.rand(n_nodes_hl1, n_nodes_hl2)*0.01,name="hl2weigths",dtype="float32")
    self.h2LB = tf.Variable(np.random.rand(n_nodes_hl2)*0.01,name="hl2bias",dtype="float32")
    self.h3LW = tf.Variable(np.random.rand(n_nodes_hl2, n_nodes_hl3)*0.01,name="hl3weigths",dtype="float32")
    self.h3LB = tf.Variable(np.random.rand(n_nodes_hl3)*0.01,name="hl3bias",dtype="float32")
    self.outW = tf.Variable(np.random.rand(n_nodes_hl3, n_classes)*0.01,name="outweigths",dtype="float32")
    self.outB = tf.Variable(np.random.rand(n_classes)*0.01,name="outbias",dtype="float32")
    self.trainable_variables =[self.h1LW,self.h1LB,self.h2LW,self.h2LB,self.h3LW,self.h3LB,self.outW,self.outB]          
  def __call__(self,x): 
      # Declarando la arquitectura

      l1 = tf.add(tf.matmul(x,self.h1LW), self.h1LB)
      l1 = tf.nn.relu(l1)

      l2 = tf.add(tf.matmul(l1,self.h2LW), self.h2LB)
      l2 = tf.nn.relu(l2)

      l3 = tf.add(tf.matmul(l2,self.h3LW), self.h3LB)
      l3 = tf.nn.relu(l3)

      output = tf.matmul(l3,self.outW) + self.outB
      return output

DNN = DNN_model()

"""Seleccionar un optimizador """

optimizador = tf.keras.optimizers.Adam(learning_rate=0.0001 )
#optimizador = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001)

"""### Definir las metricas a usar"""

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

"""### Calculo de gradientes y ajuste """

@tf.function
def train_step(model,tdata, labels):
  with tf.GradientTape() as tape:
    predictions = model(tdata)
    #calculo de una funcion de error 
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, predictions))
   
  gradients = tape.gradient(loss, model.trainable_variables)
  capped_grads_and_vars = [(grad,model.trainable_variables[index]) for index, grad in enumerate(gradients)]
  optimizador.apply_gradients(capped_grads_and_vars)
  train_loss(loss)
  train_accuracy(labels, predictions)

#train_step(DNN,x_train[24:30], y_train_onehot[24:30])

@tf.function
def test_step(model,tdata, labels):
  predictions = model(tdata)
  t_loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, predictions))

  test_loss(t_loss)
  test_accuracy(labels, predictions)

"""Bitácoras"""

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
report["expNumber"]=current_time
train_log_dir = '/content/drive/My Drive/A20/Tesis/Experimentos/ClassEmbedding2048/logs/' + current_time + '/train'
test_log_dir = '/content/drive/My Drive/A20/Tesis/Experimentos/ClassEmbedding2048/logs/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

current_time

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir "drive/My Drive/A20/Tesis/Experimentos/ClassEmbedding2048/logs/"$current_time

"""## función de entrenamiento  y prueba"""

def fitting(model,train_x,train_y,test_x,test_y,EPOCHS,N_batch,batch_size):

  table = Texttable()
  table.set_deco(Texttable.HEADER)
  table.set_cols_width([8,15,15,15,15])
  table.set_cols_align(["c", "r", "r", "r", "r"])
  table.set_cols_dtype(['i','e','e','e','e'])
  table.add_rows([["Epoch","Perdida", "Exactitud", "Perdida_test", "Exactitud_test"]])
  print(table.draw())

  for epoch in range(EPOCHS):
    i=0
    while i+batch_size < len(train_x) or i+batch_size<batch_size*N_batch:
      start = i
      end = i+batch_size
      batch_x = train_x[start:end]
      batch_y = train_y[start:end]
      train_step(model,batch_x,batch_y)
      i+=batch_size

    with train_summary_writer.as_default():
      tf.summary.scalar('Train Loss', train_loss.result(), step=epoch)
      tf.summary.scalar('Train Accuracy', train_accuracy.result(), step=epoch)

    test_step(model,test_x,test_y)
    with test_summary_writer.as_default():
      tf.summary.scalar('Test loss', test_loss.result(), step=epoch)
      tf.summary.scalar('Test accuracy', test_accuracy.result(), step=epoch)

    table.set_deco(Texttable.HLINES)
    table.add_rows([[int(epoch+1),train_loss.result().numpy(), (train_accuracy.result()*100).numpy(), test_loss.result().numpy(),(test_accuracy.result()*100).numpy()]])
    print(table.draw())
      
    template = 'Epoch {}, Perdida: {}, Exactitud: {}, Perdida de prueba: {}, Exactitud de prueba: {}'
    
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

fitting(DNN,x_train,y_train,x_test,y_test,250,100,92)

