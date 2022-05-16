# -*- coding: utf-8 -*-
"""Style7.ipynb
<center><img src="https://raw.githubusercontent.com/maverick-faus/Files/master/monogram.png" width="100">  </center>

<div align="center">
    <b>José Antonio  Faustinos</b><br>
CIC IPN - MCC - B190403  <br>
</div>

---
<div align="center">
    <i>Proyecto de Tesis - Dr Carlos Duchanoy & Dr. Marco Moreno Armendáriz </i><br>
   Arquitectura Style14_1<br>
   
</div>

En este notebook se presenta la implementación de la primera arquitectura del experimento Style14, utilizando el preprocesamiento Style14_128 del dataset, el cual implica un padding y un resize hasta obtener imágnes de 128*128.

La arquitectura implementada se especifica a continuación:
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

path_pickles="/content/drive/My Drive/A20/Tesis/Experimentos/Style14/dataset/pickles_Style14_128"
path_logs="/content/drive/My Drive/A20/Tesis/Experimentos/Style14"

!ls "/content/drive/My Drive/A20/Tesis/Experimentos/Style14/dataset/pickles_Style14_128"

"""# Generación de reportes

Esta función genera el reporte de entrenamiento, el método *drive.CreateFile({'id': file_id})* recupera un JSON con el contenido del Notebook actual, a partir de el se hace la búsqueda de los parámetros en la celda, esto con la finalidad de que siempre que se modifiquen, el reporte cambie de manera automática. 

El PDF es generado con el metodo Canvas de la biblioteca LabReport. Hace uso del formato del PDF vacío y posiciona el texto en el.
"""

def generateReport():
  #Obtenemos Json con el contenido del notebook para recuperar los hiperparámetros
  report["execTime"]=str(round((end - init).total_seconds()/60,2))+" min."

  arq2File=[]
  arq2File.append ("Input : "+report["volDim"])
  
  gauth = GoogleAuth()
  gauth.credentials = GoogleCredentials.get_application_default()
  drive = GoogleDrive(gauth)

  file_id = '1yO2B-i_csAr_OU8YAukNN3P_IvxP0heL'
  file = drive.CreateFile({'id': file_id})
  downloaded = file.GetContentString() 

  for cell in json.loads(downloaded)['cells']:

    if "optimizador" in cell["source"][0]:
      report["optimizer"] = cell["source"][0].split(" = ")[1]

    if "loss_object" in cell["source"][0]:
      report["loss_object"] = cell["source"][0].split(" = ")[1]

    if "#model," in cell["source"][0]:
      aux = cell["source"][3].replace(")","").split(",")
      report["epochs"]= aux[5]
      report["trainBatchN"]= aux[6]
      report["trainBatchSize"]= aux[7]
      report["testBatchN"]= aux[8]
      report["testBatchSize"]= aux[9].replace("\n","")

    if "#Metrics" in cell["source"][0]:
      report["trainLoss"]= cell["source"][1].split("=")[1].split("(")[0]
      report["trainAcc"]= cell["source"][2].split("=")[1].split("(")[0]
      report["testLoss"]= cell["source"][4].split("=")[1].split("(")[0]
      report["testAcc"]= cell["source"][5].split("=")[1].split("(")[0]

    if cell["metadata"]["id"]== "b6NpPnF9Fm49":
      src = cell["source"]
      i=0
      for elem in src:
        if "tf.nn.conv2d" in elem or "tf.nn.relu" in elem or "tf.nn.max_pool" in elem or "#Flattening" in elem or "tf.nn.dropout" in elem or ( "output" and "matmul") in elem or ("#" or "x") in elem:
          aux = elem.replace(',data_format="NHWC"',"").replace("strides","str").replace("padding","p").replace("SAME","S").replace("tf.nn.","").replace("      ","").replace(" ","").replace("\n","").replace("ksize","k").replace("#Declarandolaarquitectura","").replace("max_pool","pool")
          aux = re.sub('(,self.)[^,]*', '', aux)
          if "conv2d" in aux or "dropout" in aux:
            arq2File.append( str(DNN.trainable_variables[i].shape))
            i = i+2
          arq2File.append(aux)
          if "#" in aux:
            arq2File.append("\n")


      f= open("archi.txt","w+")
      for elem in arq2File:
        f.write(elem+"\n")
      f.close() 
    
  
  lenx = len(training_loss)

  maxY = sum(training_loss) / len(training_loss)
  fig = plt.figure(figsize=(10, 6), dpi=100)
  fig.patch.set_facecolor('#d3d9e8')
  ax = plt.axes()
  plt.text(lenx-2, training_loss[lenx-1], training_loss[lenx-1])
  plt.title('Training Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.ylim(0, maxY)
  ax.plot(range(lenx), training_loss, color="#0a99ff", label='Training Loss');
  fig.savefig('TrainingLoss.png', facecolor='#d3d9e8', dpi=300)

  maxY = sum(testing_loss) / len(testing_loss)
  fig = plt.figure(figsize=(10, 6), dpi=100)
  fig.patch.set_facecolor('#d3d9e8')
  ax = plt.axes()
  plt.text(lenx-2, testing_loss[lenx-1], testing_loss[lenx-1])
  plt.title('Testing Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.ylim(0, maxY)
  ax.plot(range(lenx), testing_loss, color="blue", label='Testing Loss');
  fig.savefig('TestingLoss.png', facecolor='#d3d9e8', dpi=300)

  fig = plt.figure(figsize=(10, 6), dpi=100)
  fig.patch.set_facecolor('#ffdfb5')
  ax = plt.axes()
  plt.text(lenx-2, training_accurracy[lenx-1], training_accurracy[lenx-1])
  plt.title('Training Accurracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accurracy %')
  ax.plot(range(lenx), training_accurracy, color="#730f1c", label='Training Accurracy');
  fig.savefig('TrainingAccurracy.png', facecolor='#ffdfb5', dpi=300)

  fig = plt.figure(figsize=(10, 6), dpi=100)
  fig.patch.set_facecolor('#ffdfb5')
  ax = plt.axes()
  plt.text(lenx-2, testing_accurracy[lenx-1], testing_accurracy[lenx-1])
  plt.title('Testing Accurracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accurracy %')
  ax.plot(range(lenx), testing_accurracy, color="#ff6c17", label='Testing Accurracy');
  fig.savefig('TestingAccurracy.png', facecolor='#ffdfb5', dpi=300) 

#---------------------------------------------------------------------

  packet = io.BytesIO()
  can = canvas.Canvas(packet, pagesize=letter)

  can.setFillColorRGB(0.6,0,0) 
  can.setFont("Courier",12)
  can.drawString(440, 648, report["expNumber"]) #ExperimentNumber

  can.setFillColorRGB(0,0,0) 
  can.setFont("Times-Roman",12)

  can.drawString(440, 665, report["expName"]) #ExperimentName
  can.drawString(95, 610,  report["dataset"]) #DatasetName
  can.drawString(330, 610, report["volDim"]) #Input dimensions
  can.drawString(510, 610, report["normData"]) #Normalized data
  can.drawString(140, 580, report["trainingSamples"]) #Training Samples
  can.drawString(340, 580, report["testingSamples"]) #Testing Samples
  can.drawString(490, 580, report["totalSamples"]) #total Samples
  can.drawString(150, 550, report["datasetObs"]) #Dataset Obs
  can.drawString(105, 485, report["optimizer"]) #Optimizer
  can.drawString(120, 460, report["loss_object"]) #Loss
  can.drawString(155, 436, report["trainBatchN"]) #TrainBatchN
  can.drawString(300, 436, report["trainBatchSize"]) #TrainBatchSize
  can.drawString(155, 410, report["testBatchN"]) #TestBatchN
  can.drawString(300, 410, report["testBatchSize"]) #TestBatchSize
  can.drawString(110, 387, report["epochs"]) #Epochs
  can.drawString(280, 387, report["regularization"]) #Regularization
  can.drawString(100, 327, report["trainLoss"]) #TrainLoss
  can.drawString(375, 327, report["trainAcc"]) #TrainAccurracy
  can.drawString(100, 303, report["testLoss"]) #TestLoss
  can.drawString(375, 303, report["testAcc"]) #TestAccurracy
  can.drawString(150, 50, report["normalFactor"]) #NormalWeights
  can.drawString(375, 50, report["softmax"]) #Softmax at Output

  can.setFont("Courier",7)
  y=260
  x= 38
  f = open("archi.txt", "r")
  for line in f:
    line = line.replace("\n","")
    can.drawString(x, y, line) #Architecture
    y=y-10
    if y <= 80:
      x = x+200
      y=260
  can.save()

  #move to the beginning of the StringIO buffer
  packet.seek(0)
  new_pdf = PdfFileReader(packet)
  # read your existing PDF
  existing_pdf = PdfFileReader(open(path_logs+"/TrainingExperimentReport2.pdf", "rb"))
  output = PdfFileWriter()
  # add the "watermark" (which is the new pdf) on the existing page
  page = existing_pdf.getPage(0)
  page.mergePage(new_pdf.getPage(0))
  output.addPage(page)
  #-----------------------------------------------------------PAGE 2
  packet2 = io.BytesIO()
  can2 = canvas.Canvas(packet2, pagesize=letter)
  can2.setFillColorRGB(0,0,0) 
  can2.setFont("Times-Roman",12)

  can2.drawString(125, 715, report["execTime"]) #ExecTime

  logo2 = ImageReader('TrainingLoss.png')
  can2.drawImage(logo2, 35, 375, mask='auto', width=542, height=305 )

  logo = ImageReader('TestingLoss.png')
  can2.drawImage(logo, 35, 45, mask='auto', width=542, height=305 )

  can2.save()


  packet2.seek(0)
  new_pdf2 = PdfFileReader(packet2)
  page2 = existing_pdf.getPage(1)
  page2.mergePage(new_pdf2.getPage(0))

  output.addPage(page2)

  #-----------------------------------------------------------PAGE 3
  packet3 = io.BytesIO()
  can3 = canvas.Canvas(packet3, pagesize=letter)
  can3.setFillColorRGB(0,0,0) 
  can3.setFont("Times-Roman",12)

  logo = ImageReader('TrainingAccurracy.png')
  can3.drawImage(logo, 35, 425, mask='auto', width=542, height=305 )

  logo = ImageReader('TestingAccurracy.png')
  can3.drawImage(logo, 35, 97, mask='auto', width=542, height=305 )

  #can2.showPage()
  can3.save()


  packet3.seek(0)
  new_pdf3 = PdfFileReader(packet3)
  page3 = existing_pdf.getPage(2)
  page3.mergePage(new_pdf3.getPage(0))

  output.addPage(page3)


  # finally, write "output" to a real file
  outputStream = open(path_logs+"/Reports/"+report["expNumber"]+".pdf", "wb")
  output.write(outputStream)
  outputStream.close()

#----------------------------------------------------------- Generate CSV Logbook
  logbookStr = report["expNumber"] +","+ report["epochs"] +","
  logbookStr += report["execTime"]+","
  logbookStr += str(max(training_accurracy))+"-"+str(training_accurracy.index(max(training_accurracy)))+","
  logbookStr += str(max(testing_accurracy))+"-"+str(testing_accurracy.index(max(testing_accurracy)))+","
  logbookStr += str(training_accurracy[-1])+","
  logbookStr += str(testing_accurracy[-1])+","
  logbookStr += report["dataset"]+","
  logbookStr += report["volDim"].replace(",","*")+","
  logbookStr += report["normData"]+","
  logbookStr += report["trainingSamples"]+","
  logbookStr += report["testingSamples"]+","
  logbookStr += report["totalSamples"]+","
  logbookStr += report["optimizer"]+","
  logbookStr += report["loss_object"]+","
  logbookStr += report["trainBatchN"]+","
  logbookStr += report["trainBatchSize"]+","
  logbookStr += report["testBatchN"]+","
  logbookStr += report["testBatchSize"]+","
  logbookStr += report["regularization"]+","
  logbookStr += report["normalFactor"]+","
  logbookStr += report["softmax"]+"\n"

  f= open(path_logs+"/logs/Logbook.csv","a+")
  f.write(logbookStr)
  f.close()

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

'''normFactor=255

for i,elem in enumerate(x_test):
  x_test[i] = elem/normFactor

for i,elem in enumerate(x_train):
  x_train[i] = elem/normFactor

report["normData"]= "/"+str(normFactor)'''
#-------------------------------------------------

report["normData"]= "False"

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print()

report["volDim"]=str(x_test[0].shape) 
report["trainingSamples"]=str(len(x_train))
report["testingSamples"]=str(len(x_test))
report["totalSamples"]=str(len(x_test)+len(x_train))

plt.imshow(x_test[0])
plt.show()

"""# Declarando la arquitectura

De aqui en adelante
"""

normalFactor = 0.001
softmaxLastLayer= False
dropoutProb = 0.8

report["normalFactor"]= str(normalFactor)
report["softmax"] = str(softmaxLastLayer)

if dropoutProb == 0:
  report["regularization"]= "None"
else: 
  report["regularization"] = "tf.nn.dropout("+str(dropoutProb)+")"

class DNN_model(object):
  def __init__(self,
               n_classes=7):
    self.h1LW = tf.Variable(np.random.rand(5,5,3,64)*normalFactor,name="hl1weigths",dtype="float32")
    self.h1LB = tf.Variable(np.random.rand(64)*normalFactor,name="hl1bias",dtype="float32")

    self.h4LW = tf.Variable(np.random.rand(64*64*64,64)*normalFactor,name="hl4weigths",dtype="float32")
    self.h4LB = tf.Variable(np.random.rand(64)*normalFactor,name="hl4bias",dtype="float32")

    self.outW = tf.Variable(np.random.rand(64, n_classes)*normalFactor,name="outweigths",dtype="float32")
    self.outB = tf.Variable(np.random.rand(n_classes)*normalFactor,name="outbias",dtype="float32")

    self.trainable_variables =[self.h1LW,self.h1LB,
                               self.h4LW,self.h4LB,self.outW,self.outB]          
  def __call__(self,x): 

      
      # Declarando la arquitectura
      x  = tf.cast(x, tf.float32)
      img = tf.reshape(x, shape=[-1,128,128,3])

      l1= tf.nn.conv2d(img,self.h1LW, strides=[1,1,1,1], padding='SAME',data_format="NHWC")  
      l1 = tf.add(l1, self.h1LB)
      l1 = tf.nn.relu(l1)
      l1 = tf.nn.max_pool(l1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
      #64x64x64

      l4=  tf.reshape(l1,[-1,64*64*64]) #Flattening

      l4 = tf.nn.dropout(l4, dropoutProb)
      l4=  tf.matmul(l4,self.h4LW)
      l4 = tf.add(l4, self.h4LB)
      l4 = tf.nn.relu(l4)
      #32*32*64, 64

      output = tf.matmul(l4,self.outW) + self.outB
      #64,7

      if softmaxLastLayer:
        output = tf.nn.softmax(output)

      return output

DNN = DNN_model()

optimizador = tf.keras.optimizers.Adam(learning_rate=0.00001)

loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

#Metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(model,tdata, labels):
  with tf.GradientTape() as tape:
    predictions = model(tdata)
    
    #calculo de funcion de error 
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, predictions))
    loss =  tf.reduce_mean(loss_object(labels, predictions))
   
  gradients = tape.gradient(loss, model.trainable_variables)
  capped_grads_and_vars = [(grad,model.trainable_variables[index]) for index, grad in enumerate(gradients)]
  optimizador.apply_gradients(capped_grads_and_vars)
  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(model,tdata, labels):
  predictions = model(tdata)
  #t_loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, predictions))
  t_loss =  tf.reduce_mean(loss_object(labels, predictions))
  
  test_loss(t_loss)
  test_accuracy(labels, predictions)

"""Bitácoras"""

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
report["expNumber"]=current_time
train_log_dir = path_logs+'/logs/' + current_time + '/train'
test_log_dir =  path_logs+'/logs/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

l=[]
training_loss =[]
training_accurracy=[]
testing_loss=[]
testing_accurracy=[]

def fitting(model,train_x,train_y,test_x,test_y,EPOCHS,N_batch,batch_size,N_batch_test,batch_size_test):

  #Carga de una imagen en tensor board 
  img = np.reshape(x_train[0], (-1, 128, 128, 3))
  with train_summary_writer.as_default(): #Abre un registro del log
    tf.summary.image("Training data", img, step=0)



  table = Texttable()
  table.set_deco(Texttable.HEADER)
  table.set_cols_width([8,15,15,15,15])
  table.set_cols_align(["c", "r", "r", "r", "r"])
  table.set_cols_dtype(['i','e','e','e','e'])
  table.add_rows([["Epoch","Perdida", "Exactitud", "Perdida_test", "Exactitud_test"]])
  print(table.draw())
  l.append(str(table.draw()))

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
    
     #se agregan datos a la bitácora de entrenamiento  
    with train_summary_writer.as_default():
      tf.summary.scalar('Train Loss', train_loss.result(), step=epoch)
      tf.summary.scalar('Train Accuracy', train_accuracy.result(), step=epoch)

    #tf.summary.trace_on(graph=True, profiler=True)

   #Test Step------------------------------------------------------------------------------------
    j=0
    while j+batch_size_test < len(test_x) or j+batch_size_test<batch_size_test*N_batch_test:
      start = j
      end = j+batch_size_test
      batch_x_test = test_x[start:end]
      batch_y_test = test_y[start:end]
      test_step(model,batch_x_test,batch_y_test)
      j+=batch_size_test

    #se agregan datos a la bitácora de prueba
    with test_summary_writer.as_default():
      tf.summary.scalar('Test loss', test_loss.result(), step=epoch)
      tf.summary.scalar('Test accuracy', test_accuracy.result(), step=epoch)

    
    table.set_deco(Texttable.HLINES)
    table.add_rows([[int(epoch+1),train_loss.result().numpy(), (train_accuracy.result()*100).numpy(), test_loss.result().numpy(),(test_accuracy.result()*100).numpy()]])
    print(table.draw())
    l.append(str(table.draw()))#For exporting
    training_loss.append(train_loss.result().numpy())
    training_accurracy.append( (train_accuracy.result()*100).numpy() )
    testing_loss.append(test_loss.result().numpy())
    testing_accurracy.append((test_accuracy.result()*100).numpy())



    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

current_time

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir "drive/My Drive/A20/Tesis/Experimentos/Style14/logs/"$current_time

#model,train_x,train_y,test_x,test_y,EPOCHS,N_batch,batch_size,N_batch_test,batch_size_test
init = datetime.datetime.now()# Ignorar esto, es para medir cuanto tiempo dura la ejecución

fitting(DNN,x_train,y_train,x_test,y_test,300,100,90,25,40)

end=datetime.datetime.now()

generateReport()

f= open(path_logs+"/Reports/"+report["expNumber"]+".txt","w+")
for elem in l:
     f.write(elem+"\n")
f.close()

