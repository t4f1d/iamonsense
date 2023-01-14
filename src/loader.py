from __future__ import print_function

from keras.preprocessing import sequence
from keras_preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D, Embedding, AveragePooling1D, MaxPooling1D, Bidirectional
from keras.datasets import imdb
from keras.utils.vis_utils import plot_model #from keras.utils import plot_model

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras import optimizers
from keras.layers import LSTM
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure


import os
import glob
import math
import numpy as np
import pandas as pd
import random


import time
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Graphic output
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from sklearn.metrics import confusion_matrix



datasets = [  base_path+"SenseThePen+/line_data/", 
              base_path+"IAM-OnDB+/line_data/", 
              base_path+"IAMonDo+/" ]

data_path = base_path+"Master TU KL/Master Thesis/online-recognition/dataset/"
#data_path = base_path+"dataset/"



# Data Statistic
def get_file_number(file):
    # get number from the filename, and return integer
    num_file = file.split('/')[-1].replace("s_", "").replace("w_", "").replace("l_", "").replace("seq_", "").replace(".csv", "").replace(".pkl", "")
    return int(num_file)

def sorted_file_list(file_list):
    # sorting the list based on the numeric filename
    file_list = sorted(file_list, key=lambda x: get_file_number(x) )
    return file_list

def get_folder_number(folder):
    # get number from the filename, and return integer
    num_file = folder.split('/')[-1].replace("p", "")
    return int(num_file)
    
def sorted_folder_list(folder_list):
    # sorting the list based on the numeric filename
    folder_list = sorted(folder_list, key=lambda x: get_folder_number(x) )
    return folder_list

def flatten(t):
    return [item for sublist in t for item in sublist]



def getFileList(name):
  file_list = []
  
  if name == "SenseThePen": 
    path = datasets[0]
    folder_list = sorted_folder_list( glob.glob( path + '*' ) )
    for folder in folder_list:
      files = sorted_file_list( glob.glob( folder + '/*.csv' ) )
      file_list.append(files)
    file_list = flatten(file_list) # flatten

  if name == "IAM-OnDB": 
    path = datasets[1]
    file_list = sorted(glob.glob( path + '**/*.csv' ))

  if name == "IAMonDo": 
    path = datasets[2]
    file_list = sorted(glob.glob( path + '*.csv' ))


  if name == "all": 
    path = datasets[0]
    folder_list = sorted_folder_list( glob.glob( path + '*' ) )
    for folder in folder_list:
      files = sorted_file_list( glob.glob( folder + '/*.csv' ) )
      file_list.append(files)
    file_list = flatten(file_list)

    path = datasets[1]
    file_list2 = sorted(glob.glob( path + '**/*.csv' ))
    for f2 in file_list2:
      file_list.append( f2 )

    path = datasets[2]
    file_list3 = sorted(glob.glob( path + '*.csv' ))
    for f3 in file_list3:
      file_list.append( f3 )
  
  # for i, f in enumerate(file_list):
  #   print(f)

  # for i, p in enumerate(file_list):
	 #  k = 10
	 #  if (i % k == 0):
	 #    print(p)
    
  return file_list




def getCls(len,df):
  list_class = []
  list_class_line = []
  len = flatten( pd.Series.tolist(len) )

  idx = 0
  for i in len:
    idx = idx+i
    list_class.append( df.iloc[[ idx - 1 ]]["class"].values[0] )
    list_class_line.append( df.iloc[[ idx - 1 ]]["class_line"].values[0] )

  return list_class, list_class_line


#def getLen(df_new,f):
def getLen(df_new):

  len_stroke = pd.DataFrame()
  len_word = pd.DataFrame()
  len_line = pd.DataFrame()

  stroke = df_new.groupby('stroke_id').count()
  len_stroke = len_stroke.append(stroke['x'])
  len_stroke = len_stroke.rename({'x': 'length'})
  len_stroke = len_stroke.T
  len_stroke["class"], len_stroke["class_line"] = getCls(len_stroke, df_new)
  #len_stroke["file"] = f
  #print(len_stroke)

  word = df_new.groupby('word_id').count()
  len_word = len_word.append(word['x'])
  len_word = len_word.rename({'x': 'length'})
  len_word = len_word.T
  len_word["class"], len_word["class_line"] = getCls(len_word, df_new)
  #len_word["file"] = f
  #print(len_word)

  line = df_new.groupby('line_id').count()
  len_line = len_line.append(line['x'])
  len_line = len_line.rename({'x': 'length'})
  len_line = len_line.T
  len_line["class"], len_line["class_line"] = getCls(len_line, df_new)
  #len_line["file"] = f
  #print(len_line)

  return len_stroke, len_word, len_line

def combData(file_list):
  comb_data = pd.DataFrame()

  len_stroke = pd.DataFrame()
  len_word = pd.DataFrame()
  len_line = pd.DataFrame()

  for f in tqdm( file_list ):

    #df_new = pd.read_csv(f, index_col=0)
    df_new = getDataPkl(f)

    #df_new = removeOutlier(df_new, dataset)

    comb_data = comb_data.append(df_new, ignore_index = True)

    s, w, l = getLen(df_new,f)
    len_stroke = len_stroke.append(s, ignore_index = True)
    len_word = len_word.append(w, ignore_index = True)
    len_line = len_line.append(l, ignore_index = True)


  class_text = comb_data[ comb_data['class'] == 0 ]
  class_math = comb_data[ comb_data['class'] == 1 ]
  class_graph = comb_data[ comb_data['class'] == 2 ]

  return comb_data, class_text, class_math, class_graph, len_stroke, len_word, len_line


def checkStats(name):
  file_list = getFileList(name)
  comb_data, class_text, class_math, class_graph, len_stroke, len_word, len_line = combData(file_list)

  print("Total csv files : \n", len(file_list))
  print("\n\n ====== Database Statistics : ====== \n", comb_data.describe())
  print("\n\n ====== Class Text Statistics : ====== \n", class_text.describe())
  print("\n\n ====== Class Math Statistics : ====== \n", class_math.describe())
  print("\n\n ====== Class Graph Statistics : ====== \n", class_graph.describe())

  print("\n\n ====== Length Stroke Statistics : ====== \n", len_stroke.describe())
  print("\n\n ====== Length Word Statistics : ====== \n", len_word.describe()) 
  print("\n\n ====== Length Line Statistics : ====== \n", len_line.describe())

  print("\n\n ====== Length Stroke Class Text Statistics : ====== \n", len_stroke[ len_stroke['class'] == 0 ].describe())
  print("\n\n ====== Length Stroke Class Math Statistics : ====== \n", len_stroke[ len_stroke['class'] == 1 ].describe())
  print("\n\n ====== Length Stroke Class Graph Statistics : ====== \n", len_stroke[ len_stroke['class'] == 2 ].describe())

  print("\n\n ====== Length Word Class Text Statistics : ====== \n", len_word[ len_word['class'] == 0 ].describe())
  print("\n\n ====== Length Word Class Math Statistics : ====== \n", len_word[ len_word['class'] == 1 ].describe())
  print("\n\n ====== Length Word Class Graph Statistics : ====== \n", len_word[ len_word['class'] == 2 ].describe())

  print("\n\n ====== Length Line Class Text Statistics : ====== \n", len_line[ len_line['class_line'] == 0 ].describe())
  print("\n\n ====== Length Line Class Math Statistics : ====== \n", len_line[ len_line['class_line'] == 1 ].describe())
  print("\n\n ====== Length Line Class Graph Statistics : ====== \n", len_line[ len_line['class_line'] == 2 ].describe())



def showDist(data, title):
  sns.set_theme(style="whitegrid")
  figure(figsize=(15, 4), dpi=80)
  plt.hist(data['length'], bins=100)
  plt.title(title)
  plt.show()

def showBoxPlot(data1, data2, data3):
  sns.set_theme(style="whitegrid")
  figure(figsize=(30, 8), dpi=80)
  
  data = [ data1['length'], data2['length'], data3['length'] ]
  labels = ['Stroke', 'Word', 'Line']

  plt.boxplot(data, vert=False, labels=labels)
  plt.title("Box Plot of Length")
  plt.xlim(-100,2000)
  plt.gca().invert_yaxis()

  plt.show()

  showBound(len_stroke, labels[0])
  showBound(len_word, labels[1])
  showBound(len_line, labels[2])


def showBound(df, labels=None):
  df = df['length']
  q1 = df.quantile(0.25)
  q2 = df.quantile(0.50)
  q3 = df.quantile(0.75)
  
  iqr = q3 - q1
  
  lower_bound = q1 - (1.5 * iqr)
  if lower_bound < 0: lower_bound = 0

  upper_bound = q3 + (1.5 * iqr)
  
  print(">>"+labels)
  print("   Lower Bound / Minimum = "+ str(lower_bound)
        +" \n   Q1 = "+ str(q1) 
        +" \n   Q2 = "+ str(q2)
        +" \n   Q3 = "+ str(q3)
        +" \n   Upper Bound / Maximum = "+ str(upper_bound)+"\n\n")








# Loader

def getList(file_list):
	# get list data
	train = pd.read_csv(file_list[0], index_col=0).values.tolist()
	val = pd.read_csv(file_list[1], index_col=0).values.tolist()
	test = pd.read_csv(file_list[2], index_col=0).values.tolist()
	return train,val,test


def getData(file_name):
	# select coloumn
	#req_cols = ['x', 'y', 'timestamp', 'pre_x', 'pre_y', 'class', 'class_line', 'stroke_id', 'word_id', 'line_id']
	#req_cols = ['x', 'y', 'class', 'class_line', 'stroke_id', 'word_id', 'line_id']

	# read data
	data = pd.read_csv(base_path+file_name, index_col=0)
	#data = pd.read_csv(base_path+file_name, usecols=req_cols) # using the selected column, to load faster

	# load time : 12:33 on sensethepen dataset
	return data


def getDataPkl(file_name):
  file_name = file_name.replace("/content/drive/Othercomputers/My MacBook Pro/","")
  file_name = file_name.replace(".csv",".pkl")

  data = pd.DataFrame()
  
  for i in range(20):
    try:
      data = pd.read_pickle(base_path+"pickle/"+file_name)
      #data = pd.read_pickle(base_path+"dataset/pickle/"+file_name)
      break # if success break loop
    except EOFError:
      time.sleep(0.5) #wait
      print("trying..."+str(i))
      print(file_name)
    
    
  #data = pd.read_pickle(base_path+"pickle/"+file_name)

  # load time : 8:18 on sensethepen dataset
  return data


def lenLevel(level):
  # https://github.com/t4f1d/online-recognition/blob/master/result/report8.md
  if dataset == "SenseThePen":
  	if level == "stroke_id":
  		length = 68
  	if level == "word_id":
  		length = 244
  	if level == "line_id":
  		length = 2420

  if dataset == "IAM-OnDB":
    if level == "stroke_id":
      length = 21
    if level == "word_id":
      length = 104
    if level == "line_id":
      length = 620

  if dataset == "IAMonDo":
    if level == "stroke_id":
      length = 11
    if level == "word_id":
      length = 44
    if level == "line_id":
      length = 159

  # if level == "stroke_id":
  #   length = 68
  # if level == "word_id":
  #   length = 244
  # if level == "line_id":
  #   length = 2420

  return length


def padding(data, level):
  # add padding
  #data = sequence.pad_sequences(sequences=data, maxlen=lenLevel(level), dtype='float64')
  data = pad_sequences(sequences=data, maxlen=lenLevel(level), dtype='float64')
  return data


def unique_class(classes):
	# get list class
	c = []
	for i in classes:
		# print(i)
		# print(np.unique(i)[0])
		c.append( int(np.unique(i)[0]) )
	return c


def stack_data(data_x, data_y, level):
	# stack data
	length = len(data_x)
	data_xy = np.zeros((length,2,lenLevel(level)))

	for i, data in enumerate(data_x):
		data_xy[i] = np.stack((data_x[i], data_y[i]), axis = 0)
	return data_xy

def stack_data_3(data_x, data_y, data_pen, level):
  # stack data
  length = len(data_x)
  data_xyp = np.zeros((length,3,lenLevel(level)))

  for i, data in enumerate(data_x):
    data_xyp[i] = np.stack((data_x[i], data_y[i], data_pen[i]), axis = 0)
  return data_xyp





# reshaping data
def getAvg(odd, even):
  data = pd.DataFrame()
  data = odd.add(even, fill_value=0).div(2) # average of two dataframe
  return data

def reducePoint(df_region, target_length):
  len = df_region.count()[0]
  df_region = df_region.reset_index(drop=True)

  while len > target_length:
    #print(len)
    for i in range(len):
      if i % 2 != 0:
        df_region = df_region.drop(i) # remove row
        len_update = df_region.count()[0] # reassign

        if len_update <= target_length:
          break
    
    df_region = df_region.reset_index(drop=True)
    len = df_region.count()[0] # reassign
    
  return df_region


def enlargePoint(df_region, target_length):
  len = df_region.count()[0]
  df_region = df_region.reset_index(drop=True)

  #return df_region

  while len < target_length:
    #print(len)
    for i in range( (len*2)-1 ):
      if i % 2 != 0:

        try:
          odd = df_region.iloc[i]
          even = df_region.iloc[i-1]
        except IndexError:
          break

        if odd['stroke_id'] == even['stroke_id']:
          between = getAvg(odd,even)
          # print(df_region)
          # print(between)
          line = pd.DataFrame(
              {"x": between['x'], 
              "y": between['y'], 
              "timestamp": between['timestamp'], 
              "pre_x": between['pre_x'], 
              "pre_y": between['pre_y'], 
              "class": between['class'], 
              "class_line": between['class_line'], 
              "stroke_id": between['stroke_id'], 
              "word_id": between['word_id'], 
              "line_id": between['line_id'], 
              }, index=[i-0.5])

          df_region = df_region.append(line, ignore_index=False)
          df_region = df_region.sort_index().reset_index(drop=True)
      
        #print(df_region)
        #visualize(df_region)
        len_update = df_region.count()[0] # reassign

        if len_update >= target_length: # if same break
          break
    
    df_region = df_region.reset_index(drop=True)
    len = df_region.count()[0] # reassign

  return df_region

def reshapingData(data,name,level):
  target_length = lenLevel(level)
  d = pd.DataFrame()
  
  for region, df_region in data.groupby(level): # check every stroke
    len = df_region.count()[0]

    if len == 1:
      d = d.append(df_region).reset_index(drop=True)
      continue # skip dot/small part

    if len > target_length:
      #print(str(len)+"reduce")
      data_new = reducePoint(df_region, target_length)
      d = d.append(data_new).reset_index(drop=True)
    else:
      #print(str(len)+"enlarge")
      data_new = enlargePoint(df_region, target_length)
      d = d.append(data_new).reset_index(drop=True)

    #print(d)
    #visualize(d,"horizontal")
    # break
    #print(region)

  return d




# Prepare Data
def prepData(file_name, level, rotate=False):
  #data = getData(file_name[0])
  data = getDataPkl(file_name[0])
  name = dataset

  if dataset == "all": # update dataset var
    if "SenseThePen" in file_name[0]:  
      name = "SenseThePen"
    if "IAM-OnDB" in file_name[0]:  
      name = "IAM-OnDB"
    if "IAMonDo" in file_name[0]:  
      name = "IAMonDo"

  if rotate:
    data = rotateData(data)
    #visualize(data,'horizontal')

  #data = removeOutlier(data, name) # remove the outlier
  data = reshapingData(data, name, level) # reshape data


  try:
    g_level = data.groupby(level) # stroke_id, word_id, line_id
  except:
    print(file_name[0])
    print("Error group by")
    print(data)
  


  data_pre_x = g_level['pre_x'].apply(list) # get pre x
  data_x = g_level['x'].apply(list) # get x
  
  data_pre_y = g_level['pre_y'].apply(list) # get pre y
  data_y = g_level['y'].apply(list) # get y

  data_pen = g_level['stroke_id'].apply(list) # get class

  if level == "line_id":
    classes = g_level['class_line'].apply(list) # get class
  else:
    classes = g_level['class'].apply(list) # get class


  data_t = g_level['timestamp'].apply(list) # get timestamp
  data_pre_t = g_level['timestamp'].apply(list) # get pre_timestamp
  for i in data_pre_t.index:
    data_pre_t[i].insert(0,data_pre_t[i][0]) # insert on first
    data_pre_t[i] = data_pre_t[i][:-1] # delete last


  # get delta x and delta y
  for i in data_x.index:
    data_x[i] = np.subtract(data_pre_x[i], data_x[i])
    data_y[i] = np.subtract(data_pre_y[i], data_y[i])
    data_pen[i] = checkPen(data_pen[i])
    data_t[i] = np.subtract(data_t[i], data_pre_t[i])

  # add padding
  data_x = padding(data_x, level)
  data_y = padding(data_y, level)
  data_pen = padding(data_pen, level)
  data_t = padding(data_t, level)

  # data and class
  #data_xy = stack_data(data_x, data_y, level) # stack data_x and data_y
  data_xy = stack_data_3(data_x, data_y, data_pen, level) # stack data_x, data_y, data_pen
  #data_xy = stack_data_4(data_x, data_y, data_t, data_pen,level)
  classes = unique_class(classes)

  return data_xy, classes


def checkPen(data):
  result = [0]*len(data)
  for i, d in enumerate(data):
    if i == 0: 
      result[i] = 1 # first touch
    else: #check if still touch
      if data[i] == data[i-1]: 
        result[i] = 1 # still touch
      else:
        result[i] = 0 # not touch
  
  return result


def removeOutlier(data,name):

  for level in ['stroke_id','word_id','line_id']: # filter data in each level

    lowerbound, upperbound = getBound(name,level) # get lowerbound and upperbound

    df = data.groupby(level).size().reset_index(name='counts') #calculate the size length
    outlier_id = df[(df['counts'] > upperbound) | (df['counts'] < lowerbound)]# get stroke_id of outlier

    #print(outlier_id)
    for idx in outlier_id[level]:
      data = data[data[level] != idx]
    
  #print(data[level].unique())
  return data


def getBound(name,level):
  # the outlier is the data above the upperbound and lower than lowerbound or 0
  if name == "SenseThePen":
    if level == "stroke_id":
      lowerbound = 0
      upperbound = 241.5
    if level == "word_id":
      lowerbound = 0
      upperbound = 836
    if level == "line_id":
      lowerbound = 0
      upperbound = 5163.25

  if name == "IAM-OnDB":
    if level == "stroke_id":
      lowerbound = 0
      upperbound = 57.5
    if level == "word_id":
      lowerbound = 0
      upperbound = 285.5
    if level == "line_id":
      lowerbound = 150.0
      upperbound = 1102.0

  if name == "IAMonDo":
    if level == "stroke_id":
      lowerbound = 0
      upperbound =  32.0
    if level == "word_id":
      lowerbound = 0
      upperbound = 151.0
    if level == "line_id":
      lowerbound = 0
      upperbound = 387.0

  return lowerbound, upperbound 




def loadDataset(name,level="stroke"):
  file_list = data_path + name +'_train.csv', data_path + name +'_val.csv', data_path + name +'_test.csv'
  train,val,test = getList(file_list)

  print("Preparing Train Data...")
  train_bar = tqdm( train )
  for file_name in train_bar:
    train_bar.set_postfix({'file': file_name})
    
    # undersampling class text with probability x,xx to add
    add_data = undersampling(file_name, name)
    if(not add_data):
      continue #skip

    # add data
    data_xy, classes = prepData(file_name, level)
    if 'x_train' in locals():
      x_train = np.append(x_train, data_xy, axis=0)
      y_train = np.append(y_train, classes, axis=0)
    else:
      x_train = data_xy
      y_train = classes

    # uppersampling class2
    is_graph = class2(file_name)
    if (is_graph):
      loop = loopCount(file_name, name)
      for i in range(loop):
        data_xy, classes = prepData(file_name, level, rotate=True)
        x_train = np.append(x_train, data_xy, axis=0)
        y_train = np.append(y_train, classes, axis=0)



  print("Preparing Validation Data...")
  val_bar = tqdm( val )
  for file_name in val_bar:
    val_bar.set_postfix({'file': file_name})

    # undersampling class text with probability x,xx to add
    add_data = undersampling(file_name, name)
    if(not add_data):
      continue #skip

    # add data
    data_xy, classes = prepData(file_name, level)
    if 'x_val' in locals():
      x_val = np.append(x_val, data_xy, axis=0)
      y_val = np.append(y_val, classes, axis=0)
    else:
      x_val = data_xy
      y_val = classes

    # uppersampling class2
    is_graph = class2(file_name)
    if (is_graph):
      loop = loopCount(file_name, name)
      for i in range(loop):
        data_xy, classes = prepData(file_name, level, rotate=True)
        x_val = np.append(x_val, data_xy, axis=0)
        y_val = np.append(y_val, classes, axis=0)



  print("Preparing Test Data...")
  test_bar = tqdm( test )
  for file_name in test_bar:
    test_bar.set_postfix({'file': file_name})

    # undersampling class text with probability x,xx to add
    add_data = undersampling(file_name, name)
    if(not add_data):
      continue #skip

    # add data
    data_xy, classes = prepData(file_name, level)
    if 'x_test' in locals():
      x_test = np.append(x_test, data_xy, axis=0)
      y_test = np.append(y_test, classes, axis=0)
    else:
      x_test = data_xy
      y_test = classes

    # uppersampling class2
    is_graph = class2(file_name)
    if (is_graph):
      loop = loopCount(file_name, name)
      for i in range(loop):
        data_xy, classes = prepData(file_name, level, rotate=True)
        x_test = np.append(x_test, data_xy, axis=0)
        y_test = np.append(y_test, classes, axis=0)




  data_train = list(zip(x_train, y_train))
  data_val = list(zip(x_val, y_val))
  data_test = list(zip(x_test, y_test))
  
  
  return x_train, y_train, x_val, y_val, x_test, y_test
  #return data_train, data_val, data_test
  #return data_test, x_test, y_test





def undersampling(file_name, name):
  if name == "SenseThePen":
    weight=[0.52, 0.48]
    

  data = getDataPkl(file_name[0]) # read
  if data['class'][0] == 0:
    prob = np.random.choice(np.arange(0, 2), p=weight)
    # print('random'+str(prob))
    # visualize(data,'horizontal')
  else:
    prob = 1
  return prob


# Rotation matrix function
def rotate_matrix (x, y, angle, x_shift=0, y_shift=0, units="DEGREES"):
    """
    Rotates a point in the xy-plane counterclockwise through an angle about the origin
    https://en.wikipedia.org/wiki/Rotation_matrix
    :param x: x coordinate
    :param y: y coordinate
    :param x_shift: x-axis shift from origin (0, 0)
    :param y_shift: y-axis shift from origin (0, 0)
    :param angle: The rotation angle in degrees
    :param units: DEGREES (default) or RADIANS
    :return: Tuple of rotated x and y
    """

    # Shift to origin (0,0)
    x = x - x_shift
    y = y - y_shift

    # Convert degrees to radians
    if units == "DEGREES":
        angle = math.radians(angle)

    # Rotation matrix multiplication to get rotated x & y
    xr = (x * math.cos(angle)) - (y * math.sin(angle)) + x_shift
    yr = (x * math.sin(angle)) + (y * math.cos(angle)) + y_shift

    return xr, yr

def rotateData(data):
  deg = random.randint(0, 360)
  #print(deg)
  data['x'], data['y'] = rotate_matrix(data['x'],data['y'], deg ) # random
  return data

def class2(file_name):
  isGraph = 0  
  data = getDataPkl(file_name[0]) # read
  if data['class'][0] == 2:
    if (data['class'] == data['class'][0]).all():
      isGraph = 1
  return isGraph

def class1(file_name):
  isMath = 0  
  data = getDataPkl(file_name[0]) # read
  if data['class'][0] == 1:
    # if (data['class'] == data['class'][0]).all():
    isMath = 1
  return isMath

def loopCount(file_name, name):
  if name == "SenseThePen":
    #weight = [0.86, 0.14] # 5.14
    #weight = [0.71, 0.29] # 5.14 x 3.17 = 16.29
    #weight = [0.82, 0.18] # 64.18
    #weight = [0.94, 0.06] # 53.06
    weight = [0.02, 0.98] # 21.98
    #loop = [4,5] # 5.14 - 1
    #loop = [63,64] # 64.18-1 
    loop = [20,21] # 21.98-1

  if name == "IAMonDo":
    weight = [0.53, 0.47]
    loop = [17,18]
    
  prob = np.random.choice(np.arange(0, 2), p=weight)
  # print(prob)
  # print(loop[prob])
  return loop[prob]





def visResult(history):
	sns.set_theme(style="whitegrid")
	val_loss = history.history['val_loss']
	loss = history.history['loss']
	accuracy = history.history['accuracy']
	val_accuracy = history.history['val_accuracy']

	epochs = range(1, len(accuracy) + 1)

	plt.rcParams['figure.figsize'] = [10, 5]
	plt.subplot(1, 2, 1)
	plt.plot(epochs, loss, label='Training loss')
	plt.plot(epochs,val_loss , label='Validation loss')
	plt.title('Training and validation loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()

	plt.subplot(1, 2, 2)
	plt.plot(epochs, accuracy, label='Training acc')
	plt.plot(epochs, val_accuracy, label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.tight_layout()
	plt.show()


def visConfusionMatrix(model, x_test, y_test):
  p = model.predict(x_test)
  y_pred = np.argmax(p, axis=1) # Get the maximum along row    

  #Generate the confusion matrix
  cf_val = confusion_matrix(y_test, y_pred)
  print(cf_val)
  cf_matrix = confusion_matrix(y_test, y_pred, normalize='true')
  print(cf_matrix)
  

  ax = sns.heatmap(cf_matrix, annot=True, color="Blue")

  ax.set_title('Confusion Matrix \n\n');
  ax.set_xlabel('\nPredicted Category')
  ax.set_ylabel('Actual Category ');

  ## Ticket labels - List must be in alphabetical order
  ax.xaxis.set_ticklabels(['Text','Math', 'Graph'])
  ax.yaxis.set_ticklabels(['Text','Math', 'Graph'])

  ## Display the visualization of the Confusion Matrix.
  plt.show()

