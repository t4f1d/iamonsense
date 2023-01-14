import dgl
import math
import torch
import random
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from dgl.data import DGLDataset
from dgl.nn import GraphConv, AvgPooling
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import confusion_matrix
from scipy.signal import butter, lfilter, freqz
from torch.utils.data.sampler import SubsetRandomSampler



import warnings
warnings.filterwarnings('ignore')

def getDataPkl(file_name):
  file_name = file_name.replace("/content/drive/Othercomputers/My MacBook Pro/","")
  file_name = file_name.replace(".csv",".pkl")

  data = pd.DataFrame()
  
  for i in range(20):
    try:
      
      # CHANGE IT
      #data = pd.read_pickle(base_path+"dataset/pickle/"+file_name)
      data = pd.read_pickle(base_path+"pickle/"+file_name)
      
      break # if success break loop
    except EOFError:
      time.sleep(0.5) #wait
      print("trying..."+str(i))
      print(file_name)
    
  #data = pd.read_pickle(base_path+"pickle/"+file_name)

  # load time : 8:18 on sensethepen dataset
  return data

def getData(file_name):
  file_name = file_name.replace("/content/drive/Othercomputers/My MacBook Pro/","")
  # read data
  #data = pd.read_csv(base_path+'dataset/'+file_name, index_col=0)
  data = pd.read_csv(base_path+'/'+file_name, index_col=0)
  return data


def getList(file_list):
  # get list data
  train = pd.read_csv(file_list[0], index_col=0).values.tolist()
  val = pd.read_csv(file_list[1], index_col=0).values.tolist()
  test = pd.read_csv(file_list[2], index_col=0).values.tolist()
  return train,val,test


def prepGraph(data):
  # stroke
  max = data.shape[0] # get last index, max-1 #print(max)
  stroke = data['stroke_id'].unique()
  for i in range( len(stroke) ):
    data = data.replace({'stroke_id': { stroke[i] : i }}) # reset
  for i in range( len(stroke) ):
    data = data.replace({'stroke_id': { i : i+max }})

  # word
  max = max+len(stroke)
  word = data['word_id'].unique()
  for i in range( len(word) ):
    data = data.replace({'word_id': { word[i] : i }}) # reset
  for i in range( len(word) ):
    data = data.replace({'word_id': { i : i+max }})
  
  # word
  max = max+len(word)
  line = data['line_id'].unique()
  for i in range( len(line) ):
    data = data.replace({'line_id': { line[i] : i }}) #reset
  for i in range( len(line) ):
    data = data.replace({'line_id': { i : i+max }})

  return data

def getEdgeSrc(data):
  src = data.index.to_numpy()
  stroke = data['stroke_id'].unique()
  word = data['word_id'].unique()
  line = data['line_id'].unique()

  src = np.append(src, stroke)
  src = np.append(src, word)
  src = np.append(src, line)
  return src

def pad(l, content, width):
  l.extend([content] * (width - len(l)))
  return l

def prepFeat(f, max_node):
  f = f.to_list()
  f = pad(f, 0, max_node) # add padding 0 
  f = torch.from_numpy( np.array(f) )
  return f




# Visualize Graph
def vizGraph(G, subgraph=True):
  #remove self loop
  G = dgl.remove_self_loop(G)

  #subgraph in degree > 0
  if subgraph:
    indeg = G.in_degrees() # get all in degress each node
    to_keep = [i for i, x in enumerate(indeg) if x !=0 ]
    G = G.subgraph(to_keep)

  #color
  #color_map = []
  indeg = G.in_degrees()
  outdeg = G.out_degrees()

  # when in degree is 0, it is stroke, else word. when out degree is 0, it is line
  color_map = ['blue' if deg == 0 else 'green' for deg in indeg] # stroke: blue, word: green
  for i, x in enumerate(outdeg):
    if x == 0: color_map[i] = 'red' # line : red
  #print(color_map)

  #nx_G = G.to_networkx()
  nx_G = G.to_networkx().to_undirected()
  pos = nx.kamada_kawai_layout(nx_G) # Kamada-Kawaii layout usually looks pretty for arbitrary graphs
  nx.draw(nx_G, pos, node_size=500, node_color=color_map)


def visConfusionMatrix(model, test_dataloader):
  model.eval()
  pred_label = []
  true_label = []
  for batched_graph, labels in test_dataloader:
      pred = model(batched_graph, batched_graph.ndata['feat'].float())
      y_pred = pred.argmax(1)
      pred_label = np.append(pred_label, y_pred )
      true_label = np.append(true_label, labels )
    
  cf_val = confusion_matrix(true_label, pred_label)
  print(cf_val)
  cf_matrix = confusion_matrix(true_label, pred_label, normalize='true')
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



# Augmentation
def butter_lowpass(cutoff, fs, order=5):
  nyq = 2 * fs # init 0.5
  normal_cutoff = cutoff / nyq # harmonic frequency
  b, a = butter(order, normal_cutoff, btype='low', analog=False)
  return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
  b, a = butter_lowpass(cutoff, fs, order=order)
  y = lfilter(b, a, data)
  return y

def lowPass(data):
  #cutoff = random.randint(5, 12)
  cutoff = 10
  order = 2
  #fs = random.randint(5, 20)      
  fs = 50 # sample rate, Hz

  # Get the filter coefficients so we can check its frequency response.
  b, a = butter_lowpass(cutoff, fs, order)

  # Plot the frequency response.
  w, h = freqz(b, a, worN=8000)
  #print (w,'\n\n\n\n\n\n',h)
  # plt.subplot(2, 1, 1)
  # plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
  # plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
  # plt.axvline(cutoff, color='k')
  # plt.xlim(0, 0.5*fs)
  # plt.title("Lowpass Filter Frequency Response")
  # plt.xlabel('Frequency [Hz]')
  # plt.grid()
  # plt.show()


  # Filter the data, and plot both the original and filtered signals.
  y = butter_lowpass_filter(data, cutoff, fs, order)

  # plt.plot(data, 'b-', label='data')
  # plt.plot(y, 'g-', linewidth=2, label='filtered data')
  # plt.legend()
  # plt.show()

  return y


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

# Rotation matrix function
def rotate_matrix (x, y, angle, x_shift=0, y_shift=0, units="DEGREES"):
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


def augment(data,i):
  a = data

  # get pen
  pen = a['stroke_id']
  pen = checkPen(pen) # 1 if pen in, 0 if pen out
  pen[0] = 0 

  delta_x = (a['x']-a['pre_x'])*pen
  delta_y = (a['y']-a['pre_y'])*pen

  gain = (i+1)*0.5
  #print(i, gain)
  b = 1 + ( gain * ( 1 - lowPass( delta_x )) )
  c = 1 + ( gain * ( 1 - lowPass( delta_y )) )

  a['x'] = a['x']-b
  a['y'] = a['y']-c

  res = a
  #res['n'] = i
  res = pd.DataFrame(res)
  
  return res




# model
class GCN1(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN1, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)
    
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h

        return dgl.mean_nodes(g, 'h')

class GCN2(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN2, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)
        self.classify = nn.Linear(h_feats, num_classes)
    
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)
    
class GCN3(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN3, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)
        self.conv3 = GraphConv(h_feats, h_feats)
        self.classify = nn.Linear(h_feats, num_classes)
    
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.conv3(g, h)
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)

class GCN4(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN4, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats*2)
        self.conv3 = GraphConv(h_feats*2, h_feats)
        self.classify = nn.Linear(h_feats, num_classes)
    
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.conv3(g, h)
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)



class GCN5(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN5, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats*2)
        self.conv3 = GraphConv(h_feats*2, h_feats)
        self.classify = nn.Linear(h_feats, num_classes)
    
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.conv3(g, h)
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)

class GCN6(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN6, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats*2)
        self.conv3 = GraphConv(h_feats*2, h_feats)
        self.classify = nn.Linear(h_feats, num_classes)
    
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.conv3(g, h)
        g.ndata['h'] = h

        hg = dgl.max_nodes(g, 'h')
        return self.classify(hg)

class GCN7(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN7, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats*2)
        self.conv3 = GraphConv(h_feats*2, h_feats*3)
        self.conv4 = GraphConv(h_feats*3, h_feats*2)
        self.conv5 = GraphConv(h_feats*2, h_feats)
        self.classify = nn.Linear(h_feats, num_classes)
    
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.conv3(g, h)
        h = F.relu(h)
        h = self.conv4(g, h)
        h = F.relu(h)
        h = self.conv5(g, h)
        g.ndata['h'] = h

        hg = dgl.max_nodes(g, 'h')
        return self.classify(hg)




# Train and Test
def train():
    model.train()
    train_bar = tqdm(train_dataloader)
    for batched_graph, labels in train_bar :
        pred = model(batched_graph, batched_graph.ndata['feat'].float())
        loss = criterion(pred, labels)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(loader):
    model.eval()
    num_correct = 0
    num_tests = 0
    loss_list = []
    for batched_graph, labels in loader:
        pred = model(batched_graph, batched_graph.ndata['feat'].float())
        num_correct += (pred.argmax(1) == labels).sum().item()
        num_tests += len(labels)
        batch_loss = criterion(pred, labels)  # Compute the batch loss.
        loss_list.append(batch_loss.item())
    loss = sum(loss_list) / len(loss_list)
    accuracy = num_correct / num_tests * 100
    return accuracy, loss





# SenseThePen Dataset
class senseDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='sense_graph')

    def process(self):
        
        self.graphs = []
        self.labels = []

        # CHANGE IT
        #data_path = base_path+"dataset/"
        data_path = base_path+"Master TU KL/Master Thesis/online-recognition/dataset/"

        file_list = data_path + 'SenseThePen_train.csv', data_path + 'SenseThePen_val.csv', data_path + 'SenseThePen_test.csv'
        train,val,test = getList(file_list)

        all = train+val+test

        all_bar = tqdm(all)
        for file in all_bar:
          all_bar.set_postfix({'file': file[0]})
          
          data = getDataPkl(file[0])
          data = prepGraph(data)

          max_node = max(data['line_id'])+1

          # features
          x = prepFeat(data['x'], max_node)
          y = prepFeat(data['y'], max_node)
          pre_x = prepFeat(data['pre_x'], max_node)
          pre_y = prepFeat(data['pre_y'], max_node)
          delta_x = torch.sub(pre_x, x)
          delta_y = torch.sub(pre_x, x)

          #features = torch.stack(( x, y, pre_x, pre_y, delta_x, delta_y), dim = 0).T    # 6 features
          features = torch.stack((delta_x, delta_y), dim = 0).T  # 2 features
          #print(features)


          # edge
          edge_index = data.index.to_numpy() # index
          edge_stroke = data['stroke_id'].to_numpy() # stroke
          # edge additional
          stroke_word = data.groupby(['stroke_id', 'word_id']).size().reset_index(name='Freq') # mapping unique stroke - word
          word_line = data.groupby(['word_id', 'line_id']).size().reset_index(name='Freq') # mapping unique word - line
          # edge source
          edges_src = np.append(edge_index, stroke_word['stroke_id'] )
          edges_src = np.append(edges_src, word_line['word_id'] )
          edges_src = torch.from_numpy(edges_src)
          # edge destination
          edges_dst = np.append(edge_stroke, stroke_word['word_id'] )
          edges_dst = np.append(edges_dst, word_line['line_id'] )
          edges_dst = torch.from_numpy(edges_dst)

          # edge features
          edge_feature = prepFeat(data['timestamp'], max_node-1)


          # node label
          #label_class = torch.from_numpy(data['class'].to_numpy())
          label_class = torch.from_numpy(data['class'].astype('int').to_numpy())
          
          for i in stroke_word.index:
            cls = data[data['stroke_id'] == stroke_word['stroke_id'][i]]['class'].unique()
            cls = int( cls[0] )
            #print(stroke_word['stroke_id'][i], cls)
            label_class = np.append(label_class, cls)
          
          for i in word_line.index:
            cls = data[data['word_id'] == word_line['word_id'][i]]['class_line'].unique()
            cls = int( cls[0] )
            #print(word_line['word_id'][i], cls)
            label_class = np.append(label_class, cls)

          line_freq = data.groupby(['line_id']).size().reset_index(name='Freq')
          for i in line_freq.index:
            cls = data[data['line_id'] == line_freq['line_id'][i]]['class_line'].unique()
            cls = int( cls[0] )
            #print(word_line['word_id'][i], cls)
            label_class = np.append(label_class, cls)

            graph_label = cls # graph label

          label = graph_label
          self.labels.append(label)
          label_class = torch.from_numpy(label_class)

          g = dgl.graph((edges_src, edges_dst), num_nodes=max_node)
          g.ndata['feat'] = features
          g.ndata['label'] = label_class
          g.edata['weight'] = edge_feature

          g = dgl.add_self_loop(g)
          self.graphs.append(g)

        
    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]
    
    def __len__(self):
        return len(self.graphs)


# IAMonDo Dataset
class iamondoDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='iamondo_graph')

    def process(self):
        
        self.graphs = []
        self.labels = []

        # CHANGE IT
        #data_path = base_path+"dataset/"
        data_path = base_path+"Master TU KL/Master Thesis/online-recognition/dataset/"

        file_list = data_path + 'IAMonDo_train.csv', data_path + 'IAMonDo_val.csv', data_path + 'IAMonDo_test.csv'
        train,val,test = getList(file_list)

        all = train+val+test

        all_bar = tqdm(all)
        for file in all_bar:
          all_bar.set_postfix({'file': file[0]})
          
          #document = getDataPkl(file[0])
          document = getData(file[0])
          #print(document)

          line_list = document.groupby(["line_id"]) # group by line level
          for line in line_list:
            data = line[1].reset_index()
            data = prepGraph(data)


            # looping for math class
            loop = [17,18]
            p = np.random.choice(np.arange(0, 2), p=[0.53, 0.47])
            looping = loop[p]

            while looping > 0 : # looping for all class, math class will be multiple



              max_node = max(data['line_id'])+1

              # features
              x = prepFeat(data['x'], max_node)
              y = prepFeat(data['y'], max_node)
              pre_x = prepFeat(data['pre_x'], max_node)
              pre_y = prepFeat(data['pre_y'], max_node)
              delta_x = torch.sub(pre_x, x)
              delta_y = torch.sub(pre_x, x)

              #features = torch.stack(( x, y, pre_x, pre_y, delta_x, delta_y), dim = 0).T    # 6 features
              features = torch.stack((delta_x, delta_y), dim = 0).T  # 2 features
              #print(features)


              # edge
              edge_index = data.index.to_numpy() # index
              edge_stroke = data['stroke_id'].to_numpy() # stroke
              # edge additional
              stroke_word = data.groupby(['stroke_id', 'word_id']).size().reset_index(name='Freq') # mapping unique stroke - word
              word_line = data.groupby(['word_id', 'line_id']).size().reset_index(name='Freq') # mapping unique word - line
              # edge source
              edges_src = np.append(edge_index, stroke_word['stroke_id'] )
              edges_src = np.append(edges_src, word_line['word_id'] )
              edges_src = torch.from_numpy(edges_src)
              # edge destination
              edges_dst = np.append(edge_stroke, stroke_word['word_id'] )
              edges_dst = np.append(edges_dst, word_line['line_id'] )
              edges_dst = torch.from_numpy(edges_dst)

              # edge features
              edge_feature = prepFeat(data['timestamp'], len(edges_dst))


              # node label
              #label_class = torch.from_numpy(data['class'].to_numpy())
              label_class = torch.from_numpy(data['class'].astype('int').to_numpy())
              #print(label_class)
              
              for i in stroke_word.index:
                cls = data[data['stroke_id'] == stroke_word['stroke_id'][i]]['class'].unique()
                cls = int( cls[0] )
                #print(stroke_word['stroke_id'][i], cls)
                label_class = np.append(label_class, cls)
              
              for i in word_line.index:
                cls = data[data['word_id'] == word_line['word_id'][i]]['class_line'].unique()
                cls = int( cls[0] )
                #print(word_line['word_id'][i], cls)
                label_class = np.append(label_class, cls)

              line_freq = data.groupby(['line_id']).size().reset_index(name='Freq')
              for i in line_freq.index:
                cls = data[data['line_id'] == line_freq['line_id'][i]]['class_line'].unique()
                cls = int( cls[0] )
                #print(word_line['word_id'][i], cls)
                label_class = np.append(label_class, cls)

                graph_label = cls # graph label


              #   # skip small data graph
              # if graph_label == 2:
              #   total_w = len(data['word_id'].unique())
              #   total_s = len(data['stroke_id'].unique())
              #   #print(total_w, total_s)
              #   if (total_w <= 2 and total_s <= 2):
              #     looping = 0
              #     continue
              #   #else:
              #   #  visualize(data, 'horizontal')




              label = graph_label
              self.labels.append(label) # add to labels
              label_class = torch.from_numpy(label_class)

              g = dgl.graph((edges_src, edges_dst), num_nodes=max_node)
              g.ndata['feat'] = features
              g.ndata['label'] = label_class
              g.edata['weight'] = edge_feature

              g = dgl.add_self_loop(g)
              self.graphs.append(g) # add to graphs


              #print(label)
              
              if label == 1:
                # vizGraph(g)
                initData = prepGraph(line[1].reset_index())
                # print(data)
                # print(initData)
                #visualize(data, 'horizontal')
                aug = augment(initData, random.randint(0,9)) # fourier augmentation
                data = rotateData(aug)
                #visualize(data, 'horizontal')
                looping-=1 # decrease
              else:
                looping = 0 # stop loop, just once



        
    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]
    
    def __len__(self):
        return len(self.graphs)