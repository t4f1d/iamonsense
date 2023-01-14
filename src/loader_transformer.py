# Prepare Data
def prepData(file_name, level, rotate=False):
  data = getData(file_name[0])
  #data = getData('/dataset/'+file_name[0]) #change this when in server
  #data = getDataPkl(file_name[0])
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

  #visualize(data,'horizontal')
  #data = removeOutlier(data, name) # remove the outlier
  data = reshapingData(data, name, level) # reshape data
  #visualize(data,'horizontal')

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
  #data_xy = stack_data_3(data_x, data_y, data_pen, level) # stack data_x, data_y, data_pen
  #data_xy = stack_data_3(data_x, data_y, data_t, level) # stack data_x, data_y, data_pen
  data_xy = stack_data_4(data_x, data_y, data_t, data_pen,level)
  classes = unique_class(classes)

  return data_xy, classes


# Visualization
def visualize(location, fig='square'):

    sns.set_style("whitegrid", {'axes.grid' : True}) # theme
    
    if fig == 'square':
        figure(figsize=(10, 10), dpi=80)
        scale = 1
    if fig == 'horizontal':
        figure(figsize=(15, 5), dpi=80)
        scale = 0.5

    # plot based on stroke_id
    strokes = location.groupby(['stroke_id'])
    for stroke in strokes:
        plt.plot(stroke[1]['x'], stroke[1]['y'], '-', linewidth=0.5, markersize=0.5, c='black' )

    plt.title("Position")
    #plt.plot(location['x'], location['y'])
    plt.scatter(location['x'], location['y'], c=location['timestamp'], s=10)
    plt.colorbar(shrink=scale)
    plt.axis('scaled')
    plt.gca().invert_yaxis()
    plt.show()


def reshapingData(data,name,level):
  target_length = lenLevel(level)
  d = pd.DataFrame()
  
  for region, df_region in data.groupby(level): # check every stroke
    len = df_region.count()[0]

    if len <= 3:
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

    # #print(d)
    # visualize(d,"horizontal")
    # # break
    # print(region)
  


  # for class 1
  for region, df_region in data.groupby('line_id'):
    df = df_region.reset_index()
    if(df['class_line'][0]==1):

      loop = loopCount(name)
      for i in range(loop):

        df = rotateData(df)
        #visualize(df,"horizontal")
        
        for subregion, subdf_region in df.groupby(level):
          len = subdf_region.count()[0]
          #print(len,target_length)

          if len <= 3:
            d = d.append(subdf_region).reset_index(drop=True)
            continue # skip dot/small part

          if len > target_length:
            #print(str(len)+"reduce")
            data_new = reducePoint(subdf_region, target_length)
            d = d.append(data_new).reset_index(drop=True)
          else:
            #print(str(len)+"enlarge")
            data_new = enlargePointT(subdf_region, target_length)
            d = d.append(data_new).reset_index(drop=True)

          #visualize(data_new,"horizontal")
      


  return d


def enlargePointT(df_region, target_length):
  len = df_region.count()[0]
  df_region = df_region.set_index('index')
  df_region = df_region.reset_index(drop=True)

  #visualize(df_region)
  #print(df_region)

  #return df_region

  while len < target_length:
    #print(len,target_length)
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

        #print(len_update,target_length)
        if len_update >= target_length: # if same break
          break
    
    df_region = df_region.reset_index(drop=True)
    len = df_region.count()[0] # reassign

  return df_region



def loopCount(name):
  if name == "IAMonDo":
    weight = [0.53, 0.47]
    loop = [17,18]
    
  prob = np.random.choice(np.arange(0, 2), p=weight)
  # print(prob)
  # print(loop[prob])
  return loop[prob]


