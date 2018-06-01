def get_labelname(labelmap, labels, lblMapHashBased=False):
  if lblMapHashBased == False:
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
      labels = [labels]
    for label in labels:
      found = False
      for i in xrange(0, num_labels):
        if label == labelmap.item[i].label:
          found = True
          labelnames.append(labelmap.item[i].display_name)
          break
      assert found == True
  else:
    #read from hash table in case Caffe is not loaded
    if type(labels) is not list:
      labels = [labels]

    labelnames = []
    for label in labels:
      labelname = labelmap[int(label)]
      labelnames.append(labelname)

  return labelnames

