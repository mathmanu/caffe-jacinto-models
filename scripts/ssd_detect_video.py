##################################################################################################
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from nms_ti import nms_core

#%matplotlib inline
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from array import array 
from enum import IntEnum
from copy import deepcopy

import pylab
import imageio
import sys, getopt
import os

from get_labelname import get_labelname
import csv

#dflt:False
write_boxes_afr_nms = True
#0,1
nms_verbose=0
print_frame_info = False

###################################################################################################
def visualize_weights(net, layer_name, padding=4, filename=''):
    # The parameters are a list of [weights, biases]
    if layer_name in net.params:
      print "net.params[layer_name]: ", net.params[layer_name] 
      data = np.copy(net.params[layer_name][0].data)
    else:
      return

    # N is the total number of convolutions
    print len(data.shape)
    if (len(data.shape) > 1):
      N = data.shape[0]*data.shape[1]
      # Ensure the resulting image is square
      filters_per_row = int(np.ceil(np.sqrt(N)))
      # Assume the filters are square
      filter_size = data.shape[2]
      # Size of the result image including padding
      result_size = filters_per_row*(filter_size + padding) - padding
      # Initialize result image to all zeros
      result = np.zeros((result_size, result_size))
 
      # Tile the filters into the result image
      filter_x = 0
      filter_y = 0
      for n in range(data.shape[0]):
          for c in range(data.shape[1]):
              if filter_x == filters_per_row:
                  filter_y += 1
                  filter_x = 0
              for i in range(filter_size):
                  for j in range(filter_size):
                      result[filter_y*(filter_size + padding) + i, filter_x*(filter_size + padding) + j] = data[n, c, i, j]
              filter_x += 1
 
      # Normalize image to 0-1
      min = result.min()
      max = result.max()
      result = (result - min) / (max - min)
 
      # Plot figure
      plt.figure(figsize=(10, 10))
      plt.axis('off')
      plt.imshow(result, cmap='gray', interpolation='nearest')
 
      # Save plot if filename is set
      if filename != '':
          print "saving :",  filename 
          plt.savefig(filename, bbox_inches='tight', pad_inches=1)
 
      #plt.show()
      #plt.close()
      #return
#######################################################################################################
def debug(var):
  print var, " = ", eval(var)

class ARType(IntEnum):
  AR_H_SAME_AS_DESC_H = 0
  AR_W_SAME_AS_DESC_W = 1
  AR_PRESERVE = 2

REC_WIDTH = 2 #1
#######################################################################################################


def readDetsFromFile(extDetFileName='', offsetX=0, offsetY=0, scaleX=1.0,
    scaleY=1.0, curScaleImage=''):

  dets_list = []
  if os.path.isfile(extDetFileName):
    f = open(extDetFileName, 'r')
    dets_list = list(csv.reader(f, delimiter= ' '))
    f.close()
  
  labels = ['background','person','trafficsign','vehicle']  
  detections = np.zeros(shape=(1,1,len(dets_list),7))
   
  for idx,row in enumerate(dets_list):
     #print row
     cat_idx = labels.index(row[0])
     #print("cat_idx", cat_idx)
     #det_label
     detections[0,0,idx,1] = cat_idx   
  
     #conf
     detections[0,0,idx,2] = float(row[15])
     #xmin    
     detections[0,0,idx,3] = (float(row[4]) - offsetX) / (scaleX * curScaleImage.shape[1])
     #ymin    
     detections[0,0,idx,4] = (float(row[5]) - offsetY) / (scaleY * curScaleImage.shape[0])
     #xmax    
     detections[0,0,idx,5] = (float(row[6]) - offsetX) / (scaleX * curScaleImage.shape[1])
     #ymax    
     detections[0,0,idx,6] = (float(row[7]) - offsetY) / (scaleY * curScaleImage.shape[0])
     #print("detection_temp[0,0,idx,:]", detection_temp[0,0,idx,:]) 
  #print("detection.shape", detections.shape)  
  #print("detection.dtype", detections.dtype)  
  #print("detection[0,0,0,:]", detections[0,0,0,:]) 
  #print np.allclose(detections, detection_temp, atol=0.001, rtol=0.01)

  return detections

##################################################################################################
def processOneCrop(curScaleImage, transformer, net, drawHandle, detBBoxesCurFrame, offsetX, 
    offsetY, scaleX, scaleY, aspectRatio, confTh, externalDet=False, extDetFileName='', labelmap='') :
  if externalDet == False:
    transformed_image = transformer.preprocess('data', curScaleImage)
    net.blobs['data'].data[...] = transformed_image

    # Forward pass.
    detections = net.forward()['detection_out']
  else:
    print("read detections from file")
    detections = readDetsFromFile(extDetFileName=extDetFileName, offsetX=offsetX, offsetY=offsetY,
        scaleX=scaleX, scaleY=scaleY,curScaleImage=curScaleImage)

  #use haskey based labelmap when external det is enabled  
  lblMapHashBased = externalDet
  #print("detections.shape", detections.shape)
  #print("detections.dtype", detections.dtype)
  #print("detections[0]", detections[0,0,0,:])

  # Parse the outputs.
  det_label = detections[0,0,:,1]
  det_conf = detections[0,0,:,2]
  det_xmin = detections[0,0,:,3]
  det_ymin = detections[0,0,:,4]
  det_xmax = detections[0,0,:,5]
  det_ymax = detections[0,0,:,6]

  # Get detections with confidence higher than confTh(def =0.6).
  det_label_list = det_label.astype(np.int).tolist()

  #indiates age of the tracked obj. In the frame it gets detected (born) set it to 0
  age=0.0
  #indicates whether current object is part of strog track or not. Gets used
  #by ObjProp
  strng_trk=0
  
  if type(confTh) is dict:
    confThList = [None] * len(det_label_list)
    for i, det_label_cur_obj in enumerate(det_label_list): 
      if(det_label_cur_obj <> -1):
        confThList[i] = confTh[str(get_labelname(labelmap,det_label_cur_obj, lblMapHashBased=lblMapHashBased)[0])] 
      else:  
        #some thing went wrong. Set conservative th
        confThList[i] = 1.0
    top_indices = [i for i, conf in enumerate(det_conf) if(conf > confThList[i])]
  else:  
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= confTh]
  
  top_conf = det_conf[top_indices]
  top_label_indices = det_label[top_indices].tolist()
  top_labels = get_labelname(labelmap, top_label_indices, lblMapHashBased=lblMapHashBased)
  top_xmin = det_xmin[top_indices]
  top_ymin = det_ymin[top_indices]
  top_xmax = det_xmax[top_indices]
  top_ymax = det_ymax[top_indices]

  colors = plt.cm.hsv(np.linspace(0, 1, 255)).tolist()

  for i in xrange(top_conf.shape[0]):
    xmin = top_xmin[i] * curScaleImage.shape[1]
    ymin = top_ymin[i] * curScaleImage.shape[0]
    xmax = top_xmax[i] * curScaleImage.shape[1]
    ymax = top_ymax[i] * curScaleImage.shape[0]
    score = top_conf[i]
    label = int(top_label_indices[i])
    label_name = top_labels[i]

    if label > 254 :
      print (label)
    color = colors[label]
    #display score and label name
    #print "xmin, ymin, xmax, ymax", xmin, " , ", ymin," , ", xmax," , ", ymax
    #print "scaleX:scaleY ", scaleX, " , ",  scaleY
    #print "offsetX:Y ", offsetX, " , ",  offsetY
    bbox = (int(round(xmin*scaleX))+offsetX, int(round(ymin*scaleY))+offsetY,
        int(round(xmax*scaleX))+offsetX, int(round(ymax*scaleY))+offsetY,
        label, score, age, strng_trk)
    #print "bbox : ", bbox  
    # store box co-ordinates along with label and score
    detBBoxesCurFrame.append(bbox)
  
  return [drawHandle,detections]

##################################################################################################
def writeOneBox(enable=False, bbox=[], label_name='', score=-1.0, fileHndl='',
    writeBboxMap=[], age=0, strng_trk=0, writeAgeAndStrng=False):
  if enable:
    # KITTI benchmarking format
    #map to category specified in writeBboxMap
    #e.g. WRITE_BBOX_MAP=[['car','vehicle'], ['train','vehicle'],['bus','vehicle']]
    #print "writeBboxMap in writeDets", writeBboxMap
    for xlation in writeBboxMap:
      #print "xlation ", xlation 
      if label_name == xlation[0]:
        label_name = xlation[1]

    if writeAgeAndStrng:
      #write out age and strongness flag too
      newLine = '{} 0 0 0 {} {} {} {} 0 0 0 0 0 0 0 {} {} {} \n'.format(label_name,
        bbox[0],bbox[1],bbox[2], bbox[3], score, age, strng_trk)
    else:  
      newLine = '{} 0 0 0 {} {} {} {} 0 0 0 0 0 0 0 {} \n'.format(label_name,
        bbox[0],bbox[1],bbox[2], bbox[3], score)

    #print "newLine: ", newLine
    fileHndl.write(newLine)

##################################################################################################
def drawBoxes(params=[], detObjs=[], drawHandle=[], writeBboxMap=[],
    labelmap=[], lblMapHashBased=False):
  #FIX_ME:SN. Propagate_obj.py also defines. Unify.
  STRNG_TRK_IDX = 7
  colors = plt.cm.hsv(np.linspace(0, 1, 255)).tolist()
  for idx in range(detObjs.shape[0]): 
    label = int(detObjs[idx][4]) 
    score = detObjs[idx][5] 
    age = detObjs[idx][6]
    strng_trk= detObjs[idx][STRNG_TRK_IDX]
    label_name = str(get_labelname(labelmap,label,lblMapHashBased=lblMapHashBased)[0])
                                                                   
    if type(params.confTh) is dict:                                
      draw_cur_reg = score > params.confTh[label_name]
    else:    
      draw_cur_reg = score > params.confTh

    #if age is young then choose only if score is high
    if (age <= 3) and params.enObjPropExp: 
      draw_cur_reg = (strng_trk==1)

    if draw_cur_reg: 
      display_txt = '%.3s: %.2f'%(label_name[:4], score)
      if label > 254 :
        print (label)
      color = colors[label]

      #/usr/share/fonts/truetype/droid/DroidSans.ttf
      fnt = ImageFont.truetype("DroidSans.ttf", 10)
      #display score and label name
      cor = detObjs[idx]
      drawHandle.text((cor[0]+REC_WIDTH, cor[1]+REC_WIDTH),display_txt,(255,255,255),font=fnt)
         
      # determine what color rect to draw
      rectColorAry = ["violate", "blue","yellow","orange","red","pink", "white", "brown", "green"]
      numColor=len(rectColorAry)
      label = min(label, numColor-1)
      rectColor = rectColorAry[label]

      #draw.rectangle(((xmin,ymin),(xmax,ymax)), outline = "blue")
      # hack to draw rectangle with larger than 1 px thickness. As ImageDraw natively does not support it.
      for i in range(-REC_WIDTH,REC_WIDTH):
        locRec = (cor[0]+i,cor[1]+i, cor[2]+i,cor[3]+i) 
        drawHandle.rectangle(locRec, outline=rectColor)   
    
  return drawHandle
##############################################################
def writeBoxes(params=[], detObjs=[], detObjFileHndl='', writeBboxMap=[],
    labelmap=[], lblMapHashBased=False):
 
  for i in range(detObjs.shape[0]): 
    label = int(detObjs[i][4])
    score = detObjs[i][5] 
    age = detObjs[i][6] 
    strng_trk = detObjs[i][7] 
    label_name = str(get_labelname(labelmap,label, lblMapHashBased=lblMapHashBased)[0])

    writeThisBox = True
    #if age is young then choose only if score is high
    if (age <= 3) and (strng_trk == False) and params.enObjPropExp:
      writeThisBox = False
    
    if writeThisBox:
      writeOneBox(enable=params.writeBbox, bbox=detObjs[i], label_name=label_name, 
        score=score, fileHndl=detObjFileHndl, writeBboxMap=writeBboxMap,
        age=age, strng_trk=strng_trk)
   
  return

##################################################################################################
def wrapMulTiles(imageCurFrame, transformer, net, params, curFrameNum=0,
    detObjsFile='', writeBboxMap=[], labelmap=''):
  imageCurFrameAry = deepcopy(imageCurFrame)

  curFrameDrawHandle = ImageDraw.Draw(imageCurFrameAry)

  if ((params.resizeW <> 0) or (params.resizeH <> 0)):
    imageCurFrame = imageCurFrame.resize((params.resizeW,params.resizeH), Image.ANTIALIAS);
  
  tileSizeX = params.tileSizeW
  tileSizeY = params.tileSizeH
  w, h = imageCurFrame.size
  
  #numTileX = ((w-1) / tileSizeX ) + 1
  #numTileY = ((h-1) / tileSizeY ) + 1
  
  #print w, h, numTileX, numTileY
  detObjRectList =[]
  combinedDetImgOp = Image.new("RGB", (w, h))
  detBBoxesCurFrame = []
 
  offsetX = range(0,w-tileSizeX/2, params.tileStepX)
  offsetY = range(0,h-tileSizeY/2 ,params.tileStepY)

  #for tileYIdx in range(0, numTileY, params.tileStepY):
  #  for tileXIdx in range(0, numTileX, params.tileStepX):
  
  for left in offsetX:
    for top in offsetY:
      #left = tileXIdx*tileSizeX
      #top = tileYIdx*tileSizeY
  
      right = left + tileSizeX
      bottom = top + tileSizeY
      if(curFrameNum==0):
        print "left, top, right, bottom : ", left, top, right, bottom
  
      imageCrop = imageCurFrame.crop((left, top, right, bottom))
      
      cropW, cropH = imageCrop.size
      #print cropW, cropH

      #processOneCrop expects np array in the range[0,1]
      imageCoreIp = np.asarray(imageCrop)
      imageCoreIp = imageCoreIp/255.0
  
      #imageCopy = Image.fromarray((imageCoreIp*255).astype(np.uint8))
      curTileDrawHandle = ImageDraw.Draw(imageCrop)
      detections = []
      # as rect is drawn inside in case of multiple tiles all the
      # offsets are dummy
      if params.writeBbox:
        #for objective eval write out all boxes
        confTh=0.01
      else:  
        confTh=params.confTh

      extDetFileName = params.externalDetpath + os.path.split(detObjsFile)[1]
      processOneCrop(imageCoreIp, transformer, net, curTileDrawHandle, detBBoxesCurFrame,
        offsetX=left, offsetY=top,scaleX=1.0, scaleY=1.0, aspectRatio=1.0, confTh=confTh, 
        externalDet=params.externalDet, extDetFileName=extDetFileName,
        labelmap=labelmap)
      combinedDetImgOp.paste(imageCrop, (left,top))

  detObjRectListNPArray = np.array(detBBoxesCurFrame)
  #print "detObjRectListNPArray: ", detObjRectListNPArray   

  #write detected boxes before NMS
  if write_boxes_afr_nms == False:
    if len(detObjRectListNPArray) and params.writeBbox and ((curFrameNum%params.decFreq) == 0):
      detObjFileHndl = open(detObjsFile, "w")
      writeBoxes(params=params,detObjs=detObjRectListNPArray, detObjFileHndl=detObjFileHndl, 
        writeBboxMap=writeBboxMap, labelmap=labelmap, lblMapHashBased=params.externalDet)
      detObjFileHndl.close()

  pick = []
  print " Objs bef,afr NMS, %d " % (len(detObjRectListNPArray )),
  print detObjRectListNPArray,
  if(params.enNMS):
    nmsTh = 0.45 #0.5
    #print detObjRectListNPArray 
    detObjRectListNPArray = nms_core(detObjRectListNPArray, nmsTh, pick,
        age_based_check=False, testMode=False,
        enObjPropExp=params.enObjPropExp, verbose=nms_verbose,
        confThH=params.confThH)
    print (len(detObjRectListNPArray )),
    print detObjRectListNPArray,

  #write detected boxes Afr NMS
  if write_boxes_afr_nms:
    if len(detObjRectListNPArray) and params.writeBbox and ((curFrameNum%params.decFreq) == 0):
      detObjFileHndl = open(detObjsFile, "w")
      writeBoxes(params=params,detObjs=detObjRectListNPArray, detObjFileHndl=detObjFileHndl, 
        writeBboxMap=writeBboxMap, labelmap=labelmap, lblMapHashBased=params.externalDet)
      detObjFileHndl.close()

  print(' ')
  if(len(detObjRectListNPArray)):
    drawBoxes(params=params,detObjs=detObjRectListNPArray, drawHandle=curFrameDrawHandle,
        writeBboxMap=writeBboxMap,labelmap=labelmap,lblMapHashBased=params.externalDet)
  return imageCurFrameAry 

##################################################################################################
def wrapMulScls(imageCurFrame, transformer, net, params, numScales=4, curFrameNum=0, 
    detObjsFile='', writeBboxMap=[], labelmap=''):

  if params.enObjProp:
    from propagate_obj import propagate_obj
  
  if(curFrameNum == 0):
    wrapMulScls.gPoolDetObjs = np.asarray([])

  imageWidth, imageHeight = imageCurFrame.size
  centerX = imageWidth/2
  centerY = imageHeight/2
  aspectRatio =  float(imageWidth)/imageHeight
  
  if(curFrameNum == 0):
    print "centerX centerY aspectRatio: ", centerX, centerY, aspectRatio

  imageCurFrameAry = deepcopy(imageCurFrame)

  curFrameDrawHandle = ImageDraw.Draw(imageCurFrameAry)

  if (params.arType == ARType.AR_H_SAME_AS_DESC_H):  
    # Initialize scale values
    maxScale = float(imageHeight)/params.tileSizeH
  elif (params.arType == ARType.AR_W_SAME_AS_DESC_W):  
    maxScale = float(imageWidth)/params.tileSizeW
  elif (params.arType == ARType.AR_PRESERVE):  
    #maxScale = float(max(imageWidth, imageHeight))/params.tileSizeW
    maxScale = max(float(imageWidth)/params.tileSizeW, float(imageHeight)/params.tileSizeH)
  else:
    print "wrong AR type"
  
  if(curFrameNum==0):
    print "maxScale ", maxScale
  if (numScales!=1) :
    scaleStep = (maxScale-1.0)/(numScales-1)
    #print "scaleStep ", scaleStep
    # 4 should be numSclales. pls check. FIX_ME:SN
    scaleAry = np.arange(4, dtype=np.float);
    scaleAry[0] = 1
  else:
    #if only 1 scale is used resize image to tile W/H
    #maxScale will take care of this
    scaleAry = np.arange(1, dtype=np.float);
    scaleAry[0] = maxScale

  for scaleIdx in range(1,numScales):
    scaleAry[scaleIdx] =  scaleAry[scaleIdx-1] + scaleStep 
    if(curFrameNum == 0):
      print "scaleIdx ", scaleIdx, scaleAry[scaleIdx]

  detBBoxesCurFrame = []
  for scaleIdx in range(0,numScales):
    curScale = scaleAry[scaleIdx]
    # get the rect for cur scale in full resolution quantities
    if (params.arType == ARType.AR_H_SAME_AS_DESC_H):  
      # use height same as desc height and get width keeping aspect ratio same as orignal image dimension
      curSclHInFullRes = min(int(params.tileSizeH*curScale), imageHeight)
      curSclWInFullRes = min(int(curSclHInFullRes*aspectRatio), imageWidth)
    elif (params.arType == ARType.AR_W_SAME_AS_DESC_W):
      # use W same as descW and get H keeping aspect ratio same as orignal image dimension
      curSclWInFullRes = min(int(params.tileSizeW*curScale), imageWidth)
      curSclHInFullRes = min(int(curSclWInFullRes/aspectRatio), imageHeight)
    elif (params.arType == ARType.AR_PRESERVE):
      # Preserve original AR 
      curSclWInFullRes = min(int(params.tileSizeW*curScale), imageWidth)
      curSclHInFullRes = min(int(params.tileSizeH*curScale), imageHeight)
    else:
      print "Invalid AR Type: Exiting Now!!!", params.arType
      sys.exit(2)
    
    offsetXmin = max(centerX - curSclWInFullRes/2,0)
    offsetYmin = max(centerY - curSclHInFullRes/2,0)

    offsetXmax = min(offsetXmin + curSclWInFullRes, imageWidth-1)
    offsetYmax = min(offsetYmin + curSclHInFullRes, imageHeight-1)

    imageCurScale = imageCurFrame.crop((offsetXmin, offsetYmin, offsetXmax, offsetYmax))
    # resize cur scale to desc size
    imageCurScale = imageCurScale.resize((params.tileSizeW,params.tileSizeH), Image.ANTIALIAS);
    curScaleX = float(curSclWInFullRes) / params.tileSizeW
    curScaleY = float(curSclHInFullRes) / params.tileSizeH

    if(curFrameNum == 0):
      print "scaleX scaleY curSclWInFullRes curSclHInFullRes offsetXmin offsetXmin,", curScaleX, curScaleY, curSclWInFullRes, curSclHInFullRes,  offsetXmin, offsetYmin

    #processOneCrop expects image array in the range[0,1]
    imageCoreIpCurScale = np.asarray(imageCurScale)
    imageCoreIpCurScale = imageCoreIpCurScale/255.0

    #print('offsetXmin', offsetXmin)
    confTh=params.confTh
    extDetFileName = params.externalDetPath + os.path.split(detObjsFile)[1]
    #print("extDetFileName: ", extDetFileName)
    [imageDetOp,raw_dets_cur_frm] =  processOneCrop(imageCoreIpCurScale, transformer, net, curFrameDrawHandle,
        detBBoxesCurFrame, offsetX=offsetXmin, offsetY=offsetYmin,scaleX=curScaleX,
        scaleY=curScaleY, aspectRatio=aspectRatio,confTh=confTh,  externalDet=params.externalDet,
        extDetFileName=extDetFileName, labelmap=labelmap)
  
  detObjRectListNPArray = np.array(detBBoxesCurFrame)
  np.set_printoptions(precision=3)
  np.set_printoptions(suppress=True)
  
  if print_frame_info:
    print "======================================="
    print "curFrameDetObjs:" 
    print detObjRectListNPArray
  
  if params.enObjProp:
    if params.enObjPropExp:
      for idx,detObj in enumerate(detObjRectListNPArray):
        detObjRectListNPArray[idx][7] = (detObjRectListNPArray[idx][5] >
            params.confThH)

    opFilenameWOExt, fileExt = os.path.splitext(params.opFileName)
    wrapMulScls.gPoolDetObjs = propagate_obj(gPoolDetObjs=wrapMulScls.gPoolDetObjs, 
        curImageFloat=imageCoreIpCurScale*255.0, curFrameNum=curFrameNum, scaleX=curScaleX, 
        scaleY=curScaleY, offsetX=offsetXmin, offsetY=offsetYmin, params=params, labelmap=labelmap,
        raw_dets_cur_frm=raw_dets_cur_frm, lblMapHashBased=params.externalDet,
        opFilenameWOExt=opFilenameWOExt)
    
    if print_frame_info:
      print "trackedObjs:"
      print wrapMulScls.gPoolDetObjs
    
    #propogate detected objs from prev frame to current frame
    if len(wrapMulScls.gPoolDetObjs) > 0 and len(detObjRectListNPArray) > 0:
      wrapMulScls.gPoolDetObjs = np.concatenate((detObjRectListNPArray, wrapMulScls.gPoolDetObjs), axis=0)
    else:
      wrapMulScls.gPoolDetObjs = detObjRectListNPArray
  else:
    wrapMulScls.gPoolDetObjs = detObjRectListNPArray

  if print_frame_info:
    print "global pool after trackedObjs merged with curImgDetObjs:"
    print "gPoolDetObjs: "
    print wrapMulScls.gPoolDetObjs

  #write detected boxes before NMS
  if write_boxes_afr_nms == False:
    if len(wrapMulScls.gPoolDetObjs) and params.writeBbox and ((curFrameNum%params.decFreq) == 0):
      detObjFileHndl = open(detObjsFile, "w")
      writeBoxes(params=params,detObjs=wrapMulScls.gPoolDetObjs, detObjFileHndl=detObjFileHndl, 
        writeBboxMap=writeBboxMap, labelmap=labelmap, lblMapHashBased=params.externalDet)
      detObjFileHndl.close()

  pick = []

  if print_frame_info:
    print " Objs bef,afr NMS, %d " % (len(wrapMulScls.gPoolDetObjs )),
  if(params.enNMS and len(wrapMulScls.gPoolDetObjs)):
    nmsTh = 0.45 #0.5
    wrapMulScls.gPoolDetObjs = nms_core(wrapMulScls.gPoolDetObjs, nmsTh, pick,
        age_based_check=True, testMode=False,
        enObjPropExp=params.enObjPropExp, verbose=nms_verbose)    
    wrapMulScls.gPoolDetObjs = np.array(wrapMulScls.gPoolDetObjs)
    if print_frame_info:
      print "type(wrapMulScls.gPoolDetObjs): ", type(wrapMulScls.gPoolDetObjs)
      print "Global pool After NMS"
      print "gPoolDetObjs.dtype: ", wrapMulScls.gPoolDetObjs.dtype
      print "gPoolDetObjs.shape: ", wrapMulScls.gPoolDetObjs.shape
      print "gPoolDetObjs: "
      print wrapMulScls.gPoolDetObjs

  #write detected boxes Afr NMS
  if write_boxes_afr_nms:
    if len(wrapMulScls.gPoolDetObjs) and params.writeBbox and ((curFrameNum%params.decFreq) == 0):
      detObjFileHndl = open(detObjsFile, "w")
      writeBoxes(params=params,detObjs=wrapMulScls.gPoolDetObjs, detObjFileHndl=detObjFileHndl, 
        writeBboxMap=writeBboxMap, labelmap=labelmap, lblMapHashBased=params.externalDet)
      detObjFileHndl.close()

  if(len(detObjRectListNPArray)):
    drawBoxes(params=params,detObjs=wrapMulScls.gPoolDetObjs, drawHandle=curFrameDrawHandle, 
        writeBboxMap=writeBboxMap, labelmap=labelmap, lblMapHashBased=params.externalDet)

  if print_frame_info:
    print "======================================="
  return imageCurFrameAry 
##########################################################################################
def vizCNN(vizCNNEn=False, net =''):
  if vizCNNEn:
    all_names = [n for n in net._layer_names]
    for layer_name in all_names: 
      print layer_name
      filename = "./VizCNN/S_Q/" + layer_name + '.jpg'
      visualize_weights(net, layer_name , filename=filename, padding=0)
##########################################################################################
def ssd_detect_video(ipFileName='', opFileName='', deployFileName='',
  modelWeights='', numFrames=1000, tileSizeW=512, tileSizeH=512, labelmapFile='', tileScaleMode=1, 
  resizeW=0, resizeH=0, enNMS=True, numScales=4, arType=0, confTh=0.6,
  enCrop=False, cropMinX=0, cropMinY=0,cropMaxX=0,cropMaxY=0, writeBbox=False, 
  tileStepX=0, tileStepY=0, meanPixVec=[104.0,117.0,123.0], ipScale=1.0,
  writeBboxMap=[], decFreq=1, enObjProp=False, start_frame_num=0,
  maxAgeTh=8, caffe_root='', externalDet=False, externalDetPath='',
  enObjPropExp=False, confThM=0.12, confThMStrngTrk=0.12, confThH=0.4):    

  ###################################################################################  
  enum_multipleTiles = 0
  enum_multipleScales = 1
  class Params:

    def displayArgs(self):
      print "=================================="
      print "ipFileName %s" , self.ipFileName
      print "opFileName %s" , self.opFileName
      print "deployFileName %s" , self.deployFileName
      print "modelWeights %s" , self.modelWeights
      print "tileSizeW %d" , self.tileSizeW
      print "tileSizeH %d" , self.tileSizeH
      print "numFrames %d" , self.numFrames
      print "start_frame_num %d" , self.start_frame_num
      print "labelmapFile %s" , self.labelmapFile
      print "tileScaleMode %d" , self.tileScaleMode
      print "resizeW %d" , self.resizeW 
      print "resizeH %d" , self.resizeH 
      print "enCrop " , self.enCrop
      print "cropMinX" , self.cropMinX
      print "cropMinY" , self.cropMinY
      print "cropMaxX" , self.cropMaxX
      print "cropMaxY" , self.cropMaxY
      print "enNMS " , self.enNMS
      print "numScales " , self.numScales
      print "arType " , self.arType
      print "confTh " , self.confTh
      print "confThM " , self.confThM
      print "confThMStrngTrk " , self.confThMStrngTrk
      print "confThH " , self.confThH
      print "writeBbox" , self.writeBbox
      print "decFreq" , self.decFreq
      print "writeBboxMap" , self.writeBboxMap
      print "tileStepX" , self.tileStepX
      print "tileStepY" , self.tileStepY
      print "meanPixVec" , self.meanPixVec
      print "ipScale" , self.ipScale
      print "enObjProp" , self.enObjProp
      print "enObjPropExp" , self.enObjPropExp
      print "maxAgeTh" , self.maxAgeTh
      print "caffe_root" , self.caffe_root
      print "externalDet", externalDet
      print "externalDetPath", externalDetPath
      print "=================================="

    def override(self):  
      if(self.tileStepX == 0):
        self.tileStepX = self.tileSizeW 
      if(self.tileStepY == 0):
        self.tileStepY = self.tileSizeH

  ################################################ 
  params=Params()

  params.ipFileName = ipFileName
  params.opFileName = opFileName
  params.deployFileName = deployFileName 
  params.modelWeights = modelWeights
  params.tileSizeW = tileSizeW
  params.tileSizeH = tileSizeH 
  params.numFrames = numFrames 
  params.start_frame_num = start_frame_num 
  params.labelmapFile = labelmapFile 
  params.resizeW = resizeW 
  params.resizeH = resizeH 
  params.cropMinX = cropMinX 
  params.cropMinY = cropMinY 
  params.cropMaxX = cropMaxX 
  params.cropMaxY = cropMaxY 
  params.tileScaleMode = tileScaleMode  
  params.enCrop = enCrop 
  params.enNMS = enNMS 
  params.numScales = numScales 
  params.arType = arType 
  params.confTh = confTh 
  params.confThM = confThM 
  params.confThMStrngTrk = confThMStrngTrk 
  params.confThH = confThH 
  params.writeBbox = writeBbox 
  params.decFreq = decFreq 
  params.writeBboxMap = writeBboxMap 
  params.tileStepX = tileStepX
  params.tileStepY = tileStepY
  params.meanPixVec = meanPixVec
  params.ipScale = ipScale
  params.enObjProp = enObjProp
  params.enObjPropExp = enObjPropExp
  params.maxAgeTh = maxAgeTh
  params.caffe_root = caffe_root
  params.externalDet = externalDet
  params.externalDetPath = externalDetPath
  for idx,data in enumerate(meanPixVec):
    params.meanPixVec[idx] = data * ipScale

  params.override()

  opPath, fileName = os.path.split(params.opFileName)
  print ("opPath : ", opPath)
  if not os.path.exists(opPath):
    os.makedirs(opPath)
    print ("opPath creatd", opPath)
  
  sys.stdout = open(opPath+'/console.log', 'a+')
  sys.stderr = open(opPath+'/console_err.log', 'a+')

  params.displayArgs()

  plt.rcParams['figure.figsize'] = (10, 10)
  plt.rcParams['image.interpolation'] = 'nearest'
  plt.rcParams['image.cmap'] = 'gray'

  curWorDir = os.getcwd()
  if externalDet == False:
    # Make sure that caffe is on the python path:
    os.chdir(params.caffe_root)
    print("params.caffe_root: ", os.getcwd())
    sys.path.insert(0, 'python')

    #read label map from the labelmap file
    from google.protobuf import text_format
    from caffe.proto import caffe_pb2
    
    # load labels
    file = open(params.labelmapFile, 'r')
    labelmap = caffe_pb2.LabelMap()
    print "labelmap: ", labelmap
    text_format.Merge(str(file.read()), labelmap)
    print "labelmap: ", labelmap


    import caffe
    caffe.set_device(1)
    caffe.set_mode_gpu()
  
    print("params.deployFileName", params.deployFileName)
    print("params.modelWeights", params.modelWeights)
    net = caffe.Net(params.deployFileName,    # defines the structure of the model
            params.modelWeights,  # contains the trained weights
            caffe.TEST)   # use test mode (e.g., don't perform dropout)

    print("After caffe.Net")
    vizCNN(vizCNNEn=False, net=net)
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array(meanPixVec)) # mean pixel
    scale_mul = ipScale * 255.0
    #print('scale_mul: ', scale_mul)
    transformer.set_raw_scale('data', scale_mul)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    # set net to batch size of 1
    net.blobs['data'].reshape(1,3,params.tileSizeH,params.tileSizeW)
  else:   
    #should match with label map file
    #FIX_ME:SN, make it read form label map file
    labelmap = ['none_of_the_above', 'person', 'trafficsign', 'vehicle']

    transformer = ''
    net = ''

  isVideoIp = True
  if(os.path.splitext(params.ipFileName)[1] ==''):
    isVideoIp = False    

  isVideoOp = True
  if(os.path.splitext(params.opFileName)[1] ==''):
    isVideoOp = False    

  fps = 30
  if isVideoIp:
    vidIp = imageio.get_reader(params.ipFileName,  'ffmpeg')
    fps = vidIp.get_meta_data()['fps']

  if isVideoOp:
    #quality=10.0
    vidOp = imageio.get_writer(params.opFileName,  'ffmpeg', fps=fps)

  if isVideoIp:
    for num,im in enumerate(vidIp):
      if num > (params.numFrames+params.start_frame_num-1):
        break;
      
      #skip init few frames
      if(num<params.start_frame_num): 
        continue;

      curFrameNum=num-params.start_frame_num

      #skip every frame in the batch of decFreq 
      if((curFrameNum%params.decFreq) != 0): 
        continue;

      print num,
      sys.stdout.flush()
      image = vidIp.get_data(num)
      #get_data gives image in "np array" with range [0-255]
      #while wrapperMultipleXXX expects an "image" with range[0-255] 
      imageCurFrame = Image.fromarray((image).astype(np.uint8))
      #print "image block: ", image[0:10,0:10] 

      if params.enCrop:
        imageCurFrame = imageCurFrame.crop((cropMinX,cropMinY,cropMaxX,cropMaxY))

      bareFile, file_extension = os.path.splitext(params.opFileName)
      detObjsFile = '{}_{:06d}.txt'.format(bareFile, num)
      
      if(params.tileScaleMode == enum_multipleTiles):
        combinedCurFrameOp = wrapMulTiles(imageCurFrame, transformer, net, params, 
           curFrameNum=curFrameNum,detObjsFile=detObjsFile,
           writeBboxMap=writeBboxMap, labelmap=labelmap) 
      else:
        combinedCurFrameOp = wrapMulScls(imageCurFrame, transformer, net, params, 
           numScales=params.numScales, curFrameNum=curFrameNum,detObjsFile=detObjsFile,
           writeBboxMap=writeBboxMap, labelmap=labelmap)

      if isVideoOp:
        vidOp.append_data(np.asarray(combinedCurFrameOp))
      else:
        image_file_name = "{}.jpg".format(num)
        combinedCurFrameOp.save(os.path.join(params.opFileName,image_file_name))

    os.chdir(curWorDir)
  else:
    # input is image folder
    for root, dirs, filenames in os.walk(params.ipFileName):
      for num,f in enumerate(sorted(filenames)):
        if num > params.numFrames-1:
          break;
        print f,
        imageCurFrame = caffe.io.load_image(os.path.join(root, f))

        #load_image gives image in "np array" with range [0-1]
        #while wrapperMultipleXXX expects an "image" with range[0-255] 
        imageCurFrame = imageCurFrame*255

        imageCurFrame = Image.fromarray(imageCurFrame.astype(np.uint8))
        if params.enCrop:
          imageCurFrame = imageCurFrame.crop((cropMinX,cropMinY,cropMaxX,cropMaxY))

        #print 'imageCurFrame.type: ', imageCurFrame.type 
        #print "image block: ", imageCurFrame[0:10,0:10] 
        
        detObjsFile = os.path.splitext(os.path.join(params.opFileName, f))[0]+'.txt'  
        print 'detObjsFile: ', detObjsFile  

        if(params.tileScaleMode == enum_multipleTiles):
           combinedCurFrameOp = wrapMulTiles(imageCurFrame, transformer, net, params, 
             curFrameNum=num, detObjsFile=detObjsFile,
             writeBboxMap=writeBboxMap, labelmap=labelmap) 
        else:
           combinedCurFrameOp = wrapMulScls(imageCurFrame, transformer, net, params, 
             numScales=params.numScales, curFrameNum=num, detObjsFile=detObjsFile,
             writeBboxMap=writeBboxMap, labelmap=labelmap)

        if isVideoOp:
          vidOp.append_data(np.asarray(combinedCurFrameOp))
        else:
          combinedCurFrameOp.save(os.path.join(params.opFileName,f))
    os.chdir(curWorDir)

  #sys.stdout.close()
  #sys.stderr.close()
  return  
