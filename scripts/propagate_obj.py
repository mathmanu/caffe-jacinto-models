##################################################################################################
# This file contains all the functions for Obejct Propagation functionality
##################################################################################################
import numpy as np
from array import array 
from enum import IntEnum
from copy import deepcopy
from get_labelname import get_labelname
import pylab
import sys, getopt
import os
import cv2

debugPrintObjProp = False
debugPrintKeyPoints = False
debugDrawTracks =  False
debugPrintFindOL = False


AGE_IDX = 6
SCORE_IDX = 5
LBL_IDX = 4
STRNG_TRK_IDX = 7

def getObjsAboveScore(modConfTh=0.12, labelmap='', lblMapHashBased=False,
    raw_dets_cur_frm='', W=0, H=0):
  ##prune low score detections
  #print "local detection.shape ", raw_dets_cur_frm.shape
  ## Parse the outputs.
  det_label = raw_dets_cur_frm[0,0,:,1]
  det_conf = raw_dets_cur_frm[0,0,:,2]
  det_xmin = raw_dets_cur_frm[0,0,:,3]
  det_ymin = raw_dets_cur_frm[0,0,:,4]
  det_xmax = raw_dets_cur_frm[0,0,:,5]
  det_ymax = raw_dets_cur_frm[0,0,:,6]

  if type(modConfTh) is dict:
    confThList = [None] * len(det_label_list)
    for i, det_label_cur_obj in enumerate(det_label_list): 
      if(det_label_cur_obj <> -1):
        modConfThList[i] = modConfTh[str(get_labelname(labelmap,det_label_cur_obj,lblMapHashBased=lblMapHashBased)[0])] 
      else:  
        #some thing went wrong. Set conservative th
        modConfThList[i] = 1.0
    top_indices = [i for i, conf in enumerate(det_conf) if(conf > modConfThList[i])]
  else:  
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= modConfTh]
  
  top_conf = det_conf[top_indices]
  top_label_indices = det_label[top_indices].tolist()
  top_labels = get_labelname(labelmap, top_label_indices, lblMapHashBased=lblMapHashBased)
  top_xmin = det_xmin[top_indices]
  top_ymin = det_ymin[top_indices]
  top_xmax = det_xmax[top_indices]
  top_ymax = det_ymax[top_indices]

  #print "cur scale image size: W: ", frame_gray.shape[1], "H:", frame_gray.shape[0]
  top_xmin = top_xmin * W
  top_ymin = top_ymin * H
  top_xmax = top_xmax * W
  top_ymax = top_ymax * H

  return [top_conf, top_label_indices, top_xmin, top_ymin, top_xmax, top_ymax] 

# main funtion for object propagation
def propagate_obj(gPoolDetObjs=[], curImageFloat=[], curFrameNum=0, scaleX =
    1.0, scaleY=1.0, offsetX=0, offsetY=0, params=[], labelmap=[],
    raw_dets_cur_frm=[], lblMapHashBased=False, opFilenameWOExt=''):
  #if overlap of tracked obj is at least have this much overlap with moderate
  #confidence detections then keep it alive
  maxOverLapTh = 0.4
  curImage = curImageFloat.astype('uint8')
  frame_gray = cv2.cvtColor(curImage, cv2.COLOR_BGR2GRAY)

  if params.enObjPropExp:
    modConfTh = 0.15
    #for strong tracks, continue trackign if it find match with 0.12
    modConfThForStrngTrk = 0.12
  else:  
    modConfTh = 0.12

  top_conf, top_label_indices, top_xmin, top_ymin, top_xmax, top_ymax = getObjsAboveScore(modConfTh=modConfTh,
      labelmap=labelmap, lblMapHashBased=lblMapHashBased,
      raw_dets_cur_frm=raw_dets_cur_frm, W=frame_gray.shape[1],
      H=frame_gray.shape[0])

  top_conf_strngTrk, top_label_indices_strngTrk, top_xmin_strngTrk, top_ymin_strngTrk, top_xmax_strngTrk, top_ymax_strngTrk = getObjsAboveScore(modConfTh=modConfThForStrngTrk, labelmap=labelmap, lblMapHashBased=lblMapHashBased, raw_dets_cur_frm=raw_dets_cur_frm,
      W=frame_gray.shape[1], H=frame_gray.shape[0])

  #det_cur_mod_conf= np.array(det_cur_mod_conf)
  #print "after pruning with moderate score",  det_cur_mod_conf.shape
  #print det_cur_mod_conf.shape
  #print "det_cur_mod_conf"
  #print det_cur_mod_conf

  if curFrameNum == 0:
    trackedObjsList = []
    trackedObjs = np.asarray(trackedObjsList)
  else: #initalization in first frame
    #cap = cv2.VideoCapture('slow.flv')
    # Parameters for lucas kanade optical flow
    #lk_params = dict( winSize  = (15,15),
    #                  maxLevel = 5,
    #                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Parameters taken for TI SFM OF code 
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 5,
                      criteria = (cv2.TERM_CRITERIA_EPS +
                      cv2.TERM_CRITERIA_COUNT, 20, 0.01))

    # Create some random colors. ma

    maxCornersForViz = 1000 #100
    color = np.random.randint(0,255,(maxCornersForViz*len(gPoolDetObjs),3))

    #bbox = (int(xmin*scaleX)+offsetX, int(ymin*scaleY)+offsetY, int(xmax*scaleX)+offsetX, int(ymax*scaleY)+offsetY, label, score)
    #Get det objs from full image to current scale by reversing the above
    #process
    if debugPrintObjProp:
      print "gPoolDetObjs: "  
      print gPoolDetObjs
    objPoolInCurScale = gPoolDetObjs.copy()
    for idx, detObj in enumerate(gPoolDetObjs):
      objPoolInCurScale[idx][0] = (objPoolInCurScale[idx][0] - offsetX) / scaleX
      objPoolInCurScale[idx][1] = (objPoolInCurScale[idx][1] - offsetY) / scaleY
      objPoolInCurScale[idx][2] = (objPoolInCurScale[idx][2] - offsetX) / scaleX
      objPoolInCurScale[idx][3] = (objPoolInCurScale[idx][3] - offsetY) / scaleY
      if(detObj[SCORE_IDX] > 0.55) and params.enObjPropExp: 
        detObj[STRNG_TRK_IDX] = 1

    if debugPrintObjProp:
      print "objPoolInCurScale: "  
      print objPoolInCurScale

    p0_init_done = False
    p0 = None
    for idx, detObj in enumerate(objPoolInCurScale ):
      if debugPrintObjProp: 
        print "XLeft-XRight: ", int(detObj[0]), ":", int(detObj[2]),
        print "YTop-YBot   : ", int(detObj[1]),":", int(detObj[3]),
      
      #initalize all pixels to zero (picture completely black)
      mask_detObjs = np.zeros(propagate_obj.old_gray.shape,np.uint8)

      #clip boxes which are outside pic boundary. 
      detObj[0] = np.clip(detObj[0],0,frame_gray.shape[1])
      detObj[2] = np.clip(detObj[2],0,frame_gray.shape[1])
      
      detObj[1] = np.clip(detObj[1],0,frame_gray.shape[0])
      detObj[3] = np.clip(detObj[3],0,frame_gray.shape[0])

      mask_detObjs[int(detObj[1]):int(detObj[3]),int(detObj[0]):int(detObj[2])] = 1
      areaCurObj = (detObj[2]-detObj[0]) *  (detObj[3]-detObj[1])
      if debugPrintObjProp: 
        print "areaCurObj: ", areaCurObj

      # params for ShiTomasi corner detection
      # OpenCV has a function, cv2.goodFeaturesToTrack(). It finds N strongest corners in the image by
      # Shi-Tomasi method (or Harris Corner Detection, if you specify it). As usual, image should be a 
      # grayscale image. Then you specify number of corners you want to find. Then you specify the 
      # quality level, which is a value between 0-1, which denotes the minimum quality of corner below
      # which everyone is rejected. Then we provide the minimum euclidean distance between corners detected.
      # With all these informations, the function finds corners in the image. All corners below quality 
      # level are rejected. Then it sorts the remaining corners based on quality in the descending order.
      # Then function takes first strongest corner, throws away all the nearby corners in the range of
      # minimum distance and returns N strongest corners.

      maxCornersCurObj = 100 #100
      #keep min distance between  minDistTnMin and minDistTnMax.
      #also make it depend on area
      minDistThMin = 4
      minDistThMax = 8
      minDisanceCurObj = int(np.clip(areaCurObj/24,minDistThMin,minDistThMax))
      feature_params = dict( maxCorners = maxCornersCurObj,  #100
                             qualityLevel = 0.01,            #0.05, 0.1,0.3
                             minDistance = minDisanceCurObj, #7
                             blockSize = 7 )
      p0_curObj = cv2.goodFeaturesToTrack(propagate_obj.old_gray, mask = mask_detObjs, **feature_params)
      if p0_curObj is None:
        if debugPrintObjProp: 
          print "no key point found on det objs"
      elif p0_init_done:
        p0 = np.concatenate((p0, p0_curObj), axis=0)
      else:  
        p0_init_done = True
        p0 = p0_curObj

    if debugPrintObjProp:
      print "gPoolDetObjs.dtype: ", gPoolDetObjs.dtype
      print "gPoolDetObjs.shape: ", gPoolDetObjs.shape
      print "gPoolDetObjs: ", gPoolDetObjs

    good_new = np.array([])
    good_old = np.array([])
    if p0 is not None:
      p1, st, err = cv2.calcOpticalFlowPyrLK(propagate_obj.old_gray, frame_gray, p0, None, **lk_params)
    
      if debugPrintObjProp:
        #print "p1.dtype: ", p1.dtype
        #print "p1.shape: ", p1.shape
        print "p1: ", p1
        print "st: ", st
        print "err: ", err

      # Select good points
      good_new = p1[st==1]
      good_old = p0[st==1]

      if debugPrintObjProp:
        print "good_old ", good_old

    trackedObjsList = []
    #find the nearset corner point to TL and BR and adjust tracked window
    for idx, detObj in enumerate(gPoolDetObjs):
      tlX = detObj[0] 
      tlY = detObj[1] 
      brX = detObj[2] 
      brY = detObj[3]

      #convert cur position from original image coordinate(after pre resize if it
      #is enabled)to current scale
      tlX = (tlX - offsetX) / scaleX
      brX = (brX - offsetX) / scaleX

      tlY = (tlY - offsetY) / scaleY
      brY = (brY - offsetY) / scaleY

      if debugPrintObjProp:
        print "=========="
        print "obj in cur scale coordinate : ", tlX, ":", tlY, ":", brX, ":", brY
      
      resultTLx, tlX, tlY = updateWithNearestKeypoint(curPosX=tlX,curPosY=tlY,tlX=tlX,
         tlY=tlY, brX=brX, brY=brY, good_old=good_old, good_new=good_new)
      resultBRy, brX, brY = updateWithNearestKeypoint(curPosX=brX,curPosY=brY,tlX=tlX,
         tlY=tlY, brX=brX, brY=brY, good_old=good_old, good_new=good_new)
      
      #if both TL and BR points are having good match
      if resultTLx and resultBRy: 
        if debugPrintObjProp:
          print "obj propogation successfull!"
          print "propogated obj in cur scale coordicates: ", tlX, ":", tlY, ":", brX, ":", brY

        trackedBoxCurScale = [tlX, tlY, brX, brY, float(detObj[LBL_IDX]), detObj[SCORE_IDX], detObj[AGE_IDX]]
        if detObj[STRNG_TRK_IDX] == 1:
          maxOverLap = findBestOverlap(trackedBox=trackedBoxCurScale, top_conf=top_conf_strngTrk, top_label_indices=top_label_indices_strngTrk, 
            top_xmin=top_xmin_strngTrk, top_ymin=top_ymin_strngTrk, top_xmax=top_xmax_strngTrk, top_ymax=top_ymax_strngTrk)
        else:  
          maxOverLap = findBestOverlap(trackedBox=trackedBoxCurScale, top_conf=top_conf, top_label_indices=top_label_indices, 
            top_xmin=top_xmin, top_ymin=top_ymin, top_xmax=top_xmax, top_ymax=top_ymax)

        if (maxOverLap < maxOverLapTh) and debugPrintObjProp:
          print "maxOverLap: ", maxOverLap 
          print "But object removed due to less overlap with moderate detectios in frame num: ", curFrameNum
      
        #convert from cur scale position to original image coordinate
        tlX = (tlX*scaleX + offsetX) 
        brX = (brX*scaleX + offsetX) 

        tlY = (tlY*scaleY + offsetY)
        brY = (brY*scaleY + offsetY)

        keep_obj_alive = shouldKeepObjAlive(params=params, detObj=detObj, labelmap=labelmap)    
        if keep_obj_alive and (maxOverLap>=maxOverLapTh):
          trackedBox = [tlX, tlY, brX, brY, float(detObj[LBL_IDX]), detObj[SCORE_IDX], detObj[AGE_IDX], detObj[STRNG_TRK_IDX]]
          trackedObjsList.append(trackedBox) 
      else:  
        if debugPrintObjProp:
          print "obj propogation failed!!"
          print "resultTLx", resultTLx
          print "resultBRy", resultBRy
      if debugPrintObjProp:  
        print "=========="

    trackedObjs = np.asarray(trackedObjsList)
    
    if debugPrintObjProp:
      print "trackedObjs.dtype: ", trackedObjs.dtype
      print "trackedObjs.shape: ", trackedObjs.shape
      print "trackedObjs: ", trackedObjs 

    if debugDrawTracks and good_old.size:
      # draw the tracks
      # Create a mask image for drawing purposes
      mask = np.zeros_like(curImage)
      for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        imageForViz = cv2.circle(curImage,(a,b),5,color[i].tolist(),-1)
      imageForViz= cv2.add(imageForViz,mask)
      cv2.imwrite('{}_debug_{}.png'.format(opFilenameWOExt,str(curFrameNum)),imageForViz)
  
  # Now update the previous frame
  propagate_obj.old_gray = frame_gray.copy()
  
  return  trackedObjs

# update cur <x,y> position by moving it as per OF of nearest keypoint
# <x,y> = <x,y> + <OF_nkp_x,OF_nkp_y>
def updateWithNearestKeypoint(curPosX=0,curPosY=0,tlX=0,
    tlY=0, brX=0, brY=0, p0=[],p1=[],good_old=[],good_new=[]):
  #if L1 dist to nearest keypoint is more than this value consider it as
  #failure
  minErrTh = 50
  result = False
  best_p0 = [] 
  best_match_p0 = [] 
  minErr = 1E10
  matchFoundTL = False
  for pointP0, pointP1 in zip(good_old, good_new):
    #consider nearest point which is inside the det objects
    if (pointP0[0] >= tlX) and (pointP0[1] >= tlY) and (pointP0[0] <= brX) and (pointP0[1] <= brY):
      err = abs(pointP0[0]-curPosX) + abs(pointP0[1]-curPosY)
      #print "err: ", err
      if (err < minErr):
        nearest_old = pointP0
        nearest_old_match = pointP1
        minErr = err
        #print "minErr: ", minErr
        matchFoundTL = True

  if matchFoundTL and (minErr < minErrTh):      
    #adjust TL with OF of best match   
    if debugPrintKeyPoints:
      print "OF_x:", nearest_old[0] - nearest_old_match[0]  
      print "OF_y:", nearest_old[1] - nearest_old_match[1]  

    curPosX += nearest_old_match[0] - nearest_old[0]
    curPosY += nearest_old_match[1] - nearest_old[1] 
    result = True
  else: 
    if debugPrintKeyPoints:
      print "minErr: ", minErr 
  return [result, curPosX,curPosY]

def shouldKeepObjAlive(params=[], detObj=[], labelmap=[], lblMapHashBased=False):    
  AGE_BASED = True
  reduceScoreTh  = 0.05

  if AGE_BASED:
    #increase age for each year(frame)
    detObj[AGE_IDX] += 1
    keep_obj_alive = (detObj[AGE_IDX] <= params.maxAgeTh)
  else: #score based.score gets reduced with each passing year (frame)
    #reduce score of tracked objects to give more preference to objs in
    #the currnt frame also it will make sure objs will die after few frames
    #didn't work better than AGE_BASED
    detObj[SCORE_IDX] = max(0.0, float(detObj[SCORE_IDX]-reduceScoreTh))
    if type(params.confTh) is dict:                                
      label = int(detObj[LBL_IDX]) 
      label_name = str(get_labelname(labelmap,label,lblMapHashBased=lblMapHashBased)[0])
      keep_obj_alive = detObj[SCORE_IDX] > params.confTh[label_name]
    else:    
      keep_obj_alive = detObj[SCORE_IDX] > params.confTh
  return keep_obj_alive     

def findBestOverlap(trackedBox=[], top_conf=[], top_label_indices=[], top_xmin=[], top_ymin=[], top_xmax=[], top_ymax=[]):
  x1 = trackedBox[0]
  y1 = trackedBox[1]
  x2 = trackedBox[2]
  y2 = trackedBox[3]
  label = trackedBox[4]
  score = trackedBox[5]
  #if age_based_check:
  #  age = boxes[:,6]
  #else:
  #  age = 0
  
  # compute the area of the bounding boxes and sort the bounding
  # boxes by the bottom-right y-coordinate of the bounding box
  area = (x2 - x1 + 1) * (y2 - y1 + 1)

  if debugPrintFindOL:
    print "=================="
    print "x1:", x1,
    print "y1:", y1,
    print "x2:", x2,
    print "y2:", y2,
    print "label:", label,
    print "score:", score,
    print "area:", area

  max_overlap = 0.0
  for cand_x1, cand_x2, cand_y1, cand_y2, cand_label, cand_score in zip(top_xmin, top_xmax, top_ymin,top_ymax, top_label_indices, top_conf):
    if cand_label == label:
      if debugPrintFindOL:
        print "cand_x1:", cand_x1,
        print "cand_y1:", cand_y1,
        print "cand_x2:", cand_x2,
        print "cand_y2:", cand_y2,
        print "cand_label:", cand_label,
        print "cand_score:", cand_score

      # find the largest (x, y) coordinates for the start of
      # the bounding box and the smallest (x, y) coordinates
      # for the end of the bounding box
      xx1 = np.maximum(x1, cand_x1)
      yy1 = np.maximum(y1, cand_y1)
      xx2 = np.minimum(x2, cand_x2)
      yy2 = np.minimum(y2, cand_y2)

      # compute the width and height of the bounding box
      w = np.maximum(0, xx2 - xx1 + 1)
      h = np.maximum(0, yy2 - yy1 + 1)

      if debugPrintFindOL:
        print "overlap w: " , w,
        print "overlap h: " , h,

      # compute the ratio of overlap
      overlap = (w * h) / area

      if debugPrintFindOL:
        print "overlap: ", overlap,

      if overlap > max_overlap:
        max_overlap = overlap
        bestMatchedBox = [cand_x1,cand_y1,cand_x2,cand_y2]

  if debugPrintFindOL:
    print "max_overlap: ", max_overlap
    print "=================="
    print bestMatchedBox 
  return max_overlap   

