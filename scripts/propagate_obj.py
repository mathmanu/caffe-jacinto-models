##################################################################################################
# This file contains all the functions for Obejct Propagation functionality
##################################################################################################
import numpy as np
from nms_ti import nms_core
from array import array 
from enum import IntEnum
from copy import deepcopy
from get_labelname import get_labelname
import pylab
import sys, getopt
import os
import cv2

# main funtion for object propagation
def propagate_obj(gPoolDetObjs=[], curImageFloat=[], curFrameNum=0, scaleX =
    1.0, scaleY=1.0, offsetX=0, offsetY=0, params=[], labelmap=[],
    raw_dets_cur_frm=[], hash_key_based=False):
  debug_print = True
  #if overlap of tracked obj is at least have this much overlap with moderate
  #confidence detections then keep it alive
  maxOverLapTh = 0.4
  AGE_IDX = 6
  SCORE_IDX = 5
  LBL_IDX = 4
  curImage = curImageFloat.astype('uint8')
  frame_gray = cv2.cvtColor(curImage, cv2.COLOR_BGR2GRAY)

  ##prune low score detections
  #print "local detection.shape ", raw_dets_cur_frm.shape
  ## Parse the outputs.
  det_label = raw_dets_cur_frm[0,0,:,1]
  det_conf = raw_dets_cur_frm[0,0,:,2]
  det_xmin = raw_dets_cur_frm[0,0,:,3]
  det_ymin = raw_dets_cur_frm[0,0,:,4]
  det_xmax = raw_dets_cur_frm[0,0,:,5]
  det_ymax = raw_dets_cur_frm[0,0,:,6]

  modConfTh = 0.12
  if type(modConfTh) is dict:
    confThList = [None] * len(det_label_list)
    for i, det_label_cur_obj in enumerate(det_label_list): 
      if(det_label_cur_obj <> -1):
        modConfThList[i] = modConfTh[str(get_labelname(labelmap,det_label_cur_obj,hash_key_based=hash_key_based)[0])] 
      else:  
        #some thing went wrong. Set conservative th
        modConfThList[i] = 1.0
    top_indices = [i for i, conf in enumerate(det_conf) if(conf > modConfThList[i])]
  else:  
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= modConfTh]
  
  top_conf = det_conf[top_indices]
  top_label_indices = det_label[top_indices].tolist()
  top_labels = get_labelname(labelmap, top_label_indices, hash_key_based=hash_key_based)
  top_xmin = det_xmin[top_indices]
  top_ymin = det_ymin[top_indices]
  top_xmax = det_xmax[top_indices]
  top_ymax = det_ymax[top_indices]

  #print "cur scale image size: W: ", frame_gray.shape[1], "H:", frame_gray.shape[0]

  top_xmin = top_xmin * frame_gray.shape[1]
  top_ymin = top_ymin * frame_gray.shape[0]
  top_xmax = top_xmax * frame_gray.shape[1]
  top_ymax = top_ymax * frame_gray.shape[0]

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
    if debug_print:
      print "gPoolDetObjs: "  
      print gPoolDetObjs
    objPoolInCurScale = gPoolDetObjs.copy()
    for idx, detObj in enumerate(gPoolDetObjs):
      objPoolInCurScale[idx][0] = (objPoolInCurScale[idx][0] - offsetX) / scaleX
      objPoolInCurScale[idx][1] = (objPoolInCurScale[idx][1] - offsetY) / scaleY
      objPoolInCurScale[idx][2] = (objPoolInCurScale[idx][2] - offsetX) / scaleX
      objPoolInCurScale[idx][3] = (objPoolInCurScale[idx][3] - offsetY) / scaleY

    if debug_print:
      print "objPoolInCurScale: "  
      print objPoolInCurScale

    p0_init_done = False
    p0 = None
    for idx, detObj in enumerate(objPoolInCurScale ):
      if debug_print: 
        print "XLeft-XRight: ", int(detObj[0]), ":", int(detObj[2]),
        print "YTop-YBot   : ", int(detObj[1]),":", int(detObj[3])
      
      #initalize all pixels to zero (picture completely black)
      mask_detObjs = np.zeros(propagate_obj.old_gray.shape,np.uint8)

      mask_detObjs[int(detObj[1]):int(detObj[3]),int(detObj[0]):int(detObj[2])] = 1
      areaCurObj = (detObj[2]-detObj[0]) *  (detObj[3]-detObj[1])
      if debug_print: 
        print "areaCurObj: ", areaCurObj 
      # params for ShiTomasi corner detection
      maxCornersCurObj = 100 #100
      minDisanceCurObj = int(max(min(areaCurObj/24,16),4))
      feature_params = dict( maxCorners = maxCornersCurObj,  #100
                             qualityLevel = 0.1,             #0.3
                             minDistance = minDisanceCurObj, #7
                             blockSize = 7 )
      p0_curObj = cv2.goodFeaturesToTrack(propagate_obj.old_gray, mask = mask_detObjs, **feature_params)
      if p0_curObj is None:
        if debug_print: 
          print "no key point found on det objs"
      elif p0_init_done:
        p0 = np.concatenate((p0, p0_curObj), axis=0)
      else:  
        p0_init_done = True
        p0 = p0_curObj

    if debug_print:
      print "gPoolDetObjs.dtype: ", gPoolDetObjs.dtype
      print "gPoolDetObjs.shape: ", gPoolDetObjs.shape
      print "gPoolDetObjs: ", gPoolDetObjs

    good_new = np.array([])
    good_old = np.array([])
    if p0 is not None:
      p1, st, err = cv2.calcOpticalFlowPyrLK(propagate_obj.old_gray, frame_gray, p0, None, **lk_params)
    
      if debug_print:
        #print "p1.dtype: ", p1.dtype
        #print "p1.shape: ", p1.shape
        print "p1: ", p1
        print "st: ", st
        print "err: ", err

      # Select good points
      good_new = p1[st==1]
      good_old = p0[st==1]

      if debug_print:
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

      if debug_print:
        print "=========="
        print "obj in cur scale coordinate : ", tlX, ":", tlY, ":", brX, ":", brY
      
      resultTLx, tlX, tlY = updateWithNearestKeypoint(curPosX=tlX,curPosY=tlY,tlX=tlX,
         tlY=tlY, brX=brX, brY=brY, good_old=good_old, good_new=good_new)
      resultBRy, brX, brY = updateWithNearestKeypoint(curPosX=brX,curPosY=brY,tlX=tlX,
         tlY=tlY, brX=brX, brY=brY, good_old=good_old, good_new=good_new)
      
      #if both TL and BR points are having good match
      if resultTLx and resultBRy: 
        if debug_print:
          print "obj propogation successfull!"
          print "propogated obj in cur scale coordicates: ", tlX, ":", tlY, ":", brX, ":", brY

        trackedBoxCurScale = [tlX, tlY, brX, brY, float(detObj[LBL_IDX]), detObj[SCORE_IDX], detObj[AGE_IDX]]
        maxOverLap = findBestOverlap(trackedBox=trackedBoxCurScale, top_conf=top_conf, top_label_indices=top_label_indices, 
            top_xmin=top_xmin, top_ymin=top_ymin, top_xmax=top_xmax, top_ymax=top_ymax)

        if (maxOverLap < maxOverLapTh) and debug_print:
          print "maxOverLap: ", maxOverLap 
          print "But object removed due to less overlap with moderate detectios in frame num: ", curFrameNum
      
        #convert from cur scale position to original image coordinate
        tlX = (tlX*scaleX + offsetX) 
        brX = (brX*scaleX + offsetX) 

        tlY = (tlY*scaleY + offsetY)
        brY = (brY*scaleY + offsetY)

        keep_obj_alive = shouldKeepObjAlive(params=params, detObj=detObj, labelmap=labelmap)    
        if keep_obj_alive and (maxOverLap>=maxOverLapTh):
          trackedBox = [tlX, tlY, brX, brY, float(detObj[LBL_IDX]), detObj[SCORE_IDX], detObj[AGE_IDX]]
          trackedObjsList.append(trackedBox) 
      else:  
        if debug_print:
          print "obj propogation failed!!"
      if debug_print:  
        print "=========="

    trackedObjs = np.asarray(trackedObjsList)
    
    if debug_print:
      print "trackedObjs.dtype: ", trackedObjs.dtype
      print "trackedObjs.shape: ", trackedObjs.shape
      print "trackedObjs: ", trackedObjs 

    draw_tracks = True
    if draw_tracks and good_old.size:
      # draw the tracks
      # Create a mask image for drawing purposes
      mask = np.zeros_like(curImage)
      for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        imageForViz = cv2.circle(curImage,(a,b),5,color[i].tolist(),-1)
      imageForViz= cv2.add(imageForViz,mask)
      cv2.imwrite('./debug/debug_{}.png'.format(str(curFrameNum)),imageForViz)
  
  # Now update the previous frame
  propagate_obj.old_gray = frame_gray.copy()
  
  return  trackedObjs

def updateWithNearestKeypoint(curPosX=0,curPosY=0,tlX=0,
    tlY=0, brX=0, brY=0, p0=[],p1=[],good_old=[],good_new=[]):
  debug_print = True
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
    if debug_print:
      print "OF_x:", nearest_old[0] - nearest_old_match[0]  
      print "OF_y:", nearest_old[1] - nearest_old_match[1]  

    curPosX += nearest_old_match[0] - nearest_old[0]
    curPosY += nearest_old_match[1] - nearest_old[1] 
    result = True
  else: 
    if debug_print:
      print "minErr: ", minErr 
  return [result, curPosX,curPosY]

def shouldKeepObjAlive(params=[], detObj=[], labelmap=[], hash_key_based=False):    
  AGE_BASED = True
  reduceScoreTh  =0.05
  AGE_IDX = 6
  SCORE_IDX = 5
  LBL_IDX = 4

  if AGE_BASED:
    #increase age for each year(frame)
    detObj[AGE_IDX] += 1
    keep_obj_alive = (detObj[AGE_IDX] <= params.maxAgeTh)
  else: #score based.score gets reduced with each passing year (frame)
    #reduce score of tracked objects to give more preference to objs in
    #the currnt frame also it will make sure objs will die after few frames
    detObj[SCORE_IDX] = max(0.0, float(detObj[SCORE_IDX]-reduceScoreTh))
    if type(params.confTh) is dict:                                
      label = int(detObj[LBL_IDX]) 
      label_name = str(get_labelname(labelmap,label,hash_key_based=hash_key_based)[0])
      keep_obj_alive = detObj[SCORE_IDX] > params.confTh[label_name]
    else:    
      keep_obj_alive = detObj[SCORE_IDX] > params.confTh
  return keep_obj_alive     

def findBestOverlap(trackedBox=[], top_conf=[], top_label_indices=[], top_xmin=[], top_ymin=[], top_xmax=[], top_ymax=[]):
  debug_print = False
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

  if debug_print:
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
      if debug_print:
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

      if debug_print:
        print "overlap w: " , w,
        print "overlap h: " , h,

      # compute the ratio of overlap
      overlap = (w * h) / area

      if debug_print:
        print "overlap: ", overlap,

      if overlap > max_overlap:
        max_overlap = overlap

  if debug_print:
    print "max_overlap: ", max_overlap
    print "=================="
  return max_overlap    

