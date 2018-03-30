# import the necessary packages
import numpy as np

def findOverlap(boxes, oIdx, iIdx, olMode='area_box_under_sup', verbose=0):
  tl_x = boxes[oIdx,0]
  tl_y = boxes[oIdx,1]
  br_x = boxes[oIdx,2]
  br_y = boxes[oIdx,3]

  tl_x2 = boxes[iIdx,0]
  tl_y2 = boxes[iIdx,1]
  br_x2 = boxes[iIdx,2]
  br_y2 = boxes[iIdx,3]

  #W and H of overlapping region
  if verbose > 0:
    print "min(br_x, br_x2): ", min(br_x, br_x2) 
    print "max(tl_x, tl_x2): ", max(tl_x, tl_x2)
    print "min(br_y, br_y2): ", min(br_y, br_y2)
    print "max(tl_y, tl_y2): ", max(tl_y, tl_y2) 
  w = max(min(br_x, br_x2) - max(tl_x, tl_x2) + 1, 0)
  h = max(min(br_y, br_y2) - max(tl_y, tl_y2) + 1, 0)

  if verbose > 0:
    print "boxes[oIdx]"
    print boxes[oIdx]
    print "boxes[iIdx]"
    print boxes[iIdx]
    print "w: ", w
    print "h: ", h
  aInter = w*h 

  if olMode == 'area_box_under_sup':
    #use area of the box under suppression
    aUnion = (br_x2-tl_x2+1)*(br_y2-tl_y2+1) 
  else:  
    #Area of union
    aUnion = (br_x-tl_x+1)*(br_y-tl_y+1) + (br_x2-tl_x2+1)*(br_y2-tl_y2+1) - aInter 

  iou = aInter / aUnion

  if verbose > 0:
    print "iou, ", iou
  return iou

def nms_core(boxes, olTh, selected, age_based_check=False, testMode=False):
  verbose = 0
  if age_based_check:
    sorted_indices = np.argsort(boxes[:,5]-boxes[:,6])   
  else:
    sorted_indices = np.argsort(boxes[:,5])   

  if verbose > 0:
    print "sorted_indices "
    print sorted_indices

  #make it decending order
  sorted_indices = (sorted_indices)[::-1]
  if verbose > 0:
    print "before sorting"
    print boxes
  boxes = boxes[sorted_indices]
  if verbose > 0:
    print "after sorting"
    print sorted_indices
    print boxes

  suppress = []
  for index,box in enumerate(boxes):
    suppress.append(False)

  for oIdx,box in enumerate(boxes):
    if suppress[oIdx] == False:
      #if cur box itself is supppressed then do not let it suppress any other box
      for iIdx in range(oIdx+1,len(boxes)):
        #let oIdx suppress only if it has same label as iIdx
        if boxes[iIdx,4] == boxes[oIdx,4]:
          if (verbose > 0):
            print "========================"
            print "oIdx,iIdx", oIdx, iIdx
          ol = findOverlap(boxes, oIdx, iIdx, olMode='area_box_under_sup', verbose=False)
          suppress[iIdx] = suppress[iIdx] or (ol>olTh)
          if (verbose > 0) and suppress[iIdx]:
            print "oIdx,iIdx", oIdx, iIdx
            print "boxes[oIdx]"
            print boxes[oIdx]
            print "boxes[iIdx]"
            print boxes[iIdx]

  selIdx = 0
  selectedBoxes=[]
  for sup,box in zip(suppress,boxes):
    if sup == False:
      selectedBoxes.append(box)

  if verbose > 0:
    print suppress      
    print "final boxes"
    print selectedBoxes

  selectedBoxes = np.asarray(selectedBoxes)

  #######Test NMSed boxes with ref implementation
  if testMode == True:
    from nms import non_max_suppression_fast
    boxes_ref = non_max_suppression_fast(boxes, olTh, selected, age_based_check=age_based_check)
    result = True
    if len(selectedBoxes) != len(boxes_ref) :
      result = False
    else:
      for box_ti,box_os in zip(selectedBoxes, boxes_ref):
        err = np.sum(box_ti-box_os)
        if err != 0:
          result = False

    if result == False:      
      print "===========NMS Failed================"
      print boxes
      print "result TI selectedBoxes"
      print selectedBoxes
      print "result boxes_ref"
      print boxes_ref
      sys.exit(0)
      
  return selectedBoxes

##################

