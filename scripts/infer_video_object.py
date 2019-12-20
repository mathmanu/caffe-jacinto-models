############################################################
# This is infeencing scripot for Caffe-Jacinto based SSD
# Run this script from ssd-caffe/examples
# This script takes set of videos as inputs and generates 
# output video with detected objects
############################################################
from ssd_detect_video import ssd_detect_video
import sys
import os

###################################
#Define Test Videos
###################################ssd720x368_ssdJacintoNetV2_iter_9600.caffemodel
ti_set1 = {}
ti_set1['IpPath'] = "/data/videos/"
ti_set1['fileNames'] = ['video1.MP4', 'video2.MP4']

ti_set2={}
ti_set2['IpPath']='/data/videos2/'
ti_set2['fileNames'] = ['video3.MP4']

ti_demo = {}
ti_demo['IpPath'] = "/data/mmcodec_video2_tier3/datasets/TI/RoadDrive/GermanyVideos/Munich_5th_Oct_Thorsten/107_VIRB/"
ti_demo['fileNames'] = ['VIRB0002.MP4', 'VIRB0003_4400sfr_5000nfr.MP4']

#############################################################################
def execute_test(params):
  #datasets = [ti_set1, ti_set2]
  datasets = [ti_demo]
  for dataset in datasets:
    #when writing bbox map category from src->dst
    WRITE_BBOX_MAP=[['car','vehicle'], ['train','vehicle'],['bus','vehicle']]
    category = "TI"
  
    print "dataset:", dataset
    for fileName in dataset['fileNames']:
      ipFileName=dataset['IpPath']+fileName
      opFileName=params.OpPath+fileName
      opFileName = opFileName.replace('.MP4', '_detOp.MP4')
      print ipFileName, opFileName
      ssd_detect_video(ipFileName=ipFileName,
          opFileName=opFileName, deployFileName=params.Deploy, modelWeights=params.ModelWeights,
          numFrames=params.NumFrames, tileSizeW=params.TileSizeW, tileSizeH=params.TileSizeH,labelmapFile=params.LabelMap, 
          tileScaleMode=params.MUL_TILE_SCL_MD, resizeW=params.ResizeW,resizeH=params.ResizeH,enNMS=params.NMS_FLAG, 
          numScales=params.NUM_SCALES,arType=params.AR_TYPE, confTh=params.CONF_TH, writeBbox=params.WRITE_BBOX, 
          meanPixVec=params.MEAN_PIX_VEC, ipScale=params.IP_SCALE, writeBboxMap=WRITE_BBOX_MAP,
          enCrop=params.enCrop, cropMinX=params.cropMinX, cropMinY=params.cropMinY,
          cropMaxX=params.cropMaxX, cropMaxY=params.cropMaxY, decFreq = params.decFreq, 
          enObjProp=params.enObjTracker, start_frame_num=params.start_frame_num, 
          maxAgeTh=params.maxAgeTh, caffe_root=params.caffe_root)
      
  return
############################################
#common params for all tests
def set_common_params(params):
  params.caffe_root = '/user/a0875091/files/work/bitbucket_ti/caffe-jacinto/'  
  #crop is the first operation done if enabled
  params.enCrop=False
  params.cropMinX = 0
  params.cropMinY = 0
  params.cropMaxX = 0
  params.cropMaxY = 0

  if params.enCrop:
    #use positive y_offset if horizon is below the middle of the picture
    y_offset = -80 #40
    crop_w = 1280 #
    crop_h = 720
    image_w = 1920
    image_h = 1080 
    params.cropMinX=(image_w-crop_w)/2
    params.cropMinY=((image_h-crop_h)/2) + y_offset
    params.cropMaxX=params.cropMinX+crop_w
    params.cropMaxY=params.cropMinY+crop_h
    
  params.NumFrames=300
  params.start_frame_num=0
  # ResizeW and ResizeH are not used when multiple sclales are used
  # When these options are non zero first crop (if enabled) then resizing will be done before anything else
  params.ResizeW=0
  params.ResizeH=0
  params.NMS_FLAG=0
  params.MUL_TILE_SCL_MD=1  # 0:multiple tiles, 1: multiple scales
  params.NUM_SCALES=1
  params.AR_TYPE=2          # 0:height according to tile height (def), 1:width according to tile width, 2: maintain original AR till it hits limit
  params.WRITE_BBOX = True
  params.decFreq = 1
  params.enObjTracker=False
  params.maxAgeTh=30
  return

############################################
class Params:
  def displayArgs(self):
    print "===========params instantiated======================="
############################################
#SET1: 
params = Params()
params.ModelWeights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/trained/object_detection/ti-720x368/JDetNet/ssd768x320_PSP_dsFac_32_hdDS8_1_kerMbox_1_smallOBj_1_51.41/sparse_50.52/ti-vgg-720x368-v2_ssdJacintoNetV2_iter_46000_50.52.caffemodel"
params.Deploy="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/trained/object_detection/ti-720x368/JDetNet/ssd768x320_PSP_dsFac_32_hdDS8_1_kerMbox_1_smallOBj_1_51.41/sparse_50.52/deploy.prototxt"
params.LabelMap="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/trained/object_detection/ti-720x368/labelmap.prototxt"
params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20180326_JDetNet_768x320_1Gmac_fac0.8_53.26_tda2xIp_test_rel/"
#if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
params.MEAN_PIX_VEC =[0,0,0]
params.IP_SCALE  = 1.0
params.TileSizeW = 768
params.TileSizeH = 320
params.CONF_TH = 0.4
params.EVAL_OBJ = False
set_common_params(params)
execute_test(params)
########################################################################################################################################################################
