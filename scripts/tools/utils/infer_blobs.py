#!/usr/bin/env python

import os
import os.path
import time
import sys
import matplotlib
import matplotlib.pyplot as plt
import ntpath

#sys.path.insert(0, '/user/a0393754/work/caffe/caffe-jacinto/python')

#model_path = '/user/a0393608/files/work/code/vision/ti/bitbucket/algoref/caffe-jacinto-quantization/trained/image_classification/imagenet_mobilenet_v2_shicai/test_quantize/deploy.prototxt'
#pretrained_path = '/user/a0393608/files/work/code/vision/ti/bitbucket/algoref/caffe-jacinto-quantization/trained/image_classification/imagenet_mobilenet_v2_shicai/test_quantize/mobilenet_v2.caffemodel'

model_path = '/user/a0393608/files/work/code/vision/ti/bitbucket/algoref/caffe-jacinto-quantization/trained/image_classification/imagenet_mobilenet_v2_newnames/test_quantize/deploy.prototxt'
pretrained_path = '/data/mmcodec_video2_tier3/users/debu/net_surgery/imagenet_mobilenetv2-1.0_2018-06-06_17-18-39_bvlcbn/initial/MobileNetV2_new_names.caffemodel'

input_name = '/data/mmcodec_video2_tier3/users/manu/shared/inputs/VID_20141005_182635.JPG'
input_mean_values = [103.94,116.78,123.68] #[0, 0, 0]
input_size = (224,224) #(512,256)

import caffe
caffe.set_mode_cpu()
from caffe.proto import caffe_pb2
import cv2
import numpy as np
import math
import string
from google.protobuf import text_format

def writeNPAryAsRaw(ipFrame, fileName, opDataType=np.float32, opScale=1):
    if opDataType != np.float32:
        opMult = opScale
        qFrame = np.rint(ipFrame * opMult)
    else:
        qFrame = ipFrame
            
    fileHandle = open(fileName, 'wb')
    #ip1DAry = np.reshape(qFrame, (1, np.prod(qFrame.shape)))
    ip1DAry = np.ndarray.flatten(qFrame)
    ip1DAry = ip1DAry.astype(opDataType)
    fileHandle.write(ip1DAry)
    fileHandle.close()
       
def predict(model_path, pretrained_path, image, frameNum, blobs=None):
    net = caffe.Net(model_path, weights=pretrained_path, phase=caffe.TEST)
    #model = type('', (), {})()
    #model.net = net

    input_dims = net.blobs['data'].shape
    #output_dims = net.blobs['prob'].shape

    #label_margin = config.MARGIN
    print ("input dim from desc", input_dims[2], input_dims[3])
    #print ("output_dim from desc", output_dims[2], output_dims[3])

    batch_size, num_channels, input_height, input_width = input_dims
    caffe_in = np.zeros(input_dims, dtype=np.float32)

    caffe_in[0] = image.transpose([2, 0, 1])
    out_blobs = net.forward_all(blobs, **{net.inputs[0]: caffe_in})
    
    return out_blobs, net

def getLayerByName(net_proto, layer_name):
    for layer in net_proto.layer:
       if layer.name == layer_name:
          return layer
    return None
    
def infer():
    caffe.set_mode_cpu()
    mean_pixel = input_mean_values
    num = 0

    net_proto = caffe_pb2.NetParameter()
    text_format.Merge(open(model_path).read(), net_proto)
    
    # moved image reading out from predict()
    image = cv2.imread(input_name, 1).astype(np.float32) - mean_pixel
    image = cv2.resize(image, input_size)

    out_blobs, net = predict(model_path, pretrained_path, image, num, blobs=[])
    
    #print(net._inputs[0])    
    #print(net._outputs[0])
    
    blob_names = list(net._blob_names)
    layer_names = list(net._layer_names)
    blob_name_to_layer_name_dict = {}
    for layer_idx, layer_name in enumerate(layer_names):
        #bottom_ids = list(net._bottom_ids(layer_idx))
        top_ids = list(net._top_ids(layer_idx))
        top_names = [blob_names[top_id] for top_id in top_ids]
        for top_name in top_names:
          blob_name_to_layer_name_dict[top_name] = layer_name
       
    if 'data' in out_blobs.keys():
        writeNPAryAsRaw(out_blobs['data'], 'data'+'_uint8'+'.bin', opDataType=np.uint8, opQ=8)       
          
    for cur_blob_idx, cur_blob_name in enumerate(net.blobs.keys()):
       cur_blob = net.blobs[cur_blob_name]
       cur_data = cur_blob.data
       
       layer_name = blob_name_to_layer_name_dict[cur_blob_name]
       #layer_id = layer_names.index(layer_name)
       #layer = net.layers[layer_id]
              
       cur_blob_name_out = cur_blob_name.replace('/','.')       
       print(layer_name, " : ", cur_blob_name, " : ", cur_blob_name_out)
       if cur_blob_name == 'conv4_1/expand':
           print(layer_name, ' ', cur_blob_name, ' ', cur_data)
                  
       layer_param =  getLayerByName(net_proto, layer_name)             
       if layer_param is not None and len(layer_param.quantization_param.qparam_out)>0:       
           opScale = layer_param.quantization_param.qparam_out[0].scale
           if layer_param.quantization_param.qparam_out[0].unsigned_data:   
               writeNPAryAsRaw(cur_data, cur_blob_name_out+'_uint8'+'.bin', opDataType=np.uint8, opScale=opScale)
           else:
               writeNPAryAsRaw(cur_data, cur_blob_name_out+'_int8'+'.bin', opDataType=np.int8, opScale=opScale)       
       else:
           writeNPAryAsRaw(cur_data, cur_blob_name_out+'_float32'+'.bin', opDataType=np.float32)  
       
    for layer_name in list(net.params.keys()):
       if True: #layer_name in net.params: #layer_name == 'conv2_2/linear/scale':
           #print(layer_name, ':', cur_blob_name, '=\n', cur_data)
           #print(layer_name, ':', 'num weight blobs:', len(net.params[layer_name]))
           for p in range(len(net.params[layer_name])):
              #print(layer_name, ':', 'params:', p, '=\n',net.params[layer_name][p].data)
              writeNPAryAsRaw(net.params[layer_name][p].data, layer_name.replace('/','.')+'_weight'+str(p)+'_float32'+'.bin', opDataType=np.float32) 
                     
def main():
    infer()

if __name__ == '__main__':
    main()
