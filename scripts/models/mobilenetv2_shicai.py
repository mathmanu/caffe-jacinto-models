from __future__ import print_function
import caffe
from models.model_libs import *
import copy
import math

###############################################################
#set to 'fused' to use NVIDIA/caffe style fused batch norm that incorporates scale_bias (faster)
BN_TYPE_TO_USE = 'bvlc' #'bvlc' #'fused'


###############################################################
def width_multiplier(value, base, min_val):
  value = int(math.floor(float(value) / base + 0.5) * base)
  value = max(value, min_val)
  return value
  
  
def width_multiplier8(value):
  return width_multiplier(value, 8, 8)
  
  
###############################################################
def ConvBNLayerMobileNetV2(net, from_layer, out_layer, use_relu=True, num_output=0,
    kernel_size=3, pad=0, stride=1, dilation=1, group=1, bn_type='bvlc',
    bn_in_place=True):
      
  conv_name = '{}'.format(out_layer)
  bn_name = '{}/bn'.format(out_layer)
  scale_name = '{}/scale'.format(out_layer)
  relu_name = out_layer.replace('conv', 'relu') #'{}/relu'.format(out_layer)

  out_layer = conv_name
  kwargs_conv = {'weight_filler': {'type': 'msra'}}
  net[out_layer] = L.Convolution(net[from_layer], num_output=num_output,
      kernel_size=kernel_size, pad=pad*dilation, stride=stride, group=group, dilation=dilation, bias_term=False, **kwargs_conv)
  from_layer = out_layer
      
  if bn_type == 'bvlc':
      out_layer = bn_name
      net[out_layer] = L.BatchNorm(net[from_layer], in_place=False) #bn_in_place)
      from_layer = out_layer
      
      out_layer = scale_name
      net[out_layer] = L.Scale(net[from_layer], bias_term=True, in_place=True)
      from_layer = out_layer
  else: #fused nvidia/caffe bn
      out_layer = bn_name
      net[out_layer] = L.BatchNorm(net[from_layer], scale_bias=True, in_place=bn_in_place)
      from_layer = out_layer

  if use_relu:
    out_layer = relu_name 
    net[out_layer] = L.ReLU(net[from_layer], in_place=True)
    from_layer = out_layer

  return out_layer
  

###############################################################
def InvertedResidualLinearBottleNeckBlock(net, from_layer, out_name, use_relu=True, num_input=0, num_output=0,
    stride=1, dilation=1, group=1, expansion_t=6, bn_type='bvlc'):
  
  input_layer = '{}'.format(from_layer)

  out_layer = '{}/expand'.format(out_name)
  out_layer = ConvBNLayerMobileNetV2(net, from_layer, out_layer, use_relu=use_relu, num_output=num_input*expansion_t,
    kernel_size=1, pad=0, stride=1, dilation=1, group=group, bn_type=bn_type)
  from_layer = out_layer

  out_layer = '{}/dwise'.format(out_name)
  out_layer = ConvBNLayerMobileNetV2(net, from_layer, out_layer, use_relu=use_relu, num_output=num_input*expansion_t,
    kernel_size=3, pad=1, stride=stride, dilation=dilation, group=num_input*expansion_t, bn_type=bn_type)
  from_layer = out_layer

  out_layer = '{}/linear'.format(out_name)
  out_layer = ConvBNLayerMobileNetV2(net, from_layer, out_layer, use_relu=False, num_output=num_output,
    kernel_size=1, pad=0, stride=1, dilation=1, group=group, bn_type=bn_type)
  from_layer = out_layer

  if stride == 1 and num_input == num_output:
    out_layer = out_name.replace('conv', 'block_') #'{}/eltwise'.format(out_name)
    net[out_layer] = L.Eltwise(net[input_layer], net[from_layer])
  
  return out_layer
    
    
###############################################################
def MobileNetV2Body(net, from_layer='data', dropout=False, freeze_layers=None, num_output=1000,
  wide_factor = 1.0, enable_fc=True, bn_type='bvlc', output_stride=32, expansion_t=6):

  num_output_fc = num_output

  if freeze_layers is None:
    freeze_layers = []

  block_labels = ['1', '2_1', '2_2', '3_1', '3_2', '4_1', '4_2', '4_3', '4_4', '4_5', '4_6', '4_7',
                  '5_1', '5_2', '5_3', '6_1', '6_2', '6_3', '6_4']

  # note: this is different from mobilenetv2 paper
  assert(output_stride==32 or output_stride==16)
  strides_s = [2, 1, 2, 2, 1, 2, 2, 1, 1] if output_stride == 32 else [2, 1, 2, 2, 1, 2, 1, 1, 1]

  channels = [32, 16, 24, 32, 64, 96, 160, 320, 1280]
  channels_c = map(lambda x: width_multiplier8(x * wide_factor), channels)
  #for the last conv layer, do not reduce below 1280
  channels_c[-1] = max(channels[-1], channels_c[-1])

  repeats_n = [1, 1, 2, 3, 4, 3, 3, 1, 1]

  block_name = 'conv{}'.format(1)
  dilation = 1
  out_layer = ConvBNLayerMobileNetV2(net, from_layer, block_name,
      num_output=channels_c[0], kernel_size=3, pad=1, stride=strides_s[0], bn_type=bn_type)
  num_input = channels_c[0]
  from_layer = out_layer

  num_stages = len(channels_c)
  block_idx = 1
  for stg_idx in range(1,num_stages-1):
      for n in range(repeats_n[stg_idx]):
          xt = 1 if stg_idx < 2 else expansion_t
          out_layer = 'conv'+block_labels[block_idx] #'conv{}_{}'.format(stg_idx+1, n+1)
          dilation = 2 if output_stride == 16 and stg_idx > 5 else 1
          stride = strides_s[stg_idx] if n == 0 else 1
          out_layer = InvertedResidualLinearBottleNeckBlock(net, from_layer, out_layer,
              num_input=num_input, num_output=channels_c[stg_idx], stride=stride, dilation=dilation, bn_type=bn_type, expansion_t=xt)
          num_input = channels_c[stg_idx]
          from_layer = out_layer
          block_idx += 1
  
  out_layer = 'conv'+block_labels[-1] #'conv{}_{}'.format(num_stages-1+1, 1)
  out_layer = ConvBNLayerMobileNetV2(net, from_layer, out_layer,
      num_output=channels_c[-1], kernel_size=1, pad=0, stride=strides_s[-1],
      dilation=dilation, bn_type=bn_type)
  from_layer = out_layer

  if enable_fc:
    # Add global pooling layer.
    out_layer = 'pool{}'.format(6)
    net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.AVE, global_pooling=True)
    from_layer = out_layer

    if dropout:
      out_layer = 'drop{}'.format(6)
      net[out_layer] = L.Dropout(net[from_layer], dropout_ratio=0.5)
      from_layer = out_layer
      
    out_layer = 'fc{}'.format(6+1)
    kwargs_conv = {'weight_filler': {'type': 'msra'}}
    net[out_layer] = L.Convolution(net[from_layer], kernel_size=1, pad=0, num_output=num_output_fc, **kwargs_conv)
  
  return out_layer


###############################################################
def mobilenetv2(net, from_layer='data', dropout=False, freeze_layers=None, bn_type=BN_TYPE_TO_USE,
  num_output=1000, wide_factor=1.0, expansion_t=6):
  return MobileNetV2Body(net, from_layer=from_layer, dropout=dropout, freeze_layers=freeze_layers,
      num_output=num_output, wide_factor=wide_factor, enable_fc=True, output_stride=32, bn_type=bn_type,
      expansion_t=expansion_t)


###############################################################        
def mobiledetnetv2(net, from_layer='data', dropout=False, freeze_layers=None, bn_type=BN_TYPE_TO_USE,
  num_output=1000, wide_factor=1.0, use_batchnorm=True, use_relu=True, num_intermediate=512, expansion_t=6):
  
  out_layer = MobileNetV2Body(net, from_layer=from_layer, dropout=dropout, freeze_layers=freeze_layers,
      num_output=num_output, wide_factor=wide_factor, enable_fc=False, output_stride=32, bn_type=bn_type,
      expansion_t=expansion_t)
  
  #---------------------------     
  #PSP style pool down
  pooling_param = {'pool':P.Pooling.MAX, 'kernel_size':2, 'stride':2, 'pad':0}
  from_layer = out_layer
  out_layer = 'pool6'
  net[out_layer] = L.Pooling(net[from_layer], pooling_param=pooling_param)
  #--
  pooling_param = {'pool':P.Pooling.MAX, 'kernel_size':2, 'stride':2, 'pad':0}
  from_layer = out_layer
  out_layer = 'pool7'
  net[out_layer] = L.Pooling(net[from_layer], pooling_param=pooling_param)
  #--
  pooling_param = {'pool':P.Pooling.MAX, 'kernel_size':2, 'stride':2, 'pad':0}
  from_layer = out_layer
  out_layer = 'pool8'
  net[out_layer] = L.Pooling(net[from_layer], pooling_param=pooling_param)  

  #for top in net.tops:
  #  print("top:", top)
  
  #---------------------------       
  out_layer_names = []
  
  #from_layer = 'relu4_1/sep'
  #out_layer = 'ctx_output????'
  #out_layer = ConvBNLayerMobileNetV2(net, from_layer, out_layer, use_relu, num_output=num_intermediate, kernel_size=1, pad=0, stride=1, group=1, dilation=1, bn_type=bn_type) 
  
  from_layer = 'relu5_5/sep'
  out_layer = 'ctx_output1'
  out_layer = ConvBNLayerMobileNetV2(net, from_layer, out_layer, use_relu, num_output=num_intermediate, kernel_size=1, pad=0, stride=1, group=1, dilation=1, bn_type=bn_type)  
  out_layer_names += [out_layer]
  
  from_layer = 'relu6/sep'
  out_layer = 'ctx_output2'
  out_layer = ConvBNLayerMobileNetV2(net, from_layer, out_layer, use_relu, num_output=num_intermediate, kernel_size=1, pad=0, stride=1, group=1, dilation=1, bn_type=bn_type) 
  out_layer_names += [out_layer]
  
  from_layer = 'pool6'
  out_layer = 'ctx_output3'
  out_layer = ConvBNLayerMobileNetV2(net, from_layer, out_layer, use_relu, num_output=num_intermediate, kernel_size=1, pad=0, stride=1, group=1, dilation=1, bn_type=bn_type)              
  out_layer_names += [out_layer]
  
  from_layer = 'pool7'
  out_layer = 'ctx_output4'
  out_layer = ConvBNLayerMobileNetV2(net, from_layer, out_layer, use_relu, num_output=num_intermediate, kernel_size=1, pad=0, stride=1, group=1, dilation=1, bn_type=bn_type)        
  out_layer_names += [out_layer]
  
  from_layer = 'pool8'
  out_layer = 'ctx_output5'
  out_layer = ConvBNLayerMobileNetV2(net, from_layer, out_layer, use_relu, num_output=num_intermediate, kernel_size=1, pad=0, stride=1, group=1, dilation=1, bn_type=bn_type)        
  out_layer_names += [out_layer]
  
  return out_layer, out_layer_names
 

###############################################################
def mobilesegnetv2(net, from_layer=None, use_batchnorm=True, use_relu=True, num_output=20, stride_list=None, dilation_list=None, freeze_layers=None, 
   upsample=True, bn_type=BN_TYPE_TO_USE): 

   out_layer = MobileNetV2Body(net, from_layer=from_layer, dropout=dropout, freeze_layers=freeze_layers,
      num_output=num_output, wide_factor=wide_factor, enable_fc=False, output_stride=16, bn_type=bn_type,
      expansion_t=expansion_t)
   
   #--   
   from_layer = out_layer
   out_layer = 'out5a'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=64, kernel_size=[3,3], pad=4, stride=1, group=2, dilation=4) 
   
   #frozen upsampling layer
   from_layer = out_layer 
   out_layer = 'out5a_up2'  
   deconv_kwargs = {  'param': { 'lr_mult': 0, 'decay_mult': 0 },
       'convolution_param': { 'num_output': 64, 'bias_term': False, 'pad': 1, 'kernel_size': 4, 'group': 64, 'stride': 2, 
       'weight_filler': { 'type': 'bilinear' } } }
   net[out_layer] = L.Deconvolution(net[from_layer], **deconv_kwargs)    
   
   from_layer = 'res3a_branch2b' if in_place else 'res3a_branch2b/bn'
   out_layer = 'out3a'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=64, kernel_size=[3,3], pad=1, stride=1, group=2, dilation=1) 
   
   from_layer = out_layer   
   out_layer = 'out3_out5_combined'
   net[out_layer] = L.Eltwise(net['out5a_up2'], net[from_layer])
   
   from_layer = out_layer
   out_layer = 'ctx_conv1'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=64, kernel_size=[3,3], pad=1, stride=1, group=1, dilation=1) 
      
   from_layer = out_layer
   out_layer = 'ctx_conv2'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=64, kernel_size=[3,3], pad=4, stride=1, group=1, dilation=4) 
   
   from_layer = out_layer
   out_layer = 'ctx_conv3'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=64, kernel_size=[3,3], pad=4, stride=1, group=1, dilation=4) 
   
   from_layer = out_layer
   out_layer = 'ctx_conv4'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=64, kernel_size=[3,3], pad=4, stride=1, group=1, dilation=4) 
       
   from_layer = out_layer
   out_layer = 'ctx_final'
   conv_kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        'weight_filler': dict(type='msra'),
        'bias_term': True, 
        'bias_filler': dict(type='constant', value=0) }   
   net[out_layer] = L.Convolution(net[from_layer], num_output=num_output, kernel_size=[3,3], pad=1, stride=1, group=1, dilation=1, **conv_kwargs) 
         
   from_layer = out_layer
   out_layer = 'ctx_final/relu'
   net[out_layer] = L.ReLU(net[from_layer], in_place=True) 
               
   #frozen upsampling layer   
   if upsample:      
       from_layer = out_layer
       out_layer = 'out_deconv_final_up2'   
       deconv_kwargs = {  'param': { 'lr_mult': 0, 'decay_mult': 0 },
           'convolution_param': { 'num_output': num_output, 'bias_term': False, 'pad': 1, 'kernel_size': 4, 'group': num_output, 'stride': 2, 
           'weight_filler': { 'type': 'bilinear' } } }       
       net[out_layer] = L.Deconvolution(net[from_layer], **deconv_kwargs)   
               
       from_layer = out_layer
       out_layer = 'out_deconv_final_up4'       
       net[out_layer] = L.Deconvolution(net[from_layer], **deconv_kwargs) 
       
       from_layer = out_layer
       out_layer = 'out_deconv_final_up8'       
       net[out_layer] = L.Deconvolution(net[from_layer], **deconv_kwargs)     
                            
   return out_layer    
   
   
