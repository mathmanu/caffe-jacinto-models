from __future__ import print_function
import caffe
from models.model_libs import *
import copy
import math

###############################################################
#set to 'fused' to use NVIDIA/caffe style fused batch norm that incorporates scale_bias (faster)
BN_TYPE_TO_USE = 'fused' #'bvlc' #'fused'
SEG_OUTPUT_STRIDE = 32  #16, 32
SEG_INTERMEDIATE_CHANS = 256

###############################################################
def width_multiplier(value, base, min_val):
  value = int(math.floor(float(value) / base + 0.5) * base)
  value = max(value, min_val)
  return value
  
  
def width_multiplier8(value):
  return width_multiplier(value, 8, 8)
  

###############################################################
def ConvBNLayerMobileNet(net, from_layer, out_layer, use_relu=True, num_output=0,
    kernel_size=3, pad=0, stride=1, dilation=1, group=1, bn_type='bvlc',
    bn_in_place=True):
      
  conv_name = '{}'.format(out_layer)
  bn_name = '{}/bn'.format(out_layer)
  scale_name = '{}/scale'.format(out_layer)
  relu_name = out_layer.replace('conv','relu') if 'conv' in out_layer else '{}/relu'.format(out_layer)
  
  #lower the decay of dw layer as per MobilenetV1 paper (seems not needed for MobilenetV2)
  #may not be working - harder to train
  #decay_mult =  0.01 if group == num_output else 1
  #kwargs_conv = {'param':{'lr_mult':1, 'decay_mult':decay_mult}, 'weight_filler': {'type': 'msra'}}
  kwargs_conv = {'weight_filler': {'type': 'msra'}}

  out_layer = conv_name
  net[out_layer] = L.Convolution(net[from_layer], num_output=num_output,
      kernel_size=kernel_size, pad=pad*dilation, stride=stride, group=group, dilation=dilation, bias_term=False, **kwargs_conv)
  from_layer = out_layer

  if bn_type == 'bvlc':
      out_layer = bn_name
      net[out_layer] = L.BatchNorm(net[from_layer], in_place=bn_in_place)
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
def ConvDWBlockMobileNet(net, from_layer, out_name, use_relu=True, num_input=0, num_output=0,
    stride=1, dilation=1, group=1, expansion_t=1, bn_type='bvlc', use_residual=False):

  input_layer = '{}'.format(from_layer)

  out_layer = '{}/dw'.format(out_name)
  out_layer = ConvBNLayerMobileNet(net, from_layer, out_layer, use_relu=use_relu, num_output=num_input*expansion_t,
    kernel_size=3, pad=1, stride=stride, dilation=dilation, group=num_input*expansion_t, bn_type=bn_type)
  from_layer = out_layer

  out_layer = '{}/sep'.format(out_name)
  out_layer = ConvBNLayerMobileNet(net, from_layer, out_layer, use_relu=use_relu, num_output=num_output,
    kernel_size=1, pad=0, stride=1, dilation=1, group=group, bn_type=bn_type)
  from_layer = out_layer

  if use_residual and stride == 1 and num_input == num_output:
    out_layer = '{}/eltwise'.format(out_name)
    net[out_layer] = L.Eltwise(net[from_layer], net[input_layer])
  
  return out_layer
  
  
###############################################################
def MobileNetBody(net, from_layer='data', dropout=False, freeze_layers=None, num_output=1000,
  wide_factor = 1.0, enable_fc=True, bn_type='bvlc', output_stride=32, expansion_t=1):

  num_output_fc = num_output

  if freeze_layers is None:
    freeze_layers = []

  block_labels = ['1', '2_1', '2_2', '3_1', '3_2', '4_1', '4_2', '5_1', '5_2', '5_3', '5_4', '5_5', '5_6', '6']

  if output_stride == 32:
    strides_s = [2,1,2,1,2,1,2,1,1,1,1,1,2,1]
  elif output_stride == 16:
    strides_s = [2,1,2,1,2,1,2,1,1,1,1,1,1,1]
  else:
    assert(output_stride==32 or output_stride==16)

  channels = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024,1024]
  channels_c = map(lambda x: width_multiplier8(x * wide_factor), channels)

  #for the last conv layer, do not reduce below 1024 (this is done in mobilenetv2)
  #channels_c[-1] = max(channels[-1], channels_c[-1])

  repeats_n = [1] * len(channels_c)

  block_name = 'conv{}'.format(1)
  dilation = 1
  out_layer = ConvBNLayerMobileNet(net, from_layer, block_name,
      num_output=channels_c[0], kernel_size=3, pad=1, stride=strides_s[0], bn_type=bn_type)
  num_input = channels_c[0]
  from_layer = out_layer

  cumulative_stride = 2
  intermediate_layer = None

  num_stages = len(channels_c)  
  num_channels = {}
  for stg_idx in range(1,num_stages):
      for n in range(repeats_n[stg_idx]):
          xt = 1 if stg_idx < 2 else expansion_t
          out_layer = 'conv{}'.format(block_labels[stg_idx])
          dilation = 2 if output_stride == 16 and stg_idx > 12 else 1
          stride = strides_s[stg_idx] if n == 0 else 1
          out_layer = ConvDWBlockMobileNet(net, from_layer, out_layer,
              num_input=num_input, num_output=channels_c[stg_idx], stride=stride,
              dilation=dilation, bn_type=bn_type, expansion_t=xt)
          num_input = channels_c[stg_idx]
          from_layer = out_layer
          num_channels[out_layer] = channels_c[stg_idx]
          cumulative_stride = cumulative_stride * stride
          if cumulative_stride == 4:
              intermediate_layer = '{}'.format(out_layer)

  if enable_fc:
    # Add global pooling layer.
    out_layer = 'pool6' #'pool{}'.format(num_stages)
    net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.AVE, global_pooling=True)
    from_layer = out_layer

    if dropout:
      out_layer = 'drop6' #'drop{}'.format(num_stages)
      net[out_layer] = L.Dropout(net[from_layer], dropout_ratio=0.5)
      from_layer = out_layer
      
    out_layer = 'fc7' #'fc{}'.format(num_stages)
    kwargs_conv = {'weight_filler': {'type': 'msra'}}
    net[out_layer] = L.Convolution(net[from_layer], kernel_size=1, pad=0, num_output=num_output_fc, **kwargs_conv)
  
  return out_layer, num_channels, intermediate_layer


###############################################################
def mobilenet(net, from_layer='data', dropout=False, freeze_layers=None, bn_type=BN_TYPE_TO_USE,
  num_output=1000, wide_factor=1.0, expansion_t=None):
  expansion_t = 1 if expansion_t is None else expansion_t
  out_layer, _ ,_ = MobileNetBody(net, from_layer=from_layer, dropout=dropout, freeze_layers=freeze_layers,
      num_output=num_output, wide_factor=wide_factor, enable_fc=True, output_stride=32, bn_type=bn_type,
      expansion_t=expansion_t)
  return out_layer


###############################################################
def mobiledetnet(net, from_layer='data', dropout=False, freeze_layers=None, bn_type=BN_TYPE_TO_USE,
  num_output=1000, wide_factor=1.0, num_intermediate=512, expansion_t=None):
  expansion_t = 1 if expansion_t is None else expansion_t
  out_layer, num_channels, _ = MobileNetBody(net, from_layer=from_layer, dropout=dropout, freeze_layers=freeze_layers,
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

  #reduce channels to decrease complexity
  #from_layer = 'relu4_1/sep'
  #out_layer = 'ctx_output????'
  #num_input = num_channels[from_layer]
  #out_layer = ConvBNLayerMobileNet(net, from_layer, out_layer, num_input=num_input, num_output=num_intermediate, bn_type=bn_type, expansion_t=1)
  #out_layer_names += [out_layer]

  #reduce channels to decrease complexity
  from_layer = 'relu5_5/sep'
  out_layer = 'ctx_output1'
  num_input = num_channels[from_layer]
  out_layer = ConvDWBlockMobileNet(net, from_layer, out_layer, num_input=num_input, num_output=num_intermediate, bn_type=bn_type, expansion_t=1)
  out_layer_names += [out_layer]
  
  from_layer = 'relu6/sep'
  out_layer = 'ctx_output2'
  num_input = num_channels[from_layer]
  out_layer = ConvDWBlockMobileNet(net, from_layer, out_layer, num_input=num_input, num_output=num_intermediate, bn_type=bn_type, expansion_t=1)
  out_layer_names += [out_layer]
  
  from_layer = 'pool6'
  out_layer = 'ctx_output3'
  out_layer = ConvDWBlockMobileNet(net, from_layer, out_layer, num_input=num_input, num_output=num_intermediate, bn_type=bn_type, expansion_t=1)
  out_layer_names += [out_layer]
  
  from_layer = 'pool7'
  out_layer = 'ctx_output4'
  out_layer = ConvDWBlockMobileNet(net, from_layer, out_layer, num_input=num_input, num_output=num_intermediate, bn_type=bn_type, expansion_t=1)
  out_layer_names += [out_layer]
  
  from_layer = 'pool8'
  out_layer = 'ctx_output5'
  out_layer = ConvDWBlockMobileNet(net, from_layer, out_layer, num_input=num_input, num_output=num_intermediate, bn_type=bn_type, expansion_t=1)
  out_layer_names += [out_layer]
  
  return out_layer, out_layer_names


def mobilesegnet(net, from_layer='data', dropout=False, freeze_layers=None, bn_type=BN_TYPE_TO_USE,
                   num_output=20, wide_factor=1.0, num_intermediate=SEG_INTERMEDIATE_CHANS,
                   expansion_t=None, use_aspp=False):
    expansion_t = 1 if expansion_t is None else expansion_t
    output_stride = SEG_OUTPUT_STRIDE
    out_layer, num_channels, intermediate_layer = MobileNetBody(net, from_layer=from_layer, freeze_layers=freeze_layers,
                                              num_output=num_output, wide_factor=wide_factor, enable_fc=False,
                                              output_stride=output_stride, bn_type=bn_type, dropout=dropout,
                                              expansion_t = expansion_t)

    from_layer = out_layer
    out_layer_names = []

    if use_aspp:
        ValueError('ASPP Module is not yet supported')
    else:
        out_layer = '{}/conv_down'.format(from_layer)
        out_layer = ConvBNLayerMobileNet(net, from_layer, out_layer, num_output=num_intermediate,
                             kernel_size=1, pad=0, stride=1,
                             dilation=1, bn_type=bn_type)
        from_layer = out_layer

    # upsample x2
    deconv_kwargs = {'param': {'lr_mult': 0, 'decay_mult': 0},
                     'convolution_param': {'num_output': num_intermediate, 'bias_term': False, 'pad': 1,
                                           'kernel_size': 4, 'group': num_intermediate, 'stride': 2,
                                           'weight_filler': {'type': 'bilinear'}}}
    out_layer = '{}/up2'.format(out_layer)
    net[out_layer] = L.Deconvolution(net[from_layer], **deconv_kwargs)
    from_layer = out_layer

    # upsample x4
    out_layer = '{}/up4'.format(out_layer)
    net[out_layer] = L.Deconvolution(net[from_layer], **deconv_kwargs)
    from_layer = out_layer

    # upsample x8 - one extra upsamplig required for output stride 32
    if output_stride > 16:
        from_layer = out_layer
        out_layer = '{}/up8'.format(out_layer)
        net[out_layer] = L.Deconvolution(net[from_layer], **deconv_kwargs)
        from_layer = out_layer

    out_shortcut_layer = '{}/conv_shortcut'.format(intermediate_layer)
    out_shortcut_layer = ConvBNLayerMobileNet(net, intermediate_layer, out_shortcut_layer,
                               num_output=num_intermediate//4, kernel_size=1, pad=0, stride=1,
                               dilation=1, bn_type=bn_type)

    out_layer = 'cat_block'
    net[out_layer] = L.Concat(net[from_layer], net[out_shortcut_layer])
    from_layer = out_layer
    num_intermediate_concat = (num_intermediate + num_intermediate // 4)

    # context blocks
    out_layer = 'ctx_block1'
    out_layer = ConvDWBlockMobileNet(net, from_layer, out_layer, num_input=num_intermediate_concat,
                              num_output=num_intermediate_concat, bn_type=bn_type, expansion_t=1 )

    from_layer = out_layer

    out_layer = 'ctx_block2'
    out_layer = ConvDWBlockMobileNet(net, from_layer, out_layer, num_input=num_intermediate_concat,
                              num_output=num_intermediate_concat, bn_type=bn_type, expansion_t=1 )
    from_layer = out_layer

    #output block
    out_layer = 'ctx_final'
    kwargs_conv = {'weight_filler': {'type': 'msra'}}
    net[out_layer] = L.Convolution(net[from_layer], kernel_size=1, pad=0, num_output=num_output, **kwargs_conv)
    from_layer = out_layer

    # upsample x8 or x16
    deconv_kwargs = {'param': {'lr_mult': 0, 'decay_mult': 0},
                     'convolution_param': {'num_output': num_output, 'bias_term': False, 'pad': 1,
                                           'kernel_size': 4, 'group': num_output, 'stride': 2,
                                           'weight_filler': {'type': 'bilinear'}}}
    out_layer = 'ctx_final/up16' if output_stride > 16 else 'ctx_final/up8'
    net[out_layer] = L.Deconvolution(net[from_layer], **deconv_kwargs)
    from_layer = out_layer

    # upsample x16 or x32
    out_layer = 'ctx_output' #''ctx_final/up32' if output_stride > 16 else 'ctx_final/up16'
    net[out_layer] = L.Deconvolution(net[from_layer], **deconv_kwargs)
    from_layer = out_layer

    out_layer_names += [out_layer]
    return out_layer, out_layer_names
