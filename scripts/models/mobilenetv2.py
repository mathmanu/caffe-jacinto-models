from __future__ import print_function
import caffe
from models.model_libs import *
import copy

def ConvBNLayerMobileNetV2(net, from_layer, out_layer, use_relu=True, num_output=0,
    kernel_size=3, pad=0, stride=1, dilation=1, group=1,
    prePostFix='', kwArgs='',isFrozen=False, bn_type='bvlc'):

  if (group <> 1) or (dilation <> 1):
    kwargs_conv_grp = kwArgs.kwargs_conv_grp_dil
  else:  
    kwargs_conv_grp = kwArgs.kwargs_conv
 
  bn_in_place = True
  #if bn_type == 'nvidia':
  #  #nvidia Caffe now allows inplace in BN layer
  #  bn_in_place = False

  conv_name = '{}{}{}'.format(prePostFix.conv_prefix, out_layer, prePostFix.conv_postfix)
  [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
  [pad_h, pad_w] = UnpackVariable(pad, 2)
  [stride_h, stride_w] = UnpackVariable(stride, 2)
  #lower wd for dw layers as per mobilenet paper - not sure if it helps
  kwargs_conv = copy.deepcopy(kwargs_conv_grp[isFrozen])
  #decay_mult = 0.01 if group == num_output else 1
  param = {'decay_mult': decay_mult}
  kwargs_conv['param'][0]['decay_mult'] = decay_mult
  print("conv_name: {} {}x{} {} - group={}".format(conv_name, kernel_w, kernel_h, num_output, group))
  if kernel_h == kernel_w:
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
        kernel_size=kernel_h, pad=pad_h, stride=stride_h, group=group,
        **kwargs_conv)
  else:
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
        kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
        stride_h=stride_h, stride_w=stride_w, group=group,
        **kwargs_conv)

  if dilation > 1:
    net.update(conv_name, {'dilation': dilation})
  
  op_layer_name = conv_name
  if bn_type <> 'none':
    bn_name = '{}{}{}'.format(prePostFix.bn_prefix, out_layer, prePostFix.bn_postfix)
    if kwArgs.bn_kwargs[isFrozen] is not None:
      net[bn_name] = L.BatchNorm(net[conv_name], in_place=bn_in_place, **kwArgs.bn_kwargs[isFrozen])
    else:
      net[bn_name] = L.BatchNorm(net[conv_name], in_place=bn_in_place)      
    op_layer_name = bn_name
    if bn_type == 'bvlc':
      #in BVLC type BN one nees explictly scale/bias layer
      sb_name = '{}{}{}'.format(prePostFix.scale_prefix, out_layer, prePostFix.scale_postfix)
      if kwArgs.sb_kwargs[isFrozen] is not None:
        net[sb_name] = L.Scale(net[bn_name], in_place=True, **kwArgs.sb_kwargs[isFrozen])
      else:
        net[sb_name] = L.Scale(net[bn_name], in_place=True)        
      op_layer_name = sb_name

  if use_relu:
    #relu_name = '{}{}'.format(conv_name, prePostFix.relu_postfix)
    relu_name = '{}{}{}'.format(prePostFix.conv_prefix, out_layer, prePostFix.conv_postfix)
    relu_name = relu_name.replace('conv', 'relu')
    net[relu_name] = L.ReLU(net[op_layer_name], in_place=True)
    op_layer_name = relu_name

  return op_layer_name  
  

def InvertedResidualLinearBottleNeckBlock(net, from_layer, out_layer, use_relu=True, num_output=0,
    stride=1, dilation=1, group=1, isFrozen=False, expansion_t=6):

  input_layer = from_layer
  out_name = out_layer

  out_layer = '{}/expand'.format(out_name)
  out_layer = ConvBNLayerMobileNetV2(net, from_layer, out_layer, use_relu=use_relu, num_output=num_output*t,
    kernel_size=1, pad=0, stride=1, dilation=1, group=group,
    prePostFix=prePostFix, kwArgs=kwArgs,isFrozen=isFrozen, bn_type=bn_type)
  from_layer = out_layer

  out_layer = '{}/dwise'.format(out_name)
  out_layer = ConvBNLayerMobileNetV2(net, from_layer, out_layer, use_relu=use_relu, num_output=num_output*t,
    kernel_size=3, pad=1*dilation, stride=stride, dilation=dilation, group=num_output*t,
    prePostFix=prePostFix, kwArgs=kwArgs,isFrozen=isFrozen, bn_type=bn_type)
  from_layer = out_layer

  out_layer = '{}/linear'.format(out_name)
  out_layer = ConvBNLayerMobileNetV2(net, from_layer, out_layer, use_relu=use_relu, num_output=num_output,
    kernel_size=1, pad=0, stride=1, dilation=1, group=group,
    prePostFix=prePostFix, kwArgs=kwArgs,isFrozen=isFrozen, bn_type=bn_type)
  from_layer = out_layer

  if stride == 1:
    out_layer = '{}/eltwise'.format(out_name)
    net[out_layer] = L.Eltwise(net[from_layer], net[input_layer])

  return out_layer


class PrePostFixMobileNetV2:
    #Class for prefix and postfix
    def __init__(self, bn_type='', ssd_mobile_chuanqi=False):
      if ssd_mobile_chuanqi:
        self.bn_prefix = ''
        self.bn_postfix = '_{}_bn'.format(bn_type)
        self.scale_prefix = ''
        self.scale_postfix = '_{}_scale'.format(bn_type)
        self.conv_prefix = ''
        self.conv_postfix = ''
        self.relu_prefix = ''
        self.relu_postfix = '_relu'
      else:
        self.bn_prefix = ''      
        self.bn_postfix = '/bn'     
        self.scale_prefix = ''     
        self.scale_postfix = '/scale'    
        self.conv_prefix = ''
        self.conv_postfix = ''
        self.relu_prefix = ''                     
        self.relu_postfix = '/relu'
   
        
        
def zeroOutLearnableParamV2(kwargs=''):
  #print("kwargs: ", kwargs)
  if (kwargs is not None) and ('param' in kwargs.keys()):
    for param in kwargs['param']:
      #print("param: ", param)
      param['lr_mult'] = 0
      param['decay_mult'] = 0
            
class KWArgsMobileNetV2(object):
    kwargs_conv = [] 
    kwargs_conv_grp_dil = []
    sb_kwargs = []
    bias_kwargs = []
    prelu_kwargs = []
    bn_kwargs = []

    #Class for keyword Args
    def __init__(self, caffe_fork='nvidia', eps=0.001, bias_term=True, param_in_sb=False, caffeForDilGrp=True, bn_type = 'bvlc'):
      
      #############################################################################################################
      # Params for enet
      #############################################################################################################

      # parameters for convolution layer with batchnorm.
      if bias_term:
        param_conv = {
          'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
          'weight_filler': dict(type='msra'),
          'bias_filler': dict(type='constant', value=0),
          'bias_term': bias_term,
          }
      else:  
        param_conv = {
          'param': [dict(lr_mult=1, decay_mult=1)],
          'weight_filler': dict(type='msra'),
          'bias_term': bias_term,
          }

      # In BVLC Caffe version, CUDNN does not support 'group' feature or dilated conv
      # So use CAFFE engine instead of CUDNN(def)
      if caffe_fork == 'bvlc':
        if bias_term:
          param_conv_grp_dil = {
              'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
              'weight_filler': dict(type='msra'),
              'bias_filler': dict(type='constant', value=0),
              'bias_term': bias_term,
              #'engine': 1, #CAFFE
              }
        else: 
          param_conv_grp_dil = {
              'param': [dict(lr_mult=1, decay_mult=1)],
              'weight_filler': dict(type='msra'),
              'bias_term': bias_term,
              #'engine': 1, #CAFFE
              }
        if caffeForDilGrp:
          param_conv_grp_dil['engine'] = 1
      else:
        param_conv_grp_dil = param_conv


      # parameters for batchnorm layer.
      if caffe_fork == "bvlc" :
        param_bn_kwargs = {
            'param': [dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)],
            'eps': eps,
            }
      else : #caffe_fork == "nvidia"
        if bn_type == 'bvlc':
          param_bn_kwargs = None      
        else: #nvidia
          param_bn_kwargs = {
            #scale, shift/bias,global mean, global var
            #'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=1,decay_mult=1),
            #  dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
            'eps': eps,
            #'scale_filler': dict(type='constant', value=1),
            #'bias_filler': dict(type='constant', value=0),
            'moving_average_fraction': 0.99,
            'scale_bias': True
            }

      param_prelu_kwargs = {
          'param': [dict(lr_mult=1, decay_mult=0)],
          'filler': dict(type='constant', value=0.0),
          'channel_shared': False
          }


      # parameters for scale bias layer after batchnorm.
      param_sb_kwargs = {
           'bias_term': True,
           #'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=1)],
           #'filler': dict(type='constant', value=1.0),
           #'bias_filler': dict(type='constant', value=0.0),
           }
      if(param_in_sb) :
        param_sb_kwargs['param'] = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=1)]

      if bias_term:
        param_bias_kwargs = {
          'param': [dict(lr_mult=1, decay_mult=0)],
          'filler': dict(type='constant', value=0.0),
          }

      self.kwargs_conv.append(param_conv)
      self.kwargs_conv_grp_dil.append(param_conv_grp_dil)
      self.bn_kwargs.append(param_bn_kwargs)
      self.prelu_kwargs.append(param_prelu_kwargs)
      self.sb_kwargs.append(param_sb_kwargs)

      if bias_term:
        self.bias_kwargs.append(param_bias_kwargs)

      print("param_conv ", param_conv)
      param_conv1 = copy.deepcopy(param_conv)
      param_conv_grp_dil1 = copy.deepcopy(param_conv_grp_dil)
      param_sb_kwargs1 = copy.deepcopy(param_sb_kwargs) 

      if bias_term:
        param_bias_kwargs1 = copy.deepcopy(param_bias_kwargs)
      param_prelu_kwargs1 = copy.deepcopy(param_prelu_kwargs)
      param_bn_kwargs1 = copy.deepcopy(param_bn_kwargs)

      zeroOutLearnableParamV2(param_conv1)
      zeroOutLearnableParamV2(param_conv_grp_dil1)
      if param_in_sb:
        zeroOutLearnableParamV2(param_sb_kwargs1)
      if bias_term:
        zeroOutLearnableParamV2(param_bias_kwargs1)
      zeroOutLearnableParamV2(param_prelu_kwargs1)
      zeroOutLearnableParamV2(param_bn_kwargs1)

      self.kwargs_conv.append(param_conv1)
      self.kwargs_conv_grp_dil.append(param_conv_grp_dil1)
      self.bn_kwargs.append(param_bn_kwargs1)
      self.prelu_kwargs.append(param_prelu_kwargs1)
      self.sb_kwargs.append(param_sb_kwargs1)
      if bias_term:
        self.bias_kwargs.append(param_bias_kwargs1)

      print("param_conv ", self.kwargs_conv[0])
      print("param_conv1 ", self.kwargs_conv[1])


###############################################################
def MobileNetV2Body(net, from_layer='data', dropout=True, freeze_layers=None, num_output=1000,
  wide_factor = 1.0, enable_fc=True, output_stride=32, expansion_factor=6):

  num_output_fc = num_output

  if freeze_layers is None:
    freeze_layers = []

  kwArgs = KWArgsMobileNetV2()
  block_labels = ['1', '2_1', '2_2', '3_1', '3_2', '4_1', '4_2', '5_1', '5_2', '5_3', '5_4', '5_5', '5_6', '6']

  mobilenetv2_t = expansion_t
  mobilenetv2_c = [32, 16, 24, 32, 64, 96, 160, 320, 1280]
  mobilenetv2_n = [1, 1, 2, 3, 4, 3, 3, 1, 1]

  if output_stride == 32:
    mobilenetv2_s = [2, 1, 2, 2, 2, 1, 2, 1, 1]
  elif output_stride == 16:
    mobilenetv2_s = [2, 1, 2, 2, 2, 1, 1, 1, 1]
  else:
    assert(output_stride==32 or output_stride==16)

  mobilenetv2_c = map(lambda x: int(round(x * wide_factor)), mobilenetv2_c)
  
  ##################
  block_name = 'conv{}'.format(block_labels[0])
  isFrozen= block_name in freeze_layers
  dilation = 1
  op_layer_name = ConvBNLayerMobileNetV2(net, from_layer, block_name,
      num_output=mobilenetv2_c[0], kernel_size=3, pad=1, stride=mobilenetv2_s[0],
      isFrozen=isFrozen,dilation=dilation)

  ##################
  num_stages = len(mobilenetv2_c)
  for stg_idx in range(1,num_stages-1):
    for n in range(mobilenetv2_n[stg_idx]):
      InvertedResidualLinearBottleNeckBlock()

  op_layer_name = ConvBNLayerMobileNetV2(net, from_layer, block_name,
      num_output=mobilenetv2_c[-1], kernel_size=3, pad=1, stride=mobilenetv2_s[-1],
      isFrozen=isFrozen,dilation=dilation)


  if enable_fc:
    #--   
    # Add global pooling layer.
    from_layer = op_layer_name
    op_layer_name = 'pool6'
    net[op_layer_name] = L.Pooling(net[from_layer], pool=P.Pooling.AVE, global_pooling=True)
         
    from_layer = op_layer_name
    op_layer_name = 'fc7' #'fc'+str(num_output)
    kwargs = { 'num_output': num_output_fc, 
      'param': [{'lr_mult': 1, 'decay_mult': 1}, {'lr_mult': 2, 'decay_mult': 0}], 
      'convolution_param': { 
         'kernel_size': 1,
         'weight_filler': { 'type': 'msra' }, 
      },
    }
    net[op_layer_name] = L.Convolution(net[from_layer], **kwargs)    
  
  return op_layer_name


###############################################################
def mobilenetv2(net, from_layer='data', fully_conv=False, reduced=False, dilated=False,
        dropout=True, freeze_layers=None, bn_type='bvlc', bn_at_start=True, caffe_fork='nvidia',
        training_type='ImageNet', depth_mul=1, ssd_mobile_chuanqi=False, dil_when_stride_removed=False,
        num_output=1000, wide_factor=1.0):
  return MobileNetV2Body(net, from_layer, fully_conv, reduced, dilated,
        dropout, freeze_layers, bn_type, bn_at_start, caffe_fork,
        training_type, depth_mul, ssd_mobile_chuanqi, dil_when_stride_removed,
        num_output, wide_factor=wide_factor)

        
def mobiledetnetv2(net, from_layer='data', fully_conv=False, reduced=False, dilated=False,
        dropout=True, freeze_layers=None, bn_type='bvlc', bn_at_start=True, caffe_fork='nvidia',
        training_type='ImageNet', depth_mul=1, ssd_mobile_chuanqi=False, dil_when_stride_removed=False,
        num_output=1000, wide_factor=1.0, use_batchnorm=True, use_relu=True, num_intermediate=512):

  op_layer_name = MobileNetV2Body(net, from_layer, fully_conv, reduced, dilated,
        dropout, freeze_layers, bn_type, bn_at_start, caffe_fork,
        training_type, depth_mul, ssd_mobile_chuanqi, dil_when_stride_removed,
        num_output, wide_factor=wide_factor, enable_fc=False)
  
  #---------------------------     
  #PSP style pool down
  pooling_param = {'pool':P.Pooling.MAX, 'kernel_size':3, 'stride':2, 'pad':1}      
  from_layer = op_layer_name 
  out_layer = 'pool6'
  net[out_layer] = L.Pooling(net[from_layer], pooling_param=pooling_param) 
  #--
  pooling_param = {'pool':P.Pooling.MAX, 'kernel_size':3, 'stride':2, 'pad':1}      
  from_layer = out_layer
  out_layer = 'pool7'
  net[out_layer] = L.Pooling(net[from_layer], pooling_param=pooling_param)  
  #--
  pooling_param = {'pool':P.Pooling.MAX, 'kernel_size':3, 'stride':1, 'pad':1}      
  from_layer = out_layer
  out_layer = 'pool8'
  net[out_layer] = L.Pooling(net[from_layer], pooling_param=pooling_param)  

  #for top in net.tops:
  #  print("top:", top)
  
  #---------------------------       
  out_layer_names = []
  
  #from_layer = 'relu4_1/sep'
  #out_layer = 'ctx_output????'
  #out_layer = ConvBNLayerDWSep1x1First(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=num_intermediate, kernel_size=3, pad=1, stride=1, group=1, dilation=1) 
  
  from_layer = 'relu5_5/sep'
  out_layer = 'ctx_output1'
  out_layer = ConvBNLayerDWSep1x1First(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=num_intermediate, kernel_size=3, pad=1, stride=1, group=1, dilation=1)  
  out_layer_names += [out_layer]
  
  from_layer = 'relu6/sep'
  out_layer = 'ctx_output2'
  out_layer = ConvBNLayerDWSep1x1First(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=num_intermediate, kernel_size=3, pad=1, stride=1, group=1, dilation=1) 
  out_layer_names += [out_layer]
  
  from_layer = 'pool6'
  out_layer = 'ctx_output3'
  out_layer = ConvBNLayerDWSep1x1First(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=num_intermediate, kernel_size=3, pad=1, stride=1, group=1, dilation=1)              
  out_layer_names += [out_layer]
  
  from_layer = 'pool7'
  out_layer = 'ctx_output4'
  out_layer = ConvBNLayerDWSep1x1First(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=num_intermediate, kernel_size=3, pad=1, stride=1, group=1, dilation=1)        
  out_layer_names += [out_layer]
  
  from_layer = 'pool8'
  out_layer = 'ctx_output5'
  out_layer = ConvBNLayerDWSep1x1First(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=num_intermediate, kernel_size=3, pad=1, stride=1, group=1, dilation=1)        
  out_layer_names += [out_layer]
  
  return out_layer, out_layer_names
 
