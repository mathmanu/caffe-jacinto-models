from __future__ import print_function
import caffe
from models.model_libs import *
import copy

def ConvBNLayerMobileNetPreAct(net, from_layer, out_layer, use_relu=True, num_output=0,
    kernel_size=3, pad=0, stride=1, dilation=1, group=1,
    prePostFix='', kwArgs='',isFrozen=False, bn_type='bvlc'):

  if (group <> 1) or (dilation <> 1):
    kwargs_conv_grp = kwArgs.kwargs_conv_grp_dil
  else:  
    kwargs_conv_grp = kwArgs.kwargs_conv
 
  bn_in_place = False #True
  #if bn_type == 'nvidia':
  #  #nvidia Caffe now allows inplace in BN layer
  #  bn_in_place = False

  op_layer_name = from_layer
  if bn_type <> 'none':
    bn_name = '{}{}{}'.format(prePostFix.bn_prefix, out_layer, prePostFix.bn_postfix)
    if kwArgs.bn_kwargs[isFrozen] is not None:
      net[bn_name] = L.BatchNorm(net[op_layer_name], in_place=bn_in_place, **kwArgs.bn_kwargs[isFrozen])
    else:
      net[bn_name] = L.BatchNorm(net[op_layer_name], in_place=bn_in_place)      
    op_layer_name = bn_name
    if bn_type == 'bvlc':
      #in BVLC type BN one nees explictly scale/bias layer
      sb_name = '{}{}{}'.format(prePostFix.scale_prefix, out_layer, prePostFix.scale_postfix)
      if kwArgs.sb_kwargs[isFrozen] is not None:
        net[sb_name] = L.Scale(net[op_layer_name], in_place=True, **kwArgs.sb_kwargs[isFrozen])
      else:
        net[sb_name] = L.Scale(net[op_layer_name], in_place=True)        
      op_layer_name = sb_name

  if use_relu:
    #relu_name = '{}{}'.format(conv_name, prePostFix.relu_postfix)
    relu_name = '{}{}{}'.format(prePostFix.conv_prefix, out_layer, prePostFix.conv_postfix)
    relu_name = relu_name.replace('conv', 'relu')
    net[relu_name] = L.ReLU(net[op_layer_name], in_place=True)
    op_layer_name = relu_name
    
  conv_name = '{}{}{}'.format(prePostFix.conv_prefix, out_layer, prePostFix.conv_postfix)
  [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
  [pad_h, pad_w] = UnpackVariable(pad, 2)
  [stride_h, stride_w] = UnpackVariable(stride, 2)
  print("conv_name: {} {}x{} {} - group={}".format(conv_name, kernel_w, kernel_h, num_output, group))
  if kernel_h == kernel_w:
    net[conv_name] = L.Convolution(net[op_layer_name], num_output=num_output,
        kernel_size=kernel_h, pad=pad_h, stride=stride_h, group=group,
        **kwargs_conv_grp[isFrozen])
  else:
    net[conv_name] = L.Convolution(net[op_layer_name], num_output=num_output,
        kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
        stride_h=stride_h, stride_w=stride_w, group=group,
        **kwargs_conv_grp[isFrozen])

  if dilation > 1:
    net.update(conv_name, {'dilation': dilation})
  
  op_layer_name = conv_name

  return op_layer_name  
  

def ConvBNLayerMobileNetPostAct(net, from_layer, out_layer, use_relu=True, num_output=0,
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
  print("conv_name: {} {}x{} {} - group={}".format(conv_name, kernel_w, kernel_h, num_output, group))
  if kernel_h == kernel_w:
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
        kernel_size=kernel_h, pad=pad_h, stride=stride_h, group=group,
        **kwargs_conv_grp[isFrozen])
  else:
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
        kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
        stride_h=stride_h, stride_w=stride_w, group=group,
        **kwargs_conv_grp[isFrozen])

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
  
  
def ConvBNLayerMobileNet(net, from_layer, out_layer, use_relu=True, num_output=0,
    kernel_size=3, pad=0, stride=1, dilation=1, group=1,
    prePostFix='', kwArgs='',isFrozen=False, bn_type='bvlc', bn_preact = False):
    if bn_preact:
      return ConvBNLayerMobileNetPreAct(net, from_layer, out_layer, use_relu=use_relu, num_output=num_output,
        kernel_size=kernel_size, pad=pad, stride=stride, dilation=dilation, group=group,
        prePostFix=prePostFix, kwArgs=kwArgs,isFrozen=isFrozen, bn_type=bn_type);
    else:
      return ConvBNLayerMobileNetPostAct(net, from_layer, out_layer, use_relu=use_relu, num_output=num_output,
        kernel_size=kernel_size, pad=pad, stride=stride, dilation=dilation, group=group,
        prePostFix=prePostFix, kwArgs=kwArgs,isFrozen=isFrozen, bn_type=bn_type);
    
      
class PrePostFixMobileNet:
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
   
        
        
def zeroOutLearnableParam(kwargs=''):
  #print("kwargs: ", kwargs)
  if (kwargs is not None) and ('param' in kwargs.keys()):
    for param in kwargs['param']:
      #print("param: ", param)
      param['lr_mult'] = 0
      param['decay_mult'] = 0
            
class KW_Args(object):
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

      zeroOutLearnableParam(param_conv1)
      zeroOutLearnableParam(param_conv_grp_dil1)
      if param_in_sb:
        zeroOutLearnableParam(param_sb_kwargs1)
      if bias_term:
        zeroOutLearnableParam(param_bias_kwargs1)
      zeroOutLearnableParam(param_prelu_kwargs1)
      zeroOutLearnableParam(param_bn_kwargs1)

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
def MobileResNetBody(net, from_layer='data', fully_conv=False, reduced=False, dilated=False,
        dropout=True, freeze_layers=None, bn_type='bvlc', bn_at_start=True, caffe_fork='nvidia',
        training_type='SSD', depth_mul=1, ssd_mobile_chuanqi=False, dil_when_stride_removed=False,
        num_output=1000, wide_factor = 1.0, bn_preact = False):

  if freeze_layers is None:
    freeze_layers = []
  
  #assert from_layer in net.keys()
 
  #"bvlc", "nvidia"
  if caffe_fork == '':
    caffe_fork = bn_type

  if bn_type == 'none': 
    bn_at_start = False
 
  prePostFix = PrePostFixMobileNet(bn_type=bn_type, ssd_mobile_chuanqi=ssd_mobile_chuanqi)

  #avoid intermediate activations. Changed from False to True
  use_relu_intermediate = False
  
  # Caffe_MobileNet does not have params in scale layer
  param_in_sb = False
  if ssd_mobile_chuanqi:
    param_in_sb = True
  kwArgs = KW_Args(caffe_fork=caffe_fork, bias_term=False, param_in_sb=param_in_sb, eps=0.00001, caffeForDilGrp=False, bn_type=bn_type)

  if ssd_mobile_chuanqi:
    block_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
  else:  
    block_labels = ['1', '2_1', '2_2', '3_1', '3_2', '4_1', '4_2', '5_1', '5_2', '5_3', '5_4', '5_5', '5_6', '6']

  #init with large number
  removed_stride_layer_idx = len(block_labels) + 2
  removed_stride_fac = 1
  if training_type == 'SSD':
    stride_list = [2,1,2,1,2,1,2,1,1,1,1,1,1,1]
    if dil_when_stride_removed:
      removed_stride_layer_idx = 12 
  else:   
    #for imagenet training
    stride_list = [2,1,2,1,2,1,2,1,1,1,1,1,2,1]

  num_dw_outputs =  [32, 32,  64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024]
  num_sep_outputs = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024,1024]

  num_dw_outputs = map(lambda x: int(round(x * wide_factor)), num_dw_outputs)
  num_sep_outputs = map(lambda x: int(round(x * wide_factor)), num_sep_outputs)
  
  ##################
  stage=1
  block_name = 'conv{}'.format(block_labels[0])
  isFrozen= block_name in freeze_layers
  dilation = 1
  op_layer_name = ConvBNLayerMobileNet(net, from_layer, block_name, bn_type=bn_type, use_relu=(not bn_preact),
      num_output=num_dw_outputs[0], kernel_size=3, pad=1, stride=stride_list[0],
      prePostFix=prePostFix, kwArgs=kwArgs,isFrozen=isFrozen,dilation=dilation,bn_preact=False)

  ##################
  num_stages = len(num_dw_outputs )
  #for num_dw_output,num_sep_output, block_label in zip(num_dw_outputs, num_sep_outputs, block_labels): 
  for stg_idx in range(1,num_stages):
    num_dw_output = num_dw_outputs[stg_idx] 
    num_sep_output = num_sep_outputs[stg_idx] 
    block_label = block_labels[stg_idx] 
    stride = stride_list[stg_idx]

    ip_layer_name = op_layer_name
    
    residual_block_end = ((stg_idx % 2) == 0)
    residual_block_start = (not residual_block_end)
    residual_stride = stride_list[stg_idx] * stride_list[stg_idx-1]    
    if residual_block_start:
      ip_layer_name_prev_stage = op_layer_name   
      num_output_prev_stage = num_sep_outputs[stg_idx-1]

    if ssd_mobile_chuanqi:
      block_name = 'conv{}_dw'.format(block_label)    
    else:  
      block_name = 'conv{}/dw'.format(block_label)
      
    isFrozen= block_name in freeze_layers
    op_layer_name = ConvBNLayerMobileNet(net, ip_layer_name, block_name, bn_type=bn_type, use_relu=(True if bn_preact else use_relu_intermediate),
        num_output=num_dw_output, kernel_size=3, pad=1, stride=stride,  prePostFix=prePostFix, 
        kwArgs=kwArgs,isFrozen=isFrozen, group=num_dw_output,dilation=dilation*removed_stride_fac,bn_preact=bn_preact)

    ip_layer_name = op_layer_name
    if ssd_mobile_chuanqi:
      block_name = 'conv{}_sep'.format(block_label)    
    else:  
      block_name = 'conv{}/sep'.format(block_label)    

    isFrozen= block_name in freeze_layers

    #have dilation for all layers after the layer where stride was removed
    if stg_idx >= removed_stride_layer_idx:
      removed_stride_fac = 2 
    op_layer_name = ConvBNLayerMobileNet(net, ip_layer_name, block_name, bn_type=bn_type, use_relu=(use_relu_intermediate if bn_preact else (not residual_block_end)),
        num_output=num_sep_output, kernel_size=1, pad=0, stride=1,  prePostFix=prePostFix, 
        kwArgs=kwArgs,isFrozen=isFrozen, group=1, dilation=dilation*removed_stride_fac,bn_preact=bn_preact)
          
    if residual_block_end:      
      if residual_stride != 1:
        op_layer_name_prev_stage = ip_layer_name_prev_stage + "/stride"
        kernel_size_stride = residual_stride #1 #kernel size 1 willl cause stride pooling/dropping - but we will need to use crop and its slow!
        net[op_layer_name_prev_stage] = L.Pooling(net[ip_layer_name_prev_stage], pool=P.Pooling.MAX, kernel_size=kernel_size_stride, stride=residual_stride)
        
        #output size is not by 2 when kernel size is 1 - remove the extra
        #why does crop layer slow down training?
        if kernel_size_stride == 1:
          net[ip_layer_name_prev_stage + "/crop"] = L.Crop(net[op_layer_name_prev_stage], net[op_layer_name])  
          op_layer_name_prev_stage = ip_layer_name_prev_stage + "/crop"          
      else:
        op_layer_name_prev_stage = ip_layer_name_prev_stage
          
      if num_sep_output > num_output_prev_stage:
        ip_layer_name = op_layer_name
        op_layer_name_res = ip_layer_name + "/slice0/res"     
        op_layer_name_res_concat = ip_layer_name + "/res"    
             
        op_layer_name_slice0 = ip_layer_name + "/slice0"     
        op_layer_name_slice1 = ip_layer_name + "/slice1"           
        tops = L.Slice(net[ip_layer_name], axis=1, slice_point=num_output_prev_stage, ntop=2)
        net[op_layer_name_slice0] = tops[0] 
        net[op_layer_name_slice1] = tops[1]
      
        ip_layer_name = op_layer_name_slice0    
        op_layer_name = ip_layer_name + "/res"
        net[op_layer_name_res] = L.Eltwise(net[op_layer_name_prev_stage], net[op_layer_name_slice0], operation=P.Eltwise.SUM)      
        net[op_layer_name_res_concat] = L.Concat(net[op_layer_name_res], net[op_layer_name_slice1])       
        op_layer_name = op_layer_name_res_concat
      else:  
        ip_layer_name = op_layer_name    
        op_layer_name = ip_layer_name + "/res"
        net[op_layer_name] = L.Eltwise(net[op_layer_name_prev_stage], net[ip_layer_name], operation=P.Eltwise.SUM)
        
    if (not bn_preact):
      net[op_layer_name+"/relu"] = L.ReLU(net[op_layer_name], in_place=True)
      op_layer_name = op_layer_name+"/relu"
    
  if bn_preact:
    net[op_layer_name+"/relu"] = L.ReLU(net[op_layer_name], in_place=True)
    op_layer_name = op_layer_name+"/relu"
    
  #--   
  # Add global pooling layer.
  from_layer = op_layer_name
  op_layer_name = 'pool6'
  net[op_layer_name] = L.Pooling(net[from_layer], pool=P.Pooling.AVE, global_pooling=True)
       
  from_layer = op_layer_name
  op_layer_name = 'fc7' #'fc'+str(num_output)
  kwargs = { 'num_output': num_output, 
    'param': [{'lr_mult': 1, 'decay_mult': 1}, {'lr_mult': 2, 'decay_mult': 0}], 
    'convolution_param': { 
       'kernel_size': 1,
       'weight_filler': { 'type': 'msra' }, 
    },
  }
  net[op_layer_name] = L.Convolution(net[from_layer], **kwargs)    
  
  return op_layer_name

#For ImageNet, we find that using pre-activation in networks with less than 100 layers 
#does not make any significant difference https://arxiv.org/pdf/1605.07146.pdf (WideResNet paper)
BN_PREACT_DEFAULT=False
###############################################################
def mobileresnet(net, from_layer='data', fully_conv=False, reduced=False, dilated=False,
        dropout=True, freeze_layers=None, bn_type='bvlc', bn_at_start=True, caffe_fork='nvidia',
        training_type='ImageNet', depth_mul=1, ssd_mobile_chuanqi=False, dil_when_stride_removed=False,
        num_output=1000, bn_preact=BN_PREACT_DEFAULT, wide_factor=1.0):
  return MobileResNetBody(net, from_layer, fully_conv, reduced, dilated,
        dropout, freeze_layers, bn_type, bn_at_start, caffe_fork,
        training_type, depth_mul, ssd_mobile_chuanqi, dil_when_stride_removed,
        num_output, wide_factor=wide_factor, bn_preact=bn_preact)

        
        
