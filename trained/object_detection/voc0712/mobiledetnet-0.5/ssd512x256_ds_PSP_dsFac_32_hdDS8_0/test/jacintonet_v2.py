from __future__ import print_function
import caffe
from models.model_libs import *

def jacintonet11_base(net, from_layer=None, use_batchnorm=True, use_relu=True, num_output=1000, stride_list=None, dilation_list=None, freeze_layers=None, in_place=True):  
   #Top and Bottom blobs must be different for NVCaffe BN caffe-0.15 (in_place=False), but no such constraint for caffe-0.16
   
   #--     
   stage=0
   stride=stride_list[stage]
   dilation=dilation_list[stage]   
   
   out_layer = 'conv1a'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=32, kernel_size=[5,5], pad=2*dilation, stride=stride, group=1, dilation=dilation, in_place=in_place)  
   
   from_layer = out_layer
   out_layer = 'conv1b'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=32, kernel_size=[3,3], pad=dilation, stride=1, group=4, dilation=dilation, in_place=in_place)       
   
   #--     
   stage=1
   stride=stride_list[stage]
   dilation=dilation_list[stage]   
        
   pooling_param = {'pool':P.Pooling.MAX, 'kernel_size':stride, 'stride':stride}   
   from_layer = out_layer
   out_layer = 'pool1'
   net[out_layer] = L.Pooling(net[from_layer], pooling_param=pooling_param)       
      
   from_layer = out_layer
   out_layer = 'res2a_branch2a'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=64, kernel_size=[3,3], pad=dilation, stride=1, group=1, dilation=dilation, in_place=in_place)   
   
   from_layer = out_layer
   out_layer = 'res2a_branch2b'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=64, kernel_size=[3,3], pad=dilation, stride=1, group=4, dilation=dilation, in_place=in_place)     
     
   #--     
   stage=2
   stride=stride_list[stage]
   dilation=dilation_list[stage] 
        
   pooling_param = {'pool':P.Pooling.MAX, 'kernel_size':stride, 'stride':stride}       
   from_layer = out_layer
   out_layer = 'pool2'
   net[out_layer] = L.Pooling(net[from_layer], pooling_param=pooling_param)     
         
   from_layer = out_layer
   out_layer = 'res3a_branch2a'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=128, kernel_size=[3,3], pad=dilation, stride=1, group=1, dilation=dilation, in_place=in_place)   
   
   from_layer = out_layer
   out_layer = 'res3a_branch2b'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=128, kernel_size=[3,3], pad=dilation, stride=1, group=4, dilation=dilation, in_place=in_place)    
   
   #--       
   stage=3
   stride=stride_list[stage]
   dilation=dilation_list[stage] 
      
   pooling_param = {'pool':P.Pooling.MAX, 'kernel_size':stride, 'stride':stride}   
   from_layer = out_layer
   out_layer = 'pool3'
   net[out_layer] = L.Pooling(net[from_layer], pooling_param=pooling_param)         
         
   from_layer = out_layer
   out_layer = 'res4a_branch2a'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=256, kernel_size=[3,3], pad=dilation, stride=1, group=1, dilation=dilation, in_place=in_place)   
   
   from_layer = out_layer 
   out_layer = 'res4a_branch2b'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=256, kernel_size=[3,3], pad=dilation, stride=1, group=4, dilation=dilation, in_place=in_place)        
      
   #--      
   stage=4
   stride=stride_list[stage]
   dilation=dilation_list[stage]         
   
   pooling_param = {'pool':P.Pooling.MAX, 'kernel_size':stride, 'stride':stride}   
   from_layer = out_layer
   out_layer = 'pool4'
   net[out_layer] = L.Pooling(net[from_layer], pooling_param=pooling_param)     

   from_layer = out_layer
   out_layer = 'res5a_branch2a'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=512, kernel_size=[3,3], pad=dilation, stride=1, group=1, dilation=dilation, in_place=in_place)   
   
   from_layer = out_layer
   out_layer = 'res5a_branch2b'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=512, kernel_size=[3,3], pad=dilation, stride=1, group=4, dilation=dilation, in_place=in_place) 
   
   return out_layer
  
  
def jacintonet11(net, from_layer=None, use_batchnorm=True, use_relu=True, num_output=1000, stride_list=None, dilation_list=None, freeze_layers=None):  
   if stride_list == None:
     stride_list = [2,2,2,2,2]
   if dilation_list == None:
     dilation_list = [1,1,1,1,1]
   
   out_layer = jacintonet11_base(net, from_layer=from_layer, use_batchnorm=use_batchnorm, use_relu=use_relu, \
      num_output=num_output, stride_list=stride_list, dilation_list=dilation_list, freeze_layers=freeze_layers)
   
   #--   
   # Add global pooling layer.
   from_layer = out_layer
   out_layer = 'pool5'
   net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.AVE, global_pooling=True)
       
   from_layer = out_layer 
   out_layer = 'fc'+str(num_output)
   kwargs = { 'num_output': num_output, 
     'param': [{'lr_mult': 1, 'decay_mult': 1}, {'lr_mult': 2, 'decay_mult': 0}], 
     'inner_product_param': { 
         'weight_filler': { 'type': 'msra' }, 
         'bias_filler': { 'type': 'constant', 'value': 0 }   
     },
   }
   net[out_layer] = L.InnerProduct(net[from_layer], **kwargs)    
   
   return out_layer
   
   
def jsegnet21(net, from_layer=None, use_batchnorm=True, use_relu=True, num_output=20, stride_list=None, dilation_list=None, freeze_layers=None, 
   upsample=True): 
   in_place = True
   if stride_list == None:
     stride_list = [2,2,2,2,1]
   if dilation_list == None:
     dilation_list = [1,1,1,1,2]

   out_layer = jacintonet11_base(net, from_layer=from_layer, use_batchnorm=use_batchnorm, use_relu=use_relu, \
      num_output=num_output, stride_list=stride_list, dilation_list=dilation_list, freeze_layers=freeze_layers)
   
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


def jdetnet21(net, from_layer=None, use_batchnorm=True, use_relu=True, num_output=20, stride_list=None, dilation_list=None, freeze_layers=None, 
   upsample=False, num_intermediate=512, output_stride=32): 
   eltwise_final = False
   if stride_list == None:
     stride_list = [2,2,2,2,1]
   if dilation_list == None:
     dilation_list = [1,1,1,1,2]

   out_layer = jacintonet11_base(net, from_layer=from_layer, use_batchnorm=use_batchnorm, use_relu=use_relu, \
      num_output=num_output, stride_list=stride_list, dilation_list=dilation_list, freeze_layers=freeze_layers)
      
   #---------------------------     
   #PSP style pool down
   pooling_param = {'pool':P.Pooling.MAX, 'kernel_size':3, 'stride':2, 'pad':1}      
   from_layer = out_layer
   out_layer = 'pool6'
   net[out_layer] = L.Pooling(net[from_layer], pooling_param=pooling_param) 
   
   #--
   pooling_param = {'pool':P.Pooling.MAX, 'kernel_size':3, 'stride':2, 'pad':1}      
   from_layer = out_layer
   out_layer = 'pool7'
   net[out_layer] = L.Pooling(net[from_layer], pooling_param=pooling_param)  
   #--
   pooling_param = {'pool':P.Pooling.MAX, 'kernel_size':3, 'stride':2, 'pad':1}      
   from_layer = out_layer
   out_layer = 'pool8'
   net[out_layer] = L.Pooling(net[from_layer], pooling_param=pooling_param)  

   
   #---------------------------        
   out_layer_names = []
  
   from_layer = 'res5a_branch2b/relu'
   out_layer = 'ctx_output1'
   out_layer = ConvBNLayerSSD(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=num_intermediate, kernel_size=[1,1], pad=0, stride=1, group=1, dilation=1)              
   out_layer_names += [out_layer]
   
   from_layer = 'pool6'
   out_layer = 'ctx_output2'
   out_layer = ConvBNLayerSSD(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=num_intermediate, kernel_size=[1,1], pad=0, stride=1, group=1, dilation=1)        
   out_layer_names += [out_layer]
   
   from_layer = 'pool7'
   out_layer = 'ctx_output3'
   out_layer = ConvBNLayerSSD(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=num_intermediate, kernel_size=[1,1], pad=0, stride=1, group=1, dilation=1)        
   out_layer_names += [out_layer]
   
   from_layer = 'pool8'
   out_layer = 'ctx_output4'
   out_layer = ConvBNLayerSSD(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=num_intermediate, kernel_size=[1,1], pad=0, stride=1, group=1, dilation=1)        
   out_layer_names += [out_layer]
   
   return out_layer, out_layer_names

def jdetnet21_s8(net, from_layer=None, use_batchnorm=True, use_relu=True, num_output=20, stride_list=None, dilation_list=None, freeze_layers=None, 
   upsample=False, num_intermediate=512, output_stride=32): 
   eltwise_final = False
   if stride_list == None:
     stride_list = [2,2,2,2,2] #[2,2,2,2,1]
   if dilation_list == None:
     dilation_list = [1,1,1,1,1] #[1,1,1,1,2]

   out_layer = jacintonet11_base(net, from_layer=from_layer, use_batchnorm=use_batchnorm, use_relu=use_relu, \
      num_output=num_output, stride_list=stride_list, dilation_list=dilation_list, freeze_layers=freeze_layers)

   from_layer = 'res5a_branch2b/relu'
   out_layer = 'conv6'
   out_layer = ConvBNLayerSSD(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=1024, kernel_size=[3,3], pad=1, stride=1, group=1, dilation=1)
      
   from_layer = out_layer
   out_layer = 'conv7'
   out_layer = ConvBNLayerSSD(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=1024, kernel_size=[1,1], pad=0, stride=1, group=1, dilation=1)

   #---------------------------       
   out_layer = 'res3a_branch2b/concat'    
   net[out_layer] = L.Concat(net['res3a_branch2a/relu'], net['res3a_branch2b/relu'])  
         
   out_layer = 'res4a_branch2b/concat'    
   net[out_layer] = L.Concat(net['res4a_branch2a/relu'], net['res4a_branch2b/relu'])  
            
   #out_layer = 'res5a_branch2b/concat'    
   #net[out_layer] = L.Concat(net['res5a_branch2a/relu'], net['res5a_branch2b/relu'])  
               
   #---------------------------     
   #PSP style pool down
   pooling_param = {'pool':P.Pooling.MAX, 'kernel_size':3, 'stride':2, 'pad':1}      
   from_layer = 'conv7/relu' #'res5a_branch2b/concat' #
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

   #---------------------------       
   out_layer_names = []
   
   from_layer = 'res3a_branch2b/concat'
   out_layer = 'ctx_output1'
   out_layer = ConvBNLayerSSD(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=num_intermediate, kernel_size=[1,1], pad=0, stride=1, group=1, dilation=1)  
   out_layer_names += [out_layer]
   
   from_layer = 'res4a_branch2b/concat'
   out_layer = 'ctx_output2'
   out_layer = ConvBNLayerSSD(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=num_intermediate, kernel_size=[1,1], pad=0, stride=1, group=1, dilation=1) 
   out_layer_names += [out_layer]
   
   from_layer = 'conv7/relu' #'res5a_branch2b/concat' #
   out_layer = 'ctx_output3'
   out_layer = ConvBNLayerSSD(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=num_intermediate, kernel_size=[1,1], pad=0, stride=1, group=1, dilation=1)              
   out_layer_names += [out_layer]
   
   from_layer = 'pool6'
   out_layer = 'ctx_output4'
   out_layer = ConvBNLayerSSD(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=num_intermediate, kernel_size=[1,1], pad=0, stride=1, group=1, dilation=1)        
   out_layer_names += [out_layer]
   
   from_layer = 'pool7'
   out_layer = 'ctx_output5'
   out_layer = ConvBNLayerSSD(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=num_intermediate, kernel_size=[1,1], pad=0, stride=1, group=1, dilation=1)        
   out_layer_names += [out_layer]
   
   from_layer = 'pool8'
   out_layer = 'ctx_output6'
   out_layer = ConvBNLayerSSD(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=num_intermediate, kernel_size=[1,1], pad=0, stride=1, group=1, dilation=1)        
   out_layer_names += [out_layer]
   
   return out_layer, out_layer_names

#To match configuration used by original SSD script
def ssdJacintoNetV2(net, from_layer=None, use_batchnorm=True, use_relu=True, num_output=20, stride_list=None, dilation_list=None, freeze_layers=None, 
   upsample=False, num_intermediate=512, output_stride=32, use_batchnorm_mbox=True, ds_type='PSP', fully_conv_at_end=True, reg_head_at_ds8=True, 
   concat_reg_head=False, base_nw_3_head=False, first_hd_same_op_ch=False,
   rhead_name_non_linear=False, chop_num_heads=0): 
   
   eltwise_final = False
   if stride_list == None:
     stride_list = [2,2,2,2,1]
   if dilation_list == None:
     dilation_list = [1,1,1,1,2]

   out_layer = jacintonet11_base(net, from_layer=from_layer, use_batchnorm=use_batchnorm, use_relu=use_relu, \
      num_output=num_output, stride_list=stride_list, dilation_list=dilation_list, freeze_layers=freeze_layers)

   last_base_layer_name = out_layer

   if concat_reg_head:
     out_layer = 'res4a_branch2b/concat'    
     net[out_layer] = L.Concat(net['res4a_branch2a/relu'], net['res4a_branch2b/relu']) 
                           
     out_layer = 'res5a_branch2b/concat'    
     net[out_layer] = L.Concat(net['res5a_branch2a/relu'], net['res5a_branch2b/relu'])  
     last_base_layer_name = out_layer

   if fully_conv_at_end:
     from_layer = last_base_layer_name 
     out_layer = 'fc6'
     out_layer = ConvBNLayerSSD(net, from_layer, out_layer, use_batchnorm_mbox, use_relu, num_output=1024, kernel_size=[3,3], pad=6, stride=1, group=1, dilation=6)
        
     from_layer = out_layer
     out_layer = 'fc7'
     out_layer = ConvBNLayerSSD(net, from_layer, out_layer, use_batchnorm_mbox, use_relu, num_output=1024, kernel_size=[1,1], pad=0, stride=1, group=1, dilation=1)
     last_base_layer_name = out_layer
      
   #---------------------------     
   out_layer_names = []
  
   #PSP style pool down
   if ds_type == 'PSP':
     if chop_num_heads < 4:
       pooling_param = {'pool':P.Pooling.MAX, 'kernel_size':2, 'stride':2, 'pad':0}      
       from_layer = out_layer
       out_layer = 'pool6'
       net[out_layer] = L.Pooling(net[from_layer], pooling_param=pooling_param) 
     
     #--
     if chop_num_heads < 3:
       pooling_param = {'pool':P.Pooling.MAX, 'kernel_size':2, 'stride':2, 'pad':0}      
       from_layer = out_layer
       out_layer = 'pool7'
       net[out_layer] = L.Pooling(net[from_layer], pooling_param=pooling_param)  

     #--
     if chop_num_heads < 2:
       pooling_param = {'pool':P.Pooling.MAX, 'kernel_size':2, 'stride':2, 'pad':0}      
       from_layer = out_layer
       out_layer = 'pool8'
       net[out_layer] = L.Pooling(net[from_layer], pooling_param=pooling_param)  

     #--
     if chop_num_heads < 1:
       pooling_param = {'pool':P.Pooling.MAX, 'kernel_size':2, 'stride':2, 'pad':0}      
       from_layer = out_layer
       out_layer = 'pool9'
       net[out_layer] = L.Pooling(net[from_layer], pooling_param=pooling_param)  

     #--
     if stride_list[4] == 1:
       pooling_param = {'pool':P.Pooling.MAX, 'kernel_size':2, 'stride':1, 'pad':0}      
       from_layer = out_layer
       out_layer = 'pool10'
       net[out_layer] = L.Pooling(net[from_layer], pooling_param=pooling_param)  
   
     #mbox_source_layers = ['res3a_branch2b_relu', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2','conv10_2']
     if rhead_name_non_linear:
       reg_head_idx=6
     else:
       reg_head_idx=1

     if reg_head_at_ds8:
       from_layer = 'res3a_branch2b/relu'
     else: 
       if concat_reg_head:
         from_layer = 'res4a_branch2b/concat'
       else:  
         from_layer = 'res4a_branch2b/relu'

     out_layer = 'ctx_output{}'.format(reg_head_idx)

     if rhead_name_non_linear:
       reg_head_idx=0

     reg_head_idx += 1
     out_layer = ConvBNLayerSSD(net, from_layer, out_layer, use_batchnorm_mbox, use_relu, num_output=num_intermediate>>first_hd_same_op_ch, kernel_size=[1,1], pad=0, stride=1, group=1, dilation=1)              
     out_layer_names += [out_layer]
     
     from_layer = last_base_layer_name 
     out_layer = 'ctx_output{}'.format(reg_head_idx)
     reg_head_idx += 1
     out_layer = ConvBNLayerSSD(net, from_layer, out_layer, use_batchnorm_mbox, use_relu, num_output=num_intermediate>>1, kernel_size=[1,1], pad=0, stride=1, group=1, dilation=1)        
     out_layer_names += [out_layer]
    
     if chop_num_heads < 4:
       from_layer = 'pool6'
       out_layer = 'ctx_output{}'.format(reg_head_idx)
       reg_head_idx += 1
       out_layer = ConvBNLayerSSD(net, from_layer, out_layer, use_batchnorm_mbox, use_relu, num_output=num_intermediate>>1, kernel_size=[1,1], pad=0, stride=1, group=1, dilation=1)        
       out_layer_names += [out_layer]
    
     if chop_num_heads < 3:
       from_layer = 'pool7'
       out_layer = 'ctx_output{}'.format(reg_head_idx)
       reg_head_idx += 1
       out_layer = ConvBNLayerSSD(net, from_layer, out_layer, use_batchnorm_mbox, use_relu, num_output=num_intermediate>>1, kernel_size=[1,1], pad=0, stride=1, group=1, dilation=1)        
       out_layer_names += [out_layer]
    
     if chop_num_heads < 2:
       from_layer = 'pool8'
       out_layer = 'ctx_output{}'.format(reg_head_idx)
       reg_head_idx += 1
       out_layer = ConvBNLayerSSD(net, from_layer, out_layer, use_batchnorm_mbox, use_relu, num_output=num_intermediate>>1, kernel_size=[1,1], pad=0, stride=1, group=1, dilation=1)        
       out_layer_names += [out_layer]
    
     if chop_num_heads < 1:
       from_layer = 'pool9'
       out_layer = 'ctx_output{}'.format(reg_head_idx)
       reg_head_idx += 1
       out_layer = ConvBNLayerSSD(net, from_layer, out_layer, use_batchnorm_mbox, use_relu, num_output=num_intermediate>>1, kernel_size=[1,1], pad=0, stride=1, group=1, dilation=1)        
       out_layer_names += [out_layer]
    
     if stride_list[4] == 1:
       from_layer = 'pool10'
       out_layer = 'ctx_output{}'.format(reg_head_idx)
       reg_head_idx += 1
       out_layer = ConvBNLayerSSD(net, from_layer, out_layer, use_batchnorm_mbox, use_relu, num_output=num_intermediate>>1, kernel_size=[1,1], pad=0, stride=1, group=1, dilation=1)        
       out_layer_names += [out_layer]
    
     if base_nw_3_head:
       #by default 2 heads are connected in base n/w are at res3a and res5a (at the end of base n/w)
       from_layer = 'res4a_branch2b/relu'
       out_layer = 'ctx_output{}'.format(reg_head_idx)
       reg_head_idx += 1
       out_layer = ConvBNLayerSSD(net, from_layer, out_layer, use_batchnorm_mbox, use_relu, num_output=num_intermediate>>1, kernel_size=[1,1], pad=0, stride=1, group=1, dilation=1)        
       out_layer_names += [out_layer]
    
     return out_layer, out_layer_names
   else:
     out_layer_names += ['res3a_branch2b/relu']
     if fully_conv_at_end:
       out_layer_names += ['fc7']
     
     ssd_size = '512x512'
     training_type = 'SSD'
     if (ssd_size == '512x512') and (training_type == 'SSD'):
                       #32x32    #16x16    #8x8      #4x4      #2x2
       num_outputs =  [256, 512, 128, 256, 128, 256, 128, 256, 128, 256,]
       kernel_sizes = [  1,   3,   1,   3,   1,   3,   1,   3,   1,   4,]
       pads =         [  0,   1,   0,   1,   0,   1,   0,   1,   0,   1,]
       strides=       [  1,   2,   1,   2,   1,   2,   1,   2,   1,   1,]
     elif (ssd_size == '300x300') and (training_type == 'SSD'):
                      #19x19     #10x10    #5x5      #3x3
       num_outputs =  [256, 512, 128, 256, 128, 256, 128, 256,]
       kernel_sizes = [  1,   3,   1,   3,   1,   3,   1,   3,]
       pads =         [  0,   1,   0,   1,   0,   0,   0,   0,]
       strides=       [  1,   2,   1,   2,   1,   1,   1,   1,]
     elif (ssd_size == '256x256') and (training_type == 'SSD'):
                      #16x16     #8x8    #8x8      #4x4
       num_outputs =  [256, 512, 128, 256, 128, 256, 128, 256,]
       kernel_sizes = [  1,   3,   1,   3,   1,   3,   1,   3,]
       pads =         [  0,   1,   0,   1,   0,   0,   0,   0,]
       strides=       [  1,   2,   1,   1,   1,   1,   1,   1,]
     elif (ssd_size == '512x512') and (training_type == 'IMGNET'):
                       #16x16    #8x8      #4x4      #2x2
       num_outputs =  [256, 512, 128, 256, 128, 256, 128, 256,]
       kernel_sizes = [  1,   3,   1,   3,   1,   3,   1,   3,]
       pads =         [  0,   1,   0,   1,   0,   1,   0,   1,]
       strides=       [  1,   2,   1,   2,   1,   2,   1,   2,]
     elif (ssd_size == '300x300') and (training_type == 'IMGNET'):
                       #10x10    #5x5      #3x3       #2x2   
       num_outputs =  [256, 512, 128, 256, 128, 256,  64, 128,]
       kernel_sizes = [  1,   3,   1,   3,   1,   3,   1,   3,]
       pads =         [  0,   1,   0,   1,   0,   1,   0,   1,]
       strides=       [  1,   2,   1,   2,   1,   2,   1,   2,]

     # index of first additional layer after base network
     first_idx = 6
     from_layer = net.keys()[-1]
     blk_idx = first_idx
     lr_mult = 1 
     bn_postfix='/bn'
     scale_postfix='/scale'
     print("num_outputs: ", num_outputs) 
     for idx in range (0, len(num_outputs)):
       print("blk_index: ", blk_idx)
       # TODO(weiliu89): Construct the name using the last layer to avoid duplication.
       one_or_two = (idx%2) + 1
       out_layer = "conv{}_{}".format(blk_idx, one_or_two)
       ConvBNLayer(net, from_layer, out_layer, use_batchnorm_mbox, use_relu, num_outputs[idx],
           kernel_sizes[idx], pads[idx], strides[idx], lr_mult=lr_mult, bn_postfix=bn_postfix,
           scale_postfix=scale_postfix)
       out_layer_names += [out_layer]
       from_layer = out_layer
       if one_or_two == 2:
         blk_idx = blk_idx + 1

     return out_layer, out_layer_names
   #---------------------------       

   
   
def jdetnet21_fpn(net, from_layer=None, use_batchnorm=True, use_relu=True, num_output=20, stride_list=None, dilation_list=None, freeze_layers=None, 
   upsample=False, num_intermediate=512, output_stride=32): 
   in_place = True   
   eltwise_final = False
   if stride_list == None:
     stride_list = [2,2,2,2,2]
   if dilation_list == None:
     dilation_list = [1,1,1,1,1]

   out_layer = jacintonet11_base(net, from_layer=from_layer, use_batchnorm=use_batchnorm, use_relu=use_relu, \
      num_output=num_output, stride_list=stride_list, dilation_list=dilation_list, freeze_layers=freeze_layers)
   
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
   pooling_param = {'pool':P.Pooling.MAX, 'kernel_size':3, 'stride':1, 'pad':1}      
   from_layer = out_layer
   out_layer = 'pool8'
   net[out_layer] = L.Pooling(net[from_layer], pooling_param=pooling_param)  

   #---------------------------   
   #FPN layers start here  
   #---------------------------  512->256  
   from_layer = out_layer
   out_layer = 'pool8_upconv'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=256, kernel_size=[3,3], pad=1, stride=1, group=1, dilation=1, in_place=in_place) 
   
   out_layer_1x1 = 'pool7_1x1'   
   out_layer_1x1 = ConvBNLayer(net, 'pool7', out_layer_1x1, use_batchnorm, use_relu, num_output=256, kernel_size=[1,1], pad=0, stride=1, group=1, dilation=1, in_place=in_place)    
            
   from_layer = out_layer   
   out_layer = 'pool7_plus'
   net[out_layer] = L.Eltwise(net[out_layer_1x1], net[from_layer])
              
   #---------------------------  512->256  
   from_layer = out_layer
   out_layer = 'pool7_upconv'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=256, kernel_size=[3,3], pad=1, stride=1, group=1, dilation=1, in_place=in_place) 
      
   #frozen upsampling layer
   from_layer = out_layer 
   out_layer = 'pool7_up2'  
   deconv_kwargs = {  'param': { 'lr_mult': 0, 'decay_mult': 0 },
       'convolution_param': { 'num_output': 256, 'bias_term': False, 'pad': 1, 'kernel_size': 4, 'group': 256, 'stride': 2, 
       'weight_filler': { 'type': 'bilinear' } } }
   net[out_layer] = L.Deconvolution(net[from_layer], **deconv_kwargs)   
            
   out_layer_1x1 = 'pool6_1x1'   
   out_layer_1x1 = ConvBNLayer(net, 'pool6', out_layer_1x1, use_batchnorm, use_relu, num_output=256, kernel_size=[1,1], pad=0, stride=1, group=1, dilation=1, in_place=in_place)     
            
   from_layer = out_layer   
   out_layer = 'pool6_plus'
   net[out_layer] = L.Eltwise(net[out_layer_1x1], net[from_layer])
          
   #---------------------------  512->256   
   from_layer = out_layer
   out_layer = 'pool6_upconv'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=256, kernel_size=[3,3], pad=1, stride=1, group=1, dilation=1, in_place=in_place) 
      
   #frozen upsampling layer
   from_layer = out_layer 
   out_layer = 'pool6_up2'  
   deconv_kwargs = {  'param': { 'lr_mult': 0, 'decay_mult': 0 },
       'convolution_param': { 'num_output': 256, 'bias_term': False, 'pad': 1, 'kernel_size': 4, 'group': 256, 'stride': 2, 
       'weight_filler': { 'type': 'bilinear' } } }
   net[out_layer] = L.Deconvolution(net[from_layer], **deconv_kwargs)   
            
   out_layer_1x1 = 'res5a_branch2b_1x1'   
   out_layer_1x1 = ConvBNLayer(net, 'res5a_branch2b/relu', out_layer_1x1, use_batchnorm, use_relu, num_output=256, kernel_size=[1,1], pad=0, stride=1, group=1, dilation=1, in_place=in_place)         
        
   from_layer = out_layer   
   out_layer = 'res5a_branch2b_plus'
   net[out_layer] = L.Eltwise(net[out_layer_1x1], net[from_layer])
              
   #---------------------------   256->256 
   from_layer = out_layer
   out_layer = 'res5a_branch2b_upconv'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=256, kernel_size=[3,3], pad=1, stride=1, group=1, dilation=1, in_place=in_place) 
      
   #frozen upsampling layer
   from_layer = out_layer 
   out_layer = 'res5a_branch2a_up2'  
   deconv_kwargs = {  'param': { 'lr_mult': 0, 'decay_mult': 0 },
       'convolution_param': { 'num_output': 256, 'bias_term': False, 'pad': 1, 'kernel_size': 4, 'group': 256, 'stride': 2, 
       'weight_filler': { 'type': 'bilinear' } } }
   net[out_layer] = L.Deconvolution(net[from_layer], **deconv_kwargs)   
       
   out_layer_1x1 = 'res4a_branch2b_1x1'   
   out_layer_1x1 = ConvBNLayer(net, 'res4a_branch2b/relu', out_layer_1x1, use_batchnorm, use_relu, num_output=256, kernel_size=[1,1], pad=0, stride=1, group=1, dilation=1, in_place=in_place)  
                         
   from_layer = out_layer   
   out_layer = 'res4a_branch2b_plus'
   net[out_layer] = L.Eltwise(net[out_layer_1x1], net[from_layer])
          
   #---------------------------   256->256 
   from_layer = out_layer
   out_layer = 'res4a_branch2b_upconv'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=256, kernel_size=[3,3], pad=1, stride=1, group=1, dilation=1, in_place=in_place)    
       
   out_layer_1x1 = 'res4a_branch2a_1x1'   
   out_layer_1x1 = ConvBNLayer(net, 'res4a_branch2a/relu', out_layer_1x1, use_batchnorm, use_relu, num_output=256, kernel_size=[1,1], pad=0, stride=1, group=1, dilation=1, in_place=in_place)   
                      
   from_layer = out_layer   
   out_layer = 'res4a_branch2a_plus'
   net[out_layer] = L.Eltwise(net[out_layer_1x1], net[from_layer])
                              
   #---------------------------  
   out_layer_names = []
   
   #SSD heads start here
   from_layer = 'res4a_branch2a_plus'
   out_layer = 'ctx_output1'
   out_layer = ConvBNLayerSSD(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=num_intermediate, kernel_size=[1,1], pad=0, stride=1, group=1, dilation=1)   
   out_layer_names += [out_layer]

   from_layer = 'res4a_branch2b_plus'
   out_layer = 'ctx_output2'
   out_layer = ConvBNLayerSSD(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=num_intermediate, kernel_size=[1,1], pad=0, stride=1, group=1, dilation=1)          
   out_layer_names += [out_layer]
   
   from_layer = 'res5a_branch2b_plus'
   out_layer = 'ctx_output3'
   out_layer = ConvBNLayerSSD(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=num_intermediate, kernel_size=[1,1], pad=0, stride=1, group=1, dilation=1)        
   out_layer_names += [out_layer]
   
   from_layer = 'pool6_plus'
   out_layer = 'ctx_output4'
   out_layer = ConvBNLayerSSD(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=num_intermediate, kernel_size=[1,1], pad=0, stride=1, group=1, dilation=1)        
   out_layer_names += [out_layer]
   
   from_layer = 'pool7_plus'
   out_layer = 'ctx_output5'
   out_layer = ConvBNLayerSSD(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=num_intermediate, kernel_size=[1,1], pad=0, stride=1, group=1, dilation=1)        
   out_layer_names += [out_layer]
   
   from_layer = 'pool8'
   out_layer = 'ctx_output6'
   out_layer = ConvBNLayerSSD(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=num_intermediate, kernel_size=[1,1], pad=0, stride=1, group=1, dilation=1)        
   out_layer_names += [out_layer]
   
   return out_layer, out_layer_names

