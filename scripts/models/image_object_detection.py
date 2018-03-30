from __future__ import print_function
import caffe
from google.protobuf import text_format
import ast
from models.model_libs import *
import models.jacintonet_v2
import models.mobilenet
import numpy as np
import math
import os
import shutil
import stat
import subprocess
import sys
import argparse
from collections import OrderedDict


def set_min_max_sizes(config_param):
  # in percent %
  if config_param.ssd_size == '512x512':
    min_ratio = 15
  elif (config_param.ssd_size == '300x300') or(config_param.ssd_size == '256x256'):
    min_ratio = 20
  
  if config_param.small_objs: 
    min_ratio = min_ratio - 5
   
  max_ratio = 90
  step = int(math.floor((max_ratio - min_ratio) / (config_param.num_steps - 2)))

  min_sizes = []
  max_sizes = []

  for ratio in xrange(min_ratio, max_ratio + 1, step):
    min_sizes.append(config_param.min_dim * ratio / 100.)
    max_sizes.append(config_param.min_dim * (ratio + step) / 100.)

  print('ratio_step_size:', step)   
  
  if config_param.ssd_size == '512x512':
    if config_param.small_objs:
      min_size_mul = 4 
      max_size_mul = 10
    else:  
      min_size_mul = 7
      max_size_mul = 15 
  elif (config_param.ssd_size == '300x300') or (config_param.ssd_size == '256x256'):
    if config_param.small_objs:
      min_size_mul = 7
      max_size_mul = 15 
    else:
      min_size_mul = 10
      max_size_mul = 20
  
  min_sizes = [config_param.min_dim * min_size_mul / 100.] + min_sizes
  max_sizes = [config_param.min_dim * max_size_mul / 100.] + max_sizes
  
  #print('min_sizes:', min_sizes)   
  #print('max_sizes:', max_sizes)  

  return min_sizes, max_sizes  

def CreateAnnotatedDataLayer(source, batch_size=32, backend=P.Data.LMDB,
        output_label=True, train=True, label_map_file='', anno_type=None,
        transform_param={}, batch_sampler=[{}], threads=1):
    if train:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
                'transform_param': transform_param,
                }
    else:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
                'transform_param': transform_param,
                }
    ntop = 1
    if output_label:
        ntop = 2
    annotated_data_param = {
        'label_map_file': label_map_file,
        'batch_sampler': batch_sampler,
        }
    if anno_type is not None:
        annotated_data_param.update({'anno_type': anno_type})
    return L.AnnotatedData(name="data", annotated_data_param=annotated_data_param,
        data_param=dict(batch_size=batch_size, backend=backend, source=source, parser_threads=threads, threads=threads),
        ntop=ntop, **kwargs)
        
def CreateMultiBoxHead(net, data_layer="data", num_classes=[], from_layers=[],
        use_objectness=False, normalizations=[], use_batchnorm=True, lr_mult=1,
        use_scale=True, min_sizes=[], max_sizes=[], prior_variance = [0.1],
        aspect_ratios=[], steps=[], img_height=0, img_width=0, share_location=True,
        flip=True, clip=True, offset=0.5, inter_layer_depth=[], kernel_size=1, pad=0,
        conf_postfix='', loc_postfix='', **bn_param):
    assert num_classes, "must provide num_classes"
    assert num_classes > 0, "num_classes must be positive number"
    if normalizations:
        assert len(from_layers) == len(normalizations), "from_layers and normalizations should have same length"
    assert len(from_layers) == len(min_sizes), "from_layers and min_sizes should have same length"
    if max_sizes:
        assert len(from_layers) == len(max_sizes), "from_layers and max_sizes should have same length"
    if aspect_ratios:
        assert len(from_layers) == len(aspect_ratios), "from_layers and aspect_ratios should have same length"
    if steps:
        assert len(from_layers) == len(steps), "from_layers and steps should have same length"
    net_layers = net.keys()
    assert data_layer in net_layers, "data_layer is not in net's layers"
    if inter_layer_depth:
        assert len(from_layers) == len(inter_layer_depth), "from_layers and inter_layer_depth should have same length"

    num = len(from_layers)
    priorbox_layers = []
    loc_layers = []
    conf_layers = []
    objectness_layers = []
    for i in range(0, num):
        from_layer = from_layers[i]

        # Get the normalize value.
        if normalizations:
            if normalizations[i] != -1:
                norm_name = "{}_norm".format(from_layer)
                net[norm_name] = L.Normalize(net[from_layer], scale_filler=dict(type="constant", value=normalizations[i]),
                    across_spatial=False, channel_shared=False)
                from_layer = norm_name

        # Add intermediate layers.
        if inter_layer_depth:
            if inter_layer_depth[i] > 0:
                inter_name = "{}_inter".format(from_layer)
                ConvBNLayerSSD(net, from_layer, inter_name, use_bn=use_batchnorm, use_relu=True, lr_mult=lr_mult,
                      num_output=inter_layer_depth[i], kernel_size=3, pad=1, stride=1, use_scale=use_scale, **bn_param)
                from_layer = inter_name

        # Estimate number of priors per location given provided parameters.
        min_size = min_sizes[i]
        if type(min_size) is not list:
            min_size = [min_size]
        aspect_ratio = []
        if len(aspect_ratios) > i:
            aspect_ratio = aspect_ratios[i]
            if type(aspect_ratio) is not list:
                aspect_ratio = [aspect_ratio]
        max_size = []
        if len(max_sizes) > i:
            max_size = max_sizes[i]
            if type(max_size) is not list:
                max_size = [max_size]
            if max_size:
                assert len(max_size) == len(min_size), "max_size and min_size should have same length."
        if max_size:
            num_priors_per_location = (2 + len(aspect_ratio)) * len(min_size)
        else:
            num_priors_per_location = (1 + len(aspect_ratio)) * len(min_size)
        if flip:
            num_priors_per_location += len(aspect_ratio) * len(min_size)
        step = []
        if len(steps) > i:
            step = steps[i]

        # Create location prediction layer.
        name = "{}_mbox_loc{}".format(from_layer, loc_postfix)
        num_loc_output = num_priors_per_location * 4;
        if not share_location:
            num_loc_output *= num_classes
       
        dilation=1
        pad = int((kernel_size + (dilation - 1) * (kernel_size - 1)) - 1) / 2
        ConvBNLayerSSD(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
            num_output=num_loc_output, kernel_size=kernel_size, pad=pad, stride=1, use_scale=use_scale, **bn_param)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        loc_layers.append(net[flatten_name])

        # Create confidence prediction layer.
        name = "{}_mbox_conf{}".format(from_layer, conf_postfix)
        num_conf_output = num_priors_per_location * num_classes;

        dilation=1
        pad = int((kernel_size + (dilation - 1) * (kernel_size - 1)) - 1) / 2
        ConvBNLayerSSD(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
            num_output=num_conf_output, kernel_size=kernel_size, pad=pad, stride=1, use_scale=use_scale, **bn_param)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        conf_layers.append(net[flatten_name])

        # Create prior generation layer.
        name = "{}_mbox_priorbox".format(from_layer)
        net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_size,
                clip=clip, variance=prior_variance, offset=offset)
        if max_size:
            net.update(name, {'max_size': max_size})
        if aspect_ratio:
            net.update(name, {'aspect_ratio': aspect_ratio, 'flip': flip})
        if step:
            net.update(name, {'step': step})
        if img_height != 0 and img_width != 0:
            if img_height == img_width:
                net.update(name, {'img_size': img_height})
            else:
                net.update(name, {'img_h': img_height, 'img_w': img_width})
        priorbox_layers.append(net[name])

        # Create objectness prediction layer.
        if use_objectness:
            name = "{}_mbox_objectness".format(from_layer)
            num_obj_output = num_priors_per_location * 2;
            dilation=1
            pad = int((kernel_size + (dilation - 1) * (kernel_size - 1)) - 1) / 2
            ConvBNLayerSSD(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
                num_output=num_obj_output, kernel_size=kernel_size, pad=pad, stride=1, use_scale=use_scale, **bn_param)
            permute_name = "{}_perm".format(name)
            net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
            flatten_name = "{}_flat".format(name)
            net[flatten_name] = L.Flatten(net[permute_name], axis=1)
            objectness_layers.append(net[flatten_name])

    # Concatenate priorbox, loc, and conf layers.
    mbox_layers = []
    name = "mbox_loc"
    net[name] = L.Concat(*loc_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_conf"
    net[name] = L.Concat(*conf_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_priorbox"
    net[name] = L.Concat(*priorbox_layers, axis=2)
    mbox_layers.append(net[name])
    if use_objectness:
        name = "mbox_objectness"
        net[name] = L.Concat(*objectness_layers, axis=1)
        mbox_layers.append(net[name])

    return mbox_layers


def get_arguments():
    parser = argparse.ArgumentParser()   
    parser.add_argument('--config_param', type=str, default=None, help='Extra config parameters')      
    parser.add_argument('--solver_param', type=str, default=None, help='Extra solver parameters')        
    return parser.parse_args()
      
def main(): 
    args = get_arguments()
   
    if args.solver_param != None:
      print(args.solver_param)
      args.solver_param = ast.literal_eval(args.solver_param) 
            
    if args.config_param != None:
      print(args.config_param)
      args.config_param = ast.literal_eval(args.config_param) 
            
    #Start populating config_param
    config_param = OrderedDict()
  
    #Names
    config_param.config_name = 'image-objdet'
    config_param.model_name = "jacintonet11"
    config_param.dataset = "nodataset"       
    config_param.pretrain_model = None
                          
    ### Modify the following parameters accordingly ###
    # The directory which contains the caffe code.
    # We assume you are running the script at the CAFFE_ROOT.
    config_param.caffe_root = os.environ['CAFFE_ROOT'] if 'CAFFE_ROOT' in os.environ else None
    if config_param.caffe_root == None:
      config_param.caffe_root = os.environ['CAFFE_HOME'] if 'CAFFE_HOME' in os.environ else None
    if config_param.caffe_root != None:
      config_param.caffe_root = config_param.caffe_root + '/build/tools/caffe.bin'
    config_param.caffe_cmd = 'train'

    print("caffe_root = : ",  config_param.caffe_root)

    # Set true if you want to start training right after generating all files.
    config_param.run_soon = False
    # Set true if you want to load from most recently saved snapshot.
    # Otherwise, we will load from the pretrain_model defined below.
    config_param.resume_training = True
    # If true, Remove old model files.
    config_param.remove_old_models = False
    config_param.display_sparsity = False
    
    # Specify the batch sampler.
    config_param.resize_width = 512
    config_param.resize_height = 512
    config_param.crop_width = config_param.resize_width
    config_param.crop_height = config_param.resize_height

    #feature stride can be 8, 16, 32. 16 provides the best radeoff
    config_param.feature_stride = 16
    config_param.num_feature = 512 #number of feature channels
    config_param.threads = 4
    # The database file for training data. Created by data/VOC0712/create_data.sh
    config_param.train_data = "/data/hdd/datasets/object-detect/other/pascal-voc/VOCdevkit/VOC0712/lmdb/VOC0712_trainval_lmdb"
    # The database file for testing data. Created by data/VOC0712/create_data.sh
    config_param.test_data = "/data/hdd/datasets/object-detect/other/pascal-voc/VOCdevkit/VOC0712/lmdb/VOC0712_test_lmdb"

    config_param.stride_list = None
    config_param.dilation_list = None

    config_param.mean_value = 128 #used in a bias layer in the net.

    
    # If true, use batch norm for all newly added layers.
    # Currently only the non batch norm version has been tested.
    config_param.use_batchnorm = False
    config_param.use_scale = False
    
    config_param.lr_mult = 1

    # Which layers to freeze (no backward) during training.
    config_param.freeze_layers = []
    # Defining which GPUs to use.
    config_param.gpus = "0,1" #gpus = "0"  

    config_param.batch_size = 32
    config_param.accum_batch_size = 32

    # Evaluate on whole test set.
    config_param.num_test_image = 4952
    config_param.test_batch_size = 8
    
    # Stores the test image names and sizes. Created by data/VOC0712/create_list.sh
    config_param.name_size_file = "/user/a0393608/files/work/code/vision/github/weiliu89_ssd/caffe/data/VOC0712/test_name_size.txt"
    # Stores LabelMapItem.
    config_param.label_map_file = "/user/a0393608/files/work/code/vision/github/weiliu89_ssd/caffe/data/VOC0712/labelmap_voc.prototxt"

    # minimum dimension of input image
    config_param.log_space_steps = False #True
    config_param.min_ratio = 10 #5 #20     # in percent %
    config_param.max_ratio = 90            # in percent %
    config_param.num_classes = 21
    
    # MultiBoxLoss parameters initialization.
    config_param.share_location = True
    config_param.background_label_id=0
    config_param.use_difficult_gt = True
    config_param.ignore_difficult_gt = False
    config_param.evaluate_difficult_gt = False
    config_param.normalization_mode = P.Loss.VALID
    config_param.code_type = P.PriorBox.CENTER_SIZE
    config_param.ignore_cross_boundary_bbox = False
    config_param.mining_type = P.MultiBoxLoss.MAX_NEGATIVE
    config_param.neg_pos_ratio = 3.
    config_param.loc_weight = (config_param.neg_pos_ratio + 1.) / 4.
    config_param.min_dim = -1
    config_param.aspect_ratios_type=0
    #need it for COCO which may have gray scale image
    config_param.force_color = 0 

    #Update from params given from outside
    #if args.config_param != None:
    #  config_param.update(args.config_param)   
    if args.config_param != None: 
      for k in args.config_param.keys():
        config_param.__setattr__(k,args.config_param[k])
        config_param.__setitem__(k,args.config_param[k])

    if config_param.min_dim == -1:
      config_param.min_dim = int((config_param.crop_width + config_param.crop_height)/2)

    if config_param.ds_fac == 16: 
      config_param.stride_list = [2,2,2,2,1]
      config_param.dilation_list = [1,1,1,1,2]
    elif config_param.ds_fac == 32: 
      config_param.stride_list = [2,2,2,2,2]
      config_param.dilation_list = [1,1,1,1,1]
   
    print("config_param.ds_fac :", config_param.ds_fac)
    print("config_param.stride_list :", config_param.stride_list)
    resize = "{}x{}".format(config_param.resize_width, config_param.resize_height)
    config_param.batch_sampler = [
            {
                    'sampler': {
                            },
                    'max_trials': 1,
                    'max_sample': 1,
            },
            {
                    'sampler': {
                            'min_scale': 0.3,
                            'max_scale': 1.0,
                            'min_aspect_ratio': 0.5,
                            'max_aspect_ratio': 2.0,
                            },
                    'sample_constraint': {
                            'min_jaccard_overlap': 0.1,
                            },
                    'max_trials': 50,
                    'max_sample': 1,
            },
            {
                    'sampler': {
                            'min_scale': 0.3,
                            'max_scale': 1.0,
                            'min_aspect_ratio': 0.5,
                            'max_aspect_ratio': 2.0,
                            },
                    'sample_constraint': {
                            'min_jaccard_overlap': 0.3,
                            },
                    'max_trials': 50,
                    'max_sample': 1,
            },
            {
                    'sampler': {
                            'min_scale': 0.3,
                            'max_scale': 1.0,
                            'min_aspect_ratio': 0.5,
                            'max_aspect_ratio': 2.0,
                            },
                    'sample_constraint': {
                            'min_jaccard_overlap': 0.5,
                            },
                    'max_trials': 50,
                    'max_sample': 1,
            },
            {
                    'sampler': {
                            'min_scale': 0.3,
                            'max_scale': 1.0,
                            'min_aspect_ratio': 0.5,
                            'max_aspect_ratio': 2.0,
                            },
                    'sample_constraint': {
                            'min_jaccard_overlap': 0.7,
                            },
                    'max_trials': 50,
                    'max_sample': 1,
            },
            {
                    'sampler': {
                            'min_scale': 0.3,
                            'max_scale': 1.0,
                            'min_aspect_ratio': 0.5,
                            'max_aspect_ratio': 2.0,
                            },
                    'sample_constraint': {
                            'min_jaccard_overlap': 0.9,
                            },
                    'max_trials': 50,
                    'max_sample': 1,
            },
            {
                    'sampler': {
                            'min_scale': 0.3,
                            'max_scale': 1.0,
                            'min_aspect_ratio': 0.5,
                            'max_aspect_ratio': 2.0,
                            },
                    'sample_constraint': {
                            'max_jaccard_overlap': 1.0,
                            },
                    'max_trials': 50,
                    'max_sample': 1,
            },
            ]
    config_param.train_transform_param = {
            'mirror': True,
            'mean_value': [0, 0, 0],
            'force_color':config_param.force_color,
            'resize_param': {
                    'prob': 1,
                    'resize_mode': P.Resize.WARP,
                    'height': config_param.resize_height,
                    'width': config_param.resize_width,
                    'interp_mode': [
                            P.Resize.LINEAR,
                            P.Resize.AREA,
                            P.Resize.NEAREST,
                            P.Resize.CUBIC,
                            P.Resize.LANCZOS4,
                            ],
                    },
            'distort_param': {
                    'brightness_prob': 0.5,
                    'brightness_delta': 32,
                    'contrast_prob': 0.5,
                    'contrast_lower': 0.5,
                    'contrast_upper': 1.5,
                    'hue_prob': 0.5,
                    'hue_delta': 18,
                    'saturation_prob': 0.5,
                    'saturation_lower': 0.5,
                    'saturation_upper': 1.5,
                    'random_order_prob': 0.0,
                    },
            'expand_param': {
                    'prob': 0.5,
                    'max_expand_ratio': 4.0,
                    },
            'emit_constraint': {
                'emit_type': caffe_pb2.EmitConstraint.CENTER,
                },
      'crop_h': config_param.crop_height,
      'crop_w': config_param.crop_width
            }
    config_param.test_transform_param = {
            'mean_value': [0, 0, 0],
            'force_color':config_param.force_color,
            'resize_param': {
                    'prob': 1,
                    'resize_mode': P.Resize.WARP,
                    'height': config_param.resize_height,
                    'width': config_param.resize_width,
                    'interp_mode': [P.Resize.LINEAR],
                    },
      'crop_h': config_param.crop_height,
      'crop_w': config_param.crop_width          
            }

        
    # Modify the job name if you want.
    #print("config_name is {}".format(config_param.config_name))
    config_param.base_name = config_param.config_name
    config_param.job_name = config_param.base_name

    # Base dir
    config_param.base_dir = config_param.job_name
    # Directory which stores the model .prototxt file.
    config_param.save_dir = config_param.job_name
    # Directory which stores the snapshot of models.
    config_param.snapshot_dir = config_param.job_name
    # Directory which stores the job script and log file.
    config_param.job_dir = config_param.job_name
    # Directory which stores the detection results.
    config_param.output_result_dir = "" #"{}/results".format(config_param.job_name)
        
    # model definition files.
    config_param.train_net_file = "{}/train.prototxt".format(config_param.save_dir)
    config_param.test_net_file = "{}/test.prototxt".format(config_param.save_dir)
    config_param.deploy_net_file = "{}/deploy.prototxt".format(config_param.save_dir)
    config_param.solver_file = "{}/solver.prototxt".format(config_param.save_dir)
    # snapshot prefix.
    config_param.snapshot_prefix = "{}/{}_{}".format(config_param.snapshot_dir, config_param.dataset, config_param.model_name)
    # job script path.
    job_file_base_name = 'run' 
    config_param.job_file_base = "{}/{}".format(config_param.job_dir, job_file_base_name)
    config_param.log_file = "{}.log".format(config_param.job_file_base)    
    config_param.job_file = "{}.sh".format(config_param.job_file_base)
   
    # MultiBoxLoss parameters.
    multibox_loss_param = {
        'loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
        'conf_loss_type': P.MultiBoxLoss.SOFTMAX,
        'loc_weight': config_param.loc_weight,
        'num_classes': config_param.num_classes,
        'share_location': config_param.share_location,
        'match_type': P.MultiBoxLoss.PER_PREDICTION,
        'overlap_threshold': 0.5,
        'use_prior_for_matching': True,
        'background_label_id': config_param.background_label_id,
        'use_difficult_gt': config_param.use_difficult_gt,
        'ignore_difficult_gt': config_param.ignore_difficult_gt,
        'mining_type': config_param.mining_type,
        'neg_pos_ratio': config_param.neg_pos_ratio,
        'neg_overlap': 0.5,
        'code_type': config_param.code_type,
        'ignore_cross_boundary_bbox': config_param.ignore_cross_boundary_bbox,
        }
    loss_param = {
        'normalization': config_param.normalization_mode,
        }

    if config_param.feature_stride != 16:
        ValueError("config_param.feature_stride {} is incorrect".format(config_param.feature_stride))
    
    if (config_param.model_name == 'jdetnet21v2'):
        config_param.steps = [16, 32, 64, 128]
        config_param.mbox_source_layers = ['ctx_output1/relu', 'ctx_output2/relu', 'ctx_output3/relu', \
          'ctx_output4/relu']
    elif (config_param.model_name == 'jdetnet21v2-s8'):
        config_param.steps = [8, 16, 32, 64, 128, 128]
        config_param.mbox_source_layers = ['ctx_output1/relu', 'ctx_output2/relu', 'ctx_output3/relu', \
          'ctx_output4/relu', 'ctx_output5/relu', 'ctx_output6/relu']
    elif (config_param.model_name == 'jdetnet21v2-fpn'):
        config_param.steps = [16, 16, 32, 64, 128, 128]
        config_param.mbox_source_layers = ['ctx_output1/relu', 'ctx_output2/relu', 'ctx_output3/relu', \
          'ctx_output4/relu', 'ctx_output5/relu', 'ctx_output6/relu']
    if (config_param.model_name == 'ssdJacintoNetV2'):
      config_param.steps = []
      #if config_param.resize_width == config_param.resize_height:
        #config_param.steps = [8, 16, 32, 64, 128, 256, 512]
      #config_param.mbox_source_layers = ['ctx_output1/relu', 'ctx_output2/relu', 'ctx_output3/relu', \
      #  'ctx_output4/relu', 'ctx_output5/relu', 'ctx_output6/relu']
      if config_param.ds_type == 'DFLT':
        config_param.mbox_source_layers = ['res3a_branch2b/relu', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2', 'conv10_2']
      else: 
        if config_param.stride_list[4] == 1:
          config_param.mbox_source_layers = ['ctx_output1/relu', 'ctx_output2/relu', 'ctx_output3/relu', 
          'ctx_output4/relu', 'ctx_output5/relu', 'ctx_output6/relu', 'ctx_output7/relu']
        else:  
          config_param.mbox_source_layers = ['ctx_output1/relu', 'ctx_output2/relu', 'ctx_output3/relu', 
          'ctx_output4/relu', 'ctx_output5/relu', 'ctx_output6/relu']
        
        if config_param.base_nw_3_head:
          config_param.mbox_source_layers.append('ctx_output{}/relu'.format(len(config_param.mbox_source_layers)+1))  
    elif config_param.model_name == 'vgg16':
        # conv4_3 ==> 38 x 38
        # fc7 ==> 19 x 19
        # conv6_2 ==> 10 x 10
        # conv7_2 ==> 5 x 5
        # conv8_2 ==> 3 x 3
        # conv9_2 ==> 1 x 1
        config_param.mbox_source_layers = ['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']
        config_param.steps = [8, 16, 32, 64, 100, 300]
    elif 'mobiledetnet' in config_param.model_name:
        config_param.mbox_source_layers = ['ctx_output1/relu', 'ctx_output2/relu', 'ctx_output3/relu', \
          'ctx_output4/relu', 'ctx_output5/relu']
        config_param.steps = [16, 32, 64, 128, 128]
    else:
        ValueError("Invalid model name")
    
    # parameters for generating priors.
    config_param.num_steps = len(config_param.mbox_source_layers)  
    config_param.step = int(math.floor((config_param.max_ratio - config_param.min_ratio) / config_param.num_steps))
    config_param.min_sizes = []
    config_param.max_sizes = []
      
    print("min_dim = {}".format(config_param.min_dim))
    min_dim_to_use = config_param.min_ratio*config_param.min_dim/100
    max_dim_to_use = config_param.max_ratio*config_param.min_dim/100
    if config_param.log_space_steps == 1:
      #log
      min_max_sizes = np.logspace(np.log2(min_dim_to_use), np.log2(max_dim_to_use), num=config_param.num_steps+1, base=2)
      config_param.min_sizes = list(min_max_sizes[0:config_param.num_steps])
      config_param.max_sizes = list(min_max_sizes[1:config_param.num_steps+1])
    elif config_param.log_space_steps == 0:
      #linear
      min_max_sizes = np.linspace(min_dim_to_use, max_dim_to_use, num=config_param.num_steps+1)
      config_param.min_sizes = list(min_max_sizes[0:config_param.num_steps])
      config_param.max_sizes = list(min_max_sizes[1:config_param.num_steps+1])
    else:
      #like original SSD
      config_param.min_sizes, config_param.max_sizes = set_min_max_sizes(config_param)
  
    print("minsizes = {}".format(config_param.min_sizes))
    print("maxsizes = {}".format(config_param.max_sizes))
   
    if config_param.aspect_ratios_type == 0:
      config_param.aspect_ratios = [[2]]*config_param.num_steps 
    else:
      #like original SSD
      config_param.aspect_ratios = [[2,3]]*config_param.num_steps
      config_param.aspect_ratios[0] = [2]
      config_param.aspect_ratios[-1] = [2]
      config_param.aspect_ratios[-2] = [2]
           
    print("ARs:",config_param.aspect_ratios)
    # L2 normalize conv4_3.
    config_param.normalizations = [-1]*config_param.num_steps #[20, -1, -1, -1, -1, -1]
    # variance used to encode/decode prior bboxes.
    if config_param.code_type == P.PriorBox.CENTER_SIZE:
      config_param.prior_variance = [0.1, 0.1, 0.2, 0.2]
    else:
      config_param.prior_variance = [0.1]

    if config_param.chop_num_heads > 0:
      print("Chopping heads")
      del config_param.min_sizes[-config_param.chop_num_heads:]
      del config_param.max_sizes[-config_param.chop_num_heads:]
      del config_param.aspect_ratios[-config_param.chop_num_heads:]
      del config_param.normalizations[-config_param.chop_num_heads:]
      del config_param.mbox_source_layers[-config_param.chop_num_heads:]

      print("minsizes = {}".format(config_param.min_sizes))
      print("maxsizes = {}".format(config_param.max_sizes))
      print("aspect_ratios = {}".format(config_param.aspect_ratios))
      print(config_param.mbox_source_layers)

    config_param.flip = True
    config_param.clip = False

    # Solver parameters.
    # Defining which GPUs to use.
    config_param.gpulist = config_param.gpus.split(",")
    config_param.num_gpus = len(config_param.gpulist)
   
    # Divide the mini-batch to different GPUs.
    iter_size = int(math.ceil(config_param.accum_batch_size/config_param.batch_size))
    solver_mode = P.Solver.CPU
    device_id = 0
    batch_size_per_device = config_param.batch_size
    if config_param.num_gpus > 0:
      batch_size_per_device = int(math.ceil(float(config_param.batch_size) / config_param.num_gpus))
      iter_size = int(math.ceil(float(config_param.accum_batch_size) / (batch_size_per_device * config_param.num_gpus)))
      solver_mode = P.Solver.GPU
      device_id = int(config_param.gpulist[0])

    # Ideally test_batch_size should be divisible by num_test_image,
    # otherwise mAP will be slightly off the true value.
    test_iter = int(math.ceil(float(config_param.num_test_image) / config_param.test_batch_size))

    solver_param = {
        # Train parameters
        'type': "SGD",
        'base_lr': 1e-3,
        'max_iter': 32000, 
        'weight_decay': 0.0005,
        'lr_policy': "multistep",
        'power': 1.0,
        'stepvalue': [24000, 30000, 32000],
        'gamma': 0.1,
        'momentum': 0.9,
        'iter_size': iter_size,
        'snapshot': 2000,
        'display': 100,
        'average_loss': 10,
        'type': "SGD",
        'solver_mode': solver_mode,
        'device_id': device_id,
        'debug_info': False,
        'snapshot_after_train': True,
        # Test parameters
        'test_iter': [test_iter],
        'test_interval': 2000,
        'eval_type': "detection",
        'ap_version': "11point",
        'test_initialization': True,
        'random_seed': 33,
        'show_per_class_result': True,
        }

    #if args.solver_param != None:
    #  solver_param.update(args.solver_param)       
    if args.solver_param != None: 
      for k in args.solver_param.keys():
        solver_param.__setitem__(k,args.solver_param[k])    
        #solver_param.__setattr__(k,args.solver_param[k])
    
    # parameters for generating detection output.
    det_out_param = {
        'num_classes': config_param.num_classes,
        'share_location': config_param.share_location,
        'background_label_id': config_param.background_label_id,
        'nms_param': {'nms_threshold': 0.45, 'top_k': 400},
        'save_output_param': {
            'output_directory': config_param.output_result_dir,
            'output_name_prefix': "comp4_det_test_",
            'output_format': "VOC",
            'label_map_file': config_param.label_map_file,
            'name_size_file': config_param.name_size_file,
            'num_test_image': config_param.num_test_image,
            },
        'keep_top_k': 200,
        'confidence_threshold': 0.01,
        'code_type': config_param.code_type,
        }

    # parameters for evaluating detection results.
    det_eval_param = {
        'num_classes': config_param.num_classes,
        'background_label_id': config_param.background_label_id,
        'overlap_threshold': 0.5,
        'evaluate_difficult_gt': config_param.evaluate_difficult_gt,
        'name_size_file': config_param.name_size_file,
        }

    ### Hopefully you don't need to change the following ###
    # Check file.
    check_if_exist(config_param.train_data)
    check_if_exist(config_param.test_data)
    check_if_exist(config_param.label_map_file)
    if config_param.pretrain_model != None:    
      check_if_exist(config_param.pretrain_model)
    make_if_not_exist(config_param.base_dir)  
    make_if_not_exist(config_param.save_dir)
    make_if_not_exist(config_param.job_dir)
    make_if_not_exist(config_param.snapshot_dir)

    # Create train net.
    net = caffe.NetSpec()
    net.data, net.label = CreateAnnotatedDataLayer(config_param.train_data, batch_size=config_param.batch_size,
            train=True, output_label=True, label_map_file=config_param.label_map_file,
            transform_param=config_param.train_transform_param, batch_sampler=config_param.batch_sampler, 
            threads=config_param.threads)

    out_layer = 'data'
    bias_kwargs = { #fixed value with lr_mult=0
        'param': [dict(lr_mult=0, decay_mult=0)],
        'filler': dict(type='constant', value=(-config_param.mean_value)),
        }       
    net['data/bias'] = L.Bias(net[out_layer], in_place=False, **bias_kwargs)
    out_layer = 'data/bias'           

    def core_network(net, from_layer):
        if config_param.model_name == 'jdetnet21v2':
            out_layer = models.jacintonet_v2.jdetnet21(net, from_layer=from_layer,\
              num_output=config_param.num_feature,stride_list=config_param.stride_list,dilation_list=config_param.dilation_list,\
              freeze_layers=config_param.freeze_layers, output_stride=config_param.feature_stride)
        elif config_param.model_name == 'jdetnet21v2-s8':
            out_layer = models.jacintonet_v2.jdetnet21_s8(net, from_layer=from_layer,\
              num_output=config_param.num_feature,stride_list=config_param.stride_list,dilation_list=config_param.dilation_list,\
              freeze_layers=config_param.freeze_layers, output_stride=config_param.feature_stride)              
        elif config_param.model_name == 'jdetnet21v2-fpn':
            out_layer = models.jacintonet_v2.jdetnet21_fpn(net, from_layer=from_layer,\
              num_output=config_param.num_feature,stride_list=config_param.stride_list,dilation_list=config_param.dilation_list,\
              freeze_layers=config_param.freeze_layers, output_stride=config_param.feature_stride)
        elif config_param.model_name == 'ssdJacintoNetV2':
            out_layer = models.jacintonet_v2.ssdJacintoNetV2(net, from_layer=from_layer,\
              num_output=config_param.num_feature,stride_list=config_param.stride_list,\
              dilation_list=config_param.dilation_list,\
              freeze_layers=config_param.freeze_layers, output_stride=config_param.feature_stride,\
              ds_type=config_param.ds_type, use_batchnorm_mbox=config_param.use_batchnorm_mbox,fully_conv_at_end=config_param.fully_conv_at_end, 
              reg_head_at_ds8=config_param.reg_head_at_ds8, concat_reg_head=config_param.concat_reg_head,
              base_nw_3_head=config_param.base_nw_3_head, first_hd_same_op_ch=config_param.first_hd_same_op_ch,
              num_intermediate=config_param.num_intermediate, rhead_name_non_linear=config_param.rhead_name_non_linear,
              chop_num_heads=config_param.chop_num_heads)
        elif 'mobiledetnet' in config_param.model_name:
            #out_layer = models.mobilenet.mobiledetnet(net, from_layer=from_layer,\
            #  num_output=config_param.num_feature,stride_list=config_param.stride_list,dilation_list=config_param.dilation_list,\
            #  freeze_layers=config_param.freeze_layers, output_stride=config_param.feature_stride, wide_factor=wide_factor)
            wide_factor = float(config_param.model_name.split('-')[1])
            out_layer = models.mobilenet.mobiledetnet(net, from_layer=from_layer, wide_factor=wide_factor)
        else:
            ValueError("Invalid model name")
        return net, out_layer
    
    net, out_layer = core_network(net, out_layer)
    mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=config_param.mbox_source_layers,
            use_batchnorm=config_param.use_batchnorm, use_scale=config_param.use_scale, min_sizes=config_param.min_sizes, max_sizes=config_param.max_sizes,
            aspect_ratios=config_param.aspect_ratios, steps=config_param.steps, normalizations=config_param.normalizations,
            num_classes=config_param.num_classes, share_location=config_param.share_location, flip=config_param.flip, clip=config_param.clip,
            prior_variance=config_param.prior_variance, kernel_size=config_param.ker_mbox_loc_conf, pad=1, lr_mult=config_param.lr_mult)

    # Create the MultiBoxLossLayer.
    name = "mbox_loss"
    mbox_layers.append(net.label)
    net[name] = L.MultiBoxLoss(*mbox_layers, multibox_loss_param=multibox_loss_param,
            loss_param=loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')),
            propagate_down=[True, True, False, False])

    with open(config_param.train_net_file, 'w') as f:
        print(config_param.train_net_file)
        print('name: "{}"'.format(config_param.model_name), file=f)
        print(net.to_proto(), file=f)
    #shutil.copy(train_net_file, job_dir)

    # Create test net.
    net = caffe.NetSpec()
    net.data, net.label = CreateAnnotatedDataLayer(config_param.test_data, batch_size=config_param.test_batch_size,
            train=False, output_label=True, label_map_file=config_param.label_map_file,
            transform_param=config_param.test_transform_param, threads=config_param.threads)

    out_layer = 'data'
    bias_kwargs = { #fixed value with lr_mult=0
        'param': [dict(lr_mult=0, decay_mult=0)],
        'filler': dict(type='constant', value=(-config_param.mean_value)),
        }       
    net['data/bias'] = L.Bias(net[out_layer], in_place=False, **bias_kwargs)
    out_layer = 'data/bias'    
    
    net, out_layer = core_network(net, out_layer)
    mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=config_param.mbox_source_layers,
            use_batchnorm=config_param.use_batchnorm, use_scale=config_param.use_scale, min_sizes=config_param.min_sizes, max_sizes=config_param.max_sizes,
            aspect_ratios=config_param.aspect_ratios, steps=config_param.steps, normalizations=config_param.normalizations,
            num_classes=config_param.num_classes, share_location=config_param.share_location, flip=config_param.flip, clip=config_param.clip,
            prior_variance=config_param.prior_variance, kernel_size=config_param.ker_mbox_loc_conf, pad=1, lr_mult=config_param.lr_mult)

    conf_name = "mbox_conf"
    if multibox_loss_param["conf_loss_type"] == P.MultiBoxLoss.SOFTMAX:
      reshape_name = "{}_reshape".format(conf_name)
      net[reshape_name] = L.Reshape(net[conf_name], shape=dict(dim=[0, -1, config_param.num_classes]))
      softmax_name = "{}_softmax".format(conf_name)
      net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
      flatten_name = "{}_flatten".format(conf_name)
      net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
      mbox_layers[1] = net[flatten_name]
    elif multibox_loss_param["conf_loss_type"] == P.MultiBoxLoss.LOGISTIC:
      sigmoid_name = "{}_sigmoid".format(conf_name)
      net[sigmoid_name] = L.Sigmoid(net[conf_name])
      mbox_layers[1] = net[sigmoid_name]

    net.detection_out = L.DetectionOutput(*mbox_layers,
        detection_output_param=det_out_param,
        include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    net.detection_eval = L.DetectionEvaluate(net.detection_out, net.label,
        detection_evaluate_param=det_eval_param,
        include=dict(phase=caffe_pb2.Phase.Value('TEST')))

    with open(config_param.test_net_file, 'w') as f:
        print('name: "{}_test"'.format(config_param.model_name), file=f)
        print(net.to_proto(verbose=False), file=f)
    if config_param.save_dir!=config_param.job_dir:
      shutil.copy(config_param.test_net_file, config_param.job_dir)

    # Create deploy net.
    # Remove the first and last layer from test net.
    deploy_net = net
    with open(config_param.deploy_net_file, 'w') as f:
        net_param = deploy_net.to_proto()
        # Remove the first (AnnotatedData) and last (DetectionEvaluate) layer from test net.
        del net_param.layer[0]
        del net_param.layer[-1]
        net_param.name = '{}_deploy'.format(config_param.model_name)
        net_param.input.extend(['data'])
        net_param.input_shape.extend([
            caffe_pb2.BlobShape(dim=[1, 3, config_param.resize_height, config_param.resize_width])])
        print(net_param, file=f)
    if config_param.save_dir!=config_param.job_dir:        
      shutil.copy(config_param.deploy_net_file, config_param.job_dir)

    # Create solver.
    solver = caffe_pb2.SolverParameter(
            train_net=config_param.train_net_file,
            test_net=[config_param.test_net_file],
            snapshot_prefix=config_param.snapshot_prefix,
            **solver_param)
            
    with open(config_param.solver_file, 'w') as f:
        print(solver, file=f)
    if config_param.save_dir!=config_param.job_dir:        
      shutil.copy(solver_file, job_dir)

    max_iter = 0
    # Find most recent snapshot.
    for file in os.listdir(config_param.snapshot_dir):
      if file.endswith(".solverstate"):
        basename = os.path.splitext(file)[0]
        iter = int(basename.split("{}_iter_".format(config_param.model_name))[1])
        if iter > max_iter:
          max_iter = iter

    train_src_param = None
    if config_param.pretrain_model != None:
      train_src_param = '--weights="{}" \\\n'.format(config_param.pretrain_model)
    if config_param.resume_training:
      if max_iter > 0:
        train_src_param = '--snapshot="{}_iter_{}.solverstate" \\\n'.format(config_param.snapshot_prefix, max_iter)

    if config_param.remove_old_models:
      # Remove any snapshots smaller than max_iter.
      for file in os.listdir(config_param.snapshot_dir):
        if file.endswith(".solverstate"):
          basename = os.path.splitext(file)[0]
          iter = int(basename.split("{}_iter_".format(config_param.model_name))[1])
          if max_iter > iter:
            os.remove("{}/{}".format(config_param.snapshot_dir, file))
        if file.endswith(".caffemodel"):
          basename = os.path.splitext(file)[0]
          iter = int(basename.split("{}_iter_".format(config_param.model_name))[1])
          if max_iter > iter:
            os.remove("{}/{}".format(config_param.snapshot_dir, file))

    # Create job file.
    with open(config_param.job_file, 'w') as f:
      #f.write('cd {}\n'.format(config_param.caffe_root))
      f.write('{} {} \\\n'.format(config_param.caffe_root, config_param.caffe_cmd))    
      if(config_param.caffe_cmd == 'test' or config_param.caffe_cmd == 'test_detection'):
        f.write('--model="{}" \\\n'.format(config_param.test_net_file))
        f.write('--iterations="{}" \\\n'.format(solver_param['test_iter'][0]))       
        if config_param.display_sparsity:
          f.write('--display_sparsity=1 \\\n')
      else:
        f.write('--solver="{}" \\\n'.format(config_param.solver_file))      
      if train_src_param != None:
        f.write(train_src_param)
      if solver_param['solver_mode'] == P.Solver.GPU:
        f.write('--gpu "{}" 2>&1 | tee {}\n'.format(config_param.gpus, config_param.log_file))
      else:
        f.write('2>&1 | tee {}\n'.format(config_param.log_file))

    # Copy the python script to job_dir.
    py_file = os.path.abspath(__file__)
    shutil.copy(py_file, config_param.job_dir)
    
    #copy some other utils scripts
    shutil.copy(os.getcwd() + '/train_image_object_detection.sh', config_param.job_dir)
    shutil.copy(os.getcwd() + '/models/jacintonet_v2.py', config_param.job_dir)

    # Run the job.
    os.chmod(config_param.job_file, stat.S_IRWXU)
    if config_param.run_soon:
      subprocess.call(config_param.job_file, shell=True)
  
  
if __name__ == "__main__":
  main()  
