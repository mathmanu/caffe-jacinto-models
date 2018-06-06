import caffe
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old_model', type=str, required=True, help='Old model name')    
    parser.add_argument('--old_weights', type=str, required=True, help='Pretrained caffemodel name')  
    parser.add_argument('--new_model', type=str, required=True, help='New model name')    
    parser.add_argument('--new_weights', type=str, required=True, help='Output pretrained caffemodel name')           
    return parser.parse_args()
    
def change_names():
    args = get_arguments()

    #Old model and weights
    net_old = caffe.Net(args.old_model, args.old_weights, caffe.TEST)

    #Initialize with old wights -> some layers will not get initialized
    net_new = caffe.Net(args.new_model, args.old_weights, caffe.TEST)

    #Take care of the layers (which have atleast one blob) with names cahanges

    '''
    net_old_names = ['conv1', 'conv2_1/expand', 'conv2_1/dwise', 'conv2_1/linear',
                     'conv2_2/expand', 'conv2_2/dwise', 'conv2_2/linear',
                     'conv3_1/expand', 'conv3_1/dwise', 'conv3_1/linear',
                     'conv3_2/expand', 'conv3_2/dwise', 'conv3_2/linear',
                     'conv4_1/expand', 'conv4_1/dwise', 'conv4_1/linear',
                     'conv4_2/expand', 'conv4_2/dwise', 'conv4_2/linear',
                     'conv4_3/expand', 'conv4_3/dwise', 'conv4_3/linear',
                     'conv4_4/expand', 'conv4_4/dwise', 'conv4_4/linear',
                     'conv4_5/expand', 'conv4_5/dwise', 'conv4_5/linear',
                     'conv4_6/expand', 'conv4_6/dwise', 'conv4_6/linear',
                     'conv4_7/expand', 'conv4_7/dwise', 'conv4_7/linear',
                     'conv5_1/expand', 'conv5_1/dwise', 'conv5_1/linear',
                     'conv5_2/expand', 'conv5_2/dwise', 'conv5_2/linear',
                     'conv5_3/expand', 'conv5_3/dwise', 'conv5_3/linear',
                     'conv6_1/expand', 'conv6_1/dwise', 'conv6_1/linear',
                     'conv6_2/expand', 'conv6_2/dwise', 'conv6_2/linear',
                     'conv6_3/expand', 'conv6_3/dwise', 'conv6_3/linear',
                     'conv6_4', 'fc7'

                     ]


    net_new_names = ['conv1', 'conv2_1/expand', 'conv2_1/dwise', 'conv2_1/linear',
                     'conv3_1/expand', 'conv3_1/dwise', 'conv3_1/linear',
                     'conv3_2/expand', 'conv3_2/dwise', 'conv3_2/linear',
                     'conv4_1/expand', 'conv4_1/dwise', 'conv4_1/linear',
                     'conv4_2/expand', 'conv4_2/dwise', 'conv4_2/linear',
                     'conv4_3/expand', 'conv4_3/dwise', 'conv4_3/linear',
                     'conv5_1/expand', 'conv5_1/dwise', 'conv5_1/linear',
                     'conv5_2/expand', 'conv5_2/dwise', 'conv5_2/linear',
                     'conv5_3/expand', 'conv5_3/dwise', 'conv5_3/linear',
                     'conv5_4/expand', 'conv5_4/dwise', 'conv5_4/linear',
                     'conv6_1/expand', 'conv6_1/dwise', 'conv6_1/linear',
                     'conv6_2/expand', 'conv6_2/dwise', 'conv6_2/linear',
                     'conv6_3/expand', 'conv6_3/dwise', 'conv6_3/linear',
                     'conv7_1/expand', 'conv7_1/dwise', 'conv7_1/linear',
                     'conv7_2/expand', 'conv7_2/dwise', 'conv7_2/linear',
                     'conv7_3/expand', 'conv7_3/dwise', 'conv7_3/linear',
                     'conv8_1/expand', 'conv8_1/dwise', 'conv8_1/linear',
                     'conv9_1','fc10'
                     ]

    print('Started copying..') 
    for old_name, new_name in zip(net_old_names, net_new_names):
        for i in range(len(net_old.params[old_name])):
            net_new.params[new_name][i].data[...] = net_old.params[old_name][i].data[...]
        if old_name+'/bn' in list(net_old.params.keys()):
          for i in range(len(net_old.params[old_name+'/bn'])):
            net_new.params[new_name+'/bn'][i].data[...] = net_old.params[old_name+'/bn'][i].data[...]
        if old_name+'/scale' in list(net_old.params.keys()):
          for i in range(len(net_old.params[old_name+'/scale'])):
            net_new.params[new_name+'/scale'][i].data[...] = net_old.params[old_name+'/scale'][i].data[...]
    '''
    
    for old_name, new_name in zip(net_old.params.keys(), net_new.params.keys()):
        for i in range(len(net_old.params[old_name])):
            print('copying:', old_name, ' to ', new_name)
            net_new.params[new_name][i].data[...] = net_old.params[old_name][i].data[...]
            
                 
    print('Completed copying..')  
    net_new.save(args.new_weights)

def main():
    change_names()
    
if __name__ == "__main__":
    main() 
