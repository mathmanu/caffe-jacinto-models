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
    net_new = caffe.Net(args.new_model, args.new_weights, caffe.TEST)

    #Take care of the layers (which have atleast one blob) with names cahanges

    
    for old_name, new_name in zip(net_old.params.keys(), net_new.params.keys()):
        for i in range(len(net_old.params[old_name])):
            
            if (net_new.params[new_name][i].data[...] == net_old.params[old_name][i].data[...]).all():
                print('All weights matched for {} and {}'.format(old_name, new_name))
            else:
                print('All weights didnt match for {} and {}'.format(old_name, new_name))
                print(net_new.params[new_name][i].data[...]-net_old.params[old_name][i].data[...])
            
            
                 
    print('Completed matching..')  
    net_new.save(args.new_weights)

def main():
    change_names()
    
if __name__ == "__main__":
    main() 
