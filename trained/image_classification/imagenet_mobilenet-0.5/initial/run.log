This model was converted from a similar mobilenet-0.5 model which had seperate scale layer after BN to the Fused BN used in this model. 
This conversion is straight forward and can be done using the script:
scripts/tools/utils/convert_weights_bvlccaffe2nv.py

Now the default BN style is set to Fused BN in mobilenet.py. This is slightly faster.
