#!/usr/bin/env python
#https://stackoverflow.com/questions/43517959/given-a-tensor-flow-model-graph-how-to-find-the-input-node-and-output-node-name

import tensorflow as tf
#gf = tf.GraphDef()
gf = tf.compat.v1.GraphDef()

#gf.ParseFromString(open('/your/path/to/graphname.pb','rb').read())
gf.ParseFromString(open('ssd_mobilenet_v1_coco_2017_11_17.pb','rb').read())

result1 = [n.name + '=>' +  n.op for n in gf.node if n.op in ( 'Softmax','Placeholder')]
print("\n result1: ", result1)

result2 = [n.name + '=>' +  n.op for n in gf.node if n.op in ( 'Softmax','Mul')]
print("\n result2: ", result2)

'''
$ strings ssd_mobilenet_v1_coco_2017_11_17.pb | grep FeatureExtractor | grep 'V1/MobilenetV1/Conv2d_0' | grep mul_1
KFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/batchnorm/mul_1
KFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/batchnorm/mul_1
'''

