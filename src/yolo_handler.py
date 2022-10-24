from __future__ import annotations
import numpy as np
import configparser
from keras.regularizers import l2
from keras.layers import (Conv2D, Input, ZeroPadding2D, Add,
                          UpSampling2D, MaxPooling2D, Concatenate)
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.models import Model
from src.cnnModels import *
from collections import defaultdict
import io

class Darknet(object):
    def __init__(self, cfgPath: str, weightsPath: str, dataPath: str):
        self.configPath =  cfgPath
        self.weightsPath = weightsPath
        self.modelPath = 'teste.h5'
        self.dataPath = dataPath
        self.init_parser()
        self.init_model()
        self.load_train_param()

    #Based in https://github.com/qqwweee/keras-yolo3/blob/master/convert.py
    def init_parser(self):

        section_counters = defaultdict(int)
        output_stream = io.StringIO()
        with open(self.configPath) as fin:
            for line in fin:
                if line.startswith('['):
                    section = line.strip().strip('[]')
                    _section = section + '_' + str(section_counters[section])
                    section_counters[section] += 1
                    line = line.replace(section, _section)
                output_stream.write(line)
        output_stream.seek(0)

        self.cfg_parser = configparser.ConfigParser()
        self.cfg_parser.read_file(output_stream)
    
    #Based in https://github.com/qqwweee/keras-yolo3/blob/master/convert.py
    def init_model(self):
        print('Creating Keras model.')
        input_layer = Input(shape=(None, None, 3))
        prev_layer = input_layer
        out_index = []
        all_layers = []
        count = 0

        weight_decay = float(self.cfg_parser['net_0']['decay']
                         ) if 'net_0' in self.cfg_parser.sections() else 5e-4

        weights_file = open(self.weightsPath, 'rb')

        for section in self.cfg_parser.sections():
            print('Parsing section {}'.format(section))
            if section.startswith('convolutional'):
                filters = int(self.cfg_parser[section]['filters'])
                size = int(self.cfg_parser[section]['size'])
                stride = int(self.cfg_parser[section]['stride'])
                pad = int(self.cfg_parser[section]['pad'])
                activation = self.cfg_parser[section]['activation']
                batch_normalize = 'batch_normalize' in self.cfg_parser[section]

                padding = 'same' if pad == 1 and stride == 1 else 'valid'

                # Setting weights.
                # Darknet serializes convolutional weights as:
                # [bias/beta, [gamma, mean, variance], conv_weights]
                prev_layer_shape = K.int_shape(prev_layer)

                weights_shape = (size, size, prev_layer_shape[-1], filters)
                darknet_w_shape = (filters, weights_shape[2], size, size)
                weights_size = np.product(weights_shape)

                print('conv2d', 'bn'
                    if batch_normalize else '  ', activation, weights_shape)

                conv_bias = np.ndarray(
                    shape=(filters, ),
                    dtype='float32',
                    buffer=weights_file.read(filters * 4))
                count += filters

                if batch_normalize:
                    bn_weights = np.ndarray(
                        shape=(3, filters),
                        dtype='float32',
                        buffer=weights_file.read(filters * 12))
                    count += 3 * filters

                    bn_weight_list = [
                        bn_weights[0],  # scale gamma
                        conv_bias,  # shift beta
                        bn_weights[1],  # running mean
                        bn_weights[2]  # running var
                    ]

                conv_weights = np.ndarray(
                    shape=darknet_w_shape,
                    dtype='float32',
                    buffer=weights_file.read(weights_size * 4))
                count += weights_size

                # DarkNet conv_weights are serialized Caffe-style:
                # (out_dim, in_dim, height, width)
                # We would like to set these to Tensorflow order:
                # (height, width, in_dim, out_dim)
                conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
                conv_weights = [conv_weights] if batch_normalize else [
                    conv_weights, conv_bias
                ]

                # Handle activation.
                act_fn = None
                if activation == 'leaky':
                    pass  # Add advanced activation later.
                elif activation == 'swish':
                    pass
                elif activation == 'logistic':
                    pass
                elif activation != 'linear':
                    raise ValueError(
                        'Unknown activation function `{}` in section {}'.format(
                            activation, section))

                # Create Conv2D layer
                if stride>1:
                    # Darknet uses left and top padding instead of 'same' mode
                    prev_layer = ZeroPadding2D(((1,0),(1,0)))(prev_layer)
                conv_layer = (Conv2D(
                    filters, (size, size),
                    strides=(stride, stride),
                    kernel_regularizer=l2(weight_decay),
                    use_bias=not batch_normalize,
                    weights=conv_weights,
                    activation=act_fn,
                    padding=padding))(prev_layer)

                if batch_normalize:
                    conv_layer = (BatchNormalization(
                        weights=bn_weight_list))(conv_layer)
                prev_layer = conv_layer

                if activation == 'linear':
                    all_layers.append(prev_layer)
                elif activation == 'leaky':
                    act_layer = LeakyReLU(alpha=0.1)(prev_layer)
                    prev_layer = act_layer
                    all_layers.append(act_layer)
                elif activation == 'swish':
                    act_layer = swish(prev_layer)
                    prev_layer = act_layer
                    all_layers.append(act_layer)
                elif activation == 'logistic':
                    act_layer = swish(prev_layer)
                    prev_layer = act_layer
                    all_layers.append(act_layer)

            elif section.startswith('route'):
                ids = [int(i) for i in self.cfg_parser[section]['layers'].split(',')]
                layers = [all_layers[i] for i in ids]
                if len(layers) > 1:
                    print('Concatenating route layers:', layers)
                    concatenate_layer = Concatenate()(layers)
                    all_layers.append(concatenate_layer)
                    prev_layer = concatenate_layer
                else:
                    skip_layer = layers[0]  # only one layer to route
                    all_layers.append(skip_layer)
                    prev_layer = skip_layer

            elif section.startswith('maxpool'):
                size = int(self.cfg_parser[section]['size'])
                stride = int(self.cfg_parser[section]['stride'])
                all_layers.append(
                    MaxPooling2D(
                        pool_size=(size, size),
                        strides=(stride, stride),
                        padding='same')(prev_layer))
                prev_layer = all_layers[-1]

            elif section.startswith('shortcut'):
                index = int(self.cfg_parser[section]['from'])
                activation = self.cfg_parser[section]['activation']
                assert activation == 'linear', 'Only linear activation supported.'
                all_layers.append(Add()([all_layers[index], prev_layer]))
                prev_layer = all_layers[-1]

            elif section.startswith('upsample'):
                stride = int(self.cfg_parser[section]['stride'])
                assert stride == 2, 'Only stride=2 supported.'
                all_layers.append(UpSampling2D(stride)(prev_layer))
                prev_layer = all_layers[-1]

            elif section.startswith('yolo'):
                out_index.append(len(all_layers)-1)
                all_layers.append(None)
                prev_layer = all_layers[-1]

            elif section.startswith('net'):
                pass

            else:
                raise ValueError(
                    'Unsupported section header type: {}'.format(section))

        if len(out_index)==0: out_index.append(len(all_layers)-1)
        self.model = Model(inputs=input_layer, outputs=[all_layers[i] for i in out_index])
        self.model.save('{}'.format(self.modelPath))

    def load_train_param(self):
        self.parameters = self.cfg_parser['net_0']

    def load_data(self):
        with open(os.path.join(self.dataPath,'train.txt')) as trainFile:
            self.train_annotations = trainFile.read().splitlines()

        with open(os.path.join(self.dataPath,'val.txt')) as valFile:
            self.val_annotations = valFile.read().splitlines()

        with open(os.path.join(self.dataPath,'classes.txt')) as classesFile:
            self.classes = classesFile.read().splitlines()

    def load_anchors(self):
        anchors = self.cfg_parser['yolo_0']['anchors']
        anchors = [float(x) for x in anchors.split(',')]
        self.anchors = np.array(anchors).reshape(-1, 2)
        




if __name__ == '__main__':
    darknet = Darknet('cfg/yolov3.cfg', 'weights/yolov3.weights', 'data')
    darknet.load_anchors()