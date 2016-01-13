from __future__ import true_divide
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2, activity_l2
from keras.layers.extra import TimeDistributedConvolutional1D as TDConv1D
from keras.layers.extra import TimeDistributedMaxPooling1D as TDMaxPool1D
from keras.layers.extra import TimeDistributedAvgPooling1D as TDAvgPool1D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from general_net import NetFromConfig

class TimeDistributedGoogleNet1D(NetFromConfig):
    #TODO: return shapes of all output layers as list
    def __init__(self,graph, input_node, input_shape, config):
        self.graph = graph
        self.input_node = input_node
        self.config = config

        self.input_shape = input_shape
        self.dim_ordering = config['dim_ordering']
        if self.dim_ordering == 'th':
            self.depth_axis = 2
            self.steps_axis = 3
        else:
            self.depth_axis = 3
            self.steps_axis = 2

        self.final_pool_size = config['googlenet_config']['output_pooling']['size']
        self.final_pool_type = config['googlenet_config']['output_pooling']['type']


        self.W_regularizer = l2(config['W_regularizer_value'])
        self.b_regularizer = l2(config['b_regularizer_value'])
        self.activity_regularizer = activity_l2(config['activity_regularizer_value'])
        self.init = config['init']
        if config['encoder_activator'] == 'prelu':
            self.activator = PReLU(init=self.init)
        #if want to try different activator need to specify here
        else:
            self.activator = Activation(config['encoder_activator'])

        output_name,output_shape = self.first_conv_layers()
        inception = TDInception(self.graph, output_name,output_shape,config)
        output_name,output_shape = inception.result
        self.result, self.output_shape = self.final_pool(output_name, output_shape)

    def first_conv_layers(self):
        input_node = self.input_node
        input_shape = self.input_shape
        layers = self.config['googlenet_config']['initial_conv_layers']
        for i,layer in enumerate(layers):
            name = "initial_layer_%d"%i
            input_shape = self.add_config_tuple(input_node, input_shape, name, self.graph, layer)
            input_node = name
        return input_node, input_shape
    def final_pool(self, input_name, input_shape):
        name = 'final_pool'
        if self.final_pool_type == 'max':
            layer = TDMaxPool1D
        else:
            layer = TDAvgPool1D
        self.graph.add_node(layer(pool_length=self.final_pool_size,
            stride=1, border_mode='valid', dim_ordering=self.dim_ordering), name=name, input=input_name)
        output_shape = copy.copy(input_shape)
        if output_shape[self.steps_axis] != self.final_pool_size:
            raise Exception('''sizes wrong, output not flat: final pool input shape:
(%d,%d,%d,%d), final pool size: %d, final output shape: (%d,%d,%d,%d)''' % (input_shape[0],input_shape[1],input_shape[2],
                input_shape[3], self.final_pool_size, output_shape[0], output_shape[1], output_shape[2], output_shape[3]))
        output_shape[self.steps_axis] = 1
        return name, output_shape

class TimeDistributedBackwardsGoogleNet1D(NetFromConfig):
    def __init__(self,graph, input_node, input_shape, forward_shapes,config):
        self.graph = graph
        self.input_node = input_node
        self.config = config

        self.input_shape = input_shape
        self.dim_ordering = config['dim_ordering']
        if self.dim_ordering == 'th':
            self.depth_axis = 2
            self.steps_axis = 3
        else:
            self.depth_axis = 3
            self.steps_axis = 2

        #TODO: from here
        self.initial_upsampling_size = config['googlenet_config']['output_pooling']['size']
        self.initial_upsampling_type = config['googlenet_config']['output_pooling']['type']


        self.W_regularizer = l2(config['W_regularizer_value'])
        self.b_regularizer = l2(config['b_regularizer_value'])
        self.activity_regularizer = activity_l2(config['activity_regularizer_value'])
        self.init = config['init']
        self.activator = Activation(config['decoder_activator'])

        output_name, output_shape = self.initial_upsampling()
        inception = TDBackwardsInception(self.graph, output_name,output_shape,forward_shapes, config)
        output_name,output_shape = inception.result
        self.result,self.output_shape = self.reverse_conv_layers(output_name,output_shape)

    def first_conv_layers(self, input_node, input_shape):
        layers = self.config['googlenet_config']['initial_conv_layers']
        layers.reverse()
        for i,layer in enumerate(layers):
            name = "deconv_layer_%d"%i
            input_shape = self.add_backwards_config_tuple(input_node, input_shape, name, self.graph, layer)
            input_node = name
        return input_node, input_shape
    def initial_upsampling(self):
        input_name = self.input_node
        input_shape = self.input_shape
        name = 'initial_upsampling'
        if self.initial_upsampling_type == 'max':
            layer = TDMaxUpsampling1D
        else:
            layer = TDAvgUpsampling1D
        self.graph.add_node(layer(length=self.initial_upsampling_size,
                dim_ordering=self.dim_ordering), name=name, input=input_name)
        output_shape = copy.copy(input_shape)
        output_shape[self.steps_axis] = self.initial_upsampling_size
        return name, output_shape
