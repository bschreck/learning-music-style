import copy
from keras.layers.extra import TimeDistributedMaxPooling1D as TDMaxPool1D
from keras.layers.extra import TimeDistributedAvgPooling1D as TDAvgPool1D
from keras.layers.extra import TimeDistributedConvolutional1D as TDConv1D
class NetFromConfig(object):
    def __init(self):
        self.steps_axis = None
        self.depth_axis = None
        self.dim_ordering = None
        self.init = None
        self.activator = None
        self.W_regularizer = None
        self.activity_regularizer = None
        self.b_regularizer = None
    def add_config_tuple(self,input_node, input_shape, name, graph, config_tuple):
        output_shape = copy.copy(input_shape)
        size = config_tuple[0]
        stride = config_tuple[2]
        border_mode = config_tuple[3]
        if border_mode == 'valid':
            cutoff = 2*(size // 2)
            output_shape[self.steps_axis] = math.ceil((output_shape[self.steps_axis]-cutoff)/stride)
        elif border_mode == 'same':
            output_shape[self.steps_axis] = math.ceil(output_shape[self.steps_axis]/stride)
        if type(config_tuple[1]) == int:
            depth = config_tuple[1]
            node = TDConv1D(depth,size,subsample_length=stride,border_mode=border_mode,
                    input_shape=input_shape, dim_ordering=self.dim_ordering,
                    init=self.init, activator=self.activator,W_regularizer=self.W_regularizer,
                    activity_regularizer=self.activity_regularizer,b_regularizer=self.b_regularizer)
            output_shape[self.depth_axis] = depth
        else:
            if config_tuple[1] == 'max':
                layer = TDMaxPool1D
            elif config_tuple[1] == 'avg':
                layer = TDAvgPool1D
            else:
                raise Exception("wrong pooling layer %s"%config_tuple[1])
            node = layer(pool_length=size,stride=stride,border_mode=border_mode)
        graph.add_node(node, name=name, input=input_node)
        return output_shape
    def add_backwards_config_tuple(self,input_node,input_shape,output_shape, name,graph,config_tuple):
        output_shape = copy.copy(input_shape)
        size = config_tuple[0]
        stride = config_tuple[2]
        border_mode = config_tuple[3]
        if border_mode == 'valid':
            cutoff = 2*(size // 2)
            output_shape[self.steps_axis] = math.ceil((output_shape[self.steps_axis]-cutoff)/stride)
        pass
