from keras.layers.extra import TimeDistributedConvolutional1D as TDConv1D
from keras.layers.extra import TimeDistributedMaxPooling1D as TDMaxPool1D
from keras.layers.extra import TimeDistributedAvgPooling1D as TDAvgPool1D
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2, activity_l2
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import math
import copy
from general_net import NetFromConfig



class TimeDistributedInception(NetFromConfig):
    def __init__(self, graph, input_node, input_shape, config):
        self.graph = graph
        self.input_node = input_node
        self.input_shape = input_shape
        self.config= config

        self.dim_ordering = config['dim_ordering']
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.W_regularizer = l2(config['W_regularizer_value'])
        self.b_regularizer = l2(config['b_regularizer_value'])
        self.activity_regularizer = activity_l2(config['activity_regularizer_value'])
        self.init = config['init']
        if config['encoder_activator'] == 'prelu':
            self.activator = PReLU(init=self.init)
        #if want to try different activator need to specify here
        else:
            self.activator = Activation(config['encoder_activator'])
        self.inception_config = config['googlenet_config']['inception_config']
        self.result,self.output_shape = self.walk_config_graph()

    def walk_config_graph(self):
        input_node = self.input_node
        input_shape = self.input_shape
        for i,inception_module in enumerate(self.inception_config):
            base_name = "inception_%d_"%i
            module_outputs = []
            module_output_shapes = []
            for j,path in enumerate(inception_module):
                sub_name = base_name+"path_"+str(j)
                path_input_node = input_node
                path_input_shape = input_shape
                for k,layer in enumerate(path):
                    path_name = sub_name + "_layer_" + str(k)
                    if type(layer) == tuple:
                        path_output_shape = self.add_config_tuple(path_input_node, path_input_shape, path_name, self.graph, layer)
                        path_input_shape = path_output_shape
                        path_input_node = path_name
                        if k == len(path)-1:
                            module_outputs.append(path_name)
                            module_output_shapes.append(path_output_shape)
                    else:#type(layer) == list, which means split into two parallel branches
                        #assume these come at the end, otherwise we would have to run another for loop for each
                        for l,sublayer in enuemrate(layer):
                            sub_path_name = path_name + "_subpath_" + str(l)
                            path_output_shape = self.add_config_tuple(path_input_node, path_input_shape, sub_path_name, self.graph, layer)
                            module_outputs.append(sub_path_name)
                            module_output_shapes.append(path_output_shape)
            self.concat_branches(module_outputs, base_name+"concat")
            input_node = base_name+"concat"
            input_shape = self.merge_output_shapes(module_output_shapes)
        return input_node, input_shape
    def merge_output_shapes(self, output_shapes):
        if self.dim_ordering == 'tf':
            depth_axis = 3
        elif self.dim_ordering == 'th':
            depth_axis = 2
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        new_output_shape = output_shapes[0]
        for shape in output_shapes[1:]:
            for axis,size in enumerate(shape):
                if axis == depth_axis:
                    new_output_shape[axis] += size
                else:
                    assert size == new_output_shape[axis], "shapes dont match on dim %d: "%axis+str(new_output_shape)+"vs. "+str(shape)
        return new_output_shape


    def concat_branches(self, branches, name):
        if self.dim_ordering == 'tf':
            concat_axis = 3
        elif self.dim_ordering == 'th':
            concat_axis = 2
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        self.graph.add(Activation('linear'), input=branches,
                merge_mode='concat', concat_axis=concat_axis, name=name)
