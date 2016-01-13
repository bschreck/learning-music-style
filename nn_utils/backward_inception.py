from keras.layers.convolutional import Convolutional1D as Conv1D
from keras.layers.extra import TimeDistributedConvolutional1D as TDConv1D
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2, activity_l2
from keras.layers.convolutional import MaxPooling1D as Pool
from keras.layers.extra import TimeDistributedMaxPooling1D as TDPool
from keras.layers.extra import TimeDistributedUpsampling1D as TDUpsampling1D
from keras.layers.extra import TimeDistributedSlice1D as TDSlice1D
from keras.layers.normalization import BatchNormalization
from keras import backend as K

#TODO: take this out. originally I was going to do an autoencoder
#but I think it's not worth it and better to just do a classifier
class TimeDistributedBackwardsInception(object):
    #TODO: redo
    def __init__(graph, input_node, dim_ordering, output_num_channels, num_base_filters):
        #input should be the same dimension asn output of concatentation of forwards inception layer
        self.graph = graph
        self.input_node = input_node
        #output_num_channels should be the number of channels
        #that the original signal fed into the forward inception unit had
        self.output_num_channels = output_num_channels

        self.num_base_filters = num_base_filters

        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

        self.border_mode = 'same'
        self.W_regularizer = l2(0.01)
        self.b_regularizer = l2(0.01)
        self.activity_regularizer = activity_l2(0.01)
        self.W_constraint = None
        self.b_constraint = None
        self.init = 'glorot_uniform'
        self.activator = Activation('hard_sigmoid')

        self.split_inputs()
        left_branch = self.left_branch()
        left_center_branch = self.left_center_branch()
        right_center_branch = self.right_center_branch()
        right_branch = self.right_branch()
        #avg or sum or max?
        self.result = self.combine_branches(left_branch, left_center_branch,
                    right_center_branch, right_branch, 'sum')
    def split_inputs(self):
        #split inputs into 4 different inputs to the 4 branches
        self.left_input_node = 'left_input_slice'
        graph.add_node(TDSlice1D(self.num_base_filters,0), name=self.left_input_node, input=input_node)
        self.left_input_node = 'left_center_input_slice'
        graph.add_node(TDSlice1D(self.num_base_filters,self.num_base_filters), name=self.left_center_input_node, input=input_node)
        self.left_input_node = 'right_center_input_slice'
        graph.add_node(TDSlice1D(self.num_base_filters,self.num_base_filters*2), name=self.right_center_input_node, input=input_node)
        self.left_input_node = 'right_input_slice'
        graph.add_node(TDSlice1D(self.num_base_filters,self.num_base_filters*3), name=self.right_input_node, input=input_node)
    def left_branch(self):
        layer_sizes = [1,3,3]
        depths = [self.num_base_filters,self.num_base_filters, self.output_num_channels]
        input_node = self.left_input_node
        base_name = "inception_left_branch_"
        for i,size in enumerate(layer_sizes):
            name = base_name + str(i)
            graph.add_node(TDConv1D(depths[i],size, init=self.init, activator=self.activator,
                border_mode=self.border_mode, subsample_length = 1,
                W_regularizer=self.W_regularizer, activity_regularizer=self.activity_regularizer,
                b_regularizer=self.b_regularizer, W_constraint=self.W_constraint,b_constraint=self.b_constraint,
                input_shape=input_shape, input_length=input_length, dim_ordering=self.dim_ordering), name=name,input=input_node)
            input_node = name
        graph.add_node(TDUpsampling1D(length=2, dim_ordering=self.dim_ordering),name=base_name+'upsample',input=input_node)
        return base_name+'upsample'
    def left_center_branch(self):
        layer_sizes = [1,3]
        depths = [self.num_base_filters,self.output_num_channels]
        input_node = self.left_center_input_node
        base_name = "inception_left_center_branch_"
        for i,size in enumerate(layer_sizes):
            name = base_name + str(i)
            graph.add_node(TDConv1D(depths[i],size, init=self.init, activator=self.activator,
                border_mode=self.border_mode, subsample_length = 1,
                W_regularizer=self.W_regularizer, activity_regularizer=self.activity_regularizer,
                b_regularizer=self.b_regularizer, W_constraint=self.W_constraint,b_constraint=self.b_constraint,
                input_shape=input_shape, input_length=input_length, dim_ordering=self.dim_ordering), name=name,input=input_node)
            input_node = name
        graph.add_node(TDUpsampling1D(length=2, dim_ordering=self.dim_ordering),name=base_name+'upsample',input=input_node)
        return base_name+'upsample'
    def right_center_branch(self):
        base_name = "inception_right_center_branch_"
        graph.add_node(TDUpsampling1D(length=2, dim_ordering=self.dim_ordering),name=base_name+'upsample',input=self.right_center_input_node)
        graph.add_node(TDConv1D(self.output_num_channels,1, init=self.init, activator=self.activator,
                border_mode=self.border_mode, subsample_length = 1,
                W_regularizer=self.W_regularizer, activity_regularizer=self.activity_regularizer,
                b_regularizer=self.b_regularizer, W_constraint=self.W_constraint,b_constraint=self.b_constraint,
                input_shape=input_shape, input_length=input_length, dim_ordering=self.dim_ordering),name=base_name+"1", input=base_name+"upsample")
        return base_name+'1'
    def right_branch(self):
        base_name = "inception_right_branch_"
        graph.add_node(TDConv1D(self.output_num_channels,1, init=self.init, activator=self.activator,
                border_mode=self.border_mode, subsample_length = 1,
                W_regularizer=self.W_regularizer, activity_regularizer=self.activity_regularizer,
                b_regularizer=self.b_regularizer, W_constraint=self.W_constraint,b_constraint=self.b_constraint,
                input_shape=input_shape, input_length=input_length, dim_ordering=self.dim_ordering),name=base_name+"1", input=self.right_input_node)
        graph.add_node(TDUpsampling1D(length=2, dim_ordering=self.dim_ordering),name=base_name+'upsample',input=base_name+'1')
        return base_name+'upsample'
    def combine_branches(self, left_branch, left_center_branch,
                                right_center_branch, right_branch, combine_method):
        #TODO: CHECK ON CONCAT AXIS
        if self.dim_ordering == 'tf':
            concat_axis = 3
        elif self.dim_ordering == 'th':
            concat_axis = 2
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        name = "inception_decode_output"
        self.graph.add(Activation('linear'), input=[left_branch, left_center_branch, right_center_branch, right_branch],
                merge_mode=combine_method, concat_axis=concat_axis, name=name)
        return name
