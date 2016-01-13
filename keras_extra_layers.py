# Extra Layers that I have added to Keras
# Layers that have been added to the Keras master branch will be noted both in the ReadMe and removed from extra.py.
#
# Copyright Aran Nayebi, 2015
# anayebi@stanford.edu
#
# If you already have Keras installed, for this to work on your current installation, please do the following:
# 1. Upgrade to the newest version of Keras (since some layers may have been added from here that are now commented out):
#    sudo pip install --upgrade git+git://github.com/fchollet/keras.git
# or, if you don't have super user access, just run:
#    pip install --upgrade git+git://github.com/fchollet/keras.git --user
#
# 2. Add this file to your Keras installation in the layers directory (keras/layers/)
#
# 3. Now, to use any layer, just run:
#    from keras.layers.extra import layername
#
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import numpy as np

from .. import activations, initializations, regularizers, constraints
from theano.tensor.T import TensorType
from .. import backend as K
from ..utils.generic_utils import make_tuple
from ..regularizers import ActivityRegularizer, Regularizer

from six.moves import zip

from ..layers.core import Layer
def get_placeholder(ndim):
    if ndim < 5:
        return K.placeholder(ndim=ndim)
    elif K._BACKEND == 'tensorflow':
        return K.placeholder(ndim=ndim)
    elif K._BACKEND == 'theano':
        return TensorType('float32', (False,)*ndim)()
    else:
        raise Exception('Invalid backend: ' + K._BACKEND)


def conv_output_length(input_length, filter_size, border_mode, stride):
    if input_length is None:
        return None
    assert border_mode in {'same', 'full', 'valid'}
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'full':
        output_length = input_length + filter_size - 1
    elif border_mode == 'valid':
        output_length = input_length - filter_size + 1
    return (output_length + stride - 1) // stride

def pool_output_length(input_length, pool_size, border_mode, stride):
    if input_length is None:
        return None
    if border_mode == 'valid':
        output_length = input_length - pool_size + 1
        output_length = (output_length + stride - 1) // stride
    elif border_mode =='same':
        if pool_size == input_length:
            output_length = min(input_length, stride - stride % 2)
            if output_length <= 0:
                output_length = 1
        elif stride >= pool_size:
            output_length = (input_length + stride - 1) // stride
        else:
            output_length = (input_length - pool_size + stride - 1) // stride
            if output_length <= 0:
                output_length = 1
            else:
                output_length += 1
    else:
        raise Exception('Invalid border mode: ' + str(border_mode))
    return output_length

class TimeDistributedFlatten(Layer):
    # This layer reshapes input to be flat across timesteps (cannot be used as the first layer of a model)
    # Input shape: (num_samples, num_timesteps, *)
    # Output shape: (num_samples, num_timesteps, num_input_units)
    # Potential use case: For stacking after a Time Distributed Convolution/Max Pooling Layer or other Time Distributed Layer
    def __init__(self, **kwargs):
        super(TimeDistributedFlatten, self).__init__(**kwargs)

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], input_shape[1], np.prod(input_shape[2:]))

    def get_output(self, train=False):
        X = self.get_input(train)
        size = K.prod(X[0].shape) // X[0].shape[0]
        nshape = (X.shape[0], X.shape[1], size)
        return K.reshape(X, nshape)
class TimeDistributedUpsampling1D(Layer):
    input_ndim = 4

    def __init__(self, length=2, dim_ordering='th',**kwargs):
        super(TimeDistributedUpSampling1D, self).__init__(**kwargs)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering
        self.length = length
        self.input = K.placeholder(ndim=4)

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.dim_ordering == 'tf':
            return (input_shape[0], input_shape[1], input_shape[2], self.length * input_shape[3])
        elif self.dim_ordering == 'th':
            return (input_shape[0], input_shape[1], self.length * input_shape[2], input_shape[3])
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def get_output(self, train=False):
        X = self.get_input(train)

        if self.dim_ordering == 'tf':
            X = K.permute_dimensions(X, (0,1,3,2))
        elif self.dim_ordering != 'th':
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        newshape = (X.shape[0]*X.shape[1], X.shape[2], X.shape[3])
        Y = K.reshape(X, newshape) #collapse num_samples and num_timesteps

        output = K.repeat_elements(Y, self.length, axis=1)

        if self.dim_ordering == 'tf':
            output = K.permute_dimensions(output, (0, 2, 1))
        newshape = (X.shape[0], X.shape[1], output.shape[1], output.shape[2])
        #shape is (num_samples, num_timesteps, stack_size, new_nb_row, new_nb_col) for th
        # and (num_samples, num_timesteps, new_nb_row, new_nb_col, stack_size) for tf
        return K.reshape(output, newshape)

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'length': self.length}
        base_config = super(TimeDistributedUpSampling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class TimeDistributedSlice1D(Layer):
    input_ndim = 4
    def __init__(self, num_items, offset, axis,**kwargs):
        super(Slice, self).__init__(**kwargs)
        self.num_items = num_items
        self.offset = offset
        self.axis = axis
        self.input = K.placeholder(ndim=4)
    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.num_items+self.offset <= self.input_shape[self.axis]:
            raise Exception("Not enough items in input to slice: offset %d, num_items %d, input_size %d"%(self.offset,self.num_items,self.input_shape[self.axis]))
        output_shape = (input_shape[0], input_shape[1], input_shape[2], input_shape[3])
        output_shape[self.axis] = self.num_items
        return output_shape

    def get_output(self, train=False):
        X = self.get_input(train)
        if K._BACKEND == 'tensorflow':
            begin = [0,0,0,0]
            begin[self.axis] = self.offset
            size = [-1,-1,-1,-1]
            size[self.axis] = self.num_items
            output = tf.slice(X, begin, size, name=None)
        elif K._BACKEND == 'theano':
            roll_axis = [0,1,2,3]
            roll_axis[0] = axis
            roll_axis[axis] = 0
            permuted = K.permute_dimensions(X,roll_axis)
            output = X[self.offset:self.offset+self.num_items, :,:,:]
            output = K.permute_dimensions(X,roll_axis)
        else:
            raise Exception("Backend not implemented: %s"%backend)

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'num_items': self.num_items,
                  'offset': self.offset,
                  'axis': self.axis}
        base_config = super(TimeDistributedSlice1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
class TimeDistributedConvolution2D(Layer):
    # This layer performs 2D Convolutions with the extra dimension of time
    # For TF backend:
    # Input shape: (num_samples, num_timesteps, num_rows, num_cols, stack_size)
    # Output shape: (num_samples, num_timesteps, num_rows, num_cols, num_filters), Note: num_rows and num_cols could have changed
    # For TH backend:
    # Input shape: (num_samples, num_timesteps, stack_size, num_rows, num_cols)
    # Output shape: (num_samples, num_timesteps, num_filters, num_rows, num_cols), Note: num_rows and num_cols could have changed
    # Potential use case: For connecting a Convolutional Layer with a Recurrent or other Time Distributed Layer

    input_ndim = 5

    def __init__(self, nb_filter, nb_row, nb_col,
        init='glorot_uniform', activation='linear', weights=None,
        border_mode='valid', subsample=(1, 1), dim_ordering='th',
        W_regularizer=None, b_regularizer=None, activity_regularizer=None,
        W_constraint=None, b_constraint=None, **kwargs):

        if border_mode not in {'valid', 'full', 'same'}:
            raise Exception('Invalid border mode for TimeDistributedConvolution2D:', border_mode)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.border_mode = border_mode
        self.subsample = tuple(subsample)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        self.initial_weights = weights
        super(TimeDistributedConvolution2D,self).__init__(**kwargs)

    def build(self):
        stack_size = self.input_shape[2]
        self.input = get_placeholder(5)
        if self.dim_ordering == 'tf':
            self.W_shape = (self.nb_filter, self.nb_row, self.nb_col,stack_size)
        elif self.dim_ordering == 'th':
            self.W_shape = (self.nb_filter, stack_size, self.nb_row, self.nb_col)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        self.W = self.init(self.W_shape)
        self.b = K.zeros((self.nb_filter,))

        self.params = [self.W, self.b]

        self.regularizers = []

        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.dim_ordering == 'tf':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.dim_ordering == 'th':
            rows = input_shape[3]
            cols = input_shape[4]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        rows = conv_output_length(rows, self.nb_row, self.border_mode, self.subsample[0])
        cols = conv_output_length(cols, self.nb_col, self.border_mode, self.subsample[1])
        if self.dim_ordering == 'tf':
            return (input_shape[0], input_shape[1], rows, cols, self.nb_filter)
        else:
            return (input_shape[0], input_shape[1], self.nb_filter, rows, cols)

    def get_output(self, train=False):
        X = self.get_input(train)
        newshape = (X.shape[0]*X.shape[1], X.shape[2], X.shape[3], X.shape[4])
        Y = K.reshape(X, newshape) #collapse num_samples and num_timesteps
        conv_out = K.conv2d(Y, self.W, strides=self.subsample,
                            border_mode=self.border_mode,
                            dim_ordering=self.dim_ordering,
                            image_shape=newshape,
                            filter_shape=self.W_shape)
        if self.dim_ordering == 'th':
            output = conv_out + K.reshape(self.b, (1, self.nb_filter, 1, 1))
        elif self.dim_ordering == 'tf':
            output = conv_out + K.reshape(self.b, (1, 1, 1, self.nb_filter))
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        output = self.activation(output)
        newshape = (X.shape[0], X.shape[1], output.shape[1], output.shape[2], output.shape[3])
        return K.reshape(output, newshape)


    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "nb_filter": self.nb_filter,
                  "nb_row": self.nb_row,
                  "nb_col": self.nb_col,
                  "init": self.init.__name__,
                  "activation": self.activation.__name__,
                  "border_mode": self.border_mode,
                  "subsample": self.subsample,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  "W_constraint": self.W_constraint.get_config() if self.W_constraint else None,
                  "b_constraint": self.b_constraint.get_config() if self.b_constraint else None}
        base_config = super(TimeDistributedConvolution2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class TimeDistributedConvolution1D(Layer):
    # This layer performs 1D Convolutions with the extra dimension of time
    # For TF backend:
    # Input shape: (num_samples, num_timesteps, num_steps, stack_size)
    # Output shape: (num_samples, num_timesteps, num_steps, num_filters), Note: num_steps could have changed
    # For Th backend:
    # Input shape: (num_samples, num_timesteps, stack_size, num_steps)
    # Output shape: (num_samples, num_timesteps, num_filters, num_steps), Note: num_steps could have changed
    # Potential use case: For connecting a Convolutional Layer with a Recurrent or other Time Distributed Layer

    input_ndim = 4

    def __init__(self, nb_filter, nb_steps,
        init='glorot_uniform', activation='linear', weights=None,
        border_mode='valid', subsample=1, dim_ordering='th',
        W_regularizer=None, b_regularizer=None, activity_regularizer=None,
        W_constraint=None, b_constraint=None, **kwargs):

        if border_mode not in {'valid', 'full', 'same'}:
            raise Exception('Invalid border mode for TimeDistributedConvolution2D:', border_mode)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

        self.nb_filter = nb_filter
        self.nb_steps = nb_steps
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.border_mode = border_mode
        self.subsample = tuple(subsample)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        self.initial_weights = weights
        super(TimeDistributedConvolution1D,self).__init__(**kwargs)

    def build(self):
        stack_size = self.input_shape[2]
        self.input = K.placeholder(ndim=4)
        self.W_shape = (self.nb_filter, stack_size, self.nb_steps)
        self.W = self.init(self.W_shape)
        self.b = K.variable(np.zeros((self.nb_filter,)))

        self.params = [self.W, self.b]

        self.regularizers = []

        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.dim_ordering == 'tf':
            steps = input_shape[2]
        elif self.dim_ordering == 'th':
            steps = input_shape[3]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        steps = conv_output_length(steps, self.nb_steps, self.border_mode, self.subsample)
        if self.dim_ordering == 'tf':
            return (input_shape[0], input_shape[1], steps, self.nb_filter)
        else:
            return (input_shape[0], input_shape[1], self.nb_filter, steps)

    def get_output(self, train=False):
        #if dim_ordering = 'tf':
            #input shape: (batch_size, num_samples, num_steps (rows), channels)
        #if dim_ordering = 'th':
            #input shape: (batch_size, num_samples, channels, num_steps (rows))
        #add empty dimension for "cols"
        #permute to: (batch_size, num_samples, channels, rows, "cols")
        #reshape to: (batch_size*num_samples, channels, rows, cols)
        X = self.get_input(train)
        X = K.expand_dims(X, -1)  # add a dimension to the right
        if self.dim_ordering == 'tf':
            X = K.permute_dimensions(X, (0, 1, 3, 2, 4))
        elif self.dim_ordering != 'th':
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        newshape = (X.shape[0]*X.shape[1], X.shape[2], X.shape[3], X.shape[4])
        Y = K.reshape(X, newshape)
        conv_out = K.conv2d(Y, self.W, strides=self.subsample,
                            border_mode=self.border_mode,
                            dim_ordering='th',
                            image_shape=newshape,
                            filter_shape = self.W_shape)
        output = conv_out + K.reshape(self.b, (1, self.nb_filter, 1, 1))
        output = K.squeeze(output, 4)  # remove the dummy 4th dimension
        if self.dim_ordering == 'tf':
            output = K.permute_dimensions(output, (0, 2, 1))
        output = self.activation(output)
        #reshape to separate batch_size and num_samples
        newshape = (X.shape[0], X.shape[1], output.shape[1], output.shape[2])
        return K.reshape(output,newshape)


    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "nb_filter": self.nb_filter,
                  "nb_steps": self.nb_steps,
                  "init": self.init.__name__,
                  "activation": self.activation.__name__,
                  "border_mode": self.border_mode,
                  "subsample": self.subsample,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  "W_constraint": self.W_constraint.get_config() if self.W_constraint else None,
                  "b_constraint": self.b_constraint.get_config() if self.b_constraint else None}
        base_config = super(TimeDistributedConvolution1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class TimeDistributedMaxPooling2D(Layer):
    # This layer performs 2D Max Pooling with the extra dimension of time
    # Input shape: (num_samples, num_timesteps, stack_size, num_rows, num_cols)
    # Output shape: (num_samples, num_timesteps, stack_size, new_num_rows, new_num_cols)
    # Potential use case: For stacking after a Time Distributed Convolutional Layer or other Time Distributed Layer

    input_ndim = 5

    def __init__(self, pool_size=(2, 2), stride=None, dim_ordering='th', border_mode='valid', **kwargs):
        super(TimeDistributedMaxPooling2D,self).__init__(**kwargs)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering
        self.input = get_placeholder(5)
        self.pool_size = tuple(pool_size)
        if stride is None:
            stride = self.pool_size
        self.stride = tuple(stride)
        self.border_mode = border_mode

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.dim_ordering == 'tf':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.dim_ordering == 'th':
            rows = input_shape[3]
            cols = input_shape[4]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        rows = pool_output_length(rows, self.pool_size[0], self.border_mode, self.stride[0])
        cols = pool_output_length(cols, self.pool_size[1], self.border_mode, self.stride[1])
        if self.dim_ordering == 'tf':
            return (input_shape[0], input_shape[1], rows, cols, input_shape[4])
        else:
            return (input_shape[0], input_shape[1], input_shape[2], rows, cols)

    def get_output(self, train):
        X = self.get_input(train)
        newshape = (X.shape[0]*X.shape[1], X.shape[2], X.shape[3], X.shape[4])
        Y = K.reshape(X, newshape) #collapse num_samples and num_timesteps
        output = K.pool2d(Y, self.pool_size, self.stride, border_mode=self.border_mode,
                    dim_ordering = self.dim_ordering,pool_mode='max')
        newshape = (X.shape[0], X.shape[1], output.shape[1], output.shape[2], output.shape[3])
        #shape is (num_samples, num_timesteps, stack_size, new_nb_row, new_nb_col) for th
        # and (num_samples, num_timesteps, new_nb_row, new_nb_col, stack_size) for tf
        return K.reshape(output, newshape)

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "pool_size": self.pool_size,
                  "ignore_border": self.ignore_border,
                  "stride": self.stride}
        base_config = super(TimeDistributedMaxPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
class TimeDistributedMaxPooling1D(Layer):
    # This layer performs 1D Max Pooling with the extra dimension of time
    # If TF backend:
    # Input shape: (num_samples, num_timesteps, num_steps, stack_size)
    # Output shape: (num_samples, num_timesteps, num_steps, stack_size)
    # If TH backend:
    # Input shape: (num_samples, num_timesteps, stack_size, num_steps)
    # Output shape: (num_samples, num_timesteps, stack_size, new_num_steps)
    # Potential use case: For stacking after a Time Distributed Convolutional Layer or other Time Distributed Layer

    input_ndim = 4

    def __init__(self, pool_size=2, stride=None, dim_ordering = 'th', border_mode='valid', **kwargs):
        super(TimeDistributedMaxPooling1D,self).__init__(**kwargs)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering
        self.input = get_placeholder(4)
        self.pool_size = tuple(pool_size)
        if stride is None:
            stride = self.pool_size
        self.stride = tuple(stride)
        self.border_mode = border_mode

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.dim_ordering == 'tf':
            steps = input_shape[2]
        elif self.dim_ordering == 'th':
            steps = input_shape[3]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        steps = pool_output_length(steps, self.pool_size, self.border_mode, self.stride)
        if self.dim_ordering == 'tf':
            return (input_shape[0], input_shape[1], steps, input_shape[3])
        else:
            return (input_shape[0], input_shape[1], input_shape[2], steps)

    def get_output(self, train):
        X = self.get_input(train)
        X = K.expand_dims(X, -1)
        if self.dim_ordering == 'tf':
            X = K.permute_dimensions(X, (0,1,3,2,4))
        elif self.dim_ordering != 'th':
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        newshape = (X.shape[0]*X.shape[1], X.shape[2], X.shape[3], X.shape[4])
        Y = K.reshape(X, newshape) #collapse num_samples and num_timesteps
        output = K.pool2d(Y, self.pool_size, self.stride, border_mode=self.border_mode,
                    dim_ordering = 'th',pool_mode='max')
        output = K.squeeze(output, 4)
        if self.dim_ordering == 'tf':
            output = K.permute_dimensions(output, (0, 2, 1))
        newshape = (X.shape[0], X.shape[1], output.shape[1], output.shape[2])
        #shape is (num_samples, num_timesteps, stack_size, new_nb_row, new_nb_col) for th
        # and (num_samples, num_timesteps, new_nb_row, new_nb_col, stack_size) for tf
        return K.reshape(output, newshape)

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "pool_size": self.pool_size,
                  "ignore_border": self.ignore_border,
                  "stride": self.stride}
        base_config = super(TimeDistributedMaxPooling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
