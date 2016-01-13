from keras.models import Graph
from keras.layers.core import TimeDistributedDense
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.layers.containers import Graph as GraphContainer
from keras import backend as K
from inception import TimeDistributedInception as Inception
from backwards_inception import TimeDistributedBackwardsInception as BackwardsInception
from keras.layers.extra import TimeDistributedConvolutional1D as TDConv1D
from keras.layers.extra import TimeDistributedSlice1D as TDSlice1D
from googlenet import TimeDistributedGoogleNet1D as TDGoogleNet1D

# #132300 x 2 (265k)
# #subtract 1 sample to make odd

# #132299 x 2 (900k)
# conv(3,16,stride=2,pad='v')
# #66149 x 16 (1M)
# conv(3, 32,stride=2,pad='s')
# #33075 x 32 (1M)
# conv(3, 64,stride=2,pad='v')
# #16537 x 64 (1M)
# max_pool(3,stride=2,pad='s')
# #8269 x 64 (500k)
# conv(1,80,pad='s')
# #8269 x 80 (600k)
# conv(3,128,stride=2, pad='s')
# #4135 x 128 (500k)
# max_pool(3,stride=2,pad='v')
# #2067 x 128 (260k)

# #inception1:
# 1. conv(1,64, stride=4,pad='s') #517 x 64
# 2. conv(1,48).conv(5,64, stride=4) # 517 x 64
# 3. conv(1,64).conv(3,96).conv(3,96,stride=4) #517 x 96
# 4. max_pool(3, stride=4, pad='s') #517 x 128
# #517 x 352 (180k)

# #inception2:
# 1. conv(3,384,stride=4,pad='v') # 129 x 384
# 2. conv(1,64).conv(3,96).conv(3,96,stride=4,pad='v') # 129 x 96
# 3. max_pool(3, stride=4, pad='v').conv(1,64) # 129 x 64
# #129 x 768 (70k)

# #inception3:
# 1. conv(1, 192).conv(3,320, stride=2, pad='v') # 64 x 320
# 2. conv(1, 192).conv(7,192).conv(3,192,stride=2,pad='v') # 64 x 192
# 3. max_pool(3, stride=2,pad='v') # 64 x 768
# #64 x 1280 (82k)

# #inception4:
# 1. conv(1, 320) # 33 x 320
# 2. conv(1,384).{1. conv(3, 384), 2. conv(3,384)} # 33 x 768
# 3. conv(1, 448).conv(3,384).{1. conv(3,384), 2. conv(3,384)} # 33 x 768
# 4. avg_pool(3).conv(1, 192) # 33 x 192
# #64 x 2048 (130k)

# avg_pool(64, pad='v')
#2048 x 1 (2048)
#dropout(.8)

config = {
        'use_lstm': False,
        'batch_size': 32,
        'num_time_chunks': 30,
        'num_seconds_per_chunk': 3,
        'sample_rate': 44100,
        'num_input_channels': 2,
        'num_recurrent_layers':1,
        'pooling_type': 'avg',
        'dim_ordering': 'th',
        'init': 'glorot_uniform',
        'W_regularizer_value':0.01,
        'b_regularizer_value':0.01,
        'activity_regularizer_value':0.01,
        'encoder_activator': 'prelu',
        'decoder_activator': 'hard_sigmoid',
        'loss': 'mse',
        'constrain_decoder_to_encoder': True,
        'googlenet_config': {
            'initial_conv_layers':[
                (3,16,2,'valid'),
                (3,32,2,'same'),
                (3,64,2,'valid'),
                (3,'max',2,'same'),
                (1,80,1,'same'),
                (3,128,2,'same'),
                (3,'max',2,'valid')
            ],
            'inception_config':
                [
                    [
                     [(1,64,4,'same')],
                     [(1,48,1,'same'),(5,64,4,'same')],
                     [(1,64,1,'same'),(3,96,4,'same'),(3,96,4,'same')],
                     [(3,'max',4,'same')]
                    ],
                    [
                     [(3,384,4,'valid')],
                     [(1,64,1,'same'),(3,96,1,'same'),(3,96,4,'valid')],
                     [(3,'max',4,'valid'),(1,64,1,'same')]
                    ],
                    [
                     [(1,192,1,'same'), (3,320,2,'valid')],
                     [(1,192,1,'same'),(7,192,1,'same'),(3,192,2,'valid')],
                     [(3,'max',2,'valid')]
                    ],
                    [
                     [(1,320,1,'same')],
                     [(1,384,1,'same'),[(3,384,1,'same')],[(3,384,1,'same')]],
                     [(1,448,1,'same'),(3,384,1,'same'),[(3,384,1,'same')],[(3,384,1,'same')]],
                     [(3,'avg',1,'same'), (1,192,1,'same')]
                    ]
                ],
            'output_pooling': {
                'size': 64,
                'type': 'avg'
                }
            }
        }


def calc_input_shape(config):
    batch_size = config['batch_size']
    num_time_chunks = config['num_time_chunks']
    num_input_channels = config['num_input_channels']
    steps_per_sample = config['sample_rate']*config['num_seconds_per_chunk']


    if config['dim_ordering'] == 'th':
        depth_axis = 2
        steps_axis = 3
        input_shape = (batch_size,
                       num_time_chunks,
                       num_input_channels,
                       steps_per_sample-1)#minus one to make it odd
    elif config['dim_ordering'] == 'tf':
        depth_axis = 3
        steps_axis = 2
        input_shape = (batch_size,
                       num_time_chunks,
                       num_input_channels,
                       steps_per_sample-1)#minus one to make it odd
    else:
        raise Exception("dim_ordering must be in {tf, th}, not %s"%config['dim_ordering'])
    return input_shape,depth_axis,steps_axis
def create_cnn_network(config):
    #also look at GaussianNoise and GaussianDropout

    # epsilon = 1e-06
    # mode = 0
    # momentum=0.9
    # weights=weights
    # model.add(BatchNormalization(epsilon=epsilon, mode=mode, momentum=momentum, weights=weights))
    #TODO: add dropout and batchnorm to encoders and decoders


    graph = Graph()
    input_shape,depth_axis,steps_axis = calc_input_shape(config)
    graph.add_input(name='input', input_shape=input_shape)
    if config['dim_ordering'] == 'tf':
        steps_axis = 2
    elif config['dim_ordering'] == 'th':
        steps_axis = 3
    else:
        raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
    #make num input steps odd
    graph.add_node(TDSlice1D(input_shape[steps_axis]-1,0), name='sliced_input', input='input')
    #TODO: add input distorter here as data augmentation
    googlenet = TDGoogleNet1D(graph, 'sliced_input', input_shape, config)
    googlenet_output = googlenet.result
    output_shape = googlenet.output_shape
    #make sure we output a flattened layer
    assert output_shape[steps_axis] == 1
    squeezed_shape = (output_shape[0], output_shape[1], output_shape[depth_axis])
    squeezed_output = 'squeezed_output'
    graph.add_node(Lambda(lambda x: K.squeeze(x,steps_axis),output_shape = squeezed_shape),
            name=squeezed_output,input=googlenet_output)
    recurrent_input = squeezed_output
    recurrent_input_shape = squeezed_shape

    #could add a fully connected layer to change dimension instead of using the output dimension
    #graph.add(TimeDistributedDense(input_dim=squeezed_shape[2], output_dim=config['num_hidden_dimensions']),
    #               name="encoder2rnn_fc",input=squeezed_output)
    #recurrent_input = "encoder2rnn_fc"
    #recurrent_input_shape = squeezed_shape
    #recurrent_input_shape[2] = config['num_hidden_dimensions']

    if use_lstm:
        cell = LSTM
    else:
        cell = GRU
    for cur_unit in xrange(num_recurrent_units):
        rnn_cell_name = "rnn_cell_%d"%cur_unit
        graph.add_node(cell(input_dim=recurrent_input_shape[2], output_dim=recurrent_input_shape[2], return_sequences=True),input=recurrent_input,name=name)
        recurrent_input = rnn_cell_name
    graph.add_output(name="output", input=rnn_cell_name)
    graph.compile(optimizer='rmsprop', loss={'output':'mse'})

    #TODO:add classifier here

    return graph

# def create_decoder(encoder_output_shape, squeezed_googlenet_output_shape,config):
    # graph = GraphContainer()
    # graph.add_input(name='input', input_shape=encoder_output_shape)
    # expand_input = 'input'
    # #TODO: figure out how to initialize weights to inverse of encoder

    # #if added fully connected layer to change dimension before inputting to rnn
    # #graph.add(TimeDistributedDense(input_dim=encoder_output_shape[depth_axis], output_dim=squeezed_googlenet_output_shape[depth_axis]),
    # #       name="rnn2decoder_fc", input='input')
    # #expand_input = "rnn2decoder_fc"

    # expanded_shape = squeezed_googlenet_output_shape + [1]
    # if steps_axis == 2:
        # expanded_shape[3] = expanded_shape[2]
        # expanded_shape[2] = 1
    # model.add_node(Lambda(lambda x: expand_dims_and_permute(x, steps_axis),output_shape = expanded_shape), name="expand_decode_input", input=expand_input)
    # backwards_googlenet = TDBackwardsGoogleNet1D(graph, 'expand_decode_input', expanded_shape, config)
    # return graph

# def expand_dims_and_permute(input_tensor, steps_axis):
    # expanded = K.expand_dims(input_tensor)
    # if steps_axis == 3:
        # return expand_dims
    # else:
        # return K.permute_dimensions(K.expand_dims(x), (0,1,3,2))


