## ---------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------- ##
#   FUNCTIONS - NN networks (direct classifiers, composite networks, autoencoders)
## ---------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------- ##

import numpy as np
import json, types
import tensorflow as tf

import keras.layers
from keras.layers import Input, Dense, TimeDistributed, BatchNormalization, merge, Lambda, concatenate, Bidirectional, LSTM, GRU, RNN
from keras.layers.core import*
from keras.layers.convolutional import*

from keras import backend as K
from keras.activations import relu
from keras.models import Sequential, Model

#### MOVED to ****functions_keras.py****
##SEED_tf=42; tf.compat.v1.set_random_seed(SEED_tf) 

relu_size = 16
dict_rnn = {'RNN':RNN, 'GRU':GRU, 'LSTM':LSTM}


## ############################################################################ ##
## --------------------------------------------------------------- ##
## RESIDUAL BLOCK, WAVENET CONFIGURATION
## --------------------------------------------------------------- ##
def residual_block(x, 
                   sizenet, 
                   m_dilation_rate, 
                   m_stack, 
                   causal = False, 
                   m_activation='wavenet', 
                   drop_frac = 0.25, 
                   kernel_size=3, 
                   type='convolution', 
                   name=''):   
    """---------------------------------------------------------------------------
        @summary: residual block for TCN implementation (cf. Wavenet reference). 
        @note: "WaveNet: A Generative Model for Raw Audio", 
                                van den Oord+2016, [arXiv:1609.03499v2]
        @param x: Keras layer.
        @param sizenet: int, size of the network.
        @param m_dilation_rate: int, dilation rate.
        @param m_stack: int, number of stacks/layers.
        @param causal: boolean, if causal padding. 
        @param m_activation: str, activation function.
        @param drop_frac: float(<1), fraction of dropout in keras architecture.
        @param kernel_size: int, size of 1D conv filters.
        @param name: str, layer name.
        @return (x_residual, x): tuple, residual model layer and skip connection.
   ---------------------------------------------------------------------------"""   
    x_init = x    
    m_padding='causal' if causal else 'same'
    if type=='deconvolution' :
        x = Conv1DTranspose(x,
                            filters=sizenet, 
                            kernel_size=kernel_size, 
                            dilation_rate=m_dilation_rate,
                            padding='same',
                            name=name+'B{}_L{}_dilC{}_N{}'.format(m_stack, m_dilation_rate, 
                                                                      kernel_size, sizenet))
    else:
        x = Conv1D(filters=sizenet, 
                   kernel_size=kernel_size, 
                   dilation_rate=m_dilation_rate,
                   padding=m_padding,
                   name=name+'B{}_L{}_dilC{}_N{}'.format(m_stack, m_dilation_rate, 
                                                             kernel_size, sizenet))(x)
                               
    if m_activation=='wavenet': 
        x = keras.layers.multiply([Activation('tanh', name=name+'B{}_L{}_activation_tanh'.format(m_stack, m_dilation_rate))(x), 
                                   Activation('sigmoid', name=name+ 'B{}_L{}_activation_sigmoid'.format(m_stack, m_dilation_rate))(x)],
                                  name=name+'B{}_L{}_activation_multiply'.format(m_stack, m_dilation_rate))
    else:
        x = Activation(m_activation, name=name+'B{}_L{}_activation_{}'.format(m_stack, m_dilation_rate, m_activation))(x)        
    
    x = SpatialDropout1D(rate=drop_frac, 
                         name=name+'B{}_L{}_sdrop{}'.format(m_stack, m_dilation_rate, int(drop_frac*100)))(x)
    
    if type=='deconvolution' :
        x = Conv1DTranspose(x,
                            filters=sizenet,
                            kernel_size=1,
                            padding='same',
                            name=name+'B{}_L{}_dilC{}_N{}'.format(m_stack, m_dilation_rate, 1, sizenet))
    else:
        x = Conv1D(filters=sizenet,
                   kernel_size=1,
                   padding='same',
                   name=name+'B{}_L{}_dilC{}_N{}'.format(m_stack, m_dilation_rate, 1, sizenet))(x)
                            
    x_residual = keras.layers.add([x_init, x], name=name+'B{}_L{}_add'.format(m_stack, m_dilation_rate)) 
    
    return x_residual, x


## ############################################################################ ##
## --------------------------------------------------------------- ##
## DECONVOLUTION 1D (:= transpose convolution), Flatten 2D 
## --------------------------------------------------------------- ##
def Conv1DTranspose(x, 
                    filters, 
                    kernel_size, 
                    strides=1, 
                    dilation_rate=1,
                    activation=None, 
                    padding='same', 
                    name=''):
    
    x_layer = Lambda(lambda x: K.expand_dims(x, axis=2), name=name+'_lambda')(x)
    x_layer = Conv2DTranspose(filters=filters,
                              kernel_size=(kernel_size, 1), 
                              activation=activation,
                              strides=(strides, 1),
                              dilation_rate=(dilation_rate, dilation_rate), 
                              padding=padding, 
                              name=name+'_deconv')(x_layer)
    x_layer = Lambda(lambda x: K.squeeze(x, axis=2), name=name)(x_layer)
    return x_layer


## ############################################################################ ##
## ############################################################################ ##
## --------------------------------------------------------------- ##
##     MLP [0] - METADATA
## --------------------------------------------------------------- ##
def classifier_MLP_meta(output_size,
                      add_meta = None,
                      **kwargs):
    
    param_dict = {key: value for key, value in locals().items()}
    param_dict.update(kwargs)
    param_dict = {k: v for k, v in param_dict.items()
                  if k not in ['kwargs']
                  and not isinstance(v, types.FunctionType)}
    
    meta_input = Input(shape=(add_meta,), name='meta_input') 
    
    x = meta_input 
    x = Dense(units=relu_size, 
              activation='relu', name='relu_dense')(x)
    x = Dense(units=output_size, 
              activation='softmax', 
              name='clf_softmax_dense')(x)

    #---------------MODEL---------------#
    input_layer = meta_input 
    output_layer = x
    model = Model(input_layer, output_layer)
    
    param_str = 'Classifier_MLP_out{}'.format(output_size)
    
    print('>>>> param_str = ', param_str)
    return model, param_str, param_dict



## ############################################################################ ##
## ############################################################################ ##
## --------------------------------------------------------------- ##
##     DIRECT CLASSIFIER [1] - RNN
## --------------------------------------------------------------- ##
def classifier_direct_RNN(input_dimension, output_size, sizenet,
                           nb_passbands = 1, max_lenght = None,
                           num_layers = 1,
                           drop_frac = 0.0,
                           model_type='LSTM',
                           bidirectional = True,
                           return_sequences=False,
                           add_meta = None,
                           add_dense = False,
                           **kwargs):
      
    param_dict = {key: value for key, value in locals().items()}
    param_dict.update(kwargs)
    param_dict = {k: v for k, v in param_dict.items()
                  if k not in ['kwargs']
                  and not isinstance(v, types.FunctionType)}
    
    layer = dict_rnn[model_type]
    meta_input = Input(shape=(add_meta,), name='meta_input') if add_meta is not None else None
    
    main_input_list = []
    for j in range(nb_passbands):
        if max_lenght is None:
            ndatapoints = None
        else:
            ndatapoints = max_lenght if type(max_lenght)==int else max_lenght[j]
        main_input = Input(shape=(ndatapoints, input_dimension), name='main_input_pb{}'.format(j))
        main_input_list.append(main_input)
    
    #---------------ENCODER---------------#
    encode_list = []
    for j in range(nb_passbands):
        encode = main_input_list[j] 
        for i in range(num_layers):
            wrapper = Bidirectional if bidirectional else lambda x: x
            if bidirectional:
                encode = wrapper(layer(units=sizenet, 
                              return_sequences=True if return_sequences else (i<num_layers-1)),
                        name='encode_pb{}_x{}'.format(j,i)) (encode)
            else:
                encode = wrapper(layer(units=sizenet, 
                              return_sequences=True if return_sequences else (i<num_layers-1),
                              name='encode_pb{}_x{}'.format(j,i))) (encode)
            if drop_frac > 0.0:
                encode = Dropout(rate=drop_frac, 
                            name='encode_drop_pb{}_x{}'.format(j,i))(encode)
        
        ## Output per band ##
        encode_list.append(encode)
    
    #---------------CLASSIFIER---------------#
    encode_merged = concatenate(encode_list, name='encode_concat', axis=1) if nb_passbands>1 else encode_list[0]
    if not return_sequences:
        encode_merged = concatenate([encode_merged, meta_input])  if (add_meta is not None) else encode_merged
        encode_merged = Dense(units=relu_size, 
                         activation='relu', 
                         name='relu_dense')(encode_merged) 
        encode_merged = Dense(units=output_size, 
                         activation='softmax',
                         name='clf_softmax_dense')(encode_merged) 
    else:
        if (nb_passbands>1):
            encode_merged = Dense(units=relu_size, 
                             activation='relu', 
                             name='relu_dense')(encode_merged) 
        encode_merged = TimeDistributed(Dense(units=output_size, 
                                         activation='linear'), 
                                   name='time_dist'.format(j))(encode_merged)
                      
    #---------------MODEL---------------#
    input_layer = main_input_list+[meta_input] if add_meta is not None else main_input_list
    output_layer = encode_merged
    m_model = Model(input_layer, output_layer)
    
    param_str = 'Classifier_{}_pb{}_n{}_x{}_drop{}_out{}'.format(model_type, nb_passbands, sizenet,
                            num_layers, int(drop_frac*100), output_size)
    if bidirectional:
        param_str+='_bidir'   
    
    print('>>>> param_str = ', param_str)
    return m_model, param_str, param_dict




## ############################################################################ ##
## --------------------------------------------------------------- ##
##     DIRECT CLASSIFIER [2] - TEMPORAL CNN
## --------------------------------------------------------------- ##
def classifier_direct_tCNN(input_dimension, output_size, sizenet,
                            kernel_size,
                            nb_passbands = 1, max_lenght = None,
                            causal = False,
                            num_layers = 1,
                            drop_frac = 0.25,
                            m_reductionfactor = 2,
                            m_activation='wavenet',
                            return_sequences=True,
                            add_meta = None,
                            add_dense = False,
                            **kwargs):
    
    param_dict = {key: value for key, value in locals().items()}
    param_dict.update(kwargs)
    param_dict = {k: v for k, v in param_dict.items()
                  if k not in ['kwargs']
                  and not isinstance(v, types.FunctionType)}
    
    m_padding='causal' if causal else 'same'
    meta_input = Input(shape=(add_meta,), name='meta_input') if add_meta is not None else None
    
    main_input_list = []
    for j in range(nb_passbands):
        if max_lenght is None:
            ndatapoints = None
        else:
            ndatapoints = max_lenght if type(max_lenght)==int else max_lenght[j]
        main_input = Input(shape=(ndatapoints, input_dimension), name='main_input_pb{}'.format(j))
        main_input_list.append(main_input)
    
    #---------------ENCODER---------------#
    encode_list = []
    for j in range(nb_passbands):
        encode = main_input_list[j]
        for i in range(num_layers):
            
            if type(sizenet)==int:
                size_n  = sizenet  
            else:
                size_n  = sizenet[i] if len(sizenet)==num_layers else sizenet
            size_filter = size_n//(i+1)  
            
            ## Convolution ##
            encode = Conv1D(filters=size_filter, ##### sizenet//(2**(i)), 
                       kernel_size=kernel_size, 
                       strides=m_reductionfactor,
                       activation=m_activation, #'relu'
                       input_shape=(max_lenght, input_dimension),
                       padding=m_padding,
                       name='encode_pb{}_x{}_conv1d'.format(j,i))(encode)  
            
            ## Dropout ##
            if drop_frac>0:
                encode = SpatialDropout1D(drop_frac, 
                                     name='encode_pb{}_x{}_sdrop{}'.format(j,i, int(drop_frac*100)))(encode)
                
        ## Output per band ##
        if (not return_sequences): 
            if max_lenght is None:  
                 encode = Lambda(lambda y: y[:,-1,:], name='encode_pb{}_lambda'.format(j))(encode) 
            else:
                 encode = Flatten(name='encode_pb{}_flat'.format(j))(encode) 
       
            
        ## Store ##
        encode_list.append(encode)
    
    #---------------CLASSIFIER---------------#
    encode_merged = concatenate(encode_list, name='encode_concat', axis=1) if nb_passbands>1 else encode_list[0]
    if not return_sequences:
        encode_merged = concatenate([encode_merged, meta_input]) if (add_meta is not None) else encode_merged
        encode_merged = Dense(units=relu_size, activation='relu', name='relu_dense')(encode_merged)
        encode_merged = Dense(units=output_size, activation='softmax', name='clf_softmax_dense')(encode_merged)
    else:
        if (nb_passbands>1):
             encode_merged = Dense(units=relu_size, activation='relu', name='relu_dense')(encode_merged) 
    
        encode_merged = TimeDistributed(Dense(units=output_size, activation='linear'), 
                                        name='time_dist'.format(j))(encode_merged)
        
    #---------------MODEL---------------#
    input_layer  = main_input_list+[meta_input] if add_meta is not None else main_input_list
    output_layer = encode_merged
    model = Model(input_layer, output_layer)
    
    param_str = 'Classifier_tCNN_pb{}_n{}_x{}_drop{}_cv{}_out{}'.format(nb_passbands, sizenet, num_layers,
                            int(drop_frac*100), kernel_size, output_size)
    if m_activation=='wavenet':
        param_str+='_aW'
    if causal:
        param_str+='_causal'
    
    print('>>>> param_str = ', param_str)
    return model, param_str, param_dict



## ############################################################################ ##
## --------------------------------------------------------------- ##
##     DIRECT CLASSIFIER [3] - DILATED-TCN
## --------------------------------------------------------------- ##
def classifier_direct_dTCN(input_dimension, output_size, sizenet,
                            n_stacks, list_dilations = None, max_dilation = None,
                            output_size_cw = None,
                            nb_passbands = 1, max_lenght = None,
                            causal = False,
                            drop_frac = 0.25,
                            config_wavenet = True,
                            m_activation='wavenet',
                            use_skip_connections = True,
                            kernel_size=3, kernel_wavenet = 1, 
                            return_sequences=False,
                            add_meta = None,
                            add_dense = False,
                            **kwargs):
        
    if output_size_cw is None:
        output_size_cw = output_size
     
    if list_dilations is None:
        list_dilations = list(range(max_dilation))
    
    param_dict = {key: value for key, value in locals().items()}
    param_dict.update(kwargs)
    param_dict = {k: v for k, v in param_dict.items()
                  if k not in ['kwargs']
                  and not isinstance(v, types.FunctionType)}
    
    dilation_depth = len(list_dilations)
    m_dilations = [2**i for i in list_dilations] 
    
    m_padding='causal' if causal else 'same'
    meta_input = Input(shape=(add_meta,), name='meta_input') if add_meta is not None else None
    
    main_input_list = []; 
    for j in range(nb_passbands):
        if max_lenght is None: ##
             ndatapoints = None
        else:
            ndatapoints = max_lenght if type(max_lenght)==int else max_lenght[j]
        main_input = Input(shape=(ndatapoints, input_dimension), name='main_input_pb{}'.format(j))
        main_input_list.append(main_input)
        
    #---------------ENCODER---------------#
    encode_list = []
    for j in range(nb_passbands):
        encode = main_input_list[j]
        
        ## Convolution ##
        encode = Conv1D(filters=sizenet,
                   kernel_size=kernel_size,
                   padding=m_padding,
                   name='encode_pb{}_conv1d'.format(j))(encode)
        encode = SpatialDropout1D(rate=drop_frac, 
                             name='encode_pb{}_sdrop{}'.format(j,int(drop_frac*100)))(encode)
        
        ## Residuals ##
        skip_connections = []
        for m_stack in range(n_stacks):
            for m_dilation_rate in m_dilations:
                encode, skip_x = residual_block(encode, sizenet, m_dilation_rate, m_stack, causal,
                                           m_activation, drop_frac, kernel_size, name='encode_pb{}_'.format(j))
                skip_connections.append(skip_x)
        if use_skip_connections:
            encode = keras.layers.add(skip_connections, name='encode_pb{}_add_skip_connections'.format(j))
            
        ## Wavenet config ##
        encode = Activation('relu', name='encode_pb{}_relu'.format(j))(encode)
        if config_wavenet:
            encode = Conv1D(filters=sizenet, 
                       kernel_size=kernel_wavenet, 
                       padding='same',
                       activation='relu',
                       name='encode_pb{}_n{}_cv{}_cw'.format(j, sizenet, kernel_wavenet))(encode)
            
            encode = Conv1D(filters=output_size_cw,
                       kernel_size=kernel_wavenet, 
                       padding='same',
                       name='encode_pb{}_n{}_cvw{}_cw'.format(j, output_size_cw, kernel_wavenet))(encode)    
                               
        ## Output per band ##
        if not return_sequences:
            if max_lenght is None:   ### 050620
                 encode = Lambda(lambda y: y[:, -1,:], name='encode_pb{}_lambda'.format(j))(encode)
            else:
                encode = Flatten(name='encode_pb{}_flat'.format(j))(encode)

        ## Store ##
        encode_list.append(encode)
        
    #---------------CLASSIFIER---------------#
    encode_merged = concatenate(encode_list, name='encode_concat', axis=1) if nb_passbands>1 else encode_list[0]
    if not return_sequences:
        encode_merged = concatenate([encode_merged, meta_input])  if (add_meta is not None) else encode_merged
        encode_merged = Dense(units=relu_size, 
                         activation='relu', 
                         name='relu_dense')(encode_merged) 
        encode_merged = Dense(units=output_size, 
                         activation='softmax',
                         name='clf_softmax_dense')(encode_merged) 
    else:
        if (nb_passbands>1):
            encode_merged = Dense(units=relu_size, 
                             activation='relu', 
                             name='relu_dense')(encode_merged) 
        encode_merged = TimeDistributed(Dense(units=output_size, 
                                         activation='linear'), 
                                   name='time_dist'.format(j))(encode_merged)
        
    #---------------MODEL---------------#
    input_layer  = main_input_list+[meta_input] if add_meta is not None else main_input_list
    output_layer = encode_merged
    m_model = Model(input_layer, output_layer)
    
    param_str = 'Classifier_dTCN_pb{}_n{}_drop{}_stack{}_dil{}_cv{}_cvW{}_out{}_outW{}'.format(nb_passbands, sizenet,int(drop_frac*100), 
                            n_stacks, dilation_depth, kernel_size, kernel_wavenet, output_size, output_size_cw)
    if m_activation=='wavenet':
        param_str+='_aW'
    if config_wavenet:
        param_str+='_cW'
    if causal:
        param_str+='_causal'
     
    print('>>>> param_str = ', param_str)
    return m_model, param_str, param_dict



## ############################################################################ ##
## ############################################################################ ##
## --------------------------------------------------------------- ##
##     COMPOSITE NETWORK [1] -  RNN
## --------------------------------------------------------------- ##
def composite_net_RNN(input_dimension, sizenet,
                      embedding, # bottleneck
                      output_size, #n_classes
                      nb_passbands = 1, max_lenght = None,
                      num_layers = 1,
                      drop_frac = 0.0,
                      model_type='LSTM',
                      bidirectional = True,
                      aux_in = True,
                      gp_down_size = None,
                      add_meta = None,
                      add_dense = False,
                      **kwargs):
    
    param_dict = {key: value for key, value in locals().items()}
    param_dict.update(kwargs)
    param_dict = {k: v for k, v in param_dict.items()
                  if k not in ['kwargs']
                  and not isinstance(v, types.FunctionType)}
    
    layer = dict_rnn[model_type]
    meta_input = Input(shape=(add_meta,), name='meta_input') if add_meta is not None else None
    
    main_input_list = []; 
    for j in range(nb_passbands):
        ndatapoints = max_lenght if type(max_lenght)==int else max_lenght[j]
        main_input = Input(shape=(ndatapoints, input_dimension), name='main_input_pb{}'.format(j))
        main_input_list.append(main_input)
       
    total_datapoints = np.sum(max_lenght) if gp_down_size is None else gp_down_size
    if aux_in:
        aux_input_concat=[]
        for j in range(nb_passbands):
            aux_input_concat.append( Input(shape=(total_datapoints, input_dimension-1), 
                                           name='aux_input_concat_pb{}'.format(j)))
    else :
        aux_input_concat = None
        

    #---------------ENCODER---------------#
    encode_list = []  
    for j in range(nb_passbands):
        encode = main_input_list[j]    
        for i in range(num_layers):
            mcond = (i<num_layers-1)
            wrapper = Bidirectional if bidirectional else lambda x: x
            if bidirectional:
                encode = wrapper(layer(units=sizenet, return_sequences=mcond),
                                 name='encode_pb{}_x{}'.format(j,i))  (encode)
            else:
                encode = wrapper(layer(units=sizenet, return_sequences=mcond,
                                       name='encode_pb{}_x{}'.format(j,i)))  (encode)
            
            ## Dropout ##
            if drop_frac > 0.0:
                encode = Dropout(rate=drop_frac, name='encode_drop_pb{}_x{}'.format(j,i))(encode)   
        
        ## Dense, Emb ##
        encode = Dense(activation='linear', 
                       name='encode_pb{}_embedding'.format(j), #'encode_dense_pb{}'.format(j), 
                       units=embedding)(encode)  
                       
        encode_list.append(encode)
                
                
    #---------------DECODER---------------#
    decode_list = [];
    for j in range(nb_passbands):
        decode = encode_list[j]
        
        ## Reshape ##
        decode = RepeatVector(total_datapoints, name='decode_pb{}_repeat'.format(j))(decode)
        if aux_in :   
            decode = concatenate([aux_input_concat[j], decode], name='decode_pb{}_aux_concat'.format(j))
            
        for i in range(num_layers):
            
            ## Dropout ##
            if (drop_frac>0.0) and (i>0):  # skip for 1st layer
                decode = Dropout(rate=drop_frac,
                                              name='drop_decode_pb{}_x{}'.format(j,i))(decode)
            wrapper = Bidirectional if bidirectional else lambda x: x
            if bidirectional:
                decode = wrapper(layer(units=sizenet, return_sequences=True),
                                           name='decode_pb{}_x{}'.format(j,i))(decode)
            else:
                decode = wrapper(layer(units=sizenet, return_sequences=True,
                                           name='decode_pb{}_x{}'.format(j,i)))(decode)
        
        ## Output ##
        decode = TimeDistributed(Dense(units=1, activation='linear'), 
                                 name='decode_pb{}_time_dist'.format(j))(decode)   
         
         ## Store ##
        decode_list.append(decode)
    
    '''
    #--------------- RECONSTRUCTION ---------------#
    if True:
        rec_feature_list = []  
        for j in range(nb_passbands):
            xtrue = Lambda(lambda x: x[:,:,1], name="main_slice")(main_input_list[j])
            xrec  = Lambda(lambda x: x[:,:,0], name="decode_slice")(decode_list[j])
            mfeats = loss_features(xtrue, xrec)
            ## Store ##
            rec_feature_list.append(mfeats)
        rec_feature_list_ = concatenate(rec_feature_list, name='cc_rec_') if nb_passbands>1 else rec_feature_list[0]
        meta_input2 = concatenate([rec_feature_list_, meta_input], name='meta_input2')
    else:
        meta_input2=meta_input
    '''
    
    #---------------CLASSIFIER---------------#
    encode_merged = concatenate(encode_list, name='encode_concat') if nb_passbands>1 else encode_list[0]
    if (nb_passbands>1):
        encode_merged = Dense(units=embedding, activation='relu', name='encode_relu_dense')(encode_merged) 
    classifier_out = encode_merged
    classifier_out = concatenate([classifier_out, meta_input])  if (add_meta is not None) else classifier_out
    classifier_out = Dense(units=relu_size, activation='relu', name='clf_relu_dense')(classifier_out) 
    classifier_out = Dense(units=output_size, activation='softmax', name='clf_softmax_dense')(classifier_out) 
    
    #---------------MODEL---------------#
    input_layer = main_input_list
    if (add_meta is not None)&aux_in:
        for j in range(nb_passbands):
            input_layer = input_layer+[aux_input_concat[j]]
        input_layer = input_layer+[meta_input]
    if (add_meta is None)&aux_in:
         for j in range(nb_passbands):
            input_layer = input_layer+[aux_input_concat[j]]
    if (add_meta is not None)&(not aux_in):
         input_layer = input_layer+[meta_input]
    
    output_layer = []
    for j in range(nb_passbands):
         output_layer.append(decode_list[j])
    output_layer.append(classifier_out)
    
    m_model = Model(input_layer, output_layer)
    
    param_str = 'Composite_{}_pb{}_n{}_x{}_drop{}_emb{}_out{}'.format(model_type, nb_passbands, sizenet, 
                          num_layers, int(drop_frac*100), embedding, output_size)
    if bidirectional:
        param_str+='_bidir'   
    
    print('>>>> param_str = ', param_str)
    return m_model, param_str, param_dict




## ############################################################################ ##
## --------------------------------------------------------------- ##
##     COMPOSITE NETWORK [2] -  TEMPORAL CNN
## --------------------------------------------------------------- ##
def composite_net_tCNN(input_dimension, sizenet, 
                       kernel_size,
                       embedding, ## bottleneck
                       output_size, #n_classes
                       nb_passbands = 1, max_lenght = None,
                       causal = False,
                       num_layers = 1,
                       drop_frac = 0.25,
                       m_reductionfactor = 2,
                       m_activation='tanh', #'relu' has an activation range [0-1], tanh has an activation [-1,1]
                       aux_in = True,
                       gp_down_size = None,
                       add_meta = None, #
                       add_dense = False, #
                       **kwargs):
    
    param_dict = {key: value for key, value in locals().items()}
    param_dict.update(kwargs)
    param_dict = {k: v for k, v in param_dict.items()
                  if k not in ['kwargs']
                  and not isinstance(v, types.FunctionType)}
    
    m_padding='causal' if causal else 'same'    
    meta_input = Input(shape=(add_meta,), name='meta_input') if add_meta is not None else None
    
    main_input_list = []
    for j in range(nb_passbands):
        ndatapoints = max_lenght if type(max_lenght)==int else max_lenght[j]    
        main_input = Input(shape=(ndatapoints, input_dimension), name='main_input_pb{}'.format(j))
        main_input_list.append(main_input)
  
    total_datapoints = np.sum(max_lenght) if gp_down_size is None else gp_down_size
    if aux_in:
        aux_input_concat=[]
        for j in range(nb_passbands):
            aux_input_concat.append( Input(shape=(total_datapoints, input_dimension-1), 
                                           name='aux_input_concat_pb{}'.format(j)))
    else :
        aux_input_concat = None
    
     
    config_init = True
    #---------------ENCODER---------------#
    encode_list = []; ntimes_list = []
    for j in range(nb_passbands):
        encode = main_input_list[j] 
        for i in range(num_layers):
            if type(sizenet)==int:
                size_n  = sizenet  
            else:
                size_n  = sizenet[i] if len(sizenet)==num_layers else sizenet           
            size_filter = size_n//(i+1)  ### *
            ## ------------------------------ ## ## ------------------------------ ##
            ## Convolution ##    
            if config_init:
                encode = Conv1D(filters=size_filter, 
                                kernel_size=kernel_size, 
                                strides=m_reductionfactor,
                                activation=m_activation, #'relu',
                                padding=m_padding,
                                name='encode_pb{}_xa{}_C{}_N{}'.format(j, i, kernel_size, size_n))(encode)
            else:
                encode = Conv1D(filters=size_filter, 
                                kernel_size=kernel_size, 
                                padding=m_padding,
                                name='encode_pb{}_xa{}_C{}_N{}'.format(j, i, kernel_size, size_n))(encode)
                
                ## Activation ##                 
                if m_activation=='wavenet': 
                    encode = keras.layers.multiply([Activation('tanh', name='encode_pb{}_x{}_activation_tanh'.format(j,i))(encode),
                                                    Activation('sigmoid', name='encode_pb{}_x{}_activation_sigmoid'.format(j,i))(encode)],
                                                    name='encode_pb{}_x{}_activation_multiply'.format(j,i))
                else:
                    encode = Activation(m_activation, 
                                        name='encode_pb{}_x{}_activation_{}'.format(j, i,m_activation))(encode)        
                
                ## Pooling ##
                if m_reductionfactor>0:
                    encode = MaxPooling1D(pool_size = m_reductionfactor, 
                                          name='encode_pb{}_x{}_maxpool{}'.format(j, i, m_reductionfactor))(encode)
            ## ------------------------------ ## ## ------------------------------ ##
            
            ## Dropout ##
            if drop_frac>0:
                encode = SpatialDropout1D(drop_frac, 
                                          name='encode_pb{}_x{}_sdrop{}'.format(j,i, int(drop_frac*100)))(encode)
                
        ## Flatten ##
        ntimes = encode.get_shape().as_list()[1] #max_lenght
        ntimes_list.append(ntimes)  
        ###encode = Lambda(lambda y: y[:, -1,:], name='encode_lambda'.format(j))(encode) ### embeddings
        encode = Flatten(name='encode_pb{}_flat'.format(j))(encode)
        encode = Dense(units=embedding, 
                       activation=m_activation, 
                       name='encode_pb{}_embedding'.format(j))(encode) #encode_pb{}_dense_embedding'
        
        ## Store ##
        encode_list.append(encode)
        
    
    #---------------DECODER---------------#
    decode_list = [];
    for j in range(nb_passbands):
        decode = encode_list[j]
        
        ## Reshape ##
        ndtp = ntimes_list[j] 
        
        decode = Dense(ndtp*embedding, 
                       activation=m_activation, 
                       name='decode_pb{}_dense'.format(j))(decode)
        decode = Reshape((ndtp,embedding), name='decode_pb{}_reshape'.format(j))(decode)
        ##decode = RepeatVector(total_datapoints, name='decode_repeat')(decode)
        
        if aux_in&(ndtp==total_datapoints):   
            decode = concatenate([aux_input_concat[j], decode], name='decode_pb{}_aux_concat'.format(j))
            
        size_filter_dec = size_filter 
        for i in range(num_layers):
            if type(sizenet)==int:
                size_n = sizenet  
            else:
                size_n = sizenet[-i-1] if (len(sizenet)==num_layers) else sizenet
            size_filter_dec = size_filter *(i+1)    ## //(i+1)
            
            ## ------------------------------ ## ## ------------------------------ ##
            ## Deconvolution ## 
            if config_init:
                decode = Conv1DTranspose(decode,
                                         filters=size_filter_dec,
                                         kernel_size=kernel_size, 
                                         activation=m_activation, #'relu',
                                         strides=m_reductionfactor,
                                         padding='same', 
                                         name='decode_pb{}_xa{}_C{}_N{}'.format(j,i, kernel_size, size_n))
            else:
                ## Upsampling ##
                if m_reductionfactor>0:
                    decode = UpSampling1D(m_reductionfactor, 
                                          name='decode_pb{}_x{}_maxpool{}'.format(j,i, m_reductionfactor))(decode)
                decode = Conv1DTranspose(decode,
                                         filters=size_filter_dec,
                                         kernel_size=kernel_size, 
                                         padding='same', 
                                         name='decode_pb{}_xa{}_C{}_N{}'.format(j,i, kernel_size, size_n))
                ## Activation ##                 
                if m_activation=='wavenet': 
                    decode = keras.layers.multiply([Activation('tanh', name='decode_pb{}_x{}_activation_tanh'.format(j,i))(decode),
                                                    Activation('sigmoid', name='decode_pb{}_x{}_activation_sigmoid'.format(j,i))(decode)],
                                                    name='decode_pb{}_x{}_activation_multiply'.format(j,i))
                else:
                    decode = Activation(m_activation, 
                                        name='decode_pb{}_x{}_activation_{}'.format(j,i,m_activation))(decode)        
            ## ------------------------------ ## ## ------------------------------ ##
            
            ## Dropout ##
            if drop_frac >0:
                decode = SpatialDropout1D(drop_frac, 
                                          name='decode_pb{}_x{}_sdrop_{}'.format(j,i, int(drop_frac*100)))(decode)
                        
        ## Output ##
        decode = Conv1DTranspose(decode,
                                 filters=1,
                                 kernel_size=kernel_size,
                                 activation='sigmoid',
                                 padding='same',
                                 name='decode_pb{}_time_dist'.format(j))
        
         ## Store ##
        decode_list.append(decode)
    
    #---------------CLASSIFIER---------------#
    encode_merged = concatenate(encode_list, name='encode_concat') if nb_passbands>1 else encode_list[0]
    if (nb_passbands>1):
        encode_merged = Dense(units=embedding,
                              activation=m_activation, 
                              name='encode_relu_dense')(encode_merged) 
    classifier_out = encode_merged
    classifier_out = concatenate([classifier_out, meta_input])  if (add_meta is not None) else classifier_out
    classifier_out = Dense(units=relu_size, activation='relu', 
                           name='clf_relu_dense')(classifier_out) 
    classifier_out = Dense(units=output_size, activation='softmax', 
                           name='clf_softmax_dense')(classifier_out) 

    #---------------MODEL---------------#
    input_layer = main_input_list
    if (add_meta is not None)&aux_in:
        for j in range(nb_passbands):
             input_layer = input_layer+[aux_input_concat[j]]
        input_layer = input_layer+[meta_input]
    if (add_meta is None)&aux_in:
         for j in range(nb_passbands):
            input_layer = input_layer+[aux_input_concat[j]]
    if (add_meta is not None)&(not aux_in):
         input_layer = input_layer+[meta_input]

    output_layer = []
    for j in range(nb_passbands):
         output_layer.append(decode_list[j])
    output_layer.append(classifier_out)
    
    m_model = Model(input_layer, output_layer)
    
    param_str = 'Composite_tCNN_pb{}_n{}_x{}_drop{}_cv{}_emb{}_out{}'.format(nb_passbands, sizenet, num_layers,
                              int(drop_frac*100), kernel_size, embedding, output_size)
    if m_activation=='wavenet':
        param_str+='_aW'
    if causal:
        param_str+='_causal'
    
    print('>>>> param_str = ', param_str)
    return m_model, param_str, param_dict



## ############################################################################ ##
## --------------------------------------------------------------- ##
##     COMPOSITE NETWORK [3] -  DILATED TCN
## --------------------------------------------------------------- ##
def composite_net_dTCN(input_dimension, sizenet,
                       embedding, ## bottleneck
                       output_size, ##n_classes
                       n_stacks, list_dilations = None, max_dilation = None,
                       nb_passbands = 1, max_lenght = None,
                       causal = False,
                       drop_frac = 0.25,
                       config_wavenet = True,
                       m_activation='wavenet',
                       use_skip_connections = True,
                       kernel_size=3, kernel_wavenet = 1, 
                       aux_in = True,
                       gp_down_size = None,
                       add_meta = None,
                       add_dense = False,
                       **kwargs):
    
    if list_dilations is None:
        list_dilations = list(range(max_dilation))
        
    param_dict = {key: value for key, value in locals().items()}
    param_dict.update(kwargs)
    param_dict = {k: v for k, v in param_dict.items()
                  if k not in ['kwargs']
                  and not isinstance(v, types.FunctionType)}
    for key, value in param_dict.items():
            if isinstance(value, np.int64): 
                param_dict[key] = int(value) 
    
    dilation_depth = len(list_dilations)
    m_dilations = [2**i for i in list_dilations] 
    
    ##########
    m_activation2 = 'tanh'
    ##########
    
    m_padding='causal' if causal else 'same'
    meta_input = Input(shape=(add_meta,), name='meta_input') if add_meta is not None else None
    
    main_input_list = []; #aux_input_list = []
    for j in range(nb_passbands):
        ndatapoints = max_lenght if type(max_lenght)==int else max_lenght[j]
        main_input = Input(shape=(ndatapoints, input_dimension), name='main_input_pb{}'.format(j))
        main_input_list.append(main_input)
    
    total_datapoints = np.sum(max_lenght) if gp_down_size is None else gp_down_size
    if aux_in:
        aux_input_concat=[]
        for j in range(nb_passbands):
            aux_input_concat.append( Input(shape=(total_datapoints, input_dimension-1), 
                                           name='aux_input_concat_pb{}'.format(j)))
    else :
        aux_input_concat = None
        
    #---------------ENCODER---------------#
    encode_list = []
    for j in range(nb_passbands):
        encode = main_input_list[j]        
        
        ## Convolution ##
        encode = Conv1D(filters=sizenet, 
                        kernel_size=kernel_size, 
                        padding=m_padding,
                        name='encode_pb{}_initconv'.format(j))(encode)
        encode = SpatialDropout1D(rate=drop_frac,
                                  name='encode_pb{}_sdrop{}'.format(j,int(drop_frac*100)))(encode)
        
        ## Residuals, Skip_connections ##
        skip_connections = []
        for m_stack in range(n_stacks):
            for m_dilation_rate in m_dilations:
                encode, skip_encode = residual_block(encode, sizenet, m_dilation_rate, m_stack, causal,
                                                     m_activation, drop_frac, kernel_size, name='encode_pb{}_'.format(j))
                skip_connections.append(skip_encode)   
        if use_skip_connections:
            encode = keras.layers.add(skip_connections, name='encode_pb{}_add_skip_connections'.format(j))           
        encode = Activation('relu', name='encode_pb{}_relu'.format(j))(encode)
        
        ## Wavenet config ##
        if config_wavenet:
            encode = Conv1D(filters=sizenet, 
                            kernel_size=kernel_wavenet, 
                            activation='relu',
                            padding='same',
                            name='encode_pb{}_n{}_cv{}_cw'.format(j, sizenet, kernel_wavenet))(encode)
            
            encode = Conv1D(filters=embedding,
                            kernel_size=kernel_wavenet, 
                            ## activation='softmax',
                            padding='same',
                            name='encode_pb{}_n{}_cv{}_cw'.format(j, embedding, kernel_wavenet))(encode)
        else:
            encode = Dense(units=embedding, activation='relu', name='encode_pb{}_dense'.format(j))(encode)
            
        ## Encoding ##
        encode = Flatten(name='encode_pb{}_flat'.format(j))(encode)
        encode = Dense(units=embedding, 
                       activation=m_activation2, #'relu', 
                       name='encode_pb{}_embedding'.format(j))(encode)
        
        ## Store
        encode_list.append(encode)
   
    
    #---------------DECODER---------------#
    decode_list = [];
    for j in range(nb_passbands):
        decode = encode_list[j]
    
        ## Reshape ##
        decode = Dense(total_datapoints*embedding, 
                       activation=m_activation2, #'relu', 
                       name='decode_pb{}_dense'.format(j))(decode)
        decode = Reshape((total_datapoints,embedding), name='decode_pb{}_reshape'.format(j))(decode)
        if aux_in:
            decode = concatenate([aux_input_concat[j], decode], name='decode_pb{}_aux_concat'.format(j))
            
        ## Deconvolution ##
        decode = Conv1DTranspose(decode,
                                 filters=sizenet,
                                 kernel_size=kernel_size, 
                                 padding='same',
                                 name='decode_pb{}_initconv'.format(j))
        
        decode = SpatialDropout1D(rate=drop_frac, 
                                  name='decode_pb{}_sdrop{}'.format(j,int(drop_frac*100)))(decode)
        
        ## Residuals, Skip_connections ##
        skip_connections = []
        for m_stack in range(n_stacks):
            for m_dilation_rate in m_dilations:
                decode, skip_decode = residual_block(decode, sizenet, m_dilation_rate, m_stack, causal,
                                                     m_activation, drop_frac, kernel_size, type='deconvolution', 
                                                     name='decode_pb{}_'.format(j))
                skip_connections.append(skip_decode)            
        if use_skip_connections:
            decode = keras.layers.add(skip_connections, name='decode_pb{}add_skip_connections'.format(j))                
        decode = Activation('relu', name='decode_pb{}_relu'.format(j))(decode)  
        
        ## Wavenet configuration ##
        if config_wavenet:
            decode = Conv1DTranspose(decode,
                                     filters=sizenet, 
                                     kernel_size=kernel_wavenet, 
                                     activation='relu',
                                     padding='same',
                                     name='decode_pb{}_n{}_cv{}_cw'.format(j,sizenet, kernel_wavenet)) 
            
            decode = Conv1DTranspose(decode,
                                     filters=1,
                                     kernel_size=kernel_wavenet, 
                                     padding='same',
                                     name='decode_pb{}_n{}_cv{}_cw'.format(j,1, kernel_wavenet))
            
        ## Output ##
        decode = Conv1DTranspose(decode,
                                 filters=1,
                                 kernel_size=kernel_wavenet,
                                 activation='sigmoid',
                                 padding='same',
                                 name='decode_pb{}_time_dist'.format(j))
        
         ## Store ##
        decode_list.append(decode)
    
    #---------------CLASSIFIER---------------#
    ## Merge ##
    encode_merged = concatenate(encode_list, name='encode_concat') if nb_passbands>1 else encode_list[0]
    if (nb_passbands>1):
        encode_merged = Dense(units=embedding, 
                              activation=m_activation2, 
                              name='encode_relu_dense')(encode_merged)
    classifier_out = encode_merged
    classifier_out = concatenate([classifier_out, meta_input])  if (add_meta is not None) else classifier_out
    classifier_out = Dense(units=relu_size, activation='relu', name='clf_relu_dense')(classifier_out) 
    classifier_out = Dense(units=output_size, activation='softmax', name='clf_softmax_dense')(classifier_out) 
    
    #---------------MODEL---------------#
    input_layer = main_input_list
    if (add_meta is not None)&aux_in:
        for j in range(nb_passbands):
            input_layer = input_layer+[aux_input_concat[j]]
        input_layer = input_layer+[meta_input]
    if (add_meta is None)&aux_in:
        for j in range(nb_passbands):
             input_layer = input_layer+[aux_input_concat[j]]
    if (add_meta is not None)&(not aux_in):
         input_layer = input_layer+[meta_input]
    
    output_layer = []
    for j in range(nb_passbands):
         output_layer.append(decode_list[j])
    output_layer.append(classifier_out)
    
    m_model = Model(input_layer, output_layer)
    
    param_str = 'Composite_dTCN_pb{}_n{}_drop{}_stack{}_dil{}_cv{}_cvW{}_emb{}_out{}'.format(nb_passbands, sizenet,
                                int(drop_frac*100), n_stacks, dilation_depth, kernel_size, 
                                 kernel_wavenet, embedding, output_size)
    if m_activation=='wavenet':
        param_str+='_aW'
    if config_wavenet:
        param_str+='_cW'
    if causal:
        param_str+='_causal'
    
    print('>>>> param_str = ', param_str)
    return m_model, param_str, param_dict




## ############################################################################ ##
## ############################################################################ ##
## --------------------------------------------------------------- ##
##     AUTOENCODER NETWORK [1] -  RNN
## --------------------------------------------------------------- ##
def autoencoder_RNN (input_dimension, sizenet,
                     embedding, # bottleneck
                     output_size, #n_classes
                     nb_passbands = 1, max_lenght = None,
                     num_layers = 1,
                     drop_frac = 0.0,
                     model_type='LSTM',
                     bidirectional = True,
                     aux_in = True,
                     gp_down_size = None,
                     add_dense = False,
                     **kwargs):
    
    param_dict = {key: value for key, value in locals().items()}
    param_dict.update(kwargs)
    param_dict = {k: v for k, v in param_dict.items()
                  if k not in ['kwargs']
                  and not isinstance(v, types.FunctionType)}
    
    layer = dict_rnn[model_type]
    
    main_input_list = []; 
    for j in range(nb_passbands):
        ndatapoints = max_lenght if type(max_lenght)==int else max_lenght[j]
        main_input = Input(shape=(ndatapoints, input_dimension), name='main_input_pb{}'.format(j))
        main_input_list.append(main_input)
       
    total_datapoints = np.sum(max_lenght) if gp_down_size is None else gp_down_size
    if aux_in:
        aux_input_concat=[]
        for j in range(nb_passbands):
            aux_input_concat.append( Input(shape=(total_datapoints, input_dimension-1), 
                                           name='aux_input_concat_pb{}'.format(j)))
    else :
        aux_input_concat = None
        

    #---------------ENCODER---------------#
    encode_list = []  
    for j in range(nb_passbands):
        encode = main_input_list[j]    
        for i in range(num_layers):
            mcond = (i<num_layers-1)
            wrapper = Bidirectional if bidirectional else lambda x: x
            if bidirectional:
                encode = wrapper(layer(units=sizenet, return_sequences=mcond),
                                 name='encode_pb{}_x{}'.format(j,i))(encode)
            else:
                encode = wrapper(layer(units=sizenet, return_sequences=mcond,
                                       name='encode_pb{}_x{}'.format(j,i)))  (encode)
            
            ## Dropout ##
            if drop_frac > 0.0:
                encode = Dropout(rate=drop_frac, name='encode_drop_pb{}_x{}'.format(j,i))(encode)   
        
        ## Dense, Emb ##
        encode = Dense(activation='linear', 
                       name='encode_pb{}_embedding'.format(j), #'encode_dense_pb{}'.format(j), 
                       units=embedding)(encode)  
                       
        encode_list.append(encode)
                
                
    #---------------DECODER---------------#
    decode_list = [];
    for j in range(nb_passbands):
        decode = encode_list[j]
        
        ## Reshape ##
        decode = RepeatVector(total_datapoints, name='decode_pb{}_repeat'.format(j))(decode)
        if aux_in :   
            decode = concatenate([aux_input_concat[j], decode], name='decode_pb{}_aux_concat'.format(j))
            
        for i in range(num_layers):
            
            ## Dropout ##
            if (drop_frac>0.0) and (i>0):  # skip for 1st layer 
                decode = Dropout(rate=drop_frac,
                                 name='drop_decode_pb{}_x{}'.format(j,i))(decode)
            wrapper = Bidirectional if bidirectional else lambda x: x
            if bidirectional:
                decode = wrapper(layer(units=sizenet, return_sequences=True),
                                 name='decode_pb{}_x{}'.format(j,i))(decode)
            else:
                decode = wrapper(layer(units=sizenet, return_sequences=True,
                                       name='decode_pb{}_x{}'.format(j,i)))  (decode)
        
        ## Output ##
        decode = TimeDistributed(Dense(units=1, activation='linear', 
                                 name='decode_pb{}_time_dist'.format(j)))(decode)   
         
         ## Store ##
        decode_list.append(decode)
    
    
    #---------------MODEL---------------#
    input_layer = main_input_list
    if aux_in:
         for j in range(nb_passbands):
                input_layer = input_layer+[aux_input_concat[j]]
    
    output_layer = []
    for j in range(nb_passbands):
         output_layer.append(decode_list[j])
    
    m_model = Model(input_layer, output_layer)
    
    param_str = 'Autoencoder_{}_pb{}_n{}_x{}_drop{}_emb{}_out{}'.format(model_type, nb_passbands, sizenet, 
                              num_layers, int(drop_frac*100), embedding, output_size)
    if bidirectional:
        param_str+='_bidir'   
    
    print('>>>> param_str = ', param_str)
    return m_model, param_str, param_dict




## ############################################################################ ##
## --------------------------------------------------------------- ##
##     COMPOSITE NETWORK [2] -  TEMPORAL CNN
## --------------------------------------------------------------- ##
def autoencoder_tCNN(input_dimension, sizenet,
                     kernel_size,
                     embedding, ## bottleneck
                     output_size, #n_classes
                     nb_passbands = 1, max_lenght = None,
                     causal = False,
                     num_layers = 1,
                     drop_frac = 0.25,
                     m_reductionfactor = 2,
                     m_activation='tanh', #'relu' range [0-1], tanh [-1,1]
                     aux_in = True,
                     gp_down_size = None,
                     add_dense = False, #
                     **kwargs):
    
    param_dict = {key: value for key, value in locals().items()}
    param_dict.update(kwargs)
    param_dict = {k: v for k, v in param_dict.items()
                  if k not in ['kwargs']
                  and not isinstance(v, types.FunctionType)}
    
    m_padding='causal' if causal else 'same'    
    
    main_input_list = []
    for j in range(nb_passbands):
        ndatapoints = max_lenght if type(max_lenght)==int else max_lenght[j]
        main_input = Input(shape=(ndatapoints, input_dimension), name='main_input_pb{}'.format(j))
        main_input_list.append(main_input)
  
    total_datapoints = np.sum(max_lenght) if gp_down_size is None else gp_down_size
    if aux_in:
        aux_input_concat=[]
        for j in range(nb_passbands):
            aux_input_concat.append(Input(shape=(total_datapoints, input_dimension-1), 
                                          name='aux_input_concat_pb{}'.format(j)))
    else :
        aux_input_concat = None
    
     
    config_init = True
    #---------------ENCODER---------------#
    encode_list = []; ntimes_list = []
    for j in range(nb_passbands):
        encode = main_input_list[j] 
        for i in range(num_layers):
            if type(sizenet)==int:
                size_n = sizenet  
            else:
                size_n = sizenet[i] if len(sizenet)==num_layers else sizenet           
            size_filter = size_n //(i+1)  ### *
            ## ------------------------------ ## ## ------------------------------ ##
            ## Convolution ##    
            if config_init:
                encode = Conv1D(filters=size_filter, 
                                kernel_size=kernel_size, 
                                strides=m_reductionfactor,
                                activation=m_activation, #'relu',
                                padding=m_padding,
                                name='encode_pb{}_xa{}_C{}_N{}'.format(j,i, kernel_size, size_n))(encode)
            else:
                encode = Conv1D(filters=size_filter, 
                                kernel_size=kernel_size, 
                                padding=m_padding,
                                name='encode_pb{}_xa{}_C{}_N{}'.format(j,i, kernel_size, size_n))(encode)
                
                ## Activation ##                 
                if m_activation=='wavenet': 
                    encode = keras.layers.multiply([Activation('tanh', name='encode_pb{}_x{}_activation_tanh'.format(j,i))(encode),
                                                    Activation('sigmoid', name='encode_pb{}_x{}_activation_sigmoid'.format(j,i))(encode)],
                                                    name='encode_pb{}_x{}_activation_multiply'.format(j,i))
                else:
                    encode = Activation(m_activation, 
                                        name='encode_pb{}_x{}_activation_{}'.format(j,i,m_activation))(encode)        
                
                ## Pooling ##
                if m_reductionfactor>0:
                    encode = MaxPooling1D(pool_size = m_reductionfactor, 
                                          name='encode_pb{}_x{}_maxpool{}'.format(j,i, m_reductionfactor))(encode)
            ## ------------------------------ ## ## ------------------------------ ##
            
            ## Dropout ##
            if drop_frac>0:
                encode = SpatialDropout1D(drop_frac, 
                                          name='encode_pb{}_x{}_sdrop{}'.format(j,i, int(drop_frac*100)))(encode)
                
        ## Flatten ##
        ntimes = encode.get_shape().as_list()[1] #max_lenght
        ntimes_list.append(ntimes)  
        ###encode = Lambda(lambda y: y[:, -1,:], name='encode_lambda'.format(j))(encode) ### embeddings
        encode = Flatten(name='encode_pb{}_flat'.format(j))(encode)
        encode = Dense(units=embedding, 
                       activation=m_activation, 
                       name='encode_pb{}_embedding'.format(j))(encode) #encode_pb{}_dense_embedding'
        
        ## Store ##
        encode_list.append(encode)
        
    
    #---------------DECODER---------------#
    decode_list = [];
    for j in range(nb_passbands):
        decode = encode_list[j]
        
        ## Reshape ##
        ndtp = ntimes_list[j] 
        
        decode = Dense(ndtp*embedding, 
                       activation=m_activation, 
                       name='decode_pb{}_dense'.format(j))(decode)
        decode = Reshape((ndtp,embedding), name='decode_pb{}_reshape'.format(j))(decode)
        ##decode = RepeatVector(total_datapoints, name='decode_repeat')(decode)
        
        if aux_in&(ndtp==total_datapoints):   
            decode = concatenate([aux_input_concat[j], decode], name='decode_pb{}_aux_concat'.format(j))
            
        size_filter_dec = size_filter 
        for i in range(num_layers):
            if type(sizenet)==int:
                size_n = sizenet  
            else:
                size_n = sizenet[-i-1] if (len(sizenet)==num_layers) else sizenet
            size_filter_dec = size_filter *(i+1)    ## //(i+1)
            
            ## ------------------------------ ## ## ------------------------------ ##
            ## Deconvolution ## 
            if config_init:
                decode = Conv1DTranspose(decode,
                                         filters=size_filter_dec,
                                         kernel_size=kernel_size, 
                                         activation=m_activation, #'relu',
                                         strides=m_reductionfactor,
                                         padding='same', 
                                         name='decode_pb{}_xa{}_C{}_N{}'.format(j,i, kernel_size, size_n))
            else:
                ## Upsampling ##
                if m_reductionfactor>0:
                    decode = UpSampling1D(m_reductionfactor, 
                                          name='decode_pb{}_x{}_maxpool{}'.format(j,i, m_reductionfactor))(decode)
                decode = Conv1DTranspose(decode,
                                         filters=size_filter_dec,
                                         kernel_size=kernel_size, 
                                         padding='same', 
                                         name='decode_pb{}_xa{}_C{}_N{}'.format(j,i, kernel_size, size_n))
                ## Activation ##                 
                if m_activation=='wavenet': 
                    decode = keras.layers.multiply([Activation('tanh', name='decode_pb{}_x{}_activation_tanh'.format(j,i))(decode),
                                                    Activation('sigmoid', name='decode_pb{}_x{}_activation_sigmoid'.format(j,i))(decode)],
                                                     name='decode_pb{}_x{}_activation_multiply'.format(j,i))
                else:
                    decode = Activation(m_activation, 
                                        name='decode_pb{}_x{}_activation_{}'.format(j,i,m_activation))(decode)        
            ## ------------------------------ ## ## ------------------------------ ##
            
            ## Dropout ##
            if drop_frac >0:
                decode = SpatialDropout1D(drop_frac, 
                                          name='decode_pb{}_x{}_sdrop_{}'.format(j,i, int(drop_frac*100)))(decode)
                        
        ## Output ##
        decode = Conv1DTranspose(decode,
                                 filters=1,
                                 kernel_size=kernel_size,
                                 activation='sigmoid',
                                 padding='same',
                                 name='decode_pb{}_time_dist'.format(j))
        
         ## Store ##
        decode_list.append(decode)
    
    #---------------MODEL---------------#
    input_layer = main_input_list
    if  aux_in:
         for j in range(nb_passbands):
                input_layer = input_layer+[aux_input_concat[j]]
    
    output_layer = []
    for j in range(nb_passbands):
         output_layer.append(decode_list[j])
    
    m_model = Model(input_layer, output_layer)
    
    param_str = 'Autoencoder_tCNN_pb{}_n{}_x{}_drop{}_cv{}_emb{}_out{}'.format(nb_passbands, sizenet, 
                                    num_layers, int(drop_frac*100), kernel_size, embedding, output_size)
    if m_activation=='wavenet':
        param_str+='_aW'
    if causal:
        param_str+='_causal'
    
    print('>>>> param_str = ', param_str)
    return m_model, param_str, param_dict



## ############################################################################ ##
## --------------------------------------------------------------- ##
##     COMPOSITE NETWORK [3] -  DILATED TCN
## --------------------------------------------------------------- ##
def autoencoder_dTCN (input_dimension, sizenet,
                       embedding, ## bottleneck
                       output_size, ##n_classes
                       n_stacks, list_dilations = None, max_dilation = None,
                       nb_passbands = 1, max_lenght = None,
                       causal = False,
                       drop_frac = 0.25,
                       config_wavenet = True,
                       m_activation='wavenet',
                       use_skip_connections = True,
                       kernel_size=3, kernel_wavenet = 1, 
                       aux_in = True,
                       gp_down_size = None,
                       add_dense = False,
                       **kwargs):
    
    if list_dilations is None:
        list_dilations = list(range(max_dilation))
        
    param_dict = {key: value for key, value in locals().items()}
    param_dict.update(kwargs)
    param_dict = {k: v for k, v in param_dict.items()
                  if k not in ['kwargs']
                  and not isinstance(v, types.FunctionType)}
    for key, value in param_dict.items():
            if isinstance(value, np.int64): 
                param_dict[key] = int(value) 
    
    dilation_depth = len(list_dilations)
    m_dilations = [2**i for i in list_dilations] 
    
    ##########
    m_activation2 = 'tanh'
    ##########
    
    m_padding='causal' if causal else 'same'
    
    main_input_list = []; #aux_input_list = []
    for j in range(nb_passbands):
        ndatapoints = max_lenght if type(max_lenght)==int else max_lenght[j]
        main_input = Input(shape=(ndatapoints, input_dimension), name='main_input_pb{}'.format(j))
        main_input_list.append(main_input)
    
    total_datapoints = np.sum(max_lenght) if gp_down_size is None else gp_down_size
    if aux_in:
        aux_input_concat=[]
        for j in range(nb_passbands):
            aux_input_concat.append(Input(shape=(total_datapoints, input_dimension-1), 
                                          name='aux_input_concat_pb{}'.format(j)))
    else :
        aux_input_concat = None
        
    #---------------ENCODER---------------#
    encode_list = []
    for j in range(nb_passbands):
        encode = main_input_list[j]        
        
        ## Convolution ##
        encode = Conv1D(filters=sizenet,
                        kernel_size=kernel_size, 
                        padding=m_padding,
                        name='encode_pb{}_initconv'.format(j))(encode)
        encode = SpatialDropout1D(rate=drop_frac, 
                                  name='encode_pb{}_sdrop{}'.format(j,int(drop_frac*100)))(encode)
        
        ## Residuals, Skip_connections ##
        skip_connections = []
        for m_stack in range(n_stacks):
            for m_dilation_rate in m_dilations:
                encode, skip_encode = residual_block(encode, sizenet, m_dilation_rate, m_stack, 
                                                     causal, m_activation, drop_frac, kernel_size, 
                                                     name='encode_pb{}_'.format(j))
                skip_connections.append(skip_encode)   
        if use_skip_connections:
            encode = keras.layers.add(skip_connections, name='encode_pb{}_add_skip_connections'.format(j))           
        encode = Activation('relu', name='encode_pb{}_relu'.format(j))(encode)
        
        ## Wavenet config ##
        if config_wavenet:
            encode = Conv1D(filters=sizenet, 
                            kernel_size=kernel_wavenet, 
                            activation='relu',
                            padding='same',
                            name='encode_pb{}_n{}_cv{}_cw'.format(j, sizenet, kernel_wavenet))(encode)
            
            encode = Conv1D(filters=embedding,
                            kernel_size=kernel_wavenet, 
                            ## activation='softmax',
                            padding='same',
                            name='encode_pb{}_n{}_cv{}_cw'.format(j, embedding, kernel_wavenet))(encode)
        else:
            encode = Dense(units=embedding, 
                           activation='relu', 
                           name='encode_pb{}_dense'.format(j))(encode)
            
        ## Encoding ##
        encode = Flatten(name='encode_pb{}_flat'.format(j))(encode)
        encode = Dense(units=embedding, 
                       activation=m_activation2, #'relu', 
                       name='encode_pb{}_embedding'.format(j))(encode)
        
        ## Store
        encode_list.append(encode)
   
    
    #---------------DECODER---------------#
    decode_list = [];
    for j in range(nb_passbands):
        decode = encode_list[j]
    
        ## Reshape ##
        decode = Dense(total_datapoints*embedding, 
                       activation=m_activation2, #'relu', 
                       name='decode_pb{}_dense'.format(j))(decode)
        decode = Reshape((total_datapoints,embedding), name='decode_pb{}_reshape'.format(j))(decode)
        if aux_in:
            decode = concatenate([aux_input_concat[j], decode], name='decode_pb{}_aux_concat'.format(j))
            
        ## Deconvolution ##
        decode = Conv1DTranspose(decode,
                                 filters=sizenet, 
                                 kernel_size=kernel_size, 
                                 padding='same',
                                 name='decode_pb{}_initconv'.format(j))
        
        decode = SpatialDropout1D(rate=drop_frac, 
                                  name='decode_pb{}_sdrop{}'.format(j,int(drop_frac*100)))(decode)
        
        ## Residuals, Skip_connections ##
        skip_connections = []
        for m_stack in range(n_stacks):
            for m_dilation_rate in m_dilations:
                decode, skip_decode = residual_block(decode, sizenet, m_dilation_rate, m_stack, causal, m_activation, 
                                                     drop_frac, kernel_size, type='deconvolution', name='decode_pb{}_'.format(j))
                skip_connections.append(skip_decode)            
        if use_skip_connections:
            decode = keras.layers.add(skip_connections, name='decode_pb{}add_skip_connections'.format(j))                
        decode = Activation('relu', name='decode_pb{}_relu'.format(j))(decode)  
        
        ## Wavenet configuration ##
        if config_wavenet:
            decode = Conv1DTranspose(decode,
                                     filters=sizenet, 
                                     kernel_size=kernel_wavenet, 
                                     activation='relu',
                                     padding='same',
                                     name='decode_pb{}_n{}_cv{}_cw'.format(j,sizenet, kernel_wavenet)) 
            
            decode = Conv1DTranspose(decode,
                                     filters=1,
                                     kernel_size=kernel_wavenet, 
                                     padding='same',
                                     name='decode_pb{}_n{}_cv{}_cw'.format(j,1, kernel_wavenet))
            
        ## Output ##
        decode = Conv1DTranspose(decode,
                                 filters=1,
                                 kernel_size=kernel_wavenet,
                                 activation='sigmoid',
                                 padding='same',
                                 name='decode_pb{}_time_dist'.format(j))
        
         ## Store ##
        decode_list.append(decode)
    
    #---------------MODEL---------------#
    input_layer = main_input_list
    if aux_in:
         for j in range(nb_passbands):
                input_layer = input_layer+[aux_input_concat[j]]
    
    output_layer = []
    for j in range(nb_passbands):
         output_layer.append(decode_list[j])
    
    m_model = Model(input_layer, output_layer)
    
    param_str = 'Autoencoder_dTCN_pb{}_n{}_drop{}_stack{}_dil{}_cv{}_cvW{}_emb{}_out{}'.format(nb_passbands, sizenet,
                                    int(drop_frac*100),n_stacks, dilation_depth,
                                    kernel_size, kernel_wavenet, embedding, output_size)
    if m_activation=='wavenet':
        param_str+='_aW'
    if config_wavenet:
        param_str+='_cW'
    if causal:
        param_str+='_causal'
    
    print('>>>> param_str = ', param_str)
    return m_model, param_str, param_dict



