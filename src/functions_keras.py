## ---------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------- ##
#   FUNCTIONS - Configuration and Keras functions
## ---------------------------------------------------------------------------------------- ##
## ** Note/Credits **
##
##   Following functions & attributes initially forked from 
##       "Naul, Bloom, Perez & Van der Walt, 2017 [DOI: 10.1038/s41550-017-0321-z]" :
##          <Class>      LogDirLogger, TimedCSVLogger
##          <Functions>  noisify_samples, parse_model_args, get_run_id, 
##                       limited_memory_session, train_and_log
##   (All aforementioned functions are modified in the current version)
##
##   Added:
##          <Functions>   generator_lc, run_autoencoder_gen, run_autoencoder_pad
##
## ---------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------- ##


import argparse, csv
from functools import wraps
from itertools import cycle, islice
from math import ceil

import sys, os, time, datetime 
import shutil, joblib, json, types, copy
import numpy as np

SEED_tf=42
import tensorflow as tf
tf.compat.v1.set_random_seed(SEED_tf)

import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import (Callback, TensorBoard, EarlyStopping, ModelCheckpoint, CSVLogger, ProgbarLogger)
from keras.models import load_model
from collections import Iterable, OrderedDict

SEED=0
np.random.seed(SEED)


import nets_ae_clf as m_nets #m_dln
dict_nruns = {0 : 'classifier_MLP_meta',
              #
              1 : 'classifier_direct_RNN',
              2 : 'classifier_direct_tCNN',
              3 : 'classifier_direct_dTCN',
              #
              4 : 'autoencoder_RNN',
              5 : 'autoencoder_tCNN',
              6 : 'autoencoder_dTCN',
              #
              7 : 'composite_net_RNN',
              8 : 'composite_net_tCNN',
              9 : 'composite_net_dTCN',
             }

dict_nfuncs = {0 : m_nets.classifier_MLP_meta, 
               #
               1 : m_nets.classifier_direct_RNN,
               2 : m_nets.classifier_direct_tCNN,
               3 : m_nets.classifier_direct_dTCN,
               #
               4 : m_nets.autoencoder_RNN,
               5 : m_nets.autoencoder_tCNN,
               6 : m_nets.autoencoder_dTCN,
               #
               7 : m_nets.composite_net_RNN,
               8 : m_nets.composite_net_tCNN,
               9 : m_nets.composite_net_dTCN,
              }
list_clf_meta = [0]
list_clf = [1,2,3]
list_ae  = [4,5,6]
list_composite = [7,8,9] 



## ######################################################################################################## ##
from IPython import get_ipython             
if 'get_ipython' in vars() and get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
    from keras_tqdm import TQDMNotebookCallback as Progbar
else:
    from keras_tqdm import TQDMCallback
    import sys
    class Progbar(TQDMCallback):  # redirect TQDMCallback to stdout
        def __init__(self): #, output_file):
            TQDMCallback.__init__(self)
            #self.output_file = output_file 
            self.output_file = sys.stdout
#import resource
#class MemoryCallback(Callback):
#    def on_epoch_end(self, epoch, log={}):
#        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


## ######################################################################################################## ##
class LogDirLogger(Callback):
    def __init__(self, log_dir):
        self.log_dir = log_dir 

    #def on_epoch_begin(self, epoch, logs=None):
    #    print('\n ' + self.log_dir + '\n')

    
## ######################################################################################################## ##
class TimedCSVLogger(CSVLogger):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if not self.writer:
            self.keys = sorted(logs.keys())

            class CustomDialect(csv.excel):
                delimiter = self.sep

            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=['epoch', 'time'] + self.keys,
                                         dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'epoch': epoch, 'time': str(datetime.datetime.now())})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

        
## ######################################################################################################## ##
def noisify_samples(inputs, outputs, errors, batch_size=500, sample_weight=None):
    """ ---------------------------------------------------------
        @summary: Generate noisier versions from the data
     ---------------------------------------------------------- """
    if sample_weight is None:
        sample_weight = np.ones(errors.shape)
    X = inputs['main_input']
    X_aux = inputs['aux_input']
    shuffle_inds = np.arange(len(X))
    while True:
        # New epoch
        np.random.shuffle(shuffle_inds)
        noise = errors * np.random.normal(size=errors.shape)
        X_noisy = X.copy()
        X_noisy[:, :, 1] += noise
        # Re-scale to have mean 0 and std dev 1; TODO make this optional
        X_noisy[:, :, 1] -= np.atleast_2d(np.nanmean(X_noisy[:, :, 1], axis=1)).T
        X_noisy[:, :, 1] /= np.atleast_2d(np.std(X[:, :, 1], axis=1)).T

        for i in range(ceil(len(X) / batch_size)):
            inds = shuffle_inds[(i * batch_size):((i + 1) * batch_size)]
            yield ([X_noisy[inds], X_aux[inds]], X_noisy[inds, :, 1:2], sample_weight[inds])

            
## ######################################################################################################## ##
def print_args(args):
    sep = 42*'-'
    print(f"\n\t# {sep} #\n\t# --------[ SESSION - HYPERPARAMS ] -------- # \n\t# {sep} #\n")
    for arg in vars(args):
            print( '\t',arg,'\t:', getattr(args, arg))    


## ######################################################################################################## ##
def parse_model_args(arg_dict=None):
    """ ---------------------------------------------------------
        @summary: Parse command line arguments  
    ---------------------------------------------------------- """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_id", type=str, default='')
    parser.add_argument("--run_id", type=int, default=1)
    parser.add_argument("--sim_type", type=str, default='')
    
    ## Directory
    parser.add_argument("--data_store", type=str, default='')
    parser.add_argument("--output_store", type=str, default='')
    
    
    ## NETS (TCN & RNN) params
    parser.add_argument("--nb_passbands", type=int, default=1)
    
    parser.add_argument("--sizenet", type=int)
    parser.add_argument("--embedding", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--drop_frac", type=float, default=0.25)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nb_epoch", type=int, default=250)
                  
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--learning_rate", type=float)     
    
    parser.add_argument("--decode_type", type=str, default=None)
    parser.add_argument("--decode_layers", type=int, default=None)
    
    parser.add_argument("--bidirectional", dest='bidirectional', action='store_true')
    
    
    ## TEMPO NET params
    parser.add_argument("--output_size_cw", type=int, default=None)
    parser.add_argument("--n_stacks", type=int, default=None)
    parser.add_argument("--max_dilation", type=int, default=None)
    parser.add_argument("--m_reductionfactor", type=int, default=2)
    
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--kernel_wavenet", type=int, default=1)
    parser.add_argument("--m_activation", type=str, default='wavenet')
    parser.add_argument("--do_featurizer", dest='do_featurizer', action='store_true')
    
    parser.add_argument("--config_wavenet", dest='config_wavenet', action='store_true')
    parser.add_argument("--use_skip_connections", dest='use_skip_connections', action='store_true')
                        
    parser.add_argument("--add_dense", dest= 'add_dense', action='store_true')
    
    
    ## General params
    parser.add_argument("--use_raw", dest='use_raw', action='store_true')
    #
    ##parser.add_argument("--add_meta", type=int, default=None)
    ##parser.add_argument("--catalog_meta", type=int, default=None)
    #
    #
    parser.add_argument("--add_metadata", dest='add_metadata', action='store_true')
    #
    #
    ##parser.add_argument("--meta_augm", dest= 'meta_augm', action='store_true')
    parser.add_argument("--causal", dest='causal', action='store_true')
    parser.add_argument("--aux_in", dest='aux_in', action='store_true')
    
    parser.add_argument("--categorical", dest='categorical', action='store_true')
    
    #parser.add_argument("--loss", type=str, default='mse')
    #parser.add_argument("--metrics", type=str, default='')
    #parser.add_argument("--loss_weights", type=float, nargs='*')
    parser.add_argument("--loss_weights_list", help='delimited list input', type=str, default=None)
    parser.add_argument("--validation_split", type=float, default=0.0)
    #
    parser.add_argument("--loss_AE",     type=str, default='mae') #'mse'
    parser.add_argument("--loss_CLF",    type=str, default='categorical_crossentropy')
    parser.add_argument("--metrics_CLF", type=str, default='')
    parser.add_argument("--metrics_AE",  type=str, default=None)
    
    
    ## Data params
    parser.add_argument("--n_min",       type=int,   default=None)
    parser.add_argument("--m_max",       type=float, default=20.)
    parser.add_argument("--ss_resid",    type=float, default=None)
    parser.add_argument("--lomb_score",  type=float, default=None)
    parser.add_argument("--survey_files", type=str,  nargs='*')
    
    ##parser.add_argument("--rebin", dest='rebin', action='store_true')
    ##parser.add_argument("--gp_down", dest='gp_down', action='store_true')
    parser.add_argument("--nbpoints",    type=int, default=200)
    parser.add_argument("--padding",     dest='padding', action='store_true')
    parser.add_argument("--diff_time",   dest='diff_time', action='store_true')
    parser.add_argument("--period_fold", dest='period_fold', action='store_true')
    
    
    ## ++ params
    parser.add_argument("--add_freqs", type=int, default=None)
    parser.add_argument("--gpu_frac",  type=float, default=None)
    parser.add_argument("--no_train",  dest='no_train', action='store_true')
    #parser.add_argument("--noisify",   dest='noisify', action='store_true')
    parser.add_argument("--patience",  type=int, default=20)
    
    parser.add_argument("--store_local", dest='store_local', action='store_true')
    
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--finetune_rate", type=float, default=None)
    
    parser.add_argument("--pool", type=int, default=None) #u
    parser.add_argument("--sigma", type=float, default=2e-9)
    ##parser.add_argument("--first_N", type=int, default=None)
    
    ##parser.add_argument("--even",   dest='even', action='store_true')
    ##parser.add_argument("--uneven", dest='even', action='store_false')
    
    
    parser.set_defaults(#even=False, 
                        bidirectional=False, noisify=False, period_fold=False)
    # Don't read argv if arg_dict present
    args = parser.parse_args(None if arg_dict is None else [])

    if arg_dict:  # merge additional arguments w/ defaults
        args = argparse.Namespace(**{**args.__dict__, **arg_dict})
        
    #required_args = ['sizenet', 'num_layers', 'drop_frac', 'lr', 'model_type', 'sim_type', 'nbpoints']
    required_args = ['run_id', 'sizenet', 'drop_frac', 'learning_rate', 'model_type'] 
    for key in required_args:
        if getattr(args, key) is None:
            parser.error("Missing argument {}".format(key))

    return args


## ######################################################################################################## ##
def get_run_id(data_id, model_type, sizenet, num_layers, learning_rate, batch_size, nb_epoch, 
               n_min=None, drop_frac=0.0, embedding=None, diff_time=None, period_fold=None, 
               add_metadata=None,  padding=None, aux_in=None, categorical=None, decode_type=None, 
               decode_layers=None, param_str=None, add_freqs=None, loss_weights=None, use_raw=None,
               m_activation=None,  **kwargs):
    #ss_resid=None, #add_meta=None, catalog_meta=None,  #gp_down=None,
               
    """ ---------------------------------------------------------
        @summary: Generate unique ID from model parameters
    ---------------------------------------------------------- """
    run = '[{}]'.format(data_id)
    
    if param_str is None:
        run += "{}_n{:03d}_x{}_drop{}".format(model_type, sizenet, num_layers, 
                                              int(100 * drop_frac)).replace('e-', 'm')
        if embedding:
            run += '_emb{}'.format(embedding)
        if decode_type:
            run += '_decode{}'.format(decode_type)
            if decode_layers:
                run += '_x{}'.format(decode_layers)
    else:
        run += param_str
           
    if learning_rate:
        run += '_lr{:1.0e}'.format(learning_rate)
        
    if batch_size:
        run += '_batch{}'.format(batch_size)
    
    if nb_epoch:
        run += '_nepoch{}'.format(nb_epoch)

    #if ss_resid is not None:
    #    if ss_resid>0:
    #        run+='_ssr{}'.format(int(ss_resid*100))
        
    #if n_min is not None:
    #    if n_min>0:
    #        run+='_nmin{}'.format(n_min)
            
    #if add_meta is not None:
    #    #run+='_meta{}'.format(add_meta)
    #    if catalog_meta is not None:
    #        run+='_metactg{}'.format(catalog_meta)
    #else:
    #    run+='_nometa'
     
    if add_metadata is not None:
        run+='_metadata'
            
    else:
        run+='_nometa'
        
    if categorical:
        run+='_ctg'
            
    if aux_in:
        run+='_aux'

    if padding:
        run+='_pad'
        #if gp_down:
        #    run+='_padgp'
        #else:
        #    run+='_padval'
    else:
        run+='_gen'

    if diff_time:
        run+='_dtimes'
    else:
        run+= '_times'
    
    if period_fold:
        run+='_folded'
        
    if add_freqs:
        run+='_freqclf'
        
    if loss_weights:
        for u in loss_weights.keys():
            if u =='clf_softmax_dense':
                mkey = 'CLF'
            elif u == 'decode_pb0_time_dist':
                mkey = 'AE0'
            elif u == 'decode_pb1_time_dist':
                mkey = 'AE1'
            else:
                mkey = ''
            if loss_weights[u]==1:
                lw = 1
            else:
                lw = "%.1e"%loss_weights[u]   #"%1.0e"
            run+=f'_{mkey}lossw{lw}'
           
        if use_raw:
            run+='_rawdata'
            
        if (model_type not in ['LSTM', 'GRU']) & (m_activation is not None) :
            run+=f'_{m_activation}'
    

    return run


## ######################################################################################################## ##
def limited_memory_session(gpu_frac):
    
    init_op = tf.global_variables_initializer() #tf.initialize_all_variables() ##deprecated
    if gpu_frac <= 0.0:
        m_session = tf.Session()
        m_session.run(init_op)
        tf.set_random_seed(SEED_tf)
        K.set_session(m_session)
    else:
        gpu_opts  = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac))
        m_session = tf.Session(config=gpu_opts)
        m_session.run(init_op)
        tf.set_random_seed(SEED_tf)
        K.set_session(m_session)

        
        
## ############################################################################################### ##
def generator_lc(list_lcs, label_lcs, idx, nb_epoch=100, data_id=None, meta_liste=None, sel_keys=None):
    
    dict_filters = {'red':0, 'blue':1}
    
    counter_epochs = 0
    nclasses = label_lcs.shape[1]
    
    while counter_epochs <= nb_epoch: #True:
        counter_epochs+=1
        
        sel_keys =list(list_lcs.keys())
        
        for i in idx:
            counter_pb=-1
            X_input={}
            for mkey in sel_keys:
                counter_pb+=1
                
                lc = list_lcs[mkey][i]
                
                length_lc = len(lc.times)
               
                X = np.ndarray((1,length_lc, 3))
                X[0,:,0] = lc.times  
                X[0,:,1] = lc.measurements  
                X[0,:,2] = 0 if data_id is None else dict_filters[data_id]
    
                #sample_weight = np.ndarray((1,length_lc))
                #sample_weight[0,:] = 1/lc.errors
                
                X_input[f'main_input_pb{counter_pb}']=X
                
            Y = np.ndarray((1, nclasses), dtype=np.int8)
            Y[0,:] = label_lcs[i,:] # categorical
                
            if meta_liste is not None:
                X_input['meta_input']=meta_liste[[i],:] # meta per object
            
            
            yield X_input,Y #,err_norm
            
            

## ######################################################################################################## ##
def train_and_log(X, X_list, Y, run, model, nb_epoch, 
                  batch_size, learning_rate, loss, sim_type, metrics=[], data_id=None,
                  loss_weights=None, store_local=True, data_storedpath=None, output_store=None,
                  m_generator=True, idx_training=None, idx_validation=None, meta_liste=None,
                  sample_weight=None, sample_weight_mode=None, class_weight=None, 
                  diff_time=False, no_train=False, patience=20, finetune_rate=None,
                  validation_split=0.2, validation_data=None, gpu_frac=None,
                  noisify=False, errors=None, pretrain_weights=None, **kwargs):
    """ ---------------------------------------------------------
        @summary: Train model and write logs/model/history/weights to `outputs/keras_models/{run_id}/`
            if weights already exist, load.
        @return history :
        @return args_session :   
    ---------------------------------------------------------- """
    """Train model and write logs/weights to `outputs/keras_models/{run_id}/`.
    
    If weights already existed, they will be loaded and training will be skipped.
    """
    
    ##    limited_memory_session(gpu_frac)
    log_dir = os.path.join(output_store, sim_type, run)
    print(output_store, '\n', log_dir)
    
    
    weights_path = os.path.join(log_dir, 'weights.h5')
    model_path   = os.path.join(log_dir, 'model.h5')
    history_path = os.path.join(log_dir, 'history.h5')
    m_arg_path   = os.path.join(log_dir, 'args_session.h5')
    
    history=[]; args_session=None
    loaded = False
    if os.path.exists(history_path)&os.path.exists(model_path)&os.path.exists(m_arg_path) :
        print("\nLoading {}...".format(model_path))
        model = load_model(model_path) 
        #
        print("\nLoading {}...".format(history_path)) 
        history = joblib.load(history_path)
        #
        print("\nLoading {}...".format(m_arg_path)) 
        args_session = joblib.load(m_arg_path)
        #
        loaded = True
    
    #elif (no_train) or (finetune_rate):
    #    #raise FileNotFoundError("No weights found in {}.".format(log_dir))
    
    if finetune_rate:  # write logs to new directory
        log_dir += "_ft{:1.0e}".format(finetune_rate).replace('e-', 'm')
        
    if (not loaded):
        
        optimizer = Adam(lr=learning_rate if not finetune_rate else finetune_rate)  #, decay=decay
        if sample_weight_mode is None:
            sample_weight_mode = 'temporal' if sample_weight is not None else None
            
        if loss_weights is not None:
            model.compile(optimizer=optimizer,
                          loss=loss, 
                          metrics=metrics,
                          loss_weights=loss_weights,
                          sample_weight_mode=sample_weight_mode)
        else:
             model.compile(optimizer=optimizer, 
                           loss=loss, 
                           metrics=metrics,
                           sample_weight_mode=sample_weight_mode)
        print('Compiled model.')
        
    if (not no_train):
        if (not loaded):
            shutil.rmtree(log_dir, ignore_errors=True)
            os.makedirs(log_dir)
        
        param_log = {key: value for key, value in locals().items()}
        param_log.update(kwargs)
        param_log = {k: v for k, v in param_log.items()
                     if k not in ['X', 'Y', 'X_list', 'model', 'optimizer',
                                      'sample_weight','class_weight',
                                      'kwargs', 'validation_data', 'errors',
                                      'idx_training', 'idx_validation','meta_liste']
                        and not isinstance(v, types.FunctionType)}
        json.dump(param_log, open(os.path.join(log_dir, 'param_log.json'), 'w'),
                  sort_keys=True, indent=2)
        
        if pretrain_weights:
            model.load_weights(pretrain_weights, by_name=True)
        
        verbose=2 
        if (m_generator):
            history = model.fit_generator(generator_lc(X_list, Y, 
                                                       idx_training, nb_epoch, 
                                                       data_id, meta_liste), 
                                          #class_weight=class_weights_train,
                                          validation_data=generator_lc(X_list, Y, 
                                                                       idx_validation, nb_epoch, 
                                                                       data_id, meta_liste),
                                          epochs=nb_epoch, 
                                          steps_per_epoch= len(idx_training),
                                          validation_steps= len(idx_validation),
                                          callbacks=[#Progbar(),
                                              #TensorBoard(log_dir=log_dir, write_graph=False),
                                              EarlyStopping(patience=patience), 
                                              TimedCSVLogger(os.path.join(log_dir, 'training.csv'), 
                                                             append=True),
                                              ModelCheckpoint(weights_path, save_weights_only=True, 
                                                              save_best_only=True, verbose=False),
                                              LogDirLogger(log_dir)],
                                          verbose=verbose) 
            
        else:
            if True: #(not noisify):
                history = model.fit(X, Y, 
                                    epochs=nb_epoch, 
                                    batch_size=batch_size,
                                    sample_weight=sample_weight,
                                    class_weight=class_weight,
                                    callbacks=[#Progbar(),  
                                               #TensorBoard(log_dir=log_dir, write_graph=True),
                                               TimedCSVLogger(os.path.join(log_dir, 
                                                                           'training.csv'),append=True),
                                               EarlyStopping(patience=patience), 
                                               ModelCheckpoint(weights_path, save_weights_only=True, 
                                                               save_best_only=True, verbose=False),
                                               LogDirLogger(log_dir)], 
                                    verbose=verbose,
                                    validation_split=validation_split,
                                    validation_data=validation_data)
            '''
            else:
                history = model.fit_generator(noisify_samples(X, Y, errors, batch_size,sample_weight),
                                              samples_per_epoch=len(Y), epochs=nb_epoch,
                                              callbacks=[#Progbar(),
                                                         #TensorBoard(log_dir=log_dir, write_graph=False),
                                                         EarlyStopping(patience=patience), 
                                                         TimedCSVLogger(os.path.join(log_dir, 'training.csv'),
                                                                         append=True),
                                                         ModelCheckpoint(weights_path, save_weights_only=True, 
                                                                                       save_best_only=True, 
                                                         verbose=False),
                                                         LogDirLogger(log_dir)],
                                              verbose=verbose,
                                              validation_data=validation_data)
                '''
        print('--> Model trained.')
        
        model_path = os.path.join(log_dir, 'model.h5')
        model.save(model_path)
        
        ## moved to run_autoencoder_** 
        #joblib.dump(history, history_path)        
    
    
    ## moved to run_autoencoder_** 
    #K.clear_session()                              
    
    return history, args_session


## ############################################################################################### ##
def run_autoencoder_gen(arg_dict, input_lcs, input_metadata, output_dict):
    
    ## ------------------- DATA ------------------- ##
    LC_types    = output_dict['LC_types']
    Y_label_int = output_dict['Y_label_int']
    Y_label_cat = output_dict['Y_label_cat']
    
    meta_liste = input_metadata['selected'].values if arg_dict.add_metadata else None
    
    nlc = Y_label_cat.shape[0]
    nclasses = Y_label_cat.shape[1]
    
    if arg_dict.data_id=='multiple':
        pb_keys=['red', 'blue'] 
    else:
        pb_keys = [arg_dict.data_id]
        
    arg_dict.nb_passbands = len(pb_keys) if arg_dict.data_id=='multiple' else 1
    
    X_list={}; 
    for mkey in pb_keys:
        X_list[mkey] = input_lcs[mkey]
    
    
    ## ------------------- TRAIN/VALIDATION ------------------- ##
    from sklearn.model_selection import StratifiedKFold
    m_validation_split = arg_dict.validation_split
    
    stratif1 =  StratifiedKFold(n_splits=int(1/m_validation_split),shuffle=True, 
                                random_state=SEED).split(X_list[pb_keys[0]], 
                                                         Y_label_int) 
    idx_training_full, idx_test = list(stratif1)[0]
    
    stratif2 =  StratifiedKFold(n_splits=int(1/m_validation_split),shuffle=True, 
                                random_state=SEED).split(X_list[pb_keys[0]][idx_training_full,],#:,:], 
                                                         Y_label_int[idx_training_full,]) 
    idx_training_, idx_validation_ = list(stratif2)[0]
    idx_training   = idx_training_full[idx_training_]
    idx_validation = idx_training_full[idx_validation_]
    
    if True:   
        print('len(X_list[mkey]) = ',[len(X_list[mkey]) for mkey in pb_keys],\
                '\nY_label_int.shape = ',  Y_label_int.shape,\
                '\nY_label_cat.shape = ',  Y_label_cat.shape,\
                 'len(idx_training) = ', len(idx_training),\
                '\nlen(idx_validation) = ',  len(idx_validation),\
                '\nlen(idx_test) = ',  len(idx_test))
    
    
     ## ---------------------- ## CLASS FREQUENCY ## ---------------------- ##
    classnames = np.r_[0, np.unique(Y_label_int)] 
    counter_all = np.sum(Y_label_cat, axis=0) #Counter(Y_label_int)
    sum_all     = np.sum(counter_all)
    
    mprecision = 3
    dict_ratios = {}
    for i in classnames:
        dict_ratios[i]=np.around((counter_all[i]/sum_all), mprecision)
    dict_ratios[np.argmax(counter_all)]+=(1-np.sum(list(dict_ratios.values())))
    
    dict_ratios_init = copy.deepcopy(dict_ratios)
    dict_ratios_init.pop(0,None)

    freqs_     = np.asarray([dict_ratios[Y_label_int[i]] for i in range(len(Y_label_int))])
    freqs_cat  = [Y_label_cat[i,:]*freqs_[i] for i in range(len(freqs_))]
    freqs_cat  = np.asarray(freqs_cat)
    
    
    ## ------------------- ARCHITECTURE ------------------- ##
    input_dimension = 3 #(times, mags, passbands)
    net_func = dict_nfuncs[arg_dict.run_id]
    
    m_model=None; param_str=None; param_dict=None
    
    if arg_dict.gpu_frac is not None:
        #ku.limited_memory_session(arg_dict.gpu_frac)
        init_op = tf.global_variables_initializer() #tf.initialize_all_variables() ##deprecated
        if arg_dict.gpu_frac <= 0.0:
            m_session = tf.Session()
            m_session.run(init_op)
            tf.set_random_seed(SEED_tf)
            K.set_session(m_session)
        else:
            gpu_opts = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=arg_dict.gpu_frac))
            m_session = tf.Session(config=gpu_opts)
            m_session.run(init_op)
            tf.set_random_seed(SEED_tf)
            K.set_session(m_session)
            
    m_model, param_str, param_dict = net_func(input_dimension, 
                                              output_size = nclasses if arg_dict.categorical else 1,
                                              max_lenght=None,
                                              return_sequences=False,
                                              add_meta=meta_liste.shape[1] if arg_dict.add_metadata else None,
                                              **vars(arg_dict)
                                             )                                      
    print(m_model.summary())
    
    
    
    ## ------------------- CLASS WEIGHTS ------------------- ##
    class_weights_train = None
    if (arg_dict.run_id not in list_ae) & (arg_dict.add_freqs is not None): # CLF || DUAL
        class_weights_train = {'clf_softmax_dense': freqs_cat[idx_training,:] if arg_dict.categorical else dict_ratios_init}  
        
    ## ------------------- SAMPLE_WEIGHT_MODE ------------------- ##
    if arg_dict.run_id in list_ae:
        sample_weight_mode = {'decode_time_dist'  :'temporal'}
    #
    elif arg_dict.run_id in list_composite:
        sample_weight_mode = {'decode_time_dist'  :'temporal',   'clf_softmax_dense' : None}
    else:
        sample_weight_mode = {'clf_softmax_dense' : None}
        
    
    ## ------------------- COMPILE & TRAIN ------------------- ##
    m_history=None; args_session=None
    m_run = get_run_id(**vars(arg_dict),param_str=param_str)
    m_history, args_session = train_and_log(X=None,
                                            Y=Y_label_cat,
                                            class_weight=class_weights_train, 
                                            sample_weight=None, 
                                            #
                                            run=m_run,
                                            model=m_model, 
                                            param_dict=param_dict,
                                            #
                                            m_generator=True,
                                            #
                                            X_list=X_list,
                                            idx_training=idx_training,
                                            idx_validation=idx_validation,
                                            meta_liste=meta_liste,
                                            #
                                            **vars(arg_dict)
                                           )    

    if (args_session is None) & (~arg_dict.no_train):
        args_session = {'data_id' : arg_dict.data_id,
                        'run_id' : dict_nruns[arg_dict.run_id],
                        'classnames' : LC_types,
                        #
                        'idx_test' : idx_test, 
                        'idx_training_full' : idx_training_full,  
                        'idx_training' : idx_training,
                        'idx_validation' : idx_validation,
                        #
                        'loss ' : arg_dict.loss,
                        'metrics ' : str(arg_dict.metrics),
                        'sample_weight' : False,
                        'class_weight'  : False
                       }
        
        m_arg_path = os.path.join(arg_dict.output_store, arg_dict.sim_type, m_run, 'args_session.h5')
        joblib.dump(args_session,m_arg_path)
    
    K.clear_session()
    
    return 1;






## ############################################################################################### ##
def run_autoencoder_pad(arg_dict, input_lcs, input_metadata, output_dict):
    
    ## ------------------- DATA ------------------- ##
    LC_types    = output_dict['LC_types']
    Y_label_int = output_dict['Y_label_int']
    Y_label_cat = output_dict['Y_label_cat']
    
    meta_liste  = input_metadata['selected'].values if arg_dict.add_metadata else None
    
    nlc      = Y_label_cat.shape[0]
    nclasses = Y_label_cat.shape[1]
    
    if arg_dict.data_id=='multiple':
        pb_keys=['red', 'blue']
    else:
        pb_keys = [arg_dict.data_id]
        
    arg_dict.nb_passbands = len(pb_keys) if arg_dict.data_id=='multiple' else 1
    
    sample_weights={}
    X_phot={}; err_phot={}; sample_weights={}
    for mkey in pb_keys:
        X_phot[mkey]   = input_lcs[mkey]
        err_phot[mkey] = X_phot[mkey][:,:,2]        # store errors separately
        X_phot[mkey]   = X_phot[mkey][:,:,[0,1,3]]  # retain mag + times + passbands
        
        if (arg_dict.run_id in np.r_[list_ae, list_composite]):
            if arg_dict.loss['decode_pb0_time_dist'] =='mse':
                ## NAUL+18 use sample_weights=1/sigma in MSE
                sample_weights[mkey] = 1/(err_phot[mkey]) ## 1/(err_phot[mkey]**2)
            
            elif arg_dict.loss['decode_pb0_time_dist']=='mae':
                sample_weights[mkey] = 1/(err_phot[mkey])
        else:
            sample_weights[mkey] = 1/(err_phot[mkey])
        
        sample_weights[mkey][np.isnan(sample_weights[mkey])] = 0.

    
    
    ## ------------------- TRAIN/VALIDATION ------------------- ##
    from sklearn.model_selection import StratifiedKFold
    m_validation_split = arg_dict.validation_split
    
    stratif1 =  StratifiedKFold(n_splits=int(1/m_validation_split),shuffle=True, 
                                random_state=SEED).split(X_phot[pb_keys[0]], 
                                                         Y_label_int) 
    idx_training_full, idx_test = list(stratif1)[0]
    
    stratif2 =  StratifiedKFold(n_splits=int(1/m_validation_split),shuffle=True, 
                                random_state=SEED).split(X_phot[pb_keys[0]][idx_training_full,:,:], 
                                                         Y_label_int[idx_training_full,]) 
    idx_training_, idx_validation_ = list(stratif2)[0]
    idx_training    = idx_training_full[idx_training_]
    idx_validation = idx_training_full[idx_validation_]
    
    ## ------------------ DATA ------------------- ##
    X_ae_train={}; X_ae_valid={}
    X_phot_train={}; err_phot_train={}
    X_phot_valid={}; err_phot_valid={}
    for mkey in pb_keys:
        X_phot_train[mkey]   = X_phot[mkey][idx_training,:,:]
        err_phot_train[mkey] = err_phot[mkey][idx_training,:]
        #
        X_phot_valid[mkey]   = X_phot[mkey][idx_validation,:,:]
        err_phot_valid[mkey] = err_phot[mkey][idx_validation,:]
        #
        X_ae_train[mkey]     = X_phot_train[mkey][:,:,[1]] #mags
        X_ae_valid[mkey]     = X_phot_valid[mkey][:,:,[1]] #mags
    
    Y_label_int_train = Y_label_int[idx_training]
    Y_label_cat_train = Y_label_cat[idx_training,:]
    #
    Y_label_int_valid = Y_label_int[idx_validation]
    Y_label_cat_valid = Y_label_cat[idx_validation,:]

    #
    meta_liste_train = meta_liste[idx_training,:]   if arg_dict.add_metadata is not None else None
    meta_liste_valid = meta_liste[idx_validation,:] if arg_dict.add_metadata is not None else None
    
    
    ## ---------------------- ## CLASS FREQUENCY ## ---------------------- ##
    classnames  = np.r_[0, np.unique(Y_label_int)] 
    counter_all = np.sum(Y_label_cat, axis=0) 
    sum_all     = np.sum(counter_all)
    
    mprecision = 3
    dict_ratios = {}
    for i in classnames:
        dict_ratios[i]=np.around((counter_all[i]/sum_all), mprecision)
    dict_ratios[np.argmax(counter_all)]+=(1-np.sum(list(dict_ratios.values())))
    
    dict_ratios_init = copy.deepcopy(dict_ratios)
    dict_ratios_init.pop(0,None)

    freqs_       = np.asarray([dict_ratios[Y_label_int[i]] for i in range(len(Y_label_int))])
    freqs_cat  = [Y_label_cat[i,:]*freqs_[i] for i in range(len(freqs_))]
    freqs_cat  = np.asarray(freqs_cat)
    
    ## ------------------- CONSOLE ------------------- ##
    if True:         
         for mkey in pb_keys:
            print(f'\n\n********** X_phot_full[{mkey}].shape \t=', X_phot[mkey].shape,
                     f'\n********** err_phot_full[{mkey}].shape \t=', err_phot[mkey].shape,
                     '\n********** Y_label_int_full.shape \t=', Y_label_int.shape,
                     '\n********** Y_label_cat_full.shape \t=', Y_label_cat.shape,
                     )
            print(f'\n\n********** X_phot_train[{mkey}].shape \t=', X_phot_train[mkey].shape,
                    f'\n********** err_phot_train[{mkey}].shape \t=', err_phot_train[mkey].shape,
                    '\n********** Y_label_int_train.shape \t=', Y_label_int_train.shape,
                    '\n********** Y_label_cat_train.shape \t=', Y_label_cat_train.shape,
                     )
            print(f'\n\n********** X_phot_valid[{mkey}].shape \t=', X_phot_valid[mkey].shape,
                    f'\n********** err_phot_valid[{mkey}].shape \t=', err_phot_valid[mkey].shape,
                    '\n********** Y_label_int_valid.shape \t=', Y_label_int_valid.shape,
                    '\n********** Y_label_cat_valid.shape \t=', Y_label_cat_valid.shape,
                     )
            print()
            
            print(f'\n********** sample_weights[{mkey}].shape \t= ', 
                  sample_weights[mkey].shape if (arg_dict.run_id in np.r_[list_ae, list_composite]) else None)
            print('********** freqs_cat.shape \t\t= ',  
                  freqs_cat.shape if arg_dict.add_freqs else None, '\n')
  
         if arg_dict.add_metadata is not None:
            print('********** meta_liste_train.shape \t=', meta_liste_train.shape, 
                   '\n********** meta_liste_valid.shape \t=', meta_liste_valid.shape,
                   '\n' )
        
    ## ------------------- ARCHITECTURE ------------------- ##
    sel_keys = pb_keys[:-1] if arg_dict.data_id=='multiple' else pb_keys
    max_lenght      = [X_phot[mkey].shape[1] for mkey in sel_keys]
    input_dimension = X_phot[pb_keys[0]].shape[-1]    # = 3,  times + mags + passbands
    
    
    net_func = dict_nfuncs[arg_dict.run_id]
    m_model=None; param_str=None; param_dict=None
    m_datapoints = max_lenght[0] if arg_dict.padding else None
    
    if arg_dict.gpu_frac is not None:
        #limited_memory_session(arg_dict.gpu_frac)
        init_op = tf.global_variables_initializer() #tf.initialize_all_variables() ##deprecated
        if arg_dict.gpu_frac <= 0.0:
            m_session = tf.Session()
            m_session.run(init_op)
            tf.set_random_seed(SEED_tf)
            K.set_session(m_session)
        else:
            gpu_opts = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=arg_dict.gpu_frac))
            m_session = tf.Session(config=gpu_opts)
            m_session.run(init_op)
            tf.set_random_seed(SEED_tf)
            K.set_session(m_session)
    
    
    if (arg_dict.run_id in list_ae): # encoder-decoder
        m_model, param_str, param_dict = net_func(input_dimension, 
                                                  max_lenght = max_lenght,
                                                  gp_down_size = m_datapoints,
                                                  add_meta=meta_liste.shape[1] if arg_dict.add_metadata else None,
                                                  **vars(arg_dict))
    #    
    elif (arg_dict.run_id in list_composite): # ae-clf
        m_model, param_str, param_dict = net_func(input_dimension, 
                                                  max_lenght = max_lenght,
                                                  output_size = nclasses if arg_dict.categorical else 1,
                                                  gp_down_size = m_datapoints,
                                                  add_meta=meta_liste.shape[1] if arg_dict.add_metadata else None,
                                                  **vars(arg_dict))
    #
    elif (arg_dict.run_id in list_clf_meta): # encoder-decoder
        m_model, param_str, param_dict = net_func(output_size = nclasses if arg_dict.categorical else 1,
                                                  add_meta=meta_liste.shape[1] if arg_dict.add_metadata else None,
                                                  **vars(arg_dict))   
    #
    else: #'classifiers'
        m_model, param_str, param_dict = net_func(input_dimension, 
                                                  max_lenght = max_lenght,
                                                  output_size = nclasses if arg_dict.categorical else 1,
                                                  return_sequences=False,
                                                  add_meta=meta_liste.shape[1] if arg_dict.add_metadata else None,
                                                  **vars(arg_dict))                                              
                                                                             
    #print(m_model.summary())
    
    ## ------------------- INPUT NET ------------------- ##
    input_data_train = {}
    input_data_valid = {}
    
    counter_pb=-1
    for mkey in sel_keys:
        counter_pb+=1
        input_data_train[f'main_input_pb{counter_pb}'] = X_phot_train[mkey]
        input_data_valid[f'main_input_pb{counter_pb}'] = X_phot_valid[mkey]
    
    if (arg_dict.aux_in) & (arg_dict.run_id in list_ae+list_composite):
        counter_pb=-1
        for mkey in sel_keys:
            counter_pb+=1
            input_data_train[f'aux_input_concat_pb{counter_pb}'] = np.delete(X_phot_train[mkey], 1, axis=2)
            input_data_valid[f'aux_input_concat_pb{counter_pb}'] = np.delete(X_phot_valid[mkey], 1, axis=2)
    #
    if (arg_dict.add_metadata) & ((arg_dict.run_id not in [list_ae])|arg_dict.run_id not in [list_composite]):
        input_data_train['meta_input'] = meta_liste_train
        input_data_valid['meta_input'] = meta_liste_valid
    
    if arg_dict.aux_in ==True:
            counter_pb=-1
            for mkey in sel_keys:
                counter_pb+=1
                print('********** AUX \t= ' , input_data_train.keys(),
                      input_data_train[f'aux_input_concat_pb{counter_pb}'].shape,
                      input_data_valid[f'aux_input_concat_pb{counter_pb}'].shape)
                
                
    ## ------------------- OUTPUT NET ------------------- ##
    output_data_train = {}
    output_data_valid = {}
    
    if (arg_dict.run_id in list_ae):
        counter_pb=-1
        for mkey in sel_keys:
            counter_pb+=1
            output_data_train[f'decode_pb{counter_pb}_time_dist'] = X_ae_train[mkey]
            output_data_valid[f'decode_pb{counter_pb}_time_dist'] = X_ae_valid[mkey]
            
    #
    elif (arg_dict.run_id in list_composite):
        counter_pb=-1
        for mkey in sel_keys:
            counter_pb+=1
            output_data_train[f'decode_pb{counter_pb}_time_dist'] = X_ae_train[mkey]
            output_data_valid[f'decode_pb{counter_pb}_time_dist'] = X_ae_valid[mkey]
        
        output_data_train['clf_softmax_dense'] = Y_label_cat_train if arg_dict.categorical else  Y_label_int_train
        output_data_valid['clf_softmax_dense'] = Y_label_cat_valid if arg_dict.categorical else  Y_label_int_valid
    #
    else:
        output_data_train['clf_softmax_dense'] = Y_label_cat_train if arg_dict.categorical else Y_label_int_train
        output_data_valid['clf_softmax_dense'] = Y_label_cat_valid if arg_dict.categorical else Y_label_int_valid
    
    
    ## ------------------- SAMPLE WEIGHTS ------------------- ##
    sample_weights_train = None
    sample_weights_valid = None
    if (arg_dict.run_id in np.r_[list_ae, list_composite]): 
        sample_weights_train={}; sample_weights_valid={}
        counter_pb=-1
        for mkey in sel_keys:
            counter_pb+=1
            sample_weights_train[f'decode_pb{counter_pb}_time_dist'] = sample_weights[mkey][idx_training,:]
            sample_weights_valid[f'decode_pb{counter_pb}_time_dist'] = sample_weights[mkey][idx_validation,:]
        
    
    ## ------------------- CLASS WEIGHTS ------------------- ##
    class_weights_train = None
    if (arg_dict.run_id not in list_ae) & (arg_dict.add_freqs is not None): # CLF || DUAL
        class_weights_train = {'clf_softmax_dense': freqs_cat[idx_training,:] if arg_dict.categorical else dict_ratios_init}  
        
        
    ## ------------------- VALIDATION DATA ------------------- ##
    validation_data = (input_data_valid, output_data_valid, sample_weights_valid)
    
    
    ## ------------------- SAMPLE_WEIGHT_MODE ------------------- ##
    if arg_dict.run_id in list_ae:
        sample_weight_mode ={}
        counter_pb=-1
        for mkey in sel_keys:
            counter_pb+=1
            sample_weight_mode[f'decode_pb{counter_pb}_time_dist']='temporal'
    #
    elif arg_dict.run_id in list_composite:
        sample_weight_mode ={}
        counter_pb=-1
        for mkey in sel_keys:
            counter_pb+=1
            sample_weight_mode[f'decode_pb{counter_pb}_time_dist']='temporal'
        sample_weight_mode['clf_softmax_dense']= None
    else:
        sample_weight_mode = {'clf_softmax_dense' : None}
    
    
    ## ------------------- CONSOLE ------------------- ##
    if True:
        print()
        print('********** input_data_train.keys \t=',  input_data_train.keys())
        print('********** output_data_train.keys \t=', output_data_train.keys())
        print()
        print('********** input_data_valid.keys \t=',  input_data_valid.keys())
        print('********** output_data_valid.keys \t=', output_data_valid.keys())
        
        print()
        if sample_weights_train is not None:
            counter_pb=-1
            for mkey in sel_keys:
                counter_pb+=1
                print('********** sample_weights \t\t= ' , 
                      sample_weights_train.keys(),
                      #sample_weights_train[f'decode_pb{counter_pb}_time_dist'].shape,
                      #sample_weights_valid[f'decode_pb{counter_pb}_time_dist'].shape
                     )
        else:
            print('********** sample_weights_train \t=  None')
       
        if class_weights_train is not None:
            print('********** class_weights_train \t\t= ' , 
                  class_weights_train.keys(), 
                  class_weights_train['clf_softmax_dense'].shape if arg_dict.categorical else class_weights_train)
        else:
            print('********** class_weights_train \t\t=  None')
            
        print('********** sample_weight_mode \t\t= ', sample_weight_mode, '\n')
   
    print(m_model.summary())
   

    ## ------------------- COMPILE & TRAIN ------------------- ##
    m_history=None; args_session=None
    m_run = get_run_id(**vars(arg_dict),param_str=param_str)
    m_history, args_session = train_and_log(X=input_data_train,
                                            Y=output_data_train,
                                            class_weight=class_weights_train, 
                                            sample_weight=sample_weights_train,
                                            #
                                            run=m_run,
                                            model=m_model, 
                                            param_dict=param_dict,
                                            #
                                            validation_data=validation_data, 
                                            #
                                            errors=err_phot_train, 
                                            #
                                            m_generator=False,
                                            #
                                            X_list=None,
                                            idx_training=None,
                                            idx_validation=None,
                                            meta_liste=None,
                                            #
                                            sample_weight_mode=sample_weight_mode,
                                            #
                                            **vars(arg_dict)
                                           )    
    if (args_session is None) & (~arg_dict.no_train):
        args_session = {'data_id' : arg_dict.data_id,
                        'run_id' : dict_nruns[arg_dict.run_id],
                        'classnames' : LC_types,
                        #
                        'idx_test' : idx_test,
                        'idx_training_full' : idx_training_full,  
                        'idx_training' : idx_training,
                        'idx_validation' : idx_validation,
                        #
                        'loss ' : arg_dict.loss,
                        'metrics '  : str(arg_dict.metrics),
                        'sample_weight' : True if sample_weights_train is not None else False,
                        'class_weight' : True if class_weights_train is not None else False
                       }
        
        m_arg_path = os.path.join(arg_dict.output_store, arg_dict.sim_type, m_run, 'args_session.h5')
        joblib.dump(args_session,m_arg_path)
        
    
    K.clear_session()
    
    return 1;
