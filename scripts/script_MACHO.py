## ---------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------- ##
#   SCRIPT - classification of MACHO preprocessed data
## ---------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------- ##

import sys, os, errno
import numpy as np
import time, datetime
import copy, joblib, shutil
from collections import Counter

#module_path = os.path.dirname(os.getcwd())+'/../src/'
module_path = os.getcwd()+'/src/'
if module_path not in sys.path:
    sys.path.append(module_path)


#### MOVED to ****functions_keras.py****
##import tensorflow as tf
##SEED_tf=42; tf.compat.v1.set_random_seed(SEED_tf) 

import keras.backend as K
from light_curve import LightCurve
import functions_preprocess as m_preprocess #cf
import functions_keras as m_func #ku

SEED=0
np.random.seed(SEED)

dict_filters = {'red':0, 'blue':1}
from functions_keras import dict_nfuncs, dict_nruns, list_ae, list_composite, list_clf_meta, list_clf


LC_types = { 1: 'RR_Lyrae_AB',        # RR Lyraes, fundamental mode pulsators
             2: 'RR_Lyrae_C',         # RR Lyraes, 1st overtone pulsators
             3: 'RR_Lyrae_E',         # RR Lyraes, 2nd overtone pulsators
             4: 'Cepheid_Fund',       # Cepheids, fundamental mode pulsators 
             5: 'Cepheid_1st',        # Cepheid, 1st overtone pulsators
             6: 'LPV_WoodA',          # Long-Period Variables, Wood Sequence A
             7: 'LPV_WoodB',          # Long-Period Variables, Wood Sequence B
             8: 'LPV_WoodC',          # Long-Period Variables, Wood Sequence C
             9: 'LPV_WoodD',          # Long-Period Variables, Wood Sequence D
            10: 'Eclipsing_Binary',   # Eclipsing Binaries
            11:' RR_Lyrae_GB',        # RRL + GB blends (?)
           }

## ############################################################################ ##
def run_network(arg_dict, input_lcs, input_metadata, output_dict):
    
    if (arg_dict.run_id in np.r_[list_ae, list_composite])&(arg_dict.padding): ## fixed-length  
        print('TRAIN BACTH')
        m_func.run_network_pad(arg_dict, input_lcs, input_metadata, output_dict)
    else:
        print('TRAIN GENERATORS')
        m_func.run_network_gen(arg_dict, input_lcs, input_metadata, output_dict) 
        
    return 1;
    
            
## ############################################################################ ##
def get_data(arg_dict, fileformat='pkl'):
    ''' ---------------------------------------------------------
        Load stored preprocessed data 
    ---------------------------------------------------------- '''
    
    import pandas as pd
    
    input_lcs={}; 
    input_metadata={}

    data_id  = arg_dict.data_id
    data_dir = arg_dict.data_store+'preprocessed_data/'
    
    meta_dir = data_dir+'MACHO_metadata.xlsx'
    if not os.path.isfile(meta_dir):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), meta_dir)
    else:
        ## Ylabels
        output_dict = np.load(data_dir+'MACHO_labels.npz', allow_pickle=True)
        
        ## Xmeta
        input_metadata['selected']  = pd.read_excel(meta_dir, 
                                                    sheet_name='selected_metadata',  #'full_metadata'
                                                    index_col=0)
        ## Xphot
        input_lcs['data_id']     = data_id
        input_lcs['data_norm']   = 'raw_LCs'       if arg_dict.use_raw     else 'normalized_LCs'
        input_lcs['data_type']   = 'phasefold'     if arg_dict.period_fold else 'timeseries'
        input_lcs['data_format'] = 'fixed_lengths' if arg_dict.padding     else 'initial_lengths' 
        
        m_folder = os.path.join(data_dir, 
                                'pkl_fileformat' if fileformat=='pkl' else 'npz_fileformat',
                                input_lcs['data_type'], 
                                input_lcs['data_format']
                                )
        
        data_norm  = 'raw' if arg_dict.use_raw else 'norm'
        data_fmt   = 'fold' if arg_dict.period_fold else 'times'
        filename   = f'X{data_norm}_{data_fmt}' if arg_dict.padding else f'listLC_{data_fmt}'
        m_ext      = '.pkl' if fileformat=='pkl' else '.npz'
        
        if data_id=='multiple':
            m_path_b = f'{m_folder}/{filename}_blue{m_ext}'
            m_path_r = f'{m_folder}/{filename}_red{m_ext}'
            input_lcs['blue'] = joblib.load(m_path_b) if fileformat=='pkl' else np.load(m_path_b)
            input_lcs['red']  = joblib.load(m_path_r) if fileformat=='pkl' else np.load(m_path_r)
        else:
            m_path = f'{m_folder}/{filename}_{data_id}{m_ext}'
            input_lcs[data_id] = joblib.load(m_path) if fileformat=='pkl' else np.load(m_path)
        
    return input_lcs, input_metadata, output_dict



## ############################################################################ ##
def set_params_cline(args):    
    ''' ---------------------------------------------------------
        Set parameters 
    ---------------------------------------------------------- '''
    
    import datetime
    m_dateformat = '%m%d%Y'
    m_date = datetime.datetime.now().strftime(m_dateformat)
    
    args.sim_type = args.sim_type+('_fixedlength' if args.padding else '_generator')
    args.sim_type += f'_{m_date}'
    
    args.data_store = os.getcwd()+ args.data_store
    args.output_store = os.getcwd()+ args.output_store
    
    nb_passbands=2 if args.data_id=='multiple' else 1
           
    if args.loss_weights_list is None:
        loss_w1 = 1.
        loss_w2 = 1.
    else:
        loss_weights_list = [float(item) for item in args.loss_weights_list.split(':')]
        loss_w1 = loss_weights_list[0]
        loss_w2 = loss_weights_list[1]
    
    ## ------------------ COMPOSITE NETWORKS ------------------ ##
    if args.run_id in list_composite:
        args.metrics={}; args.loss={}; args.loss_weights={}
        
        ## ENCODER-CLF BRANCH
        args.loss         ['clf_softmax_dense'] = args.loss_CLF    
        args.metrics      ['clf_softmax_dense'] = args.metrics_CLF 
        args.loss_weights ['clf_softmax_dense'] = loss_w1
        
        ## ENCODER-DECODER BRANCH
        for id_passband in range(nb_passbands) :
            idn = 'decode_pb{}_time_dist'.format(id_passband)
            args.loss         [idn] = args.loss_AE 
            args.metrics      [idn] = []
            args.loss_weights [idn] = loss_w2
            
    ## ------------------ AUTOENCODERS NETS ------------------ ##
    elif args.run_id in list_ae:
        args.metrics={}; args.loss={}; args.loss_weights={}
        
        ## ENCODER-DECODER BRANCH
        for id_passband in range(nb_passbands) :
            idn = 'decode_pb{}_time_dist'.format(id_passband)
            args.loss         [idn] = args.loss_AE  
            args.metrics      [idn] = [] if args.metrics_AE is None else args.metrics_AE
            args.loss_weights [idn] = loss_w2
            
    ## ------------------ DIRECT CLASSIFIERS ------------------ ##
    else :
        ## ENCODER-CLF BRANCH
        args.metrics={}; args.loss={}; args.loss_weights={}
        args.loss         ['clf_softmax_dense'] = args.loss_CLF 
        args.metrics      ['clf_softmax_dense'] = args.metrics_CLF
        args.loss_weights ['clf_softmax_dense'] = loss_w1
        
    args.n_min = 200  
    m_func.print_args(args)
    
    return args



## ############################################################################ ##
def main(args=None):
    
    np.random.seed(SEED)
    
    print("\n\t# ----------------------------------------------------------------------- #",
            "\n\t# ------------------------------- [MACHO] ------------------------------- #",
            "\n\t# ----------------------------------------------------------------------- #\n")
    
    stime = time.time() 
    
    if True:
        ############## 1 - Set args session ############## 
        arg_dict = set_params_cline(args)  
        
        ############## 2 - Load stored data_structures  ############## 
        input_lcs=None; input_metadata=None; output_dict=None
        input_lcs, input_metadata, output_dict = get_data(arg_dict)
        
        ############## 3 - Train network(s) ##############
        if input_lcs is not None:
             run_network(arg_dict, input_lcs, input_metadata, output_dict)
          
        
    hours, rem = divmod(time.time() - stime, 3600) #timeit.default_timer()-stime
    minutes, seconds = divmod(rem, 60)
    print("\n*Execution time : {:0>2} h {:0>2} min {:05.2f} s".format(int(hours), int(minutes), seconds))


## ############################################################################ ##
if __name__ == "__main__":
    args = m_func.parse_model_args()
    main(args)

