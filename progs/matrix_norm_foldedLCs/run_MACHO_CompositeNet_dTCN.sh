## #################### RUN ID & SETTINGS #################### ##
#   dict NNruns : { 0 : 'classifier_MLP_meta',
#                   #
#                   1 : 'classifier_direct_RNN',
#                   2 : 'classifier_direct_tCNN',
#                   3 : 'classifier_direct_dTCN',
#                   #
#                   4 : 'autoencoder_RNN',
#                   5 : 'autoencoder_tCNN',
#                   6 : 'autoencoder_dTCN',
#                   #
#                   7 : 'composite_net_RNN',
#                   8 : 'composite_net_tCNN',
#                   9 : 'composite_net_dTCN',
#                 }
## ########################################################### ##
run_id=9
sim_type='composite_net_dTCN'

data_id="${1}"
model_type="${2}"
sizenet="${3}"
num_layers="${4}"

## Composite nets & Autoencoders
embedding=8

drop_frac=0.25
batch_size=128
learning_rate=0.0005
nb_epoch=200
validation_split=0.2
gpu_frac=0.0
loss_AE='mae'
loss_CLF='categorical_crossentropy' # if categorical else 'logcosh'
metrics_CLF='categorical_accuracy'  # if categorical else 'accuracy'

data_store='../example_data/MACHO/'
output_store='../outputs/trained_models/'

##   --use_raw          ## TRUE: normalized data,   FALSE: initial obs.
##   --padding          ## TRUE: fixed-length data, FALSE: initial lengths
##   --period_fold      ## TRUE: phase-folded LCs,  FALSE: time-series
##   --add_metadata     ## TRUE: metadata as ancillary input,  FALSE: none

## -------- RNN  -------- ##
# --bidirectional

## -------- tCNN  -------- ##
m_activation='tanh'
kernel_size=5
max_dilation=2

## -------- dTCN  -------- ##
n_stacks=num_layers
kernel_size=3
kernel_wavenet=1
max_dilation=2
m_activation='wavenet'


python3.6 ../scripts/script_MACHO.py --run_id=$run_id --sim_type=$sim_type --data_id=$data_id --model_type=$model_type  --sizenet=$sizenet --num_layers=$num_layers --drop_frac=$drop_frac --batch_size=$batch_size --learning_rate=$learning_rate --nb_epoch=$nb_epoch --validation_split=$validation_split --gpu_frac=$gpu_frac --loss_AE=$loss_AE --loss_CLF=$loss_CLF --metrics_CLF=$metrics_CLF --data_store=$data_store --output_store=$output_store --add_dense --categorical --causal --padding --period_fold --add_metadata --n_stacks=$n_stacks --kernel_size=$kernel_size --kernel_wavenet=$kernel_wavenet --max_dilation=$max_dilation --m_activation=$m_activation --embedding=$embedding
