## ---------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------- ##
#   FUNCTIONS - Results & displays
## ---------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------- ##

import sys, os
import numpy as np
import pandas as pd
import joblib, json, glob
from copy import deepcopy

from collections import namedtuple
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, roc_curve, log_loss #cohen_kappa_score, roc_auc_score

_KERAS_EPS = 1e-7  ## LOG_LOSS undefined for probas P in {0,1}, P are clipped to MAX(eps, min(1-eps, P)).

import setup_notebook


def parse_logs(log_files, max_epoch=250):
    """ ---------------------------------------------------------
        @summary: load training logs (times, epochs)
    ---------------------------------------------------------- """
    logs = [pd.read_csv(f, index_col="epoch", parse_dates=[1]) for f in log_files]
    for log, f in zip(logs, log_files):
        run = f.split("/")[-2]
        log.drop(log.index[log.index > max_epoch], axis=0, inplace=True)
        if "time" not in log:
            raise ValueError("Missing time samples from {}".format(f))
        updated_cols = []
        columns = list(log.columns)
        ncols = len(log.columns)
        for ii in range(len(columns)):
            if columns[ii]=="loss":
                columns[ii] = "total_loss"
            if columns[ii]=="val_loss":
                columns[ii] = "val_total_loss"
            if ii==0:
                updated_cols.append(columns[ii])
            else:
                updated_cols.append(run+ " "+ columns[ii])
        log.columns = updated_cols
        log["time"] = log["time"].values.astype(float)
        log["time"] = (log["time"] - log["time"].min()) / 1e9 /3600
    time_logs = []; step_logs = []
    if len(logs)>0:
        step_logs = pd.concat([l.drop("time", axis=1, inplace=False) for l in logs], axis=1)
        time_logs = pd.concat([l.set_index("time") for l in logs], axis=1)
    return step_logs, time_logs



def training_plot(logs, loss_type="Valid", ylim=None,  wi=11, hei=8, siz=14,
                  ax=None, leg=None, linestyle=None, yscale="log",ylab="Loss"):
    """ ---------------------------------------------------------
        @summary: display of training logs (times, epochs)
    ---------------------------------------------------------- """
    counter=-1 if (loss_type=="loss") else -2
    for i, c in enumerate(logs.columns):
        c_sharp = c.split(" ")[1]
        if (loss_type) is not None and (loss_type == c_sharp):
            counter+=1
            to_plot = logs[c]
            to_plot.dropna(inplace=True)
            mlinestyle = linestyle if linestyle is not None else "-"
            to_plot.plot(color=color_list_display0[int(counter)], legend=False, linestyle=mlinestyle)
    if (leg is not None):
        plt.legend(leg, loc="upper right", shadow=True, ncol=1, frameon=True, fontsize=siz-2)
    LABEL_MAP = {"epoch": "Epochs", "time": "Time (hours)"}
    plt.xlabel(LABEL_MAP[logs.index.name], weight="bold", fontsize=siz-2)
    plt.ylabel(ylab, weight="bold", fontsize=siz-2)
    ax.get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.grid(b=True, which="major", color="w", linewidth=1.0)
    ax.grid(b=True, which="minor", color="w", linewidth=0.5)
    plt.xticks(fontsize=siz-2)
    plt.yticks(fontsize=siz-2)
    
        

def run_args(mpath_json):
    """ ---------------------------------------------------------
        @summary: List of main hyperparameters in a model
    ---------------------------------------------------------- """
    list_fields = ['model_type','nb_passbands','nb_epoch',
                'run_generator','batch_size','sizenet','num_layers',
                'n_stacks','max_dilation','embedding','drop_frac',
                'diff_time','categorical','add_metadata','bidirectional',
                'config_wavenet','kernel_size','kernel_wavenet',
                'm_activation','m_reductionfactor','use_skip_connections',
                'nbpoints','period_fold','padding','ss_resid','use_raw',
                'metrics','loss','loss_weights','loss_weights_list','validation_split']
    with open(mpath_json) as json_file:
        mjson_dict = json.load(json_file)
        vals = [mjson_dict[e] for idx,e in enumerate(margs_structure._fields) if e in list(mjson_dict.keys())]
        mjson_struct = namedtuple('Params_run_json', list_fields)(*vals)
        #mjson_struct = namedtuple('Params_run_json', mjson.keys())(*mjson.values())
    return mjson_dict, mjson_struct



def get_confmat(YTRUE_, YPRED_, CLASSNAMES_):
    """ ---------------------------------------------------------
        @summary: compute confusion matrices (%, counts)
    ---------------------------------------------------------- """
    m_confusionMat_cnt = confusion_matrix(YTRUE_, YPRED_, labels=CLASSNAMES_)
    temporary_vector = m_confusionMat_cnt.sum(axis=1).astype("float"); temporary_vector[temporary_vector==0]=np.nan
    m_confusionMat = m_confusionMat_cnt.astype("float")/temporary_vector[:,np.newaxis]
    return m_confusionMat, m_confusionMat_cnt



def plot_confusion_matrix(cm, cm_cnt, m_classnames, idrun, m_title):
    """ ---------------------------------------------------------
        @summary: display of confusion matrices (%, counts)
    ---------------------------------------------------------- """
    m_tot = np.sum(cm_cnt, axis=1)
    n=7; m_width=14; m_height=5; m_size=14
    fig, ax = plt.subplots(num=None, figsize=(m_width, m_height))
    plt.subplots_adjust(wspace=0.5)
    G=gridspec.GridSpec(1,2*n+2)
    ## Confusion matrix expressed in counts
    ax1 = plt.subplot(G[0,:n-1])
    s=sns.heatmap(cm_cnt, xticklabels=m_classnames, yticklabels=m_classnames, 
                  cmap=plt.cm.binary, annot = np.around(cm_cnt,2), lw=0, alpha=.5,
                  fmt="d", linecolor=color_gray, linewidths=1.,
                  cbar=False, annot_kws={"size": m_size-2})
    ax1.set_xlabel("Predicted Label",fontweight="bold", fontsize=m_size-2)
    ax1.set_ylabel("True Label", fontweight="bold", fontsize=m_size-2)
    ax1.set_aspect("equal")
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(m_size-2) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(m_size-2)
    a=len(m_classnames)+1
    for j in range(a):
        plt.plot(range(a), j*np.ones(a), color="k", linewidth=1.)
        plt.plot(j*np.ones(a), range(a), color="k", linewidth=1.)
    ax1.set_xlim(-.01,a-1+.03); ax1.set_ylim(a-1+.05, -.01)
    m_tot = np.sum(cm_cnt, axis=1)
    for jj in range(len(m_tot)) :
        posx=1.015; posy=1-(jj+1)/len(m_classnames) +1/len(m_classnames)*1/2
        ax1.text(posx,posy, f"= {m_tot[jj]}",
                 horizontalalignment="center", verticalalignment="top",
                 visible=True,size=m_size, rotation=0.,
                 ha="left", va="center",
                 bbox=dict(boxstyle="round", ec=(1.0,1.0,1.0), fc=(1.0,1.0,1.0),),
                 transform=plt.gca().transAxes,fontweight="normal",style="italic",
                 color="gray", fontsize=m_size-2, backgroundcolor=None)
    ## Confusion matrix expressed in % (/true_labels initial count)
    ax2 = plt.subplot(G[0,n+2:2*n-1])
    ax3 = plt.subplot(G[0,2*n])
    sns.heatmap(cm, xticklabels=m_classnames, yticklabels=m_classnames, 
                cmap="Blues", annot= np.around(cm,2), lw=0.5, ax = ax2,
                 linecolor=color_gray, linewidths=1.,#.5
                 cbar_ax=ax3, 
                cbar=True, annot_kws={"size": m_size-2}) 
    ax2.set_xlabel("Predicted Label",fontweight="bold", fontsize=m_size-2)
    ax2.set_ylabel("True Label", fontweight="bold", fontsize=m_size-2)
    ax2.set_aspect("equal")
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(m_size-2) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(m_size-2)
    m_title_ = m_title if m_title is not None else "Confusion matrix"
    ax1.set_position([0.050,  0.125, 0.4419, 0.9])
    ax2.set_position([0.415, 0.125, 0.7955, 0.9])
    ax3.set_position([0.990, 0.125, 0.0150, 0.9])
    yticklabels = [l for l in ax3.get_yticks() if l is not ""]
    formattedList = ["%.2f" % member for member in yticklabels]
    ax3.set_yticklabels(formattedList)
    plt.text(-25,1.05, m_title_,
             horizontalalignment="center", verticalalignment="top",
             visible=True, size=14, rotation=0.,ha="center", va="center",
             bbox=dict(boxstyle="round", ec=(0.9,0.9,0.9), fc=(0.9,0.9,0.9),),
            transform=plt.gca().transAxes, fontweight="bold", fontsize=14)
    return fig, [ax1,ax2,ax3]
    
    

def categorical_to_label(ypred_cat, dict_types):
    """ ---------------------------------------------------------
        @summary: convert categorical labels
    ---------------------------------------------------------- """
    ypred=[]; ypred_int=[]
    for i in range(len(ypred_cat)):
        idx = np.argmax(ypred_cat[i])
        ypred_int.append(idx)
        ypred.append(dict_types[idx])
    return ypred, ypred_int

