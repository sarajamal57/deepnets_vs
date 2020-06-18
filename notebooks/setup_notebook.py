## ---------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------- ##
#   FUNCTIONS - Setup
## ---------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------- ##

import sys, os, errno
import tensorflow as tf
import warnings, logging
#0 = all messages are logged (default behavior)
#1 = INFO messages are not printed
#2 = INFO and WARNING messages are not printed
#3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger('tensorflow').setLevel(logging.INFO)


## ############################################################################ ##
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#get_ipython().magic('matplotlib inline')
#get_ipython().magic('config InlineBackend.figure_format = "retina"')

sns.set_context("notebook")
sns.set_style  ("white") #"whitegrid")

m_size = 12;
m_width = 12; m_height = 4

color_red   = '#8f1402'; color_blue   = '#042e60'; color_gray   = '#d8dcd6'; 
color_green = '#06470c'; color_forest = '#154406'; color_orange = '#fdaa48' 
color_mauve = '#4b006e'; color_purple = '#bc13fe' 
color_liste_display1 = ['#b66a50', '#c85a53', '#fdaa48', '#05696b', '#cb7723', 
                        '#a4be5c', '#8cffdb', '#ce5dae', '#b1916e', '#028f1e',
                        '#8e82fe', '#735c12', '#b5485d', '#9e3623', '#db4bda']
color_liste_display2 = ['#b66a50', '#c85a53', '#fdaa48', '#ff028d', '#8e82fe', 
                        '#4b006e', '#0165fc', '#ce5dae', '#e17701', '#8eab12', 
                        '#b1916e', '#7f5e00', '#137e6d', '#9e3623', '#db4bda']
color_list_display0  = ["#030aa7","#fec615","#cb00f5","#15b01a", "#ffb07c",  
                        "#ff000d","#02d8e9","#3d1c02", "#7a687f"]

from matplotlib import rcParams
rcParams["savefig.dpi"]        = 100
rcParams["figure.dpi"]         = 100
rcParams["font.size"]          = 16
rcParams["text.usetex"]        = False
rcParams["font.family"]        = ["arial"]
rcParams["font.sans-serif"]    = ["cmss10"]
rcParams["axes.unicode_minus"] = False

m_format_f4="%4.4f" 
m_format_f8="%4.8f"
m_format_e2="%4.2e"

cmap  = plt.cm.Paired
#font = {'family':'arial','variant':'small-caps','stretch':'normal','weight':'normal','size': m_size}##'sans-serif':'cmss10'
#matplotlib.rc('font', **font)
#matplotlib.rc('legend', edgecolor=(0.1,0.1,0.1))


## ############################################################################ ##
import numpy as np
import pandas as pd
import copy, joblib, glob, time, datetime
from time import sleep
    
module_path = os.path.dirname(os.getcwd())+'/../src'
if module_path not in sys.path:
    sys.path.append(module_path)
from light_curve import LightCurve
import functions_keras as m_func

module_path = os.path.dirname(os.getcwd())+'/../scripts'
if module_path not in sys.path:
    sys.path.append(module_path)

    
from light_curve import LightCurve
import functions_preprocess as m_preprocess #cf
import functions_keras as m_func #ku


dict_filters = {'red':0, 'blue':1}




