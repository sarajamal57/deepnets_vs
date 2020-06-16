## ---------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------- ##
#   FUNCTIONS - Preprocessing
## ---------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------- ##
## ** Note/Credits **
##
##   Following functions & attributes initially forked from 
##       "Naul, Bloom, Perez & Van der Walt, 2017 [DOI: 10.1038/s41550-017-0321-z]" :
##          <Functions>  preprocess_matrix, times_to_lags, lags_to_times
##   (All aforementioned functions are modified in the current version)
##
##   Added:
##          <Functions>   period_fold, pred_GP_lc, get_FRandom_vector, 
##                        get_random_timerange, get_random_timerange
##
## ---------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------- ##
import sys, os
import numpy as np
import joblib, time, copy, random

SEED = 1 #0
random.seed(SEED)
np.random.seed(SEED)

from statsmodels.stats.weightstats import DescrStatsW
from keras.preprocessing.sequence import pad_sequences
import light_curve


def print_versions():
    """ ---------------------------------------------------------
        @summary: List of current Py packages.
    ---------------------------------------------------------- """
    lib_py = ['argparse','astropy', 'cesium','collections','copy','h5py','joblib','json','keras','keras_preprocessing', 'keras_applications', 'matplotlib','natsort',
                  'numpy', 'os', 'pandas', 'pickle','random','schwimmbad','scipy','seaborn', 'simplejson','sklearn', 'statsmodels','sys','tensorflow','time','tqdm']
    for name, module in sorted(sys.modules.items()):
        if (name in lib_py) & (hasattr(module, '__version__')):
                print ('\t *', name, module.__version__)
    
    

def normalize_lc(vect_mags, vect_errs, norm_method = "standardization", do_reject = True, alpha = 0.05):
    """ ---------------------------------------------------------
        @summary: Normalize data
    ---------------------------------------------------------- """
    mags = copy.deepcopy(vect_mags)
    errs = copy.deepcopy(vect_errs)

    if do_reject:
        keep = np.nanpercentile(mags,(alpha*100,(1-alpha)*100))
        selidx = ((mags<keep[1])&(mags>keep[0]))
    else:
        selidx = range(len(mags))

    m_mean = np.nanmean(mags[selidx])
    m_std  = np.nanstd(mags[selidx])
    m_min, m_max = (np.nanmin(mags[selidx]), np.nanmax(mags[selidx]))
    m_amplitude = m_max-m_min
    m_stats = {'mean':m_mean, 'std':m_std, 'amplitude':m_amplitude}
    
    norm_mags=None; norm_errs = None
    ## Min-Max scaling
    if norm_method == "minmax":
        norm_mags = (mags-m_min)/m_amplitude
        norm_errs = errs/m_amplitude
        if True:
            mcond0 = (norm_mags<0); mcond1 = (norm_mags>1)
            norm_errs[mcond0] = np.Inf  # weighting by 1/sigma (or 1/sigma**2) zeroes these pix contributions
            norm_mags[mcond0] = 0.; norm_mags[mcond1] = 1.
    #
    ## Standardization
    else:
        norm_mags = (mags-m_mean)/m_std
        norm_errs = errs/m_std

    return norm_mags, norm_errs, m_stats
     
                                    
                                 
def preprocess_matrix(X_raw, m_max = np.inf, m_stats = None, do_diffT = False,
                norm_method = "standardization", do_reject = True, alpha = 0.05): #alpha = 0.001)
    """ ---------------------------------------------------------
        @summary: Normalize mag measurements, pix rejection
    ---------------------------------------------------------- """
    X = copy.deepcopy(X_raw)
    wrong_units =  np.all(np.isnan(X[:, :, 1])) | (np.nanmax(X[:, :, 1], axis = 1) > m_max)
    X = X[~wrong_units, :, :]
    
    if do_diffT: # times into dt
        X[:, :, 0] = times_to_lags(X[:, :, 0])
    
    nobjects = X.shape[0]
    m_stats = np.ndarray((nobjects,3), dtype = np.float64)
    for j in range(nobjects):
        X[j,:,1], X[j,:,2], m_stats[j,:] = normalize_lc(X[j,:,1], X[j,:,2], norm_method, do_reject, alpha)
    
    ## Statistics
    # means  = np.atleast_2d(np.nanmean(X[:, :, 1], axis = 1)).T
    # scales = np.atleast_2d(np.nanstd(X[:, :, 1]-means, axis = 1)).T
    # datastats = [DescrStatsW(X[i, :, 1], weights = ((1./X_raw[:, :,2])**2)[i,:]) for i in range(len(X))]
    # w_mean = [datastats[i].mean for i in range(len(datastats))]     # weighted mean
    # w_stdm = [datastats[i].std_mean for i in range(len(datastats))] # standard-deviation of wMEAN
    # X[:,:,1] = (X[:,:,1]-means)/scales
    # X[:,:,2] = X[:,:,2]/scales
    
    return X, m_stats, wrong_units


def times_to_lags(T):
    """ ---------------------------------------------------------
        (N x n_step) matrix of times -> (N x n_step) matrix of lags.
        per line/object:
           times vector : T = [t_0,...,t_nstep]
           lags vector  : dT = [0, (t_1 - t_0), ..., t_(nstep-1)-t_nstep]
    ---------------------------------------------------------- """
    assert T.ndim==2, "T must be an (N x n_step) matrix"
    return np.c_[np.zeros(T.shape[0]), np.diff(T, axis = 1)]
    #return np.c_[np.diff(T, axis = 1), np.zeros(T.shape[0])]
    

def lags_to_times(dT,t_0 = None):
    """ ---------------------------------------------------------
        (N x n_step) matrix of lags -> (N x n_step) matrix of times
        per line/object:
           lags vector   :   dT = [0, (t_1 - t_0), ..., t_(nstep-1)-t_nstep]
           times vector :   T = [t_0,...,t_nstep]
    ---------------------------------------------------------- """
    assert dT.ndim==2, "dT must be an (N x n_step) matrix"
    if t_0.ndim==1:
        t_0 = np.expand_dims(t_0, axis = 1)
    return np.c_[np.cumsum(dT, axis = 1)]+t_0
    #return np.c_[np.zeros(dT.shape[0]), np.cumsum(dT[:,:-1], axis = 1)]
    


## ######################################################################## ## 
def period_fold(lc, period, pix_rejection = True,
                epoch_select = 'max_brightness', epoch_t0 = None,
                extend_2cycles = True, rm_dupli = True):
    ''' Phase folding for light-curves.
        @param period: float, period.
        @param pix_rejection: boolean, rejection outside 0.05 and 0.95 quantiles.
        @param epoch_select: str, {'max_brightness','min_brightness','first_epoch'}.
        @param epoch_t0: float, epoch associated to the observed max brightness per default. 
        @param extend_2cycles: boolean, if extend to 2 cycles = [-1,1]. Per default, 1 cycle = [0,1]. 
        ''' 
    if epoch_t0 is None :
        if pix_rejection :
            keep = np.nanpercentile(lc.measurements,(5,95))
            args_keep = ((lc.measurements<keep[1])&(lc.measurements>keep[0]))
        else:
            args_keep = range(len(lc.measurements))
        #    
        if epoch_select in ['max_brightness', 'min_magnitude'] :
            arg_t0 = np.argmin(lc.measurements[args_keep])
        elif epoch_select in ['min_brightness', 'max_magnitude'] :
            arg_t0 = np.argmax(lc.measurements[args_keep])
        elif epoch_select in ['first_epoch'] :
            arg_t0 = 0
         #   
        epoch_t0 = lc.times[args_keep][arg_t0]
        
    lc.epoch_t0 = {lc.name:epoch_t0}
    
    phase = np.modf((lc.times - epoch_t0)/period)[0]   # = ((lc.times-t0)%lc.p)/self.p
    phase[(phase<0)]+=1    ##(np.modf(phase)[0]+1) is equivalent to (np.modf(phase - np.floor(phase))[0])
    lc.times = phase
    
    if extend_2cycles:
        # mirror
        phase_ext = np.concatenate((phase-1, phase))
        meas_ext = np.concatenate((lc.measurements, lc.measurements))
        errors_ext = np.concatenate((lc.errors, lc.errors))
        if lc.passbands is not None:
            pb_ext = np.concatenate((lc.passbands, lc.passbands))
        
        if rm_dupli :# remove duplicates
            phase_ext, inds_ = np.unique(phase_ext, return_index = True) #return_inverse = True)
            lc.times = phase_ext
            lc.measurements = meas_ext[inds_]
            lc.errors = errors_ext[inds_]
            if lc.passbands is not None:
                lc.passbands = pb_ext[inds_]
            if lc.trend_cesium is not None :
                lc.trend_cesium = np.concatenate((lc.trend_cesium, lc.trend_cesium))[inds]
            if lc.trend_line is not None :
                lc.trend_line = np.concatenate((lc.trend_line, lc.trend_line))[inds]
            if lc.trend_spline is not None :
                lc.trend_spline = np.concatenate((lc.trend_spline, lc.trend_spline))[inds]
        else:
            lc.times = phase_ext
            lc.measurements = meas_ext
            lc.errors = errors_ext
            if lc.passbands is not None:
                lc.passbands = pb_ext
            if lc.trend_cesium is not None :
                lc.trend_cesium = np.concatenate((lc.trend_cesium, lc.trend_cesium))
            if lc.trend_line is not None :
                lc.trend_line = np.concatenate((lc.trend_line, lc.trend_line))
            if lc.trend_spline is not None :
                lc.trend_spline = np.concatenate((lc.trend_spline, lc.trend_spline))
    
    inds = np.argsort(lc.times)
    lc.times = lc.times[inds]
    lc.errors = lc.errors[inds]
    lc.measurements = lc.measurements[inds]
    if lc.passbands is not None:
        lc.passbands = lc.passbands[inds]
    if lc.trend_cesium is not None :
         lc.trend_cesium = lc.trend_cesium[inds]
    if lc.trend_line is not None :
        lc.trend_line = lc.trend_line[inds]
    if lc.trend_spline is not None :
        lc.trend_spline = lc.trend_spline[inds]

    return lc



## ######################################################################## ##
def pred_GP_lc (xtimes, ymags, yerrs, mperiod, 
                x_up, pix_rejection=True, alpha=None,
                m_variation=1e-20, kernel_type="mix_SHOs",
                do_phased=False, do_normalized=True,
                verbose=True):
    """ ---------------------------------------------------------
        @summary: GP model computed on (x,y,yerr) measurements
                    using the py EXOPLANET package and its dependencies
        @param x: float[], time/phase measurements.
        @param y: float[], mag/flux measurements.
        @param yerr        : float[], uncertainties.
        @param x_up        : float[], up/downsampled time/phase vector.
        @param mperiod     : float, period of light-curve (set to 1 if folded LC).
        @param m_variation : float, minimal std deviation.
        @return dict_gp    : dictionary, results of GP model & predictions
    --------------------------------------------------------- """
    import pymc3 as pm
    import exoplanet as xo
    #import theano; theano.config.mode = 'FAST_RUN'
    import theano.tensor as tt

    list_kernels = ['matern', 'SHO', 'rotation', 'mix_SHOs']
    if kernel_type not in list_kernels:
        kernel_type = "mix_SHOs"
        
    s2_err = False
    
    x    = copy.deepcopy(xtimes)
    y    = copy.deepcopy(ymags)
    yerr = copy.deepcopy(yerrs)
    if pix_rejection:
        alpha=5 if alpha is None else alpha
        keep=np.nanpercentile(y,(alpha,(100-alpha)))
        args_keep=((y<keep[1])&(y>keep[0]))
    else:
        args_keep = range(len(y))
    x = x[args_keep]; y=y[args_keep]; yerr=yerr[args_keep];

    #nbpoints_up = 1000 if nbpoints_up is None else nbpoints_up
    #if x_up is None:
    #    if nbpoints_up>len(x) :
    #        x_up  = np.sort(np.r_[np.random.randint(low=x[0], high=x[-1], size=nbpoints_up-len(x)), x])
    #    else :
    #        x_up  = x[np.sort(np.random.randint(low=0, high=len(x), size=nbpoints_up))]
    #        #x_up = np.random.randint(low=x[0], high=x[-1], size=nbpoints_up)


    with pm.Model() as model:

        mean = pm.Normal("mean", mu=0 if do_normalized else np.mean(y), sd=10.0)

        if kernel_type=='matern': # Matérn kernel
            logs2   = pm.Normal("logs2", mu=2*np.log(np.min(yerr)), sd=5.0)
            s2_err = True
            
            log_sigma = pm.Uniform("log_sigma", lower=m_variation, upper=1.0);

            estimated_rho = np.sqrt(3)/(4*np.pi/mperiod)
            log_rho = pm.Normal("log_rho", mu=np.log(estimated_rho), sd=100.0) ##mu=1.0, sd=100.0)

            kernel = xo.gp.terms.Matern32Term(log_sigma=log_sigma, log_rho=log_rho)
            #gp = xo.gp.GP(kernel, x, yerr**2, J=2)
            gp = xo.gp.GP(kernel, x, yerr**2+tt.exp(logs2), J=2)


        elif kernel_type=='SHO': # stochastically-driven, damped harmonic oscillator
            logs2 = pm.Normal("logs2", mu=2*np.log(np.min(yerr)), sd=5.0)
            s2_err = True
            
            
            logamp = np.log(np.var(y))

            varp = (.1 if mperiod>1 else 10.) if do_phased==False else .1
            #estimated_period = np.log(1/mperiod) if (mperiod<1) else np.log(mperiod)
            estimated_period = np.log(mperiod)
            logperiod = pm.Normal("logperiod", mu=estimated_period, sd=varp)
            period=pm.Deterministic("period", np.exp(-logperiod) if (mperiod<1) else np.exp(logperiod))

            logQ0  = pm.Normal("logQ0", mu=1, sd=10.0)
            omega0 = 4*np.pi/(mperiod+tt.sqrt(4*logQ0**2-1))
            logw0 = np.log(omega0)
            logS0 = logamp - logw0 - logQ0

            kernel = xo.gp.terms.SHOTerm(log_S0=logS0, log_w0=logw0, log_Q=logQ0)

            gp = xo.gp.GP(kernel, x, yerr**2+np.exp(logs2))#, J=4)


        elif kernel_type=='rotation' : # quasi-periodic kernel

            logs2 = pm.Normal("logs2", mu=2*np.log(np.min(yerr)), sd=5.0)
            s2_err=True
            
            ##estimated_ampl = np.log(max(y)-min(y))
            estimated_ampl = np.log(np.var(y))
            logamp = pm.Normal("logamp", mu=estimated_ampl, sd=m_variation)

            estimated_period = np.log(1/mperiod) if (mperiod<1) else np.log(mperiod)
            #estimated_period = np.log(mperiod)
            #varp = .1 if do_phased else (.1 if mperiod>1 else 10.)
            varp = 10 #.1 #10
            logperiod = pm.Normal("logperiod", mu=estimated_period, sd=varp)
            period = pm.Deterministic("period", tt.exp(-logperiod) if (mperiod<1) else tt.exp(logperiod))

            ##logQ0 = pm.Normal("logQ0", mu=1.0, sd=10.0)
            ##logdeltaQ = pm.Normal("logdeltaQ", mu=2.0, sd=10.0)
            ###Q0=100.
            Q0=100.
            logQ0 = pm.Normal("logQ0", mu=Q0, sd=10.0) #mu=1.0, sd=10.0)
            logdeltaQ = pm.Normal("logdeltaQ", mu=Q0//2, sd=Q0//4)
            mix = pm.Uniform("mix", lower=0, upper=1)
            
            kernel = xo.gp.terms.RotationTerm(log_amp=logamp,
                                              period=period,
                                              Q0=logQ0,
                                              log_deltaQ=logdeltaQ,
                                              mix=mix)
            gp = xo.gp.GP(kernel, x, yerr**2+np.exp(logs2), J=4)
            
            
        elif kernel_type=='mix_SHOs' : # mix of 2-3 SHOs, following the "rotation" kernel in celerite
            logs2 = pm.Normal("logs2", mu=2*np.log(np.min(yerr)), sd=5.0)
            s2_err=True
            
            #if False:
            #    Q0=100
            #    logQ0 = pm.Normal("logQ0", mu=Q0, sd=10.0) #mu=1.0, sd=10.0)
            #    logdeltaQ = pm.Normal("logdeltaQ", mu=Q0//2, sd=Q0//4)
            #
            #    logQ1 = tt.log(.5+Q0+tt.exp(logdeltaQ))
            #    logQ2 = tt.log(.5+Q0)
            #
            #    omega1 = 4*np.pi/(mperiod+tt.sqrt(4*logQ1**2-1))
            #    omega2 = 8*np.pi/(mperiod+tt.sqrt(4*logQ2**2-1))
            #
            #    logw1 = np.log(omega1)
            #    logw2 = np.log(omega2)
            #
            #    estimated_ampl = np.log(np.var(y))
            #    logamp = pm.Normal("logamp", mu=estimated_ampl, sd=m_variation)
            #    logS1 = logamp - logw1 - logQ1
            #    mix = pm.Uniform("mix", lower=0, upper=1)
            #    logS2 = tt.log(mix) + logamp - logw2 - logQ2
            #
            #    # Set up the kernel an GP
            #    kernel  = xo.gp.terms.SHOTerm(log_S0=logS1, log_w0=logw1, log_Q=logQ1)
            #    kernel += xo.gp.terms.SHOTerm(log_S0=logS2, log_w0=logw2, log_Q=logQ2)
            
            
            Q0=100
            logdeltaQ = pm.Normal("logdeltaQ", mu=1e-20, sd=1e-20)
            estimated_ampl = np.log(np.var(y))
            logamp = pm.Normal("logamp", mu=estimated_ampl, sd=m_variation)
                
            amp = tt.exp(logamp) #estimated_ampl
            deltaQ = tt.exp(logdeltaQ)
            mix2 = pm.Uniform("mix2", lower=0, upper=1)
            mix3 = 1 #pm.Uniform("mix3", lower=0, upper=1)
            
            # period
            Q1 = 0.5 + Q0 + deltaQ
            w1 = 4 * np.pi * Q1 / (mperiod * tt.sqrt(4 * Q1 ** 2 - 1))
            S1 = amp / (w1 * Q1)
            
            # half the period
            Q2 = 0.5 + Q0 #+ deltaQ//100
            w2 = 4 * np.pi * Q2 / ((mperiod/2) * tt.sqrt(4 * Q2 ** 2 - 1))
            S2 = mix2 * amp / (w2 * Q2)
            #
            Q3 = 0.5 + Q0//2
            w3 = 4 * np.pi * Q3 / ((mperiod/2) * tt.sqrt(4 * Q3 ** 2 - 1))
            S3 = mix3 * amp / (w3 * Q3)
            
            
            term1 = xo.gp.terms.SHOTerm(S0=S1, w0=w1, Q=Q1)
            term2 = xo.gp.terms.SHOTerm(S0=S2, w0=w2, Q=Q2)
            term3 = xo.gp.terms.SHOTerm(S0=S3, w0=w3, Q=Q3)
            
            kernel=xo.gp.terms.TermSum(term1, term2)#, term3)#
            
            gp = xo.gp.GP(kernel, x, yerr**2+np.exp(logs2))


        ### GP Likelihood
        pm.Potential("loglike", gp.log_likelihood(y-mean))

        ### Mean model prediction (check)
        pm.Deterministic("pred", gp.predict())

        ### MAP
        map_solution, minfo = xo.optimize(start=model.test_point, return_info=True, verbose=verbose) #, vars=[trend])

        ### Evaluate the MAP solution on the sampled x-axis
        mu_up, sigma2_up=xo.eval_in_model(gp.predict(x_up, return_var=True), map_solution)
        sd_up=np.sqrt(sigma2_up)
        
        #txt, bib = xo.citations.get_citations_for_model()
        #print(txt,bib)

    ## Outputs ##
    dict_gpm={'context':model, 'gp':gp, 'kernel':kernel}   
    dict_map_gp={'map_solution':map_solution,'minfo':minfo}

    vstd = np.exp(map_solution['logs2']) if s2_err else 0
    dict_pred_gp={'x_up':x_up,
                  'norm_mu_pred':mu_up,
                  'norm_sd_pred':sd_up,
                  'mu_pred': mu_up+map_solution['mean'],
                  'sd_pred':np.sqrt(np.abs(sd_up**2+vstd))
                 }
    
    return dict_gpm, dict_map_gp, dict_pred_gp


## ######################################################################## ##
def get_FRandom_vector(xmin, xmax, nbsample):
    randomF_vect=[]
    for i in range(nbsample):
        rand_0_1 = (i/(nbsample-1))*random.random()
        rand_rescaled  = xmin*1.0 +(xmax-xmin)*1.0*rand_0_1
        #rand_rescaled_ = xmin*(1.0-rand_0_1)+ xmax*rand_0_1
        randomF_vect.append(rand_rescaled)
    return randomF_vect


## ######################################################################## ##
def get_random_timerange(x,dx, nbpoints_up=200, period_crit=None, factor=10, fixed_shift=True) :
    
    ## ------------------------- DETECTABLE GAPS ------------------------- ##
    from sklearn.cluster import KMeans
    import random
    random.seed(1)

    nclusters = 2 #3
    km = KMeans(n_clusters=nclusters, random_state=0).fit(dx.reshape(-1,1))

    clusts={}
    clusts[0] = np.where(km.labels_==0)[0]
    clusts[1] = np.where(km.labels_==1)[0]
    clusts[2] = np.where(km.labels_==2)[0]
    #print(Counter(km.labels_))
    #print(km.cluster_centers_)

    id_obs = 0; id_gaps=[]
    if len(clusts[0])<len(clusts[1]):
        id_obs=2 if len(clusts[1])<len(clusts[2])else 1
    else:
        if len(clusts[0])<len(clusts[2]):
            id_obs=2

    id_gaps=[ii for ii in range(nclusters) if ii is not id_obs]

    detected_gaps = [clusts[ii] for ii in range(nclusters) if ii in id_gaps]
    detected_obs  = clusts[id_obs]
    
    gaps_end_idx    = np.asarray([list(clusts[id_gap]) for id_gap in id_gaps][0])  #detected the end of the gap #d#x=[0,diff(times)]
    gaps_start_idx  = np.asarray([list(clusts[id_gap]-1) for id_gap in id_gaps][0])
    gaps_end_time   = np.asarray([x[g] for g in gaps_end_idx])
    gaps_start_time = np.asarray([x[g] for g in gaps_start_idx])
    gaps_diff_time  = gaps_end_time - gaps_start_time
    ngaps = len(gaps_end_idx)
    #print('ngaps = ', ngaps)

    ## GAPS less than n-cycles are merged within obs data ##
    if period_crit is not None:
        gaps_reject = gaps_diff_time<period_crit

        gaps_start_time = gaps_start_time[~gaps_reject]
        gaps_end_time   = gaps_end_time  [~gaps_reject]
        gaps_start_idx  = gaps_start_idx[~gaps_reject]
        gaps_end_idx    = gaps_end_idx  [~gaps_reject]
        gaps_diff_time  = gaps_diff_time [~gaps_reject]
        ngaps = len(gaps_end_idx)
    
    
    ## ------------------------- FRAC SUBSETS ------------------------- ##
    obs_subsets_idx={}; obs_subsets_times={}
    for i in range(ngaps +1):
        xstart = 0 if i==0 else gaps_end_idx[i-1]
        xend = len(x)-1 if i==(ngaps) else gaps_end_idx[i]-1
        #print(f'xstart[{xstart}]\t xend[{xend}]\t xend-xstart[{xend-xstart}]')
        obs_subsets_idx[f'set{i}']= (xstart, xend, xend-xstart)
        obs_subsets_times[f'set{i}'] = [x[xstart]] if xstart==xend else x[xstart:xend]
        
    
    ## ------------------------- FRAC POINTS PER SUBSET ------------------------- ##
    len_subsets = [len(obs_subsets_times[f'set{xx}']) for xx in range(ngaps+1)]
    len_subsets_total = np.sum(len_subsets) #len(x)

    nbpoints_up_RED = nbpoints_up - len_subsets_total
    nbpoints_up_set = []; nbpoints_up_set_RED = []
    for i in range(ngaps+1):
        ll = obs_subsets_idx[f'set{i}'][-1]
        frac_points = ll/len_subsets_total*100
        #val_ = np.int(np.ceil(nbpoints_up*frac_points/100))
        #val_red = np.int(np.ceil(nbpoints_up_RED*frac_points/100)) if (nbpoints_up>len_subsets_total) else -1
        v_=nbpoints_up*frac_points/100
        v_RED = nbpoints_up_RED*frac_points/100
        val_ = np.int(np.floor(v_)) if v_>1 else 1
        val_ = 1 if ll==1 else val_  #si la population initiale est 1  ###added
        if (nbpoints_up>len_subsets_total) :
            val_red = np.int(np.floor(v_RED)) if v_RED>1 else int(v_RED) #1
            val_red = 0 if ll==1 else val_red  #si la population initiale est 1 ##added
            ###val_red = np.int(np.floor(v_RED)) if v_RED>1 else 1 ##removed
        else :
            val_red = -1
        #print(i, '\t', ll, '\t', m_format%(frac_points), '%', '\t',  val_,  '\t',  val_red)
        nbpoints_up_set.append(val_)
        nbpoints_up_set_RED.append(val_red)

    #print('\t >> np.sum(nbpoints_up_set) =', np.sum(nbpoints_up_set))
    if np.sum(nbpoints_up_set)<nbpoints_up:
        missing_points = nbpoints_up-np.sum(nbpoints_up_set)
        amx=np.argmax(nbpoints_up_set)
        nbpoints_up_set[amx]+=missing_points

    if np.sum(nbpoints_up_set)>nbpoints_up:
        remn_points = np.sum(nbpoints_up_set) - nbpoints_up
        amx=np.argmax(nbpoints_up_set)
        if nbpoints_up_set[amx]>remn_points:
            #print(amx)
            nbpoints_up_set[amx]-=remn_points

    nbcrit = 1
    
    amx_RED=np.argmax(nbpoints_up_set_RED)
    if (nbpoints_up>len_subsets_total):
        #print(f'N_observed[{len_subsets_total}] < N_fixed[{nbpoints_up}]')
        if np.sum(nbpoints_up_set_RED)<(nbpoints_up-len_subsets_total):
            missing_nb = (nbpoints_up-len_subsets_total)-np.sum(nbpoints_up_set_RED)
            nbpoints_up_set_RED[amx_RED]+=missing_nb

            
    #id_nb1 = []; id_nb1_RED = []
    #for s in range(len(nbpoints_up_set)):
    #    if nbpoints_up_set[s]<=nbcrit: #==1:
    #        id_nb1.append(s)
    #    if nbpoints_up_set_RED[s]<=nbcrit: #==1:
    #        id_nb1_RED.append(s)
    #
    #amx=np.argmax(nbpoints_up_set)
    #dec = int(min(10,nbpoints_up_set[amx]/10*len(id_nb1)))
    #for l in id_nb1:
    #    nbpoints_up_set[l]+=dec
    #nbpoints_up_set[amx]-=dec
    #
    #amx_RED=np.argmax(nbpoints_up_set_RED)
    #dec = int(min(10,nbpoints_up_set_RED[amx_RED]/10*len(id_nb1_RED)))
    #for l in id_nb1_RED:
    #    nbpoints_up_set_RED[l]+=dec
    #nbpoints_up_set_RED[amx_RED]-=dec

    ## ------------------------------------------------------------------------------------- ##
    dec_min_all = min(np.diff(x))/2;
    x_up1=[]; x_up2=[];


    if len_subsets_total>=nbpoints_up: #nbpoints_up<len_subsets_total
        #print(f'\n>> OPTION 1 : N_observed[{len_subsets_total}] > N_fixed[{nbpoints_up}]')
        for i in range(ngaps+1):
            x_subset = obs_subsets_times[f'set{i}']

            if len(x_subset)>nbcrit:  #crit >2datapoints..
                xmin     = x_subset[0]; xmax = x_subset[-1]

                ## TOTAL RANDOM SAMPLE WITHIN TARGET TIME_RANGE
                nbsample = (nbpoints_up_set[i] if nbpoints_up_set[i]>1 else (nbpoints_up_set[i]+1)) #nm = nbpoints_up_set[i]
                xvec1    = np.sort(random.sample(get_FRandom_vector(xmin, xmax, nbsample),
                                                     k=nbpoints_up_set[i]))

                ## SHIFTED (SAMPLED) INITIAL TIMEFRAMED
                idx_    = np.sort(random.sample(range(len(x_subset)),
                                                k=nbpoints_up_set[i])) #sample without replacement
                dec_min = min(np.diff(x_subset))/factor if fixed_shift else min(np.diff(x_subset))/random.randint(5,100)  #random.sample(np.diff(x_subset)/factor, k=1)
                xvec2   = x_subset[idx_]+dec_min
            else:
                xvec1 = copy.deepcopy(x_subset); xvec2 = copy.deepcopy(x_subset)
            x_up1.extend(xvec1);x_up2.extend(xvec2)

    else:
        #print(f'\n>> OPTION 2 : N_observed[{len_subsets_total}] < N_fixed[{nbpoints_up}]')
        for i in range(ngaps+1):
            x_subset = obs_subsets_times[f'set{i}']

            if len(x_subset)>nbcrit:  #crit >2datapoints..
                xmin     = x_subset[0]; xmax = x_subset[-1]
                ## TOTAL RANDOM SAMPLE WITHIN TARGET TIME_RANGE
                nbsample = (nbpoints_up_set[i] if nbpoints_up_set[i]>1 else (nbpoints_up_set[i]+1)) #nm = nbpoints_up_set[i]
                xvec1    = np.sort(random.sample(get_FRandom_vector(xmin, xmax, nbsample),
                                                 k=nbpoints_up_set[i]))

                ## REDUCED TOTAL_RANDOM + SHIFTED (SAMPLED) INITIAL TIMEFRAMED
                nbsample_ = (nbpoints_up_set_RED[i] if nbpoints_up_set_RED[i]>1 else (nbpoints_up_set_RED[i]+1)) #nm = nbpoints_up_set_RED[i]
                xvec2_r   = np.sort(random.sample(get_FRandom_vector(xmin, xmax, nbsample_),
                                                  k=nbpoints_up_set_RED[i]))
                idx_ = np.sort(random.sample(range(len(x_subset)),
                                             k=nbpoints_up_set[i]-nbpoints_up_set_RED[i]))  #sample without replacement
                dec_min  = min(np.diff(x_subset))/factor if fixed_shift else min(np.diff(x_subset))/random.randint(5,100) #random.sample(np.diff(x_subset)/factor, k=1)
                xvec2_w  = x_subset[idx_]+dec_min
                xvec2    = np.sort(np.r_[xvec2_r, xvec2_w])

            else:
                xvec1 = copy.deepcopy(x_subset); xvec2 = copy.deepcopy(x_subset)
            x_up1.extend(xvec1);x_up2.extend(xvec2)
 

    ## ------------------------------------------------------------------------------------- ##
    dict_km_clusters = {'clusts': clusts, 'id_gaps':id_gaps, 'id_obs':id_obs}
    
    dict_identified_subsets = {'ngaps':ngaps,
                               'detected_obs':detected_obs,
                               'obs_subsets_idx':obs_subsets_idx,
                               'obs_subsets_times': obs_subsets_times}
    
    dict_identified_gaps = {'ngaps':ngaps,
                            'detected_gaps':   detected_gaps,
                            'gaps_start_time': gaps_start_time,
                            'gaps_end_time':   gaps_end_time,
                            'gaps_diff_time':  gaps_diff_time,
                            'gaps_start_idx':  gaps_start_idx,
                            'gaps_end_idx':    gaps_end_idx,
                            }

    dict_gen_times = {'rand_gen':x_up1, 'rand_selshift':x_up2, 'dec_min':dec_min}
                      
    return dict_gen_times, dict_identified_gaps, dict_identified_subsets, dict_km_clusters

