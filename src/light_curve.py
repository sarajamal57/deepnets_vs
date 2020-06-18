## ---------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------- ##
#   CLASS - Light-curve
## ---------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------- ##
## ** Note/Credits **
##
##   Following class, functions & attributes initially forked from 
##       "Naul, Bloom, Perez & Van der Walt, 2017 [DOI: 10.1038/s41550-017-0321-z]" :
##          <Class>      LightCurve
##          <Functions>  fit_supersmoother, fit_lomb_scargle_gatspy, fit_lomb_scargle_cesium 
##   (All aforementioned functions are modified in the current version)
##
##   Added:
##          <Functions>   period_fold, evaluate_trend_linear, evaluate_trend_spline
##
## ---------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------- ##

import sys, os
import time, datetime, glob, joblib, copy
from io import StringIO

import numpy as np
import pandas as pd
import scipy, cesium, gatspy
        

## ############################################################################ ##
class LightCurve():
    
    def __init__(self, times, measurements, errors, 
                      m_mean=None,m_scale=None, passbands=None, cesium_ts=None, \
                      name=None, label=None, survey=None, period_catalog=None, \
                      p_signif=None, p_class=None, epoch_t0=None, ss_resid=None,\
                      period_gatspy=None, model_gatspy=None, score_gatspy=None,\
                      period_cesium=None, model_cesium=None, trend_cesium=None, \
                      trenderr_cesium=None, trend_spline=None, m_stats=None):
        
        ## Measurements ##
        self.times           = times
        self.errors          = errors
        self.measurements    = measurements
        self.passbands       = passbands
        
        self.mean            = m_mean
        self.scale           = m_scale
        self.mstats          = m_stats
        
        self.period_catalog  = period_catalog   
        self.name            = name
        self.label           = label
        self.survey          = survey
        
        self.p_signif        = p_signif   # deprecated
        self.p_class         = p_class    # deprecated
        
        ## Epoch, phase-folding
        self.epoch_t0        = {self.name:epoch_t0}
        
        ## Residuals from supersmoother fit
        self.ss_resid        = ss_resid
        
        ## Lomb-Scargle model [GATSPY py library]
        self.model_gatspy    = model_gatspy
        self.period_gatspy   = period_gatspy    
        self.score_gatspy    = score_gatspy    
        
        ## Lomb-Scargle model [CESIUM py library]
        self.model_cesium    = None
        self.period_cesium   = None
        self.trend_cesium    = None
        self.trenderr_cesium = None
        
        ## LSQunivariateSpline fit
        self.trend_spline    = None
        
        ## Linear fit
        self.trend_line      = None
        
        
    def __repr__(self):
        return "LightCurve(" + ', '.join("{}={}".format(k, v)
                                         for k, v in self.__dict__.items()) + ")"


    def __len__(self):
        return len(self.times)


   ## ############################################################################ ## 
    def fit_supersmoother(self, m_period = None, periodic=True, scale=True):
        ''' Residuals from SuperSmoother (Friedman 1984). [SOURCE: py supersmoother library]
        @param m_period: float, period.
        @param periodic: boolean, if the model contains a periodic component.
        @param scale: boolean, if scaling the residuals.
        '''
        from supersmoother import SuperSmoother
        if m_period is None:
            m_period = self.period_catalog
        
        model = SuperSmoother(period=m_period if periodic else None)
        try:
            model.fit(self.times, self.measurements, self.errors)
            self.ss_resid = np.sqrt(np.mean((model.predict(self.times) - self.measurements) ** 2))
            if scale:
                self.ss_resid /= np.std(self.measurements)
        except ValueError:
            self.ss_resid = np.inf

    ## ############################################################################ ##
    def fit_lomb_scargle_gatspy(self, pix_order=True, passbands=None):
        ''' Fit a Lomb-Scargle model. [SOURCE: py GATSPY library]
        @param pix_order: boolean, order pixels in time.
        '''
        from gatspy.periodic import LombScargleMultibandFast #LombScargleFast
        if pix_order:
            ord_idx = np.argsort(self.times)
            t   = self.times[ord_idx]
            err = self.errors[ord_idx]
            m   = self.measurements[ord_idx]            
        else:
            t   = self.times
            err = self.errors
            m   = self.measurements
        period_range = (0.005 * (max(t) - min(t)), 0.95 * (max(t) - min(t)))
        #self.model_gatspy = LombScargleFast(fit_period=True, silence_warnings=True, 
        #                                    optimizer_kwds={'period_range': period_range, 'quiet': True})
        self.model_gatspy = LombScargleMultibandFast(fit_period=True, 
                                                     optimizer_kwds={'period_range': period_range, 'quiet': True})
        self.model_gatspy.fit(t,m,err,passband)
        self.period_gatspy = self.model_gatspy.best_period
        self.score_gatspy   = self.model_gatspy.score(self.period_gatspy).item()

    ## ############################################################################ ##
    def fit_lomb_scargle_cesium(self, pix_order=True, sys_err=0.00, nharm=15, nfreq=1,tone_control=5.0):
        ''' Fit a Lomb-Scargle model. [SOURCE: py CESIUM library]
        @param pix_order: boolean, order pixels in time.
        @param sys_err: float, systematic error.
        @param nharm: float, number of harmonics in the model.
        @param nfreq: float, number of frequencies.
        @param tone_control: float, optimization control parameter.
        '''
        from cesium.features.lomb_scargle import lomb_scargle_model #, fit_lomb_scargle
        if pix_order:
            ord_idx = np.argsort(self.times)
            t = self.times[ord_idx]; m = self.measurements[ord_idx]; err = self.errors[ord_idx]
        else:
            t = self.times; m = self.measurements;  err = self.errors
            
        self.model_cesium    = lomb_scargle_model(t-min(t), m,  err, sys_err=sys_err, 
                                                  nharm=nharm, nfreq=nfreq, tone_control=tone_control)
        self.trend_cesium     = self.model_cesium  ['freq_fits'][0]['trend']
        self.trenderr_cesium  = self.model_cesium  ['freq_fits'][0]['trend_error']
        self.period_cesium    = 1/self.model_cesium['freq_fits'][0]['freq']
        
        
    ## ############################################################################ ## 
    def evaluate_trend_linear(self, do_weighted=False):
        ''' Trend estimation from linear fit
        @param do_weighted: boolean, if weights are used.
        '''
        from scipy.optimize import curve_fit
        def func(x, a, c):
            return a * x + c 
        if self.mean is None:
            self.mean = np.nanmean(self.measurements)
        popt, pcov = curve_fit(func, self.times, self.measurements, 
                               sigma= self.measurements if do_weighted else None,
                               p0=(0., self.mean))
        self.trend_line= func(self.times, *popt)
        
    ## ############################################################################ ##
    def evaluate_trend_spline(self, w0=None, pix_order=True):
        ''' Trend estimation from spline fit
        @param w: float[], weights.
        '''
        from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline
        if pix_order:
            ord_idx = np.argsort(self.times)
            t = self.times[ord_idx]; m = self.measurements[ord_idx]; err = self.errors[ord_idx]
        else:
            t = self.times; m = self.measurements; err = self.errors
            
        if w0 is not None:
            w0=1/(self.errors[ord_idx])**2
            
        knots = UnivariateSpline(t,m,w=w0,s=0).get_knots()
        m_spl = LSQUnivariateSpline(t,m,knots[-1:1],k=3,w=w0)
        self.trend_spline = m_spl(t)
        
        
    ## ############################################################################ ## 
    def period_fold(self, period=None, pix_rejection = True, 
                              epoch_select = 'max_brightness', epoch_t0=None, extend_2cycles = True):
        ''' Phase folding for light-curves.
        @param period: float, period.
        @param pix_rejection: boolean, rejection outside 0.05 and 0.95 quantiles.
        @param epoch_select: str, {'max_brightness','min_brightness','first_epoch'}.
        @param epoch_t0: float, epoch associated to the observed max brightness per default. 
        @param extend_2cycles: boolean, if extend to 2 cycles = [-1,1]. Per default, 1 cycle = [0,1]. 
        '''
        if period is None:
            period = self.period_catalog
            
        if epoch_t0 is None :
            if pix_rejection :
                keep = np.nanpercentile(self.measurements,(5,95))
                args_keep = ((self.measurements<keep[1])&(self.measurements>keep[0]))
            else:
                args_keep = range(len(self.measurements))
            #    
            if epoch_select in ['max_brightness', 'min_magnitude'] :
                arg_t0 = np.argmin(self.measurements[args_keep])     
            elif epoch_select in ['min_brightness', 'max_magnitude'] :
                arg_t0 = np.argmax(self.measurements[args_keep])     
            elif epoch_select in ['first_epoch'] :
                arg_t0 = 0
             #   
            epoch_t0 = self.times[args_keep][arg_t0]
            
        self.epoch_t0={self.name:epoch_t0}
        
        phase = np.modf((self.times - epoch_t0)/period)[0]   #  = ((self.times-t0)%self.p)/self.p
        phase[(phase<0)] += 1    ##(np.modf(phase)[0]+1) eq. to (np.modf(phase - np.floor(phase))[0])
        self.times = phase
        
        if extend_2cycles: ## = mirror
            phase_ext  = np.concatenate((phase-1, phase)) 
            meas_ext   = np.concatenate((self.measurements, self.measurements))
            errors_ext = np.concatenate((self.errors, self.errors))
            if self.passbands is not None :
                pb_ext = np.concatenate((self.passbands, self.passbands))
                
            # remove duplicates
            phase_ext, inds_  = np.unique(phase_ext, return_index=True) #return_inverse=True)
            self.times        = phase_ext 
            self.measurements = meas_ext[inds_]
            self.errors = errors_ext[inds_]
            if self.passbands is not None :
                self.passbands = pb_ext[inds_]
            #    
            if self.trend_cesium is not None :
                self.trend_cesium = np.concatenate((self.trend_cesium, self.trend_cesium))[inds]
            if self.trend_line is not None :
                self.trend_line   = np.concatenate((self.trend_line, self.trend_line))[inds]
            if self.trend_spline is not None :
                self.trend_spline = np.concatenate((self.trend_spline, self.trend_spline))[inds]
        
        inds               = np.argsort(self.times)
        self.times         = self.times[inds]
        self.errors        = self.errors[inds]
        self.measurements  = self.measurements[inds]
        if self.passbands is not None :
            self.passbands = self.passbands[inds]
        #
        if self.trend_cesium is not None :
            self.trend_cesium = self.trend_cesium[inds]
        if self.trend_line is not None :
            self.trend_line   = self.trend_line[inds]
        if self.trend_spline is not None :
            self.trend_spline = self.trend_spline[inds]
        
        

## ######################################################################## ## 
if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    
    ##
