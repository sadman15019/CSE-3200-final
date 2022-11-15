
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
import sys
sys.path.append('../')
import os
import argparse
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from numpy import array
from tqdm.auto import tqdm
import cv2
from termcolor import colored
import io
import matplotlib.pylab as plt
import scipy
import scipy.signal
from PIL import Image
from matplotlib.pyplot import plot, scatter, show
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
plt.style.use('classic')
from scipy.signal import argrelmax, argrelmin
import sys
from numpy import NaN, Inf, arange, isscalar, asarray
from scipy.signal import butter, filtfilt, sosfilt
from random import randint


class config:
    #---------------------------------
    # fixed params: Band-pass Fileter
    #---------------------------------
    FPS             =  30  # camera captured video using 30 FPS
    '''
    A normal resting heart rate for adults ranges from 60 to 100 beats per minute. 
    Generally, a lower heart rate at rest implies more efficient heart function 
    and better cardiovascular fitness. 
    For example, a well-trained athlete might have a 
    normal resting heart rate closer to 40 beats per minute.
    '''
    BPM_L           =  70 #randint(60, 101) #70  # min Blood Pulse per min (60-100)
    '''
    The American Heart Association (AHA) states that the maximum heart rate 
    during exercise should be roughly equal to 220 bpm minus the age of the person.
    '''
    BPM_H           =  195 #randint(170, 201) #200 #(220-31) # Max Blood Pulse per min (220) 
    ORDER           =  [2,3,4,6,9] # 0 index --> band-pass; 1 index --> transform
    _ORDER          =  randint(1,9)

    #----------------------------------
    # peak detection of PPG signal
    #----------------------------------
    DELTA           =    0.1

    #----------------------------------
    # Rate of PPG signal
    #----------------------------------    
    PPG_SAMPLE_RATE =  200

    #----------------------------------
    # Frame Rate for Extracting Frame
    #---------------------------------- 
    FRAME_RATE     =    0.0167 

    #----------------------------------
    # Frame Rate for Extracting Frame
    #---------------------------------- 
    FRAME_NUM      =    60 
    

class PPG:
    def __init__(self, data):
        '''
            args:
                data : avg. value of red channel value (1-D list).
        '''
        self.data = data
        # error check -- type
        assert type(self.data)==list,"data(r_mean channel) is not a list"

    def bandPass(self):
        '''
            * apply band-pass filter for generating PPG signal

            returns:
                band-pass filter: it's a like of reverse-ppg singal.

        '''
        b, a = butter(config.ORDER[2], [(config.BPM_L / 60) * (2 / config.FPS), (config.BPM_H / 60) * (2 / config.FPS)], btype='band')
        filtered_data = filtfilt(b, a, self.data) # This is a reverse-filtered PPG.
        return filtered_data

    def bandPassSos(self):
        '''
            * apply band-pass (sosfilt) filter for generating PPG signal

            returns:
                band-pass filter: it's a like of reverse-ppg singal.

        '''
        sos = butter(config.ORDER[2], [(config.BPM_L / 60) * (2 / config.FPS), (config.BPM_H / 60) * (2 / config.FPS)], analog=False, btype='band', output='sos')
        filtered_data = sosfilt(sos, self.data) # This is a reverse-filtered PPG.
        return filtered_data
    
    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

    def butter_bandpass_filter(self):
        lowcut = (config.BPM_L / 60) * (2 / config.FPS)
        highcut = (config.BPM_H / 60) * (2 / config.FPS)
        fs = config.FPS
        order = config._ORDER
        sos = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, self.data)
        return y
    
    def normPass(self):
        '''
            * normal calculation for PPG
        '''
        return 0

    def peakdet(self, v, delta, x = None):
        """
        Converted from MATLAB script at http://billauer.co.il/peakdet.html
        
        Returns two arrays
        
        function [maxtab, mintab]=peakdet(v, delta, x)
        %PEAKDET Detect peaks in a vector
        %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
        %        maxima and minima ("peaks") in the vector V.
        %        MAXTAB and MINTAB consists of two columns. Column 1
        %        contains indices in V, and column 2 the found values.
        %      
        %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
        %        in MAXTAB and MINTAB are replaced with the corresponding
        %        X-values.
        %
        %        A point is considered a maximum peak if it has the maximal
        %        value, and was preceded (to the left) by a value lower by
        %        DELTA.
        
        % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
        % This function is released to the public domain; Any use is allowed.
        
        """
        maxtab = []
        mintab = []
        
        if x is None:
            x = arange(len(v))
        
        v = asarray(v)
        
        if len(v) != len(x):
            sys.exit('Input vectors v and x must have same length')
        
        if not isscalar(delta):
            sys.exit('Input argument delta must be a scalar')
        
        if delta <= 0:
            sys.exit('Input argument delta must be positive')
        
        mn, mx = Inf, -Inf
        mnpos, mxpos = NaN, NaN
        
        lookformax = True
        
        for i in arange(len(v)):
            this = v[i]
            if this > mx:
                mx = this
                mxpos = x[i]
            if this < mn:
                mn = this
                mnpos = x[i]
            
            if lookformax:
                if this < mx-delta:
                    maxtab.append((mxpos, mx))
                    mn = this
                    mnpos = x[i]
                    lookformax = False
            else:
                if this > mn+delta:
                    if mn < 0.0: # <-- change condition
                        mintab.append((mnpos, mn))
                        mx = this
                        mxpos = x[i]
                        lookformax = True
        return array(maxtab), array(mintab)

    def peak3ppgwave_avg3ppgwave(self, series, maxtab, mintab):
        ''' 
            * Pick Fresh 3 PPG wave 
            args:
                series              =   1-D list of float value of PPG waves
                [maxtab, mintab]    =   [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
                                        maxima and minima ("peaks") in the vector V.
            return:
                fresh_peak_3_ppg    =   values of peak 3 PPG waves. <LIST>
                avg
        '''
        pos_maxtab = array(maxtab)[:,0]
        val_maxtab = array(maxtab)[:,1]
        pos_mintab = array(mintab)[:,0]
        val_mintab = array(mintab)[:,1]
        values, posns = max_first_three(val_maxtab)
        # print(values, posns)

        values_peak_3_ppg = []
        for idx in posns:
            left_idx = idx - 1
            right_idx = idx #+ 1
            if left_idx < 0:
                left_idx = left_idx + 1
            elif right_idx >= len(pos_mintab):
                right_idx = right_idx - 1 
                
            # print(pos_mintab[left_idx], pos_mintab[right_idx])
            l_idx = int(pos_mintab[left_idx])
            r_idx = int(pos_mintab[right_idx]) + 1
            # print(series[l_idx:r_idx])
            # print("Len: ", len(series[l_idx:r_idx]))
            values_peak_3_ppg.append(list(series[l_idx:r_idx]))

        avg_peak_3_ppg = avgNestedLists(values_peak_3_ppg)
        fresh_peak_3_ppg = sum(values_peak_3_ppg, [])  
        return fresh_peak_3_ppg, avg_peak_3_ppg, l_idx, r_idx

    ## -----------------------------------------------------------------
    def _peakFinePPG(self, series, maxtab, mintab):
        ''' 
            * Pick Fresh 3 or 1 PPG wave 
            args:
                series              =   1-D list of float value of PPG waves
                [maxtab, mintab]    =   [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
                                        maxima and minima ("peaks") in the vector V.
            return:
                fresh_peak_ppg    =   values of peak PPG wave. <LIST>
                avg
        '''
        pos_maxtab = array(maxtab)[:,0]
        val_maxtab = array(maxtab)[:,1]
        pos_mintab = array(mintab)[:,0]
        val_mintab = array(mintab)[:,1]
        values, posns = sort_descending_order(val_maxtab)
        # print(values, posns)

        cnt_ppg = 0
        values_peak_ppg = []
        for idx in posns:
            left_idx = idx - 1
            right_idx = idx #+ 1
            if left_idx < 0:
                left_idx = left_idx + 1
            elif right_idx >= len(pos_mintab):
                right_idx = right_idx - 1 
                
            # print(pos_mintab[left_idx], pos_mintab[right_idx])
            l_idx = int(pos_mintab[left_idx])
            r_idx = int(pos_mintab[right_idx]) + 1
            # print(l_idx, r_idx)

            ## check Dicrotic Notch
            sing_ppg = list(series[l_idx:r_idx])
            # print(sing_ppg)
            maxima_index = argrelmax(np.array(sing_ppg))[0]
            minima_index = argrelmin(np.array(sing_ppg))[0]
            # print(maxima_index, minima_index)

            if len(maxima_index) >= 2 and len(minima_index) >= 1:
                values_peak_ppg.append(sing_ppg) # take single ppg
                cnt_ppg += 1
                # LOG_INFO(cnt_ppg) 
                # LOG_INFO(values_peak_ppg) 
            # else:
            #     '''
            #         *when we can't find best right signal
            #     '''
            #     fresh_peak_3_ppg, avg_peak_3_ppg = self.peak3ppgwave_avg3ppgwave(series, maxtab, mintab)
            #     return fresh_peak_3_ppg, avg_peak_3_ppg

            if cnt_ppg == 1: # 1 for one PPG wave & 3 for three PPG waves 
                break

        avg_peak_ppg = avgNestedLists(values_peak_ppg)
        fresh_peak_ppg = sum(values_peak_ppg, [])  
        return fresh_peak_ppg, avg_peak_ppg, l_idx, r_idx

class PPGFeatures:
 
    def extract_ppg45(single_waveform, sample_rate=config.PPG_SAMPLE_RATE):
        def __next_pow2(x):
            return 1<<(x-1).bit_length()

        features = []

        maxima_index = argrelmax(np.array(single_waveform))[0]
        minima_index = argrelmin(np.array(single_waveform))[0]

        if len(maxima_index) >= 2 and len(minima_index) >= 1:
            pass
        else:
            f_ind = np.argmax(np.array(single_waveform))
            # print(len(single_waveform[f_ind:]))
            s_ind = f_ind + 1
            if (len(single_waveform) > f_ind + 2) and (s_ind - f_ind >= 1):
                s_val = single_waveform[s_ind]
                t_val = single_waveform[s_ind+1]
                single_waveform[s_ind] = t_val
                single_waveform[s_ind+1] = s_val

            maxima_index = argrelmax(np.array(single_waveform))[0]
            minima_index = argrelmin(np.array(single_waveform))[0]

        # print(maxima_index, minima_index)

        derivative_1 = np.diff(single_waveform, n=1) * float(sample_rate)
        # print(derivative_1)
        derivative_1_maxima_index = argrelmax(np.array(derivative_1))[0]
        derivative_1_minima_index = argrelmin(np.array(derivative_1))[0]
        derivative_2 = np.diff(single_waveform, n=2) * float(sample_rate)
        derivative_2_maxima_index = argrelmax(np.array(derivative_2))[0]
        derivative_2_minima_index = argrelmin(np.array(derivative_2))[0]
        sp_mag = np.abs(np.fft.fft(single_waveform, n=__next_pow2(len(single_waveform))*16))
        freqs = np.fft.fftfreq(len(sp_mag))
        sp_mag_maxima_index = argrelmax(sp_mag)[0]
        # x
        x = single_waveform[maxima_index[0]]
        features.append(x)

        ##========================
        # c slope
        c = 0.55*x 
        features.append(c)
        
        # t_ms
        t_ms = 0.55*(float(maxima_index[0] + 1) / float(sample_rate))
        features.append(t_ms)

        # d = immediate prev. point of diastolic point
        d = single_waveform[maxima_index[1]-1]
        features.append(d)

        # t_ip
        t_ip = float(maxima_index[1]) / float(sample_rate)
        features.append(t_ip)
        ##========================
        
        # y
        y = single_waveform[maxima_index[1]]
        features.append(y)
        # z
        z = single_waveform[minima_index[0]]
        features.append(z)
        # t_pi
        t_pi = float(len(single_waveform)) / float(sample_rate)
        features.append(t_pi)
        # t_1
        t_1 = float(maxima_index[0] + 1) / float(sample_rate)
        features.append(t_1)
        # t_2
        t_2 = float(minima_index[0] + 1) / float(sample_rate)
        features.append(t_2)
        # t_3
        t_3 = float(maxima_index[1] + 1) / float(sample_rate)
        features.append(t_3)
        # width
        single_waveform_halfmax = max(single_waveform) / 2
        width = 0
        for value in single_waveform[maxima_index[0]::-1]:
            if value >= single_waveform_halfmax:
                width += 1
            else:
                break
        for value in single_waveform[maxima_index[0]+1:]:
            if value >= single_waveform_halfmax:
                width += 1
            else:
                break
        features.append(width)
        # A_2/A_1
        features.append(sum(single_waveform[:maxima_index[0]]) / sum(single_waveform[maxima_index[0]:]))
       
        #a1
        a_1 = derivative_1[derivative_1_maxima_index[0]]
        features.append(a_1)
        #b1
        b_1 = derivative_1[derivative_1_minima_index[0]]
        features.append(b_1)
        #e1
        e_1 = derivative_1[derivative_1_maxima_index[1]]
        features.append(e_1)
        #l1
        l_1 = derivative_1[derivative_1_minima_index[1]]
        features.append(l_1)
         
        # a_2
        a_2 = derivative_2[derivative_2_maxima_index[0]]
        features.append(a_2)
        # b_2
        b_2 = derivative_2[derivative_2_minima_index[0]]
        features.append(b_2)
        # e_2
        if len(derivative_2_maxima_index)>1:
            e_2 = derivative_2[derivative_2_maxima_index[1]]
            t_e2 = float(derivative_2_maxima_index[1]) / float(sample_rate)
        elif len(derivative_2_minima_index)>1:
            e_2 = derivative_2[derivative_2_minima_index[1]]
            t_e2 = float(derivative_2_minima_index[1]) / float(sample_rate)
        else:
            e_2 = derivative_2[derivative_2_maxima_index[0]]
            t_e2 = float(derivative_2_maxima_index[0]) / float(sample_rate)
        # print(derivative_2_maxima_index, derivative_2_minima_index)
        # e_2 = derivative_2[derivative_2_maxima_index[1]]
        features.append(e_2)
        
        # t_a1
        t_a1 = float(derivative_1_maxima_index[0]) / float(sample_rate)
        features.append(t_a1)
        # t_b1
        t_b1 = float(derivative_1_minima_index[0]) / float(sample_rate)
        features.append(t_b1)
        # t_e1
        t_e1 = float(derivative_1_maxima_index[1]) / float(sample_rate)
        features.append(t_e1)
        # t_f1
        t_f1 = float(derivative_1_minima_index[1]) / float(sample_rate)
        features.append(t_f1)
        
       
        
        # t_a2
        t_a2 = float(derivative_2_maxima_index[0]) / float(sample_rate)
        features.append(t_a2)
        # t_b2
        t_b2 = float(derivative_2_minima_index[0]) / float(sample_rate)
        features.append(t_b2)
        # t_e2
        features.append(t_e2)
        
        # f_base
        f_base = freqs[sp_mag_maxima_index[0]] * sample_rate
        features.append(f_base)
        # sp_mag_base
        sp_mag_base = sp_mag[sp_mag_maxima_index[0]] / len(single_waveform)
        features.append(sp_mag_base)
        # f_2
        f_2 = freqs[sp_mag_maxima_index[1]] * sample_rate
        features.append(f_2)
        # sp_mag_2
        sp_mag_2 = sp_mag[sp_mag_maxima_index[1]] / len(single_waveform)
        features.append(sp_mag_2)
        # f_3
        f_3 = freqs[sp_mag_maxima_index[2]] * sample_rate
        features.append(f_3)
        # sp_mag_3
        sp_mag_3 = sp_mag[sp_mag_maxima_index[2]] / len(single_waveform)
        features.append(sp_mag_3)
        return features, single_waveform

    def extract_svri(single_waveform):
        def __scale(data):
            data_max = max(data)
            data_min = min(data)
            return [(x - data_min) / (data_max - data_min) for x in data]
        max_index = np.argmax(single_waveform)
        single_waveform_scaled = __scale(single_waveform)
        return np.mean(single_waveform_scaled[max_index:]) / np.mean(single_waveform_scaled[:max_index])

def LOG_INFO(msg,mcolor='blue'):
    '''
        prints a msg/ logs an update
        args:
            msg     =   message to print
            mcolor  =   color of the msg    
    '''
    print(colored("#LOG     :",'green')+colored(msg,mcolor))
#---------------------------------------------------------------
def create_dir(base,ext):
    '''
        creates a directory extending base
        args:
            base    =   base path 
            ext     =   the folder to create
    '''
    _path=os.path.join(base,ext)
    if not os.path.exists(_path):
        os.mkdir(_path)
    return _path
#-------------------------------------------------------------------
def max_first_three(lst):
    '''
        * Use to find first three PPG wave
        arg:
            lst = list 
    '''
    ranks = sorted( [(x,i) for (i,x) in enumerate(lst)], reverse=True )
    values = []
    posns = []
    for x,i in ranks:
        if x not in values:
            values.append( x )
            posns.append( i )
            if len(values) == 3:
                break
            
    return values, posns

#-------------------------------------------------------------------
def sort_descending_order(lst):
    '''
        * Use to sort descending order of PPG wave
        arg:
            lst = list 
    '''
    ranks = sorted( [(x,i) for (i,x) in enumerate(lst)], reverse=True )
    values = []
    posns = []
    for x,i in ranks:
        if x not in values:
            values.append( x )
            posns.append( i )
            
    return values, posns
#------------------------------------------------------------------
def avgNestedLists(nested_vals):
    """
    Averages a 2-D array and returns a 1-D array of all of the columns
    averaged together, regardless of their dimensions.
    """
    output = []
    maximum = 0
    for lst in nested_vals:
        if len(lst) > maximum:
            maximum = len(lst)
    for index in range(maximum): # Go through each index of longest list
        temp = []
        for lst in nested_vals: # Go through each list
            if index < len(lst): # If not an index error
                temp.append(lst[index])
        output.append(np.nanmean(temp))
    return output
#-------------------------------------------------------------------
def tolerant_mean(arrs):
    """
    Averages a 2-D array and returns a 1-D array of all of the columns
    averaged together and error, regardless of their dimensions.
    """
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)

#-------------------------------------------------------------------
def plot_time_series(sinagl, out_img_dir, color = "r", label="name", fig_save=False, verbose=1):
    '''
        plot series of input signal
        args:
            singal  =   1-D list of float value
    '''
    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.figure(figsize=(8,6))
    plt.plot(sinagl, color, linewidth=1, label=label)
    
    plt.xlim(0, len(sinagl))
    plt.legend()
    plt.tight_layout()
    if fig_save:
        _path=create_dir(out_img_dir, "ppg_imgs")
        plt.savefig(os.path.join(_path, label+".png"), dpi = 100)
    if verbose:
        plt.show()

#-------------------------------------------------------------------
def plot_time_series(sinagl, out_img_dir, color = "r", label="name", fig_save=False, verbose=1):
    '''
        plot series of input signal
        args:
            singal  =   1-D list of float value
    '''
    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.figure(figsize=(8,6))
    plt.plot(sinagl, color, linewidth=1, label=label)
    
    plt.xlim(0, len(sinagl))
    plt.legend()
    plt.tight_layout()
    if fig_save:
        _path=create_dir(out_img_dir, "ppg_imgs")
        plt.savefig(os.path.join(_path, label+".png"), dpi = 100)
    if verbose:
        plt.show()

#-------------------------------------------------------------------
def plot_certain_ppg(sinagl, xmin, xmax, out_img_dir, color = "r", label="name", fig_save=False, verbose=1):
    '''
        plot series of input signal
        args:
            singal  =   1-D list of float value
    '''
    xi = list(range(xmin, xmax))
    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.figure(figsize=(8,6))
    plt.plot(xi, sinagl,  color, linewidth=1, label=label) 
    plt.xlabel('Index')
    plt.ylabel('Magnitude') 
    plt.title('PPG Signal')
    plt.grid(False)
    plt.legend() 
    plt.tight_layout()
    if fig_save:
        _path=create_dir(out_img_dir, "ppg_imgs")
        plt.savefig(os.path.join(_path, label+".png"), dpi = 100)
    if verbose:
        plt.show()

#-------------------------------------------------------------------
def plot_final_ppg(sinagl, color = "r", label="name"):
    '''
        plot series of input signal
        args:
            singal  =   1-D list of float value
    '''
    plt.rcParams.update({'figure.max_open_warning': 0})
    fig = Figure(figsize=(8, 6), dpi=100)
    plt.plot(sinagl, color, linewidth=1, label=label)
    plt.xlim(0, len(sinagl))
    plt.legend()
    plt.tight_layout()

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    im = Image.open(img_buf)
    return im

#-------------------------------------------------------------------
def plot_sample_series(sinagl, color = "r", label="name", verbose=1):
    '''
        plot series of input signal
        args:
            singal  =   1-D list of float value
    '''
    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.figure(figsize=(8,6))
    plt.plot(sinagl, color, linewidth=1, label=label)
    
    plt.xlim(0, len(sinagl))
    plt.legend()
    plt.tight_layout()
    if verbose:
        plt.show()
#-------------------------------------------------------------------
def refilter(_sig):
    '''
    * source: https://stackoverflow.com/a/51997184/5424617

        Finally, if you want to better see the contribution of 
        some specific frequency components without the interference 
        from spectral leakage from other frequency component, 
        you may want to consider pre-filtering your time-domain 
        signal before computing the FFT. For example, if you want to 
        eliminate the effect of the constant signal bias, the slow ~0.1Hz 
        variation and the noises with frequency greater than 10Hz 
        you might use something like the following:
    '''
    b,a = scipy.signal.butter(config.ORDER[1], [0.25/10, 5/10], btype='bandpass')
    y = scipy.signal.filtfilt(b,a,scipy.signal.detrend(np.ravel(_sig), type='constant'), padlen=len(_sig)-1)
    return y
#-------------------------------------------------------------------
def FFT(_signal):
    '''
    * Apply FFT on signal
    '''
    fast_FFT = scipy.fft.fft(_signal)
    return fast_FFT.real

#--------------------------------------------------------------------
def plot_peak_detect_series(series, maxtab, mintab, out_img_dir, label="name", fig_save=False, verbose=1):
    '''
        plot peak of series (ppg)
        args:
            series              =   1-D list of float value
            [maxtab, mintab]    =   [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
                                    maxima and minima ("peaks") in the vector V.
    
    '''
    pos_maxtab = array(maxtab)[:,0]
    val_maxtab = array(maxtab)[:,1]
    pos_mintab = array(mintab)[:,0]
    val_mintab = array(mintab)[:,1]

    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.style.use('ggplot') # nicer plots
    np.random.seed(52102) 
    plt.figure(figsize=(8, 6))
    plot(series)

    scatter(array(maxtab)[:,0], array(maxtab)[:,1], color='blue')
    scatter(array(mintab)[:,0], array(mintab)[:,1], color='green')
    for i, txt in enumerate(pos_maxtab):
        plt.annotate(txt, (pos_maxtab[i], val_maxtab[i]))
        
    for i, txt in enumerate(pos_mintab):
        plt.annotate(txt, (pos_mintab[i], val_mintab[i]))
    if fig_save:
        _path=create_dir(out_img_dir, "peak detected ppg_imgs")
        plt.savefig(os.path.join(_path, label+".png"), dpi = 100)
    if verbose:
        plt.show()
#----------------------------------------------------------------------

#----------------------------------------------------------------
# class: for generating PPG signal
#----------------------------------------------------------------
class ExtractFrames(object):
    def __init__(self, video_dir):
        '''
            video_dir: directory of finger-trip videos
            img_dir  : directory of finger-trip images
        '''
        self.video_dir = video_dir
       # self.img_dir   = img_dir

    def video_to_frames(self, video_filename, frames_save=False, fps_num=None):
        """
            Extract frames from video
        """
        video_file_path = os.path.join(self.video_dir, video_filename)
        if frames_save:
            frames_dir = create_dir(self.img_dir, 'frames_'+video_filename)

        cap = cv2.VideoCapture(video_file_path)
        #if fps_num is not None:
          #  cap.set(cv2.CAP_PROP_FPS, int(fps_num))

        # fps = round(cap.get(cv2.CAP_PROP_FPS))
        fps=30
      #  video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1 # frame numbers

        #duration = video_length/fps  

        if cap.isOpened():
            count = 0
            success, image = cap.read()
            while success:
                try:
                    fps = 30
                    success, image = cap.read()
                    im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if frames_save:
                        cv2.imwrite(os.path.join(frames_dir, str(count) + '.jpg'), image)
                    yield  im_rgb, fps
                    count += 1
                except Exception as e:
                    LOG_INFO(f"Error in frame Check:{count}",mcolor="yellow")
                    LOG_INFO(f"{e}",mcolor="red") 

    def video_to_frames_fixed_frameRate(self, video_filename, frames_save=False, fps_num=None, frame_rate = 0.035):
        """
            Extract frames from video
        """
        #video_file_path = os.path.join(self.video_dir, video_filename)
        if frames_save:
            frames_dir = create_dir(self.img_dir, 'frames_'+video_filename)

        cap = cv2.VideoCapture(video_filename)
        #if fps_num is not None:
            #cap.set(cv2.CAP_PROP_FPS, int(fps_num))
        
        fps = 30
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1 # frame numbers
        return video_length
        #duration = video_length/fps  

        if cap.isOpened():
            count = 0
            # success, image = cap.read()
            '''
            * CAP_PROP_POS_MSEC 
                Python: cv.CAP_PROP_POS_MSEC
                Current position of the video file in milliseconds.
            '''
            sec = 0
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            success, image = cap.read()

            while success:
                try:
                    sec = sec + frame_rate
                    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
                    fps = round(cap.get(cv2.CAP_PROP_FPS))
                    success, image = cap.read()
                    if frames_save:
                        cv2.imwrite(os.path.join(frames_dir, str(count) + '.jpg'), image)
                    yield image, fps
                    count += 1
                except Exception as e:
                    LOG_INFO(f"Error in frame Check:{count}",mcolor="yellow")
                    LOG_INFO(f"{e}",mcolor="red")

frames_save = False
fps_num = 60
frame_rate = 0.0167
default_fps = True
fig_save = True
verbose = 0
####*IMPORANT*: Have to do this line *before* importing tensorflow
os.environ['PYTHONHASHSEED']=str(1)

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras as keras
import tensorflow.keras.layers 
import random
import pandas as pd
import numpy as np
#import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Dropout
from keras import optimizers

 
def main(data_dir,ppg_feats,age,gender):
    ##---------------------------------
    # data_dir        =   args.data_dir
    # save_dir        =   args.save_dir
    # out_img_dir     =   args.out_img_dir
    # ppg_signal_dir  =   args.ppg_signal_dir
    # ppg_feats       =   args.ppg_feats
    # xlsx_path       =   args.xlsx_path
    # ppg_labels      =   args.ppg_feats_labels
    # frames_save     =   args.frames_save
    # fps_num         =   args.fps_num
    # frame_rate      =   args.frame_rate
    # default_fps     =   args.default_fps
    # fig_save        =   args.fig_save
    # verbose         =   args.verbose
    ##----------------------------------
    file_name_lst = []
    ppg_signals_lst = []
    features_set = []

    _gen_ppg = ExtractFrames(data_dir)
    for root, dirs, files in tqdm(sorted(os.walk(data_dir))):
        for file in files:
            LOG_INFO(f"File name= {file}", mcolor="green")
            if default_fps == True:
                frames_fps = _gen_ppg.video_to_frames(file, frames_save, fps_num)
            else:
                frames_fps = _gen_ppg.video_to_frames_fixed_frameRate(file, frames_save, None, frame_rate)

            # to store avg. r, g, b value
            _signal=[]
            _cnt = 0
            for idx, (img,fps) in tqdm(enumerate(frames_fps)):
              
                try:
                    '''
                        fps = total_frames / video_duration
                    '''
                    # motive to take 600 frames 
                    # if fps <= 30:
                    #     fps = 2*fps
                        
                    # take almost 300/600 frames (10 sec video: 30 fps / 60 fps --> (10x30)=300, (10x60)=600)   
                    if idx >= 2*fps and _cnt <= 10*fps:
                        '''
                            * avoid first 2 sec. video
                            * take 10 sec. video
                        '''

                        ## to get a square right to left (ROI: 500x500)
                        h= img.shape[0]//2 - 250 # height//2 - 250. where height = 1080 px
                        w= img.shape[1] # width. where width = 1920 px
                        img = img[h:h+500, w-500:w]
                        # print("Image size: " + str(img.shape))
                        

                        ## find max and min intensity of image's red channel
                        intensity_min = img[..., 2].min()
                        intensity_max = img[..., 2].max()
                        thresh = (0.5*intensity_min)+(0.5*intensity_max)
                        # print(intensity_min, intensity_max, thresh)

                        mean_pixel = img[:, :, 2].mean()
                        if mean_pixel > thresh:
                            _signal.append(mean_pixel)
    
                        _cnt += 1

                    elif _cnt > 10*fps:
                        LOG_INFO(f"Frame number= {_cnt}", mcolor="green")
                        break
                    
                except Exception as e:
                    LOG_INFO(f"File Number Error= {idx}",mcolor="red")
                    LOG_INFO(f"{e}",mcolor="red") 

            ## check the ppg whether found or not
            try:

                ## plot red signal
               # plot_time_series(_signal, out_img_dir, 'r', str(file.split(".")[0])+'- RED Channel Signal', fig_save=fig_save, verbose=verbose)

                ## apply bandpass filter
                _ppg = PPG(_signal)
                rev_PPG_signal = _ppg.bandPass()
                _PPG_signal = rev_PPG_signal[::-1] # reverse bandpass signal (like. PPG)
                plot_time_series(_PPG_signal, "/storage/emulated/0/Download/Output", 'r', 'BandPass Signal', fig_save=fig_save, verbose=verbose)
                
                ## Save PPG singals w.r.t video
                file_name_lst.append(file)
                ppg_signals_lst.append(np.array(_PPG_signal))


                ## Peak detection of PPG signal
                series = _PPG_signal # rename or copy
                maxtab, mintab = _ppg.peakdet(series,config.DELTA)
             #   plot_peak_detect_series(series, maxtab, mintab, out_img_dir, str(file.split(".")[0])+'- Peak detect BandPass Signal', fig_save=fig_save, verbose=verbose)

                ## pick fresh 3 PPG wave and avg PPG wave
                best_systolic = False
                if best_systolic == True:
                    peak_3_ppg, avg_3_ppg, l_idx, r_idx = _ppg.peak3ppgwave_avg3ppgwave(series, maxtab, mintab)
                else:
                    '''
                        sorting decending order w.r.t. sysolic pick
                    '''
                    peak_ppg, avg_ppg, l_idx, r_idx = _ppg._peakFinePPG(series, maxtab, mintab)
                    '''
                        * if best ppg is not found then take most 3 ppg and do avg.
                    '''
                    if len(peak_ppg) == 0 or len(avg_ppg) == 0:
                        LOG_INFO("best ppg is not found!!!")
                        peak_ppg, avg_ppg, l_idx, r_idx = _ppg.peak3ppgwave_avg3ppgwave(series, maxtab, mintab)

              #  plot_certain_ppg(peak_ppg, l_idx, l_idx+len(peak_ppg), out_img_dir, 'r', str(file.split(".")[0])+"- peak PPG Wave", fig_save=fig_save, verbose=verbose)
              #  plot_certain_ppg(avg_ppg, l_idx, l_idx+len(avg_ppg), out_img_dir, 'r', str(file.split(".")[0])+"- avg PPG wave", fig_save=fig_save, verbose=verbose)

                ## Total 46 features: Extract PPG 45 features + svri feature _FFT
                '''
                    * with extracted features generate csv file
                '''
                try:
                    if best_systolic:
                        _feat_ppg49, _single_waveform = PPGFeatures.extract_ppg45(avg_3_ppg)
                        _feat_svri = PPGFeatures.extract_svri(_single_waveform)
                    else:
                        _feat_ppg49, _single_waveform = PPGFeatures.extract_ppg45(avg_ppg)
                        _feat_svri = PPGFeatures.extract_svri(_single_waveform)

                   # LOG_INFO(f"49 fetures: {_feat_ppg49}||\t\n SVRI: {_feat_svri}",mcolor="green") 
              #      plot_certain_ppg(_single_waveform, l_idx, l_idx+len(_single_waveform), out_img_dir, 'r', str(file.split(".")[0])+"- Final Single PPG", fig_save=fig_save, verbose=verbose)
                    
                    """ PPG-49 + extract_svri Feature Extraction """
                
                    #file_name = file.split(".")[0]
                    #ID = file_name.split("Hb")[-1]                    
                    _feat_ppg49.insert(0, 20)
                    _feat_ppg49.append(_feat_svri)
                    features_set.append(_feat_ppg49)
                    

        
                except Exception as e:
                    LOG_INFO(f"{e}",mcolor="red") 

            except Exception as e:
                LOG_INFO(f"PPG Signal Error: {e}",mcolor="red") 

    # ## save ppg signal of each video to .csv file
    # list_dict = {'FileName': np.array(file_name_lst, dtype=object), 'PPG_Signal': np.array(ppg_signals_lst, dtype=object)} 
    # ppg_df = pd.DataFrame(list_dict) 
    # ppg_df.to_csv(ppg_signal_dir, index=False) 

    """ 
        * Preprocess the dataset for all videos
    """
    ## make ppg features .csv file
    headers = ['ID', 'Systolic_peak(x)', 'Max. Slope(c)', 'Time of Max. Slope(t_ms)', 'Prev. point a_ip(d)', 'Time of a_ip(t_ip)', 'Diastolic_peak(y)', 'Dicrotic_notch(z)', 'Pulse_interval(tpi)', 'Systolic_peak_time(t1)', 'Diastolic_peak_time(t2)', 'Dicrotic_notch_time(t3)', 'w', 'Inflection_point_area_ratio(A2/A1)', 'a1','b1', 'e1', 'l1', 'a2','b2','e2', 'ta1', 'tb1', 'te1', 'tl1', 'ta2', 'tb2', 'te2', 'Fundamental_component_frequency(fbase)', 'Fundamental_component_magnitude(|sbase|)', '2nd_harmonic_frequency(f2nd)', '2nd_harmonic_magnitude(|s2nd|)', '3rd_harmonic_frequency(f3rd)', '3rd_harmonic_magnitude(|s3rd|)', 'Stress-induced_vascular_response_index(sVRI)']
    print(len(headers))
    df_input = pd.DataFrame(features_set, columns=headers)

    df_input.insert(0, 'Age', float(age))#pass from android
    df_input.insert(1, 'Sex(M/F)', gender)#
    #dataFr.head()
    #dataFr.shape
    df_input.to_csv(ppg_feats)
    csv=os.path.join(os.path.dirname(__file__),"preprocessed_PPG-34.csv")
    df = pd.read_csv(csv)
    df.drop(df.columns[[0, 1, 40]], axis=1, inplace=True) 
    df.shape
    Xorg = df.to_numpy()
    scaler = StandardScaler()
    Xscaled = scaler.fit_transform(Xorg)
    Xmeans = scaler.mean_
    Xstds = scaler.scale_
    Gl_means, GL_std = Xmeans[37], Xstds[37]
    Hb_means, Hb_std = Xmeans[36], Xstds[36]
    df_in = df_input.append(df, ignore_index = True)
    df_in.to_csv(ppg_feats)

    #standardization 
    Xorg_in = df_in.to_numpy()
    scaler_in = StandardScaler()
    Xscaled_in = scaler_in.fit_transform(Xorg_in)
    ## store these off for predictions with unseen data
    Xmeans_in = scaler_in.mean_
    Xstds_in = scaler_in.scale_
    X = Xscaled_in[:1, 0:36]
    get_best_ind_final_Gl = [0, 1, 2, 4, 5, 6, 7, 10, 11, 12, 13, 14, 17, 18, 20, 21, 22, 26, 27, 30, 32, 35] 
    get_best_ind_final_Hb = [0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 14, 18, 19, 20, 21, 22, 24, 26, 27, 30, 32, 35] 
    X_Gl = X[:, get_best_ind_final_Gl]
    X_Hb = X[:, get_best_ind_final_Hb]
    model_path=os.path.join(os.path.dirname(__file__),'DNN_model_Gl.h5')
    model_Gl = tf.keras.models.load_model(model_path)
    Gl = model_Gl.predict(X_Gl) ### Estimated of Gl Level
    G_estimate = (Gl * GL_std) + Gl_means
   # print("Estimated Gl (mmol/L): " + str(G_estimate))
    #Add two dataframe 
    #df_in = dataFr.append(df, ignore_index = True)
    model_path=os.path.join(os.path.dirname(__file__),'DNN_model_Hb.h5')
    model_Hb = tf.keras.models.load_model(model_path)
    Hb = model_Hb.predict(X_Hb) ### Estimated of Hb Level
    Hb_estimate = (Hb * Hb_std) + Hb_means
    #print("Estimated Hemoglobin (g/dL): " + str(Hb_estimate))
    return str(Hb_estimate)

if __name__=="__main__":
    '''
        parsing and executions
    '''
    # parser = argparse.ArgumentParser("Raw Video Data Analysis, PPG Generation, and PPG Features Set Generation Script")
    # parser.add_argument("data_dir", help="Path to source data") 
    # parser.add_argument("save_dir", help="Path to save the processed data")
    # parser.add_argument("out_img_dir", help="Path to save the output images") 
    # parser.add_argument("ppg_signal_dir", help="Path to save the ppg signal as csv file")
    # parser.add_argument("ppg_feats", help="Path to save PPG features .csv file")
    # parser.add_argument("xlsx_path", help="Path of gold standard datset in .xlsx")
    # parser.add_argument("ppg_feats_labels", help="Path to save PPG features and labels .csv file")
    # parser.add_argument("--frames_save",type=bool,required=False,default=False,help ="Whether you will save images in save dir : default=False")
    # parser.add_argument("--fps_num",type=int,default=60,help ="Fixed fps: default=30")
    # parser.add_argument("--frame_rate",type=float,default=0.0167,help ="Fixed frame rate: default=0.035")
    # parser.add_argument("--default_fps",type=bool,required=False,default=True,help ="Whether you will use default fps: default=True")
    # parser.add_argument("--fig_save",type=bool,required=False,default=True,help ="Whether you will see save plotted figure: default=True")
    # parser.add_argument("--verbose",type=int,required=False,default=0,help ="Whether you will see message/figure in terminal: default=1")

    # args = parser.parse_args()
    # main(args)
    '''
    root_path="../dataset_folder/" 
    src_path="${root_path}raw_videos/"
    save_img_path="${root_path}raw_images/"
    output_imgs_path="../output_images/"
    ppg_signal_dir="${root_path}ppg_signals.csv"
    ppg_feats="${root_path}ppg_feats.csv"
    xlsx_path="${root_path}Data.xlsx"
    ppg_feats_labels="${root_path}ppg_feats_labels.csv"
    
    python generate_dataset.py $src_path $save_img_path $output_imgs_path $ppg_signal_dir $ppg_feats $xlsx_path $ppg_feats_labels
    #--------------------------------------------------------------------------------------------------------------
    echo succeded
    '''
    
    
#     data_dir =  '/storage/emulated/0/Movies/CameraX-Video/20_abc_.mp4'
#     ppg_feats = '/storage/emulated/0/Download/ppg_feats.csv'
    frames_save = False
    fps_num = 60
    frame_rate = 0.0167
    default_fps = True
    fig_save = True
    verbose = 0
    #main(data_dir,ppg_feats)