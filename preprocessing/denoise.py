import numpy as np
from scipy.signal import savgol_filter,butter,filtfilt
from statsmodels.nonparametric.smoothers_lowess import lowess

#moving average
def moving_average(spectrum,window=3):
    c = np.ones(window)
    print(spectrum.intensity)
    spectrum.intensity = np.convolve(spectrum.intensity, c, 'valid')
    spectrum.processing_history.append ('denoise-moving_average-window-{}'.format(window))
    return spectrum

#salvitzky golay
def salvitzky_golay_smooth(spectrum,order=1,window=7):
    spectrum.intensity = savgol_filter(spectrum.intensity,polyorder=order,window_length=window)
    spectrum.processing_history.append ('denoise-salvitzky_golay_smooth-order-{}-window-{}'.format(order,window))
    return spectrum

#lowess
def lowess_smooth(spectrum,order=1,window=5):
    spectrum.intensity = lowess(spectrum.intensity,spectrum.shift,frac=window/len(spectrum.shift),return_sorted=False)
    spectrum.processing_history.append ('denoise-lowess_smooth-order-{}-window-{}'.format(order,window))
    return spectrum