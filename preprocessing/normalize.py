import numpy as np

def min_max_norm(spectrum, height = 1):
    norm_factor = max(spectrum.intensity)/height
    spectrum.intensity = spectrum.intensity/norm_factor
    spectrum.processing_history.append('normalize-min_max-height-{}'.format(height))
    return spectrum

def one_norm(spectrum):
    # norm_factor = sum(spectrum.intensity.abs())
    norm_factor = np.sum(np.abs(spectrum.intensity))
    spectrum.intensity = spectrum.intensity/norm_factor
    spectrum.processing_history.append('normalize-one-norm}')
    return spectrum

def two_norm(spectrum):
    norm_factor = sum(spectrum.intensity**2)**0.5
    spectrum.intensity = spectrum.intensity/norm_factor
    spectrum.processing_history.append('normalize-two-norm}')
    return spectrum

def inf_norm(spectrum):
    norm_factor = max(np.abs(spectrum.intensity))
    spectrum.intensity = spectrum.intensity/norm_factor
    spectrum.processing_history.append('normalize-inf-norm}')
    return spectrum

def snv_norm(spectrum):
    mean = spectrum.intesity.mean()
    sd = spectrum.intesity.std()
    spectrum.intensity = (spectrum.intensity-mean)/sd
    spectrum.processing_history.append('normalize-snv-norm}')
    return spectrum 