from numpy import polyfit
from numpy import polyval
import numpy as np

def mod_polyfit(spectrum, order=4, iterations=128):
    spectrum.baseline = spectrum.intensity
    fit = polyval(polyfit(spectrum.shift,spectrum.baseline,deg=order),spectrum.shift)
    for _ in range(iterations):
        fit = polyval(polyfit(spectrum.shift,spectrum.baseline,deg=order),spectrum.shift)
        spectrum.baseline = [min(spectrum.baseline[i],fit[i]) for i in range(len(spectrum.shift))]
    else:
        spectrum.intensity = spectrum.intensity - spectrum.baseline
        spectrum.processing_history.append('baseline-mod_polyfit-order-{}-iterations-{}'.format(order,iterations))
    return [spectrum , fit]