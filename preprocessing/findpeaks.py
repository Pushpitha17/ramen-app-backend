from scipy.signal import find_peaks


def detect_peaks(spectrum, prominence=1, threshold=0.1):
    max_peak = max(spectrum.intensity)
    peaks = find_peaks(spectrum.intensity,
                       prominence=prominence, height=max_peak*threshold)

    peaks_shift = [spectrum.shift[i] for i in peaks[0]]
    peaks_intensity = [spectrum.intensity[i] for i in peaks[0]]

    spectrum.fingerprint = list(zip(peaks_shift, peaks_intensity))
    print(spectrum.fingerprint)
    return spectrum
