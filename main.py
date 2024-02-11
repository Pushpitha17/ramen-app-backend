from distutils.log import debug
from flask import Flask, render_template, request, make_response, jsonify
from flask_cors import CORS
import numpy as np
import json


import preprocessing.baseline as baseline_func
import preprocessing.denoise as desnoising_func
import preprocessing.normalize as noramilzing_func
import preprocessing.findpeaks as findpeaks_func


class RamanSpectrum:
    def __init__(self, name, L, I) -> None:
        """initiate RamanSpectrum objcet

        Args:
            L (iterable): raman shift waves of the spectrum
            I (iterable): corresponding intensities to the raman shifts
        """
        self.shift = np.array(L)
        self.raw_intensity = np.array(I)
        self.intensity = self.raw_intensity
        self.baseline = np.zeros(len(self.shift))
        self.name = name
        self.fingerprint = "Normalize to generate the spectral fingerprint"
        self.processing_history = []


app = Flask(__name__)
CORS(app)


def makeSpectrumInstance(spectrum):
    shift = np.asarray(spectrum["shift"], dtype=float)
    intensity = np.asarray(spectrum["intensity"], dtype=float)

    return RamanSpectrum("Temp", shift, intensity)


@app.route('/')
def root():
    return "Allo Bois !!"


@app.route('/baseline/polyfit', methods=['POST'])
def baseline():
    request_data = request.get_json()
    spectrum_instance = makeSpectrumInstance(request_data['spectrum'])
    order = int(request_data['order'])
    iterations = int(request_data['iterations'])

    result = baseline_func.mod_polyfit(spectrum_instance, order, iterations)
    resp = {"intensity": result[0].intensity.tolist(), "baseline": result[0].baseline, "polynomial": result[1].tolist()}
    return jsonify(resp)


@app.route('/denoise/moving_avg', methods=['POST'])
def denoise_moving_avg():
    request_data = request.get_json()

    spectrum_instance = makeSpectrumInstance(request_data['spectrum'])
    window = int(request_data['window'])

    result = desnoising_func.moving_average(spectrum_instance, window)
    resp = {"intensity": result.intensity.tolist()}
    return jsonify(resp)


@app.route('/denoise/savgol', methods=['POST'])
def denoise_savgol():
    request_data = request.get_json()

    spectrum_instance = makeSpectrumInstance(request_data['spectrum'])
    order = int(request_data['order'])
    window = int(request_data['window'])

    result = desnoising_func.salvitzky_golay_smooth(
        spectrum_instance, order, window)
    resp = {"intensity": result.intensity.tolist()}
    return jsonify(resp)


@app.route('/denoise/lowess', methods=['POST'])
def denoise_lowess():
    request_data = request.get_json()

    spectrum_instance = makeSpectrumInstance(request_data['spectrum'])
    order = int(request_data['order'])
    window = int(request_data['window'])

    result = desnoising_func.lowess_smooth(spectrum_instance, order, window)
    resp = {"intensity": result.intensity.tolist()}
    return jsonify(resp)


@app.route('/normalize/minmax', methods=['POST'])
def normalize_minmax():
    request_data = request.get_json()

    spectrum_instance = makeSpectrumInstance(request_data['spectrum'])
    height = int(request_data['height'])

    result = noramilzing_func.min_max_norm(spectrum_instance, height)
    resp = {"intensity": result.intensity.tolist()}
    return jsonify(resp)


@app.route('/normalize/one', methods=['POST'])
def normalize_one():
    request_data = request.get_json()

    spectrum_instance = makeSpectrumInstance(request_data['spectrum'])

    result = noramilzing_func.one_norm(spectrum_instance)
    resp = {"intensity": result.intensity.tolist()}
    return jsonify(resp)


@app.route('/normalize/two', methods=['POST'])
def normalize_two():
    request_data = request.get_json()

    spectrum_instance = makeSpectrumInstance(request_data['spectrum'])

    result = noramilzing_func.two_norm(spectrum_instance)
    resp = {"intensity": result.intensity.tolist()}
    return jsonify(resp)


@app.route('/normalize/inf', methods=['POST'])
def normalize_inf():
    request_data = request.get_json()

    spectrum_instance = makeSpectrumInstance(request_data['spectrum'])

    result = noramilzing_func.one_norm(spectrum_instance)
    resp = {"intensity": result.intensity.tolist()}
    return jsonify(resp)


@app.route('/normalize/snv', methods=['POST'])
def normalize_snv():
    request_data = request.get_json()

    spectrum_instance = makeSpectrumInstance(request_data['spectrum'])

    result = noramilzing_func.one_norm(spectrum_instance)
    resp = {"intensity": result.intensity.tolist()}
    return jsonify(resp)


@app.route('/findpeaks', methods=['POST'])
def findpeaks():
    request_data = request.get_json()

    spectrum_instance = makeSpectrumInstance(request_data['spectrum'])
    prominence = int(request_data['prominence'])

    result = findpeaks_func.detect_peaks(spectrum_instance, prominence)
    resp = {"peaks": result.fingerprint}
    return jsonify(resp)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
