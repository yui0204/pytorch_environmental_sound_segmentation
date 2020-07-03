import os
import glob
import numpy as np
import cmath
from mir_eval.separation import bss_eval_sources
import soundfile as sf
from scipy import signal
import re


def _pred_dir_make(no, save_dir):
    pred_dir = os.path.join(save_dir, "prediction", str(no))
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    
    return pred_dir
    

def restore(Y_true, Y_pred, phase, no, save_dir, classes, ang_reso, label, dataset_dir):
    plot_num = classes * ang_reso

    pred_dir = _pred_dir_make(no, save_dir)
    data_dir = os.path.join(dataset_dir, "val", str(no))

    sdr_array = np.zeros((plot_num, 1))
    sir_array = np.zeros((plot_num, 1))
    sar_array = np.zeros((plot_num, 1))

    wavefile = glob.glob(data_dir + '/0__*.wav')
    X_wave, _ = sf.read(wavefile[0])

    for index_num in range(plot_num):
        if Y_true[no][index_num].max() > 0:
            Y_linear = 10 ** ((Y_pred[no][index_num] * 120 - 120) / 20)
            Y_linear = np.vstack((Y_linear, Y_linear[::-1]))

            Y_complex = np.zeros((Y_true.shape[-1] * 2, Y_true.shape[-1]), dtype=np.complex128)
            for i in range (Y_true.shape[-1] * 2):
                for j in range (Y_true.shape[-1]):
                    Y_complex[i][j] = cmath.rect(Y_linear[i][j], phase[no][i][j])

            if ang_reso == 1:
                filename = label.index[index_num]+"_prediction.wav"
            else:
                filename = label.index[index_num % ang_reso] + "_" + str((360 // ang_reso) * (index_num % ang_reso)) + "deg_prediction.wav"
                
            _, Y_pred_wave = signal.istft(Zxx=Y_complex, fs=16000, nperseg=512, input_onesided=False)
            Y_pred_wave = Y_pred_wave.real
            sf.write(pred_dir + "/" + filename, Y_pred_wave.real, 16000, subtype="PCM_16")

            # calculate SDR
            if classes == 1:
                with open(os.path.join(data_dir, "sound_direction.txt"), "r") as f:
                    directions = f.read().split("\n")[:-1]
                for direction in directions:
                    if index_num == int(re.sub("\\D", "", direction.split("_")[1])) // (360 // ang_reso):
                        class_name = direction.split("_")[0]
                        Y_true_wave, _ = sf.read(data_dir + "/" + class_name + ".wav")
            else:                
                Y_true_wave, _ = sf.read(data_dir + "/" + label.index[index_num // ang_reso] + ".wav")
            
            Y_true_wave = Y_true_wave[:len(Y_pred_wave)]
            X_wave = X_wave[:len(Y_pred_wave)]

            sdr_base, sir_base, sar_base, per_base = bss_eval_sources(Y_true_wave[np.newaxis,:], X_wave[np.newaxis,:], compute_permutation=False)
            sdr, sir, sar, per = bss_eval_sources(Y_true_wave[np.newaxis,:], Y_pred_wave[np.newaxis,:], compute_permutation=False)
            print("No.", no, "Class", index_num, label.index[index_num // ang_reso], "SDR", round(sdr[0], 2), "SDR_Base", round(sdr_base[0], 2), "SDR improvement: ", round(sdr[0] - sdr_base[0], 2))
            
            sdr_array[index_num] = sdr
            sir_array[index_num] = sir
            sar_array[index_num] = sar

    return sdr_array, sir_array, sar_array

