import os
import glob
import numpy as np
import cmath
from .sound import WavfileOperate, Stft
from mir_eval.separation import bss_eval_sources


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
    X_wave = WavfileOperate(wavefile[0]).wavedata.norm_sound            

    for class_n in range(plot_num):
        if Y_true[no][class_n].max() > 0:
            Y_linear = 10 ** ((Y_pred[no][class_n] * 120 - 120) / 20)
            Y_linear = np.vstack((Y_linear, Y_linear[::-1]))

            Y_complex = np.zeros((Y_true.shape[-1] * 2, Y_true.shape[-1]), dtype=np.complex128)
            for i in range (Y_true.shape[-1] * 2):
                for j in range (Y_true.shape[-1]):
                    Y_complex[i][j] = cmath.rect(Y_linear[i][j], phase[no][i][j])

            if ang_reso == 1:
                Y_Stft = Stft(Y_complex, 16000, label.index[class_n]+"_prediction")
            else:
                Y_Stft = Stft(Y_complex, 16000, label.index[i % ang_reso * 0] + "_" + str((360 // ang_reso) * (class_n % ang_reso)) + "deg_prediction")
                
            Y_pred_wave = Y_Stft.scipy_istft()
            Y_pred_wave.write_wav_sf(dir=pred_dir, filename=None, bit=16)

            # calculate SDR
            Y_pred_wave = Y_pred_wave.norm_sound
            Y_true_wave = WavfileOperate(data_dir + "/" + label.index[class_n] + ".wav").wavedata.norm_sound            
            Y_true_wave = Y_true_wave[:len(Y_pred_wave)]
            X_wave = X_wave[:len(Y_pred_wave)]
            
            sdr_base, sir_base, sar_base, per_base = bss_eval_sources(Y_true_wave[np.newaxis,:], X_wave[np.newaxis,:], compute_permutation=True)
            sdr, sir, sar, per = bss_eval_sources(Y_true_wave[np.newaxis,:], Y_pred_wave[np.newaxis,:], compute_permutation=True)
            print("No.", no, class_n, label.index[class_n], round(sdr[0], 2), round(sdr_base[0], 2), "SDR improvement: ", round(sdr[0] - sdr_base[0], 2))
            
            sdr_array[class_n] = sdr
            sir_array[class_n] = sir
            sar_array[class_n] = sar

    return sdr_array, sir_array, sar_array

