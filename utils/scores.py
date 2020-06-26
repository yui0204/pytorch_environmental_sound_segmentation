import numpy as np
import os

def rmse(Y_true, Y_pred, classes):
    Y_pred_db = (Y_pred * 120) # 0~120dB
    Y_true_db = (Y_true * 120)
    
    total_rmse = np.sqrt(((Y_true_db - Y_pred_db) ** 2).mean())
    #print("Total RMSE =", total_rmse)

    rms_array = np.zeros(classes + 1)
    num_array = np.zeros(classes + 1) # the number of data of each class

    area_array = np.zeros(classes + 1) # area size of each class
    duration_array = np.zeros(classes + 1) # duration size of each class
    freq_array = np.zeros(classes + 1) # frequency conponents size of each class
    
    spl_array =  np.zeros(classes + 1) #average SPL of each class 
    percent_array =  np.zeros(classes + 1) 

    for no in range(len(Y_true)):
        for class_n in range(classes):
            if Y_true[no][class_n].max() > 0: # calculate RMS of active event class               
                num_array[classes] += 1 # total number of all classes
                num_array[class_n] += 1 # the number of data of each class
                on_detect = Y_true[no][class_n].max(0) > 0.0 # active section of this spectrogram image
                
                per_rms = ((Y_true_db[no][class_n] - Y_pred_db[no][class_n]) ** 2).mean(0) # mean squared error about freq axis of this spectrogram
                rms_array[class_n] += per_rms.sum() / on_detect.sum() # mean squared error of one data
                
                per_spl = Y_true_db[no][class_n].mean(0) # mean spl about freq axis
                spl_array[class_n] += per_spl.sum() / on_detect.sum() # mean spl of one data
                
                area_array[class_n] += ((Y_true[no][class_n] > 0.0) * 1).sum() # number of active bins = area size
                duration_array[class_n] += ((Y_true[no][class_n].max(0) > 0.0) * 1).sum() # duration bins
                freq_array[class_n] += ((Y_true[no][class_n].max(1) > 0.0) * 1).sum() # duration bins
                
    rms_array[classes] = rms_array.sum()
    rms_array = np.sqrt(rms_array / num_array) # Squared error is divided by the number of data = MSE then root = RMSE

    spl_array[classes] = spl_array.sum()
    spl_array = spl_array / num_array # Sum of each spl is divided by the number of data = average spl

    area_array[classes] = area_array.sum()
    area_array = area_array // num_array # average area size of each class

    duration_array[classes] = duration_array.sum()
    duration_array = duration_array // num_array # average duration size of each class
    duration_array = duration_array * 16.0

    percent_array = rms_array / spl_array * 100 

    #print("rms\n", rms_array, "\n")     
    all_array = np.vstack([num_array, area_array, duration_array, rms_array, percent_array, spl_array])
   
    return all_array


def save_score_array(scores_array, save_dir):
    save_path = os.path.join(save_dir, "scores_array.csv")
    np.savetxt(save_path, scores_array.T, fmt ='%.3f')


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {'Overall Acc: \t': acc,
            'Mean Acc : \t': acc_cls,
            'FreqW Acc : \t': fwavacc,
            'Mean IoU : \t': mean_iu,}, cls_iu
