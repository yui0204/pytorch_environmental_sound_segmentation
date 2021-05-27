import os
import numpy as np
import matplotlib.pyplot as plt

def _pred_dir_make(no, save_dir):
    pred_dir = os.path.join(save_dir, "prediction", str(no))
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    
    return pred_dir


def plot_loss(losses, val_losses, save_dir):
    plt.plot(losses)
    plt.plot(val_losses)
    plt.title('loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xlim(0, 100)
#    plt.ylim(0.0, 0.03)
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(save_dir + "/loss.png")
    plt.close()


def plot_event(Y_true, Y_pred, no, save_dir, classes, ang_reso, label):
    pred_dir = _pred_dir_make(no, save_dir)

    plot_num = classes * ang_reso
    if ang_reso > 1 and classes == 1:
        ylabel = "angle"
    elif classes > 1 and ang_reso == 1:
        ylabel = "class index"

    if ang_reso == 1 or classes == 1:
        plt.pcolormesh((Y_true[no][0].T))
        plt.title("truth")
        plt.xlabel("time")
        plt.ylabel(ylabel)
        plt.clim(0, 1)
        plt.colorbar()
        plt.savefig(pred_dir + "true.png")
        plt.close()
        
        plt.pcolormesh((Y_pred[no][0].T))
        plt.title("prediction")
        plt.xlabel("time")
        plt.ylabel(ylabel)
        plt.clim(0, 1)
        plt.colorbar()
        plt.savefig(pred_dir + "pred.png")    
        plt.close()
    
    else: # SELD        
        Y_true_total = np.zeros(Y_true[0][0].shape)
        Y_pred_total = np.zeros(Y_pred[0][0].shape)
        for i in range(classes):
            if Y_true[no][i].max() > 0:
                
                plt.pcolormesh((Y_true[no][i]))
                plt.title(label.index[i] + "_truth")
                plt.xlabel("time")
                plt.ylabel('angle')
                plt.clim(0, 1)
                plt.colorbar()
                plt.savefig(pred_dir + label.index[i] + "_true.png")
                plt.close()
    
                plt.pcolormesh((Y_pred[no][i]))
                plt.title(label.index[i] + "_prediction")
                plt.xlabel("time")
                plt.ylabel('angle')
                plt.clim(0, 1)
                plt.colorbar()
                plt.savefig(pred_dir + label.index[i] + "_pred.png")
                plt.close()
                
            Y_true_total += (Y_true[no][i] > 0.45) * (i + 4)
            Y_pred_total += (Y_pred[no][i] > 0.45) * (i + 4)
        
        plt.pcolormesh((Y_true_total), cmap="gist_ncar")
        plt.title(str(no) + "__color_truth")
        plt.xlabel("time")
        plt.ylabel('angle')
        plt.clim(0, Y_true_total.max())
        plt.savefig(pred_dir + "color_truth.png")
        plt.close()
    
        plt.pcolormesh((Y_pred_total), cmap="gist_ncar")
        plt.title(str(no) + "__color_prediction")
        plt.xlabel("time")
        plt.ylabel('angle')
        plt.clim(0, Y_true_total.max())
        plt.savefig(pred_dir + "color_pred.png")
        plt.close()

        
def plot_mixture_stft(X, no, save_dir):
    pred_dir = _pred_dir_make(no, save_dir)

    plt.pcolormesh((X[no][0]))
    plt.xlabel("time")
    plt.ylabel('frequency')
    plt.clim(0, 1)
    plt.colorbar()
    plt.savefig(pred_dir + "/mixture.png")
    plt.close()


def plot_class_stft(Y_true, Y_pred, no, save_dir, classes, ang_reso, label):
    plot_num = classes * ang_reso
    if ang_reso > 1:
        ylabel = "angle"
    else:
        ylabel = "frequency"

    pred_dir = _pred_dir_make(no, save_dir)
        
    Y_true_total = np.zeros(Y_true[0][0].shape)
    Y_pred_total = np.zeros(Y_pred[0][0].shape)
    for i in range(plot_num):
        if Y_true[no][i].max() > 0: 
            plt.pcolormesh((Y_true[no][i]))
            if ang_reso == 1:
                plt.title(label.index[i] + "_truth")
            else:
                plt.title(label.index[i // ang_reso] + "_" + str((360 // ang_reso) * (i % ang_reso)) + "deg_truth")
            plt.xlabel("time")
            plt.ylabel(ylabel)
            plt.clim(0, 1)
            plt.colorbar()
            if ang_reso == 1:
                plt.savefig(pred_dir + "/" + label.index[i] + "_true.png")
            else:
                plt.savefig(pred_dir + "/" + label.index[i // ang_reso] + "_" + str((360 // ang_reso) * (i % ang_reso)) + "deg_true.png")
            plt.close()

            plt.pcolormesh((Y_pred[no][i]))
            if ang_reso == 1:
                plt.title(label.index[i] + "_prediction")
            else:
                plt.title(label.index[i // ang_reso] + "_" + str((360 // ang_reso) * (i % ang_reso)) + "deg_prediction")                
            plt.xlabel("time")
            plt.ylabel(ylabel)
            plt.clim(0, 1)
            plt.colorbar()
            if ang_reso == 1:
                plt.savefig(pred_dir + "/" + label.index[i] + "_pred.png")
            else:
                plt.savefig(pred_dir + "/" + label.index[i // ang_reso] + "_" + str((360 // ang_reso) * (i % ang_reso)) + "deg_pred.png")         
            plt.close()
            
        Y_true_total += (Y_true[no][i] > 0.45) * (i + 4)
        Y_pred_total += (Y_pred[no][i] > 0.45) * (i + 4)
    
    plt.pcolormesh((Y_true_total), cmap="gist_ncar")
    plt.title(str(no) + "_truth")
    plt.xlabel("time")
    plt.ylabel(ylabel)
    plt.clim(0, Y_true_total.max())
    plt.savefig(pred_dir + "/color_truth.png")
    plt.close()

    plt.pcolormesh((Y_pred_total), cmap="gist_ncar")
    plt.title(str(no) + "_prediction")
    plt.xlabel("time")
    plt.ylabel(ylabel)
    plt.clim(0, Y_true_total.max())
    plt.savefig(pred_dir + "/color_pred.png")
    plt.close()
