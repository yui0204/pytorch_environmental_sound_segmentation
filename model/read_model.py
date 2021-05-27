from model import FCN8s, UNet, CRNN, Deeplabv3plus, CRNN_SED

def read_model(model_name, n_classes, angular_resolution, input_dim):
    if model_name == "FCN8s":
        model = FCN8s(n_classes=n_classes, input_dim=input_dim)
    elif model_name == "UNet":
        model = UNet(n_classes=n_classes, angular_resolution=angular_resolution, input_dim=input_dim)
    elif model_name == "CRNN":
        model = CRNN(n_classes=n_classes, angular_resolution=angular_resolution, input_dim=input_dim)
    elif model_name == "Deeplabv3plus":
        model = Deeplabv3plus(nInputChannels=input_dim, n_classes=n_classes, angular_resolution=angular_resolution, os=16, _print=False)

    elif model_name == "CRNN_SED":
        model = CRNN_SED(n_classes=n_classes, angular_resolution=angular_resolution, input_dim=input_dim)
    
    #print(model)
        
    return model.cuda()

