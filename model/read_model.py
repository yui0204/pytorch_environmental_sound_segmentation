from model import FCN8s, UNet

def read_model(model_name, n_classes=75, input_dim=1):
    if model_name == "FCN8s":
        model = FCN8s(n_classes=n_classes, input_dim=input_dim)
    elif model_name == "UNet":
        model = UNet(n_classes=n_classes, input_dim=input_dim)
    
    print(model)
        
    return model.cuda()

