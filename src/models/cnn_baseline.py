import timm
from src.utils.seed import get_device

def build_cnn_baseline(model_name, num_classes):
    '''
        model_name: resnet18.a1_in1k, efficientnet_b0.ra_in1k, mobilenetv3_large_100.ra_in1k
        All of the models have been pretrained on ImageNet-1k
    '''
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    
    return model.to(get_device())