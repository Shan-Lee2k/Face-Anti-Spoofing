
import torch
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small
import torchvision
import torch.nn as nn




pretrained_weights_dict = {
    'MN3_CelebA': "C:/Users/PC/Documents/GitHub/Face-Anti-Spoofing/at_learner_core/at_learner_core/models/architectures/MN3_antispoof.pth",
    'ImageNet_V2_Large': "C:/Users/PC/Documents/GitHub/Face-Anti-Spoofing/at_learner_core/at_learner_core/models/architectures/mobilenet_v3_large-5c1a4163.pth",
    'ImageNet_V1_Small':  "C:/Users/PC/Documents/GitHub/Face-Anti-Spoofing/at_learner_core/at_learner_core/models/architectures/mobilenet_v3_small-047dcff4.pth",
}
last_channel = {
    'large' : 960,
    'small' :576,
}


def check_frozen_layers(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable parameter: {name} (Shape: {param.shape})")
        else:
            print(f"Frozen parameter: {name} (Shape: {param.shape})")  
    return 
    
                
class ConvBNActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation_layer):
        super(ConvBNActivation, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=1, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation_layer()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
class MobileNetV3_Custom(nn.Module):
    def __init__(self, pretrained = None, num_classes=2, mode = 'large' or 'small', out_feature = 512, drop_rate = 0.3):
        super(MobileNetV3_Custom, self).__init__()
        self.mode = mode
        self.feature_size = out_feature
        self.drop_out_rate = drop_rate
        # Load the pre-trained MobileNetV3 model
        if self.mode == 'large':
            self.features = mobilenet_v3_large(pretrained=False).features
        else: self.features = mobilenet_v3_small(pretrained=False).features
            
        # Load pre-trained weights if provided
        if pretrained is not None:
            print(f"=====Create MobileNetv3 with pre-trained successfully=====")
            pretrained_weights = torch.load(pretrained, map_location='cpu')
            if 'state_dict' in pretrained_weights.keys():
                state_dict = pretrained_weights['state_dict']
                new_dict = {}
                list_new_key = list(self.state_dict().keys())
                i = 0
                for k,v in state_dict.items():
                    if k.startswith('features'):
                        if len(v.size()) == 2:
                            new_dict[list_new_key[i]] = v.unsqueeze(-1).unsqueeze(-1)
                        else: new_dict[list_new_key[i]] = v
                        i+=1
                    else:
                        break
                
                # Load state_dict and make sure that dimension is suitable (Feature 1 to 15) Mobile 
                self.load_state_dict(new_dict, strict= False)
            else: self.load_state_dict(pretrained_weights,strict=False)
            # Freeze all feature extractor parameters
            for param in list(self.features.parameters())[:-16]:  #-42 -16
                param.requires_grad = False  # Freeze all parameters
        
        #self.dropout1 = nn.Dropout(p=self.drop_out_rate)
        # Define a new ConvBNActivation layer with desired output of 256
        self.custom_layer = ConvBNActivation(
            in_channels = last_channel[self.mode],
            out_channels= self.feature_size, # Desired output channels [512] for Large
            kernel_size=1,    # 1x1 convolution
            activation_layer=nn.Hardswish  # Hardswish activation
        )

        self.dropout2 = nn.Dropout(p=self.drop_out_rate)

    def forward(self, x):
        # Pass input through the base MobileNetV3 layers
        x = self.features(x)
        #x = self.dropout1(x)
        # Pass through the custom layer
        x = self.custom_layer(x)
        x = self.dropout2(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1,1))
        x = x.view(-1, self.feature_size)
        #x = self.classifier(x)
        return x

    

if __name__ == '__main__':
    #model = torchvision.models.mobilenet_v3_large(pretrained= torch.load("MN3_antispoof.pth", map_location= 'cpu') )
    pre_state_dict = torch.load(pretrained_weights_dict['MN3_CelebA'], map_location='cpu')
    state_dict = pre_state_dict['state_dict']
 
    #print(pre_state_dict['features.0.0.weight'])
    #print(pre_state_dict['state_dict']['features.0.0.weight'])
    #print(len(pre_state_dict['state_dict']))
    model = MobileNetV3_Custom(pretrained= pretrained_weights_dict['ImageNet_V1_Small'],mode='small',out_feature= 256)
    model_mb3 = torchvision.models.mobilenet_v3_small()
    #print(list(model_mb3.state_dict().keys()))
    #print(state_dict.keys())
    #print(list(model.features.parameters())[:20])
    
    # new_dict = {}
    # list_new_key = list(model_mb3.state_dict().keys())
    # i = 0
    # for k,v in state_dict.items():
    #     if k.startswith('features'):
    #         if len(v.size()) == 2: 
    #             new_dict[list_new_key[i]] = v.unsqueeze(-1).unsqueeze(-1)
    #         else: new_dict[list_new_key[i]] = v
    #         i+=1
    #     else:
    #         break
    # model_mb3.load_state_dict(new_dict,strict=False)
    # print(model.state_dict()['features.15.block.3.1.weight'] == state_dict['features.15.conv.8.weight'])
    check_frozen_layers(model)
    #print(model_mb3.eval())       
        
        
    
    # pre_dict = torch.load(pretrained_weights_dict['ImageNet_V2'], map_location= 'cpu')
    # for k,v in pre_dict.items():
    #     print(k)
    #print(model.state_dict()['features.0.0.weight'])

    # #print(model.eval())
    # pre_state_dict = torch.load("MN3_antispoof.pth", map_location='cpu')
    # # print(state_dict['state_dict']['features.0.0.weight'])
    # model.load_state_dict(pre_state_dict,strict=False) 
    # # print(f"Weight: {model.features[0][0].weight}")
    # #print(model.eval())

    # #print(set(pre_state_dict.keys()))
    # #print(model.eval())
    # #print(model.base_model[0][0].weight)
    # #print(model.state_dict().keys())
    # mapped_state_dict = {k.replace("conv", "block"): v for k, v in pre_state_dict['state_dict'].items()}
    # # for layer_name, param in model.state_dict().items():
    # #     print(f"{layer_name}: {param.shape}")
    # #for i in range(303): # Until end layer 15 in MobileNetv3
    # new_dict = {}
    # i = 0
    # while True:
    #     key_list = [k for k in model.state_dict().keys()]
    #     value_list = [v for k, v in pre_state_dict['state_dict'].items()]              
    #     if key_list[i].startswith("features"):
    #         new_dict[key_list[i]] = value_list[i]
    #         i += 1
    #     else:
    #         break
        
            
    # #for layer_name in model.state_dict().keys():
    #  #   print(layer_name)
    # # print(model.eval())
    # # Architecture of .ph
    # # print(type(pre_state_dict['state_dict']))
    # model.load_state_dict(new_dict,strict=False) 




    
