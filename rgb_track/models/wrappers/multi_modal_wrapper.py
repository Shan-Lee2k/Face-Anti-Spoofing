import torch.nn as nn
import torch
from at_learner_core.models.wrappers.losses import get_loss
from at_learner_core.models.wrappers.simple_classifier_wrapper import SimpleClassifierWrapper
from at_learner_core.models.architectures import get_backbone
from ..architectures.transformer import TransformerEncoder


class MultiModalWrapper(SimpleClassifierWrapper):
    def __init__(self, wrapper_config):
        super().__init__(wrapper_config)

    def _init_modules(self, wrapper_config):
        self.input_modalities = wrapper_config.input_modalities
        for modal_key in self.input_modalities:
            print(f"Modal_key: {modal_key}")
            if (modal_key == 'optical_flow') or (modal_key == 'optical_flow_start'):
                backbone, feature_size = get_backbone(wrapper_config.backbone,
                                                      pretrained=None,
                                                      get_feature_size=True,
                                                      in_channels=2)
            elif modal_key == 'random_static_image':
                backbone, feature_size_static = get_backbone(architecture_name=wrapper_config.backbone_static,
                                                      pretrained=wrapper_config.pretrained,
                                                      get_feature_size=True)
                
            else:
                backbone, feature_size = get_backbone(wrapper_config.backbone,
                                                      pretrained=None,
                                                      get_feature_size=True)

            setattr(self, 'backbone_' + modal_key, backbone)

        self.backbone_feature_size = feature_size
        self.pooling = nn.AdaptiveAvgPool2d((1, feature_size))
        self.pooling2 = nn.AdaptiveMaxPool2d((1, feature_size))
        self.pooling3 = nn.AdaptiveMaxPool2d((1, feature_size))

        self.classifier = nn.Sequential(
            nn.Linear(3 * feature_size, feature_size),
            nn.ReLU(inplace=True),
            nn.Linear(feature_size, wrapper_config.nclasses)
        )
        self.classifier = nn.Linear(3 * feature_size, wrapper_config.nclasses)

    def forward(self, x):
        B, C, W, H = x[self.input_modalities[0]].size()
        device = x[self.input_modalities[0]].device
        M = len(self.input_modalities)  # Number of modalities
        
        # Initialize an empty list to store features from each modality
        modality_features = []

        for idx, key in enumerate(self.input_modalities):
            # Get features from each modality's backbone
            feature_output = getattr(self, 'backbone_' + key)(x[key])

            # If the feature size is 512, reshape it from [B, 512] to [B, 2, 256]
            if feature_output.size(1) == 512:
                feature_output = feature_output.view(B, 2, 256)
                modality_features.append(feature_output)  # Add [B, 2, 256] to the list
            else:
                # If feature size is 256, add as [B, 1, 256]
                modality_features.append(feature_output.unsqueeze(1))

        # Concatenate all modality features along the second dimension
        features = torch.cat(modality_features, dim=1)  # Result: [B, 5, 256]
        #print(f"Concatenated features size: {features.size()}")  # Should print: [B, 5, 256]

        # Apply pooling operations
        features1 = self.pooling(features)
        features2 = self.pooling2(features)
        features3 = self.pooling3(-features)
        
        # Concatenate pooled features across the last dimension
        features = torch.cat([features1, features2, features3], axis=2)
        features = features.squeeze()
        
        # Classification
        output = self.classifier(features)
        sigmoid_output = torch.sigmoid(output)
    
        if isinstance(self.loss, nn.modules.loss.CrossEntropyLoss):
            x['target'] = x['target'].squeeze()

        output_dict = {
            'output': sigmoid_output.detach().cpu().numpy(),
            'target': x['target'].detach().cpu().numpy()
        }
        
        # Add any additional metadata if available
        for k, v in x.items():
            if k not in ['data', 'target'] + self.input_modalities:
                output_dict[k] = v

        loss = self.loss(output, x['target'])
        return output_dict, loss



if __name__ == '__main__':
    print("Experiment on Multi_Modal_Wrapper!!!")
    model = MultiModalWrapper()
    torch.Tensor().size()
    print(model)
