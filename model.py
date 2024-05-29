import torch
import torch.nn as nn
from clip import clip
import torchvision.models as models

class BottleneckCLIP(nn.Module):
    def __init__(self, output_dim, bias=False):
        super(BottleneckCLIP, self).__init__()
        model, _ = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
        in_features = 512  # Change to match ViT-B/32
        out_features = 1024

        # Take the visual encoder of CLIP and convert it to 32 bit
        self.encoder = model.visual.float()

        # Add a bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(in_features, in_features // 2, bias=bias),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 2, in_features // 2, bias=bias),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 2, out_features, bias=bias),
            nn.ReLU(inplace=True),
            nn.Linear(out_features, output_dim)
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
        x = self.bottleneck(x)
        return x

def get_CLIP_model(output_dim):
    model = BottleneckCLIP(output_dim)
    print("CLIP model ready to go!")
    return model.to("cuda" if torch.cuda.is_available() else "cpu")


def get_Vgg19_model(output_dim):
    model = models.vgg19(pretrained=True)

    # Freeze training for all "features" layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Modify the classifier
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, output_dim)

    print("Vgg19 model ready to go!")
    return model.to("cuda" if torch.cuda.is_available() else "cpu")


def get_ResNet50_model(output_dim, dropout_p=0.5):
    model = models.resnet50(pretrained=True)

    # Freeze training for all layers
    for param in model.parameters():
        param.requires_grad = False

    # Modify the fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(num_features, output_dim)
    )
    
    print("ResNet50 model ready to go!")
    return model.to("cuda" if torch.cuda.is_available() else "cpu")