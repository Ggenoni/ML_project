import torch
import torch.nn as nn
from clip import clip

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

def get_model(output_dim):
    model = BottleneckCLIP(output_dim)
    return model.to("cuda" if torch.cuda.is_available() else "cpu")
