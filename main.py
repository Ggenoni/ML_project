import torch
import torch.optim as optim
import torch.nn as nn
import yaml
import argparse
from dataset import get_data_loaders
from model import get_model
from train import train

def main(args):
    torch.manual_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if config["logger"]["wandb"]:
        import wandb
        wandb.login()
        wandb.init(project="UDA", 
                   config=config, 
                   name=f"{args.run_name}")
        
    batch_size_train = config["data"]["batch_size_train"]
    batch_size_test = config["data"]["batch_size_test"]
    num_workers = config["data"]["num_workers"]
    num_epochs = config["training"]["num_epochs"]

    # Get data loaders
    train_loader, val_loader = get_data_loaders(batch_size_train, batch_size_test, num_workers)

    # Get model
    model = get_model(output_dim=102)  # 102 classes in Flowers102 dataset

    # Freeze the CLIP model parameters
    for param in clip_model.parameters():
        param.requires_grad = False

    # Set up the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.bottleneck.parameters(), lr=1e-4)

    # Train the model
    train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, logger=config["logger"]["wandb"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--config", required=True, type=str, help="Path to the configuration file")
    parser.add_argument("--run_name", required=False, type=str, help="Name of the run")
    args = parser.parse_args()
    main(args)
