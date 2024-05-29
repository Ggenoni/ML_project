import os
import torch

def save_model(model, filename="trained_model.pt"):
    print("Saving models ...")
    save_folder = "trained_model"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_path = os.path.join(save_folder, filename)
    torch.save(model.state_dict(), save_path)
    print("The model has been successfully saved!")
