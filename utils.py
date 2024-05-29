import os
import torch

def save_model(model, filename):
    print("Saving models ...")
    save_folder = "trained_model"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_path = os.path.join(save_folder, filename)
    torch.save(model.state_dict(), save_path)
    print("The model has been successfully saved!")



def load_model(model, filename):
    print("Loading model ...")
    save_folder = "trained_model"
    save_path = os.path.join(save_folder, filename)
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        model.eval()  # Mettere il modello in modalità valutazione
        print("The model has been successfully loaded!")
    else:
        print("Model file not found!")


#use azure machine:
#scp -r -P 5012 "D:/DATA SCIENCE/MACHINE LEARNING/CLIP_project" disi@lab-b19fb86e-17c2-41af-aa77-c4a6adf27da4.westeurope.cloudapp.azure.com:/home/disi/

#pushing commits to github (terms in uppercase must be changed):
# cd to project folder
# git add .
# git commit -m "message"
# if needed: git remote add origin https://github.com/USER_NAME/REPO_NAME.git
# git push origin BRANCH_NAME


