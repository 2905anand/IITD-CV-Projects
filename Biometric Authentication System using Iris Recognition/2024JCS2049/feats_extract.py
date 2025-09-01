import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F


class IrisNet(nn.Module):
    """
    Purpose:
        A basic Convolutional Neural Network (CNN) for iris classification.
        Includes several convolution + ReLU + max-pool blocks, then fully-connected layers.
    
    Inputs (constructor):
        num_cls (int): The number of classes (unique labels) in the dataset.
    
    Outputs:
        forward(x): Returns the output logits for input tensor x.
    """
    def __init__(self, num_cls):
        super(IrisNet, self).__init__()
        self.convBlock = nn.Sequential(
            nn.Conv2d(1, 6, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(6, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.fcBlock = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_cls)
        )

    def forward(self, x):
        x = self.convBlock(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fcBlock(x)
        return x

class Trainer:
    """
    Purpose:
        A helper class to train and evaluate the IrisNet model. Handles:
        - Model initialization on CPU/GPU,
        - Loss and optimizer setup,
        - Training loop,
        - Periodic evaluation and saving.
    
    Inputs (constructor):
        model (nn.Module): The neural network model (IrisNet).
        dev (torch.device): The device for model and data (CPU or GPU).
    
    Functions:
        train_model(train_dl, test_dl, epochs):
            Runs the training loop, with periodic evaluation and saving.
        evaluate(test_dl):
            Evaluates the model's accuracy on a test dataset.
        save(path):
            Saves the model state_dict to disk.
        load(path):
            Loads a model state_dict from disk.
    """
    def __init__(self, model, dev):
        self.dev = dev
        self.model = model.to(self.dev)
        self.loss_fn = nn.CrossEntropyLoss()
        self.opt = optim.Adagrad(self.model.parameters(), lr=0.01)

    def train_model(self, train_dl, test_dl, epochs=50):
        """
        Purpose:
            Train the model for a specified number of epochs. Every 10 epochs,
            evaluate on the test set and save the model checkpoint.
        
        Inputs:
            train_dl (DataLoader): Dataloader for training samples.
            test_dl (DataLoader): Dataloader for test/validation samples.
            epochs (int): Number of epochs to train.
        
        Outputs:
            None (prints training loss and evaluation accuracy).
        """
        self.model.train()
        for ep in range(epochs):
            tot_loss = 0.0
            # Evaluate & save every 10 epochs
            if ep % 10 == 0:
                self.evaluate(test_dl)
                self.save(f"iris_epoch_{ep}.pth")

            for ims, lbs in train_dl:
                # Insert channel dimension if missing
                if ims.dim() == 3:
                    ims = ims.unsqueeze(1)

                ims = ims.to(self.dev)
                lbs = lbs.to(self.dev)

                self.opt.zero_grad()
                outs = self.model(ims)
                loss = self.loss_fn(outs, lbs)
                loss.backward()
                self.opt.step()
                tot_loss += loss.item() * ims.size(0)

            print(f"Epoch {ep+1}/{epochs} Loss: {tot_loss / len(train_dl.dataset):.4f}")

    def evaluate(self, test_dl):
        """
        Purpose:
            Evaluate the current model on a test set and print the accuracy.
        
        Inputs:
            test_dl (DataLoader): Dataloader for test/validation samples.
        
        Outputs:
            acc (float): The computed accuracy over the test dataset.
        """
        self.model.eval()
        tot = 0
        corr = 0
        with torch.no_grad():
            for ims, lbs in test_dl:
                if ims.dim() == 3:
                    ims = ims.unsqueeze(1)

                ims = ims.to(self.dev)
                lbs = lbs.to(self.dev)
                outs = self.model(ims)
                _, preds = torch.max(outs, 1)
                tot += lbs.size(0)
                corr += (preds == lbs).sum().item()
        acc = corr / tot
        print(f"Test Accuracy: {acc:.4f}")
        self.model.train()
        return acc

    def save(self, path):
        """
        Purpose:
            Save the model's state dictionary to the specified path.
        
        Inputs:
            path (str): File path to save the model state_dict.
        
        Outputs:
            None (prints a confirmation message).
        """
        torch.save(self.model.state_dict(), path)
        print(f"Saved model to {path}")

    def load(self, path):
        """
        Purpose:
            Load the model's state dictionary from a specified path.
        
        Inputs:
            path (str): File path from which to load the model.
        
        Outputs:
            None (prints a confirmation message).
        """
        self.model.load_state_dict(torch.load(path, map_location=self.dev))
        self.model.to(self.dev)
        print(f"Loaded model from {path}")


class IrisData(Dataset):
    """
    Purpose:
        A custom Dataset class for loading iris images along with their labels.
    
    Inputs (constructor):
        file_list (list of tuples): Each tuple is (image_path, label).
        transform (callable, optional): A function/transform that takes in a PIL image
                                        and returns a transformed version (e.g., for data augmentation).
    
    Outputs:
        __len__ returns the number of samples in the dataset.
        __getitem__(idx) returns the sample (image, label) at the specified index.
    """
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, lbl = self.file_list[idx]
        im = Image.open(path).convert('L')  # Convert image to grayscale
        if self.transform:
            im = self.transform(im)
        return im, lbl

def get_label(fname):
    """
    Purpose:
        Extract a label from the given filename by splitting on underscores ('_').
        Example filename: "001_01_L.png"
    
    Inputs:
        fname (str): The filename (not the full path) of an image.
    
    Outputs:
        (str): The extracted label. For example, if the filename is "001_01_L.png", 
               it might return "001_L" if there's a third underscore.
    """
    parts = fname.split('_')
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[2]}"
    return fname

def list_data(dirPath):
    """
    Purpose:
        Traverse the given directory, collecting all image file paths and their labels.
    
    Inputs:
        dirPath (str): The root directory containing images.
    
    Outputs:
        flist (list of tuples): Each tuple is (image_path, label) for all found images.
    """
    flist = []
    for root, dirs, files in os.walk(dirPath):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full = os.path.join(root, file)
                lbl = get_label(file)
                flist.append((full, lbl))
    return flist


def run_training():
    """
    Purpose:
        The main training workflow:
          1. Set up image transformations (resize, flip, normalization).
          2. Create dataset from the folder of normalized images.
          3. Split dataset into train and test subsets.
          4. Create and train the IrisNet model.
          5. Finally, evaluate and save the model checkpoint.
    """
    trans = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    folder = "Local/normalized_dataset/Local/edited_normalized_img"
    data_list = list_data(folder)
    print(f"Found {len(data_list)} images.")

    ds = IrisData(data_list, transform=trans)
    tr_size = int(0.8 * len(ds))
    te_size = len(ds) - tr_size
    tr_ds, te_ds = torch.utils.data.random_split(ds, [tr_size, te_size])

    tr_dl = DataLoader(tr_ds, batch_size=16, shuffle=True)
    te_dl = DataLoader(te_ds, batch_size=16, shuffle=False)

    # Determine the number of unique labels
    uniq = sorted(list(set([lbl for _, lbl in data_list])))
    num_cls = len(uniq)
    print("Classes:", num_cls)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IrisNet(num_cls)
    trainer = Trainer(model, dev)

    trainer.train_model(tr_dl, te_dl, epochs=500)
    trainer.evaluate(te_dl)
    trainer.save("iris_convnet_splited.pth")

def get_feats(model, dl, dev):
    """
    Purpose:
        Extract feature vectors from the CNN's convolutional block for
        all images in a given dataloader.
    
    Inputs:
        model (nn.Module): A trained IrisNet model.
        dl (DataLoader): A DataLoader for images (no augmentation needed here).
        dev (torch.device): Device on which to compute (CPU or GPU).
    
    Outputs:
        feats (torch.Tensor): A concatenated tensor of feature vectors of shape (N, D),
                              where N is number of samples and D is feature dimension.
        lbls (list): A list of labels corresponding to each feature vector.
    """
    model.eval()
    feats = []
    lbls = []
    with torch.no_grad():
        for ims, lb in dl:
            if ims.dim() == 3:
                ims = ims.unsqueeze(1)
            ims = ims.to(dev)
            # Forward pass only up to the convolutional block
            f = model.convBlock(ims)
            f = f.view(f.size(0), -1)
            feats.append(f.cpu())
            lbls.extend(lb)
    feats = torch.cat(feats, dim=0)
    return feats, lbls

def run_feature_extraction(model_path):
    """
    Purpose:
        Given a trained IrisNet model, load it and extract CNN-based features
        for the entire dataset of normalized images. Save these feature tensors
        and corresponding labels to disk for later use.
    
    Inputs:
        model_path (str): Path to the saved model checkpoint (state_dict).
    
    Outputs:
        None (prints progress, saves 'db_features.pt' and 'db_labels.pt').
    """
    trans = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    folder = "Local/normalized_dataset/Local/edited_normalized_img"
    data_list = list_data(folder)
    print(f"Found {len(data_list)} images.")

    ds = IrisData(data_list, transform=trans)
    dl = DataLoader(ds, batch_size=32, shuffle=False)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_cls = len(set([lbl for _, lbl in data_list]))

    # Instantiate model and load weights
    model = IrisNet(num_cls)
    model.load_state_dict(torch.load(model_path, map_location=dev))
    model.to(dev)

    # Extract features from the dataset
    feats, lbls = get_feats(model, dl, dev)
    print("Feature shape:", feats.shape)

    # Save the feature vectors and labels
    torch.save(feats, "Local/metadata/db_features.pt")
    torch.save(lbls, "Local/metadata/db_labels.pt")
    print("Features and labels saved.")

def main():
    """
    Purpose:
        Orchestration:
         1. Train the CNN (run_training).
         2. Extract CNN features for the entire dataset (run_feature_extraction).
    """
    run_training()
    run_feature_extraction("iris_convnet_splited.pth")

if __name__ == "__main__":
    main()
