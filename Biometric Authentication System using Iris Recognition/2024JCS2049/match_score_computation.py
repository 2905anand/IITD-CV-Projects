import os
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2
from preprocessing import preprocess, normalization

class IrisRecognition:
    """
    Purpose:
        A class that provides methods for directory-based preprocessing or querying
        a trained iris recognition model for a single image.
    """
    def __init__(self):
        pass
    def execute_directory(self, src_path, des_path, func):
        """
        Purpose:
            Apply a specified function to all images in a directory, saving the outputs 
            in another directory.

        Inputs:
            src (str): Source directory containing images.
            dst (str): Destination directory to save processed images.
            func (callable): Function that takes (source_path, destination_path) arguments 
                             and performs some operation (e.g., preprocess_image).
        
        Outputs:
            None (though processed images are saved to 'dst').
        """
        for filename in os.listdir(src_path):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                src_file = os.path.join(src_path, filename)
                dest_file = os.path.join(des_path, filename)
                func(src_file, dest_file)
    def query(self, model_path, query_image_path, db_feature_path="Local/metadata/db_features.pt", db_label_path="Local/metadata/db_labels.pt"):
        """
        Purpose:
            Given a trained model and a query image, load precomputed database features
            and labels, compute the query's feature, and return the top 5 ranked matches.

        Inputs:
            model_path (str): Path to the trained CNN model state_dict.
            query_image_path (str): Path to the query image to be matched.
            db_feature_path (str): Path to the saved database features (torch.Tensor).
            db_label_path (str): Path to the saved database labels (list or tensor).

        Outputs:
            out_lbls (list): Top 5 labels corresponding to the database entries 
                             most similar to the query image.
            top5_scores (torch.Tensor): Similarity scores (cosine) for those top 5 entries.
        """
        db_features = torch.load(db_feature_path)
        db_labels = torch.load(db_label_path)
        transform = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        from train_split_database import IrisConvNet
        num_classes = len(set(db_labels))
        model = IrisConvNet(num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        query_image = Image.open(query_image_path).convert("L")
        query_tensor = transform(query_image).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            query_feat = model.features(query_tensor)
            query_feat = query_feat.view(query_feat.size(0), -1).squeeze(0).cpu()
        ranking, scores = get_ranking(query_feat, db_features, metric='cosine')
        top5_indices = ranking[:5]
        labels = [db_labels[i] for i in top5_indices]
        top5_scores = scores[top5_indices]
        return labels, top5_scores

def compute_cosine_similarity(query_feature, db_features):
    """
    Purpose:
        Compute the cosine similarity between a single query feature vector 
        and a set of database feature vectors.

    Inputs:
        query_feat (torch.Tensor): 1D tensor representing the feature vector of the query image.
        db_feats (torch.Tensor): 2D tensor (N x D) of feature vectors, 
                                 where N is the number of database entries 
                                 and D is the feature dimension.

    Outputs:
        sim (torch.Tensor): 1D tensor of length N containing the cosine similarity 
                            scores between the query feature and each database feature.
    """
    query_norm = F.normalize(query_feature, p=2, dim=0)
    db_norm = F.normalize(db_features, p=2, dim=1)
    similarity = torch.mm(query_norm.unsqueeze(0), db_norm.t()).squeeze(0)
    return similarity

def get_ranking(query_feature, db_features, metric='cosine'):
    """
    Purpose:
        Rank the database entries according to similarity (cosine) or distance (euclidean)
        to the query feature vector.

    Inputs:
        query_feat (torch.Tensor): 1D tensor (feature vector for the query image).
        db_feats (torch.Tensor): 2D tensor of shape (N, D) representing the database features.
        metric (str): Either 'cosine' or 'euclidean'. If 'cosine', higher is better. 
                      If 'euclidean', lower is better.

    Outputs:
        ranking (torch.Tensor): 1D tensor of indices sorted by highest similarity 
                                (cosine) or lowest distance (euclidean).
        scores (torch.Tensor): 1D tensor of similarity/distance scores corresponding to each database entry.
    """
    if metric=='cosine':
        scores = compute_cosine_similarity(query_feature, db_features)
        ranking = torch.argsort(scores, descending=True)
    elif metric=='euclidean':
        scores = torch.norm(db_features - query_feature.unsqueeze(0), dim=1)
        ranking = torch.argsort(scores, descending=False)
    return ranking, scores


def main():
    left_eye = "Local/original_dataset/021_01_L.png"
    
    os.makedirs("Local/test", exist_ok=True)
    iris_rec = IrisRecognition()
    iris_rec.execute_directory("Local/test", "Local/test", preprocess)
    preprocess(left_eye, "Local/test/lefteye_preprocessed.png")
    
    normalization("Local/test/lefteye_preprocessed.png", "Local/test/lefteye_normalized.png")
    
    trained_model_path = "iris_convnet_splited.pth"
    print("_------------ 1 ------------_")
    labels1, scores1 = iris_rec.query(model_path=trained_model_path, query_image_path="Local/test/lefteye_normalized.png")
    print("_------------ 2 ------------_")
    rank1 = list(range(1,6))
    print("Rank =", rank1)
    print("Labels =", labels1)
    print("Scores =", scores1)

if __name__ == "__main__":
    main()
