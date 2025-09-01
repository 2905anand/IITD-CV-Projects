import os
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2

from preprocess import preprocess_image, do_normalization
from feats_extract import IrisNet, IrisData, list_data
from torch.utils.data import DataLoader

def compute_cos_sim(query_feat, db_feats):
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
    q_norm = F.normalize(query_feat, p=2, dim=0)
    db_norm = F.normalize(db_feats, p=2, dim=1)
    sim = torch.mm(q_norm.unsqueeze(0), db_norm.t()).squeeze(0)
    return sim

def get_rank(query_feat, db_feats, metric='cosine'):
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
    if metric == 'cosine':
        scores = compute_cos_sim(query_feat, db_feats)
        ranking = torch.argsort(scores, descending=True)
    elif metric == 'euclidean':
        scores = torch.norm(db_feats - query_feat.unsqueeze(0), dim=1)
        ranking = torch.argsort(scores, descending=False)
    return ranking, scores

def fuse_highest(left, right):
    """
    Purpose:
        Fuse two ranking lists by selecting the best (highest) rank per ID 
        from either list. If the rank is the same, select the higher similarity score.

    Inputs:
        left (list of tuples): Each tuple is (rank, ID, similarity).
        right (list of tuples): Each tuple is (rank, ID, similarity).

    Outputs:
        out (list of tuples): A combined list of (ID, best_rank, best_similarity), 
                              sorted by best_rank ascending, then similarity descending.
    """
    ld = {item[1]: (item[0], item[2]) for item in left}
    rd = {item[1]: (item[0], item[2]) for item in right}
    ids = set(ld.keys()).union(rd.keys())
    out = []
    for idd in ids:
        l_r, l_s = ld.get(idd, (6, 0.0))
        r_r, r_s = rd.get(idd, (6, 0.0))
        if l_r < r_r:
            br, bs = l_r, l_s
        elif r_r < l_r:
            br, bs = r_r, r_s
        else:
            # If ranks are equal, pick the higher similarity
            br, bs = l_r, l_s if l_s >= r_s else r_s
        out.append((idd, br, bs))
    # Sort by (best_rank asc, similarity desc, fallback to left rank asc for tie-break)
    out.sort(key=lambda x: (x[1], -x[2], ld.get(x[0], (6, 0.0))[0]))
    return out

def fuse_borda(left, right):
    """
    Purpose:
        Perform a Borda count fusion by summing the ranks from two lists 
        and using the average of their similarities for tie-breaking.

    Inputs:
        left (list of tuples): [(rank, ID, similarity), ...].
        right (list of tuples): [(rank, ID, similarity), ...].

    Outputs:
        list of tuples: [(ID, sum_of_ranks, avg_similarity), ...] sorted by sum_of_ranks ascending,
                        then avg_similarity descending, then the left rank ascending.
    """
    ld = {item[1]: (item[0], item[2]) for item in left}
    rd = {item[1]: (item[0], item[2]) for item in right}
    ids = set(ld.keys()).union(rd.keys())
    out = []
    for idd in ids:
        l_r, l_s = ld.get(idd, (6, 0.0))
        r_r, r_s = rd.get(idd, (6, 0.0))
        tot = l_r + r_r
        avg = (l_s + r_s) / 2.0
        out.append((idd, tot, avg, ld.get(idd, (6, 0.0))[0]))
    # Sort by (sum_of_ranks asc, avg_similarity desc, left_rank asc)
    out.sort(key=lambda x: (x[1], -x[2], x[3]))
    return [(idd, tot, avg) for idd, tot, avg, lr in out]

def fuse_logistic(left, right, w_l=0.5, w_r=0.5):
    """
    Purpose:
        Fuse two ranking lists by computing a weighted sum of their ranks, 
        and averaging their similarity scores.

    Inputs:
        left (list of tuples): [(rank, ID, similarity), ...].
        right (list of tuples): [(rank, ID, similarity), ...].
        w_l (float): Weight given to ranks from the left list.
        w_r (float): Weight given to ranks from the right list.

    Outputs:
        list of tuples: [(ID, fused_rank, avg_similarity), ...] sorted by fused_rank ascending,
                        then avg_similarity descending, then the original left rank ascending.
    """
    ld = {item[1]: (item[0], item[2]) for item in left}
    rd = {item[1]: (item[0], item[2]) for item in right}
    ids = set(ld.keys()).union(rd.keys())
    out = []
    for idd in ids:
        l_r, l_s = ld.get(idd, (6, 0.0))
        r_r, r_s = rd.get(idd, (6, 0.0))
        fused = w_l * l_r + w_r * r_r
        avg = (l_s + r_s) / 2.0
        out.append((idd, fused, avg, ld.get(idd, (6, 0.0))[0]))
    # Sort by (fused_rank asc, avg_similarity desc, left_rank asc)
    out.sort(key=lambda x: (x[1], -x[2], x[3]))
    return [(idd, fused, avg) for idd, fused, avg, lr in out]

def integrated_main():
    """
    Purpose:
        A demonstration of an end-to-end iris recognition workflow:
         1) Preprocess two sample images (left and right eyes).
         2) Normalize the images.
         3) Load a trained model and query the database for each image.
         4) Rank the top 5 matches for each.
         5) Perform fusion of results by three methods (highest rank, Borda, logistic).
         6) Print and compare final fused results.
    """
    left_img = "Local/original_dataset/021_01_L.png"
    right_img = "Local/original_dataset/021_06_R.png"

    # Ensure test directory exists
    os.makedirs("Local/test", exist_ok=True)

    # Preprocess and normalize
    preprocess_image(left_img, "Local/test/left_preproc.png")
    preprocess_image(right_img, "Local/test/right_preproc.png")
    do_normalization("Local/test/left_preproc.png", "Local/test/left_norm.png")
    do_normalization("Local/test/right_preproc.png", "Local/test/right_norm.png")

    # If there's no specific model path, default to "Model.pth"
    trained_model = "Model.pth" if not "" else "Model.pth"

    iris_rec = IrisRecognition()
    labels_left, scores_left = iris_rec.query(
        model_path=trained_model,
        query_image_path="Local/test/left_norm.png"
    )
    labels_right, scores_right = iris_rec.query(
        model_path=trained_model,
        query_image_path="Local/test/right_norm.png"
    )

    # Create a rank list from 1 to 5 for each eye's top 5 results
    r1 = list(range(1, 6))
    r2 = list(range(1, 6))

    # Pack results into the form (rank, label, score)
    L1 = [(r1[i], labels_left[i], scores_left[i]) for i in range(5)]
    L2 = [(r2[i], labels_right[i], scores_right[i]) for i in range(5)]

    # Perform fusion using highest rank, Borda count, and logistic weighting
    fusion_hr = fuse_highest(L1, L2)
    fusion_bc = fuse_borda(L1, L2)
    fusion_lr = fuse_logistic(L1, L2, w_l=0.5, w_r=0.5)

    # Print results
    print("Highest Rank Fusion:")
    for i, (idd, br, bs) in enumerate(fusion_hr, start=1):
        print(f"Rank {i}: {idd}, {br}, {bs:.3f}")

    print("\nBorda Count Fusion:")
    for i, (idd, tot, avg) in enumerate(fusion_bc, start=1):
        print(f"Rank {i}: {idd}, {tot}, {avg:.3f}")

    print("\nLogistic Regression Fusion:")
    for i, (idd, fused, avg) in enumerate(fusion_lr, start=1):
        print(f"Rank {i}: {idd}, {fused:.3f}, {avg:.3f}")

    # Example final identified ID from each fusion method (strip last 2 chars just as a demo)
    print(f"PERSON IDENTIFIED AS {fusion_hr[0][0][:-2]} | {fusion_bc[0][0][:-2]} | {fusion_lr[0][0][:-2]}.")

class IrisRecognition:
    """
    Purpose:
        A class that provides methods for directory-based preprocessing or querying
        a trained iris recognition model for a single image.
    """
    def __init__(self):
        pass

    def execute_directory(self, src, dst, func):
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
        for f in os.listdir(src):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                sp = os.path.join(src, f)
                dp = os.path.join(dst, f)
                func(sp, dp)

    def query(self, model_path, query_image_path,
              db_feature_path="Local/metadata/db_features.pt",
              db_label_path="Local/metadata/db_labels.pt"):
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
        # Load database features and labels
        db_feats = torch.load(db_feature_path)
        db_lbls = torch.load(db_label_path)

        # Define transformation for the query image
        trans = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Dynamically import IrisNet model to avoid circular dependency
        from feats_extract import IrisNet
        num_cls = len(set(db_lbls))

        # Load model
        net = IrisNet(num_cls)
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.to(device)

        # Process query image
        q_img = Image.open(query_image_path).convert("L")
        q_tensor = trans(q_img).unsqueeze(0).to(device)

        # Extract features
        net.eval()
        with torch.no_grad():
            q_feat = net.convBlock(q_tensor)
            q_feat = q_feat.view(q_feat.size(0), -1).squeeze(0).cpu()

        # Rank and retrieve top 5
        ranking, scores = get_rank(q_feat, db_feats, metric='cosine')
        top5 = ranking[:5]
        out_lbls = [db_lbls[i] for i in top5]
        top5_scores = scores[top5]
        return out_lbls, top5_scores

def integrated_main_wrapper():
    """
    Purpose:
        A wrapper function that calls integrated_main(), providing a clear entry point
        for running the entire demonstration.
    """
    integrated_main()

if __name__ == "__main__":
    integrated_main_wrapper()
