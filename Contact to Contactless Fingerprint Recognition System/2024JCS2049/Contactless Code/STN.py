import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse

class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),   # Output: (8, 474, 474)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),          # Output: (8, 237, 237)
            nn.Conv2d(8, 10, kernel_size=5),    # Output: (10, 233, 233)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)           # Output: (10, 116, 116)
        )
        
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 116 * 116, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 6)
        )
        
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 116 * 116)
        theta = self.fc_loc(xs).view(-1, 2, 3)
        
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

def process_images(input_folder, output_folder, device):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((480, 480)),
        transforms.ToTensor()  
    ])

    
    to_pil = transforms.ToPILImage()

    
    stn = STN().to(device)
    stn.eval()

    
    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
    image_paths = []  # Initialize as an empty list.
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))

    if not image_paths:
        print(f"No images found in {input_folder}.")
        return

    
    with torch.no_grad():
        for img_path in image_paths:
            try:
                
                image = Image.open(img_path)
                
                input_tensor = transform(image)
                
                input_tensor = input_tensor.unsqueeze(0).to(device)
                
                
                output_tensor = stn(input_tensor)
                
                output_image = to_pil(output_tensor.squeeze(0).cpu())
                
                
                filename = os.path.basename(img_path)
                out_path = os.path.join(output_folder, filename)
                output_image.save(out_path)
                print(f"Processed and saved: {out_path}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Process segmented, enhanced, normalized, binarized contactless images with STN resized to 480x480"
    )
    parser.add_argument('--input_folder', type=str, default='unbatchedContactless',
                        help='Folder containing the input contactless images')
    parser.add_argument('--output_folder', type=str, default='stn_output',
                        help='Folder to save the processed images')
    args = parser.parse_args()

    # Use GPU if available; otherwise, use CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    process_images(args.input_folder, args.output_folder, device)

if __name__ == '__main__':
    main()
