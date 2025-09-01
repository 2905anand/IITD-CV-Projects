import os
import time
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class DoubleConv(nn.Module):
    
    def _init_(self, in_channels, out_channels):
        super(DoubleConv, self)._init_()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    
    def _init_(self, n_channels, n_classes):
        super(UNet, self)._init_()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)
        logits = self.outc(x)
        return logits

class SegmentationDataset(Dataset):
    
    def _init_(self, images_dir, masks_dir, image_size=(1024, 1024)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size

        self.image_files = sorted(os.listdir(images_dir))
        self.mask_files = sorted(os.listdir(masks_dir))
        
    def _len_(self):
        return len(self.image_files)
    
    def _getitem_(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image = image.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)
        image = np.array(image).astype(np.float32) / 255.0
        mask = np.array(mask).astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  
        image = torch.tensor(image)
        mask = torch.tensor(mask).unsqueeze(0)   
        return image, mask

def train_model(train_images_dir, train_masks_dir, num_epochs, model_save_path, device):
    dataset = SegmentationDataset(train_images_dir, train_masks_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    model = UNet(n_channels=3, n_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.2f}s")
        if (epoch + 1) % 10 == 0:
            snapshot_path = f"{model_save_path}_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), snapshot_path)
            print(f"Saved snapshot: {snapshot_path}")
    torch.save(model.state_dict(), model_save_path)
    print(f"Training complete. Model saved to {model_save_path}")

def segment_image(image_path, model, device):
    image = cv2.imread(image_path)
    orig_h, orig_w = image.shape[:2]
    img_resized = cv2.resize(image, (480, 480), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    img_tensor = torch.tensor(img_transposed).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        output = torch.sigmoid(output)
        pred_mask = (output > 0.5).float()
    
    pred_mask_np = pred_mask.squeeze().cpu().numpy()
    pred_mask_np = (pred_mask_np * 255).astype(np.uint8)
    pred_mask_resized = cv2.resize(pred_mask_np, (orig_w, orig_h))
    segmented_image = cv2.bitwise_and(image, image, mask=pred_mask_resized)
    
    coords = cv2.findNonZero(pred_mask_resized)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped_image = segmented_image[y:y+h, x:x+w]
    else:
        cropped_image = segmented_image
    
    final_resized = cv2.resize(cropped_image, (480, 480), interpolation=cv2.INTER_LINEAR)
    return final_resized, pred_mask_resized, segmented_image

def segmentation_directory(src_folder, dest_folder, model, device, saving_all=False):
    for file in os.listdir(src_folder):
        src_path = os.path.join(src_folder, file)
        dest_path_resized = os.path.join(dest_folder, file.replace(".", "-resizedmask."))
        dest_path_binary = os.path.join(dest_folder, file.replace(".", "-binarymask."))
        dest_path_seg = os.path.join(dest_folder, file.replace(".", "-seg."))
        
        image = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if image is not None:
            resized_image, binary_mask, segmented_mask = segment_image(src_path, model, device)
            cv2.imwrite(dest_path_resized, resized_image)
            if saving_all:
                cv2.imwrite(dest_path_binary, binary_mask)
                cv2.imwrite(dest_path_seg, segmented_mask)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "segment"])
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.mode == "train":
        train_images_dir = "train_images"
        train_masks_dir = "train_masks"
        num_epochs = 50
        model_save_path = "unet_model.pth"
        train_model(train_images_dir, train_masks_dir, num_epochs, model_save_path, device)
    
    elif args.mode == "segment":
        input_dir = "segment_input"
        output_dir = "segment_output"
        model_path = "unet_model.pth"
        
        model = UNet(n_channels=3, n_classes=1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        segmentation_directory(input_dir, output_dir, model, device, saving_all=True)
        print(f"Segmented images saved in {output_dir}")

if __name__ == "__main__":
    main()