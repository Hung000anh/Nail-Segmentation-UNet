# Nail Segmentation using U-Net

## Overview
This project implements a U-Net model for nail segmentation using PyTorch. The dataset is sourced from Kaggle and consists of nail images with corresponding segmentation masks.

## Dataset
The dataset is downloaded using `kagglehub`:

```python
import kagglehub

data_folder = kagglehub.dataset_download("vpapenko/nails-segmentation")
print("Path to dataset files:", data_folder)
```

The dataset is then split into training and validation sets.

## Model Architecture
The U-Net model consists of an encoder, bottleneck, and decoder:

### Encoder
```python
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = TwoConvLayers(in_channels, out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.block(x)
        y = self.max_pool(x)
        return y, x
```

### Decoder
```python
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = TwoConvLayers(in_channels, out_channels)

    def forward(self, x, y):
        x = self.transpose(x)
        u = torch.cat([x, y], dim=1)
        u = self.block(u)
        return u
```

### U-Net Model
```python
class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.enc_block1 = Encoder(in_channels, 64)
        self.enc_block2 = Encoder(64, 128)
        self.enc_block3 = Encoder(128, 256)
        self.enc_block4 = Encoder(256, 512)
        self.bottleneck = TwoConvLayers(512, 1024)
        self.dec_block1 = Decoder(1024, 512)
        self.dec_block2 = Decoder(512, 256)
        self.dec_block3 = Decoder(256, 128)
        self.dec_block4 = Decoder(128, 64)
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x):
        x, y1 = self.enc_block1(x)
        x, y2 = self.enc_block2(x)
        x, y3 = self.enc_block3(x)
        x, y4 = self.enc_block4(x)
        x = self.bottleneck(x)
        x = self.dec_block1(x, y4)
        x = self.dec_block2(x, y3)
        x = self.dec_block3(x, y2)
        x = self.dec_block4(x, y1)
        return self.out(x)
```

## Training
The model is trained using Binary Cross Entropy and Dice Loss:
```python
criterion_1 = nn.BCEWithLogitsLoss()
criterion_2 = DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
```

During training, the best model is saved:
```python
if average_val_loss < best_loss:
    best_loss = average_val_loss
    torch.save(model.state_dict(), 'best_loss_unet.pt')
```

## Evaluation
The accuracy of the model is computed on the validation set:
```python
def calculate_accuracy(model, data_loader, device, threshold=0.5):
    model.eval()
    correct_pixels = 0
    total_pixels = 0
    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            predicted_masks = (torch.sigmoid(outputs) > threshold).float()
            correct_pixels += (predicted_masks == masks).sum().item()
            total_pixels += masks.numel()
    accuracy = correct_pixels / total_pixels
    return accuracy
```

## Prediction
A helper function is provided for making predictions:
```python
def predict(image_path, transformations, device):
    image = Image.open(image_path).convert('RGB')
    image = transformations(image).unsqueeze(0).to(device)
    preds = model(image).squeeze(0).to(device)
    mask_np = (nn.functional.sigmoid(preds.permute(1, 2, 0)) > 0.5).cpu().numpy().astype(np.uint8)
    return mask_np
```

## Results
Sample predictions are visualized using:
```python
for image_name in val_images:
    image_path = os.path.join(VAL_DIR, image_name)
    image, mask = predict(image_path, test_transformations, device)
    visualize_data(image, mask)
```

## Saving and Loading the Model
The trained model can be saved and loaded as follows:
```python
torch.save(model.state_dict(), 'best_loss_unet.pt')
model.load_state_dict(torch.load('best_loss_unet.pt'))
model.eval()
```

## Dependencies
- Python 3.8+
- PyTorch
- torchvision
- numpy
- matplotlib
- PIL
- kagglehub
