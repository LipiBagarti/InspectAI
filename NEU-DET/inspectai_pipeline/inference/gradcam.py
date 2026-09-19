import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms, models
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(r"c:\Users\lipib\Downloads\hackathon\NEU-DET\inspectai_pipeline")
MODELS_DIR = ROOT / "models"
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        target = output[0][class_idx]
        target.backward()
        
        # Global average pooling on gradients
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # Weighted combination of activations
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam) # ReLU for positive influence
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        return cam.cpu().detach().numpy(), class_idx, float(output[0][class_idx].item())

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(CLASSES))
    
    cls_path = MODELS_DIR / "mobilenetv2_best.pth"
    if cls_path.exists():
        model.load_state_dict(torch.load(cls_path, map_location=device))
    else:
        print("Warning: weights not found, using untrained model")
        
    model = model.to(device)
    model.eval()
    return model, device

def apply_colormap_on_image(org_im, activation, colormap_name='jet'):
    color_map = plt.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    
    # Heatmap to RGB
    heatmap = cv2.cvtColor(np.uint8(255 * no_trans_heatmap), cv2.COLOR_RGBA2RGB)
    
    # Resize to match original image
    heatmap = cv2.resize(heatmap, (org_im.shape[1], org_im.shape[0]))
    
    # Overlay
    cam = np.float32(heatmap) / 255 + np.float32(org_im) / 255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def generate_gradcam(img_path, output_path, model, device):
    img = Image.open(img_path).convert('RGB')
    org_img = np.array(img)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # Target last conv layer in MobileNetV2
    target_layer = model.features[-1]
    
    grad_cam = GradCAM(model, target_layer)
    cam, class_idx, raw_score = grad_cam(input_tensor)
    
    pred_class = CLASSES[class_idx]
    
    # Overlay
    overlay = apply_colormap_on_image(org_img, cam)
    
    # Add text
    cv2.putText(overlay, f"{pred_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return pred_class

if __name__ == "__main__":
    model, device = load_model()
    # Test
    test_img = ROOT / "dataset" / "test" / "crazing" / os.listdir(ROOT / "dataset" / "test" / "crazing")[0]
    out_dir = ROOT / "results" / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_gradcam(test_img, str(out_dir / "sample_crazing.jpg"), model, device)
    print("Grad-CAM generation successful.")
