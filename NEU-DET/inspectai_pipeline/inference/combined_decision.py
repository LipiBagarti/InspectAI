import torch
import numpy as np
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import json

ROOT = Path(r"c:\Users\lipib\Downloads\hackathon\NEU-DET\inspectai_pipeline")
MODELS_DIR = ROOT / "models"

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

# Import Autoencoder definition
import sys
sys.path.append(str(ROOT))
from training.train_autoencoder import SimpleAutoencoder

class InspectAIDecisionEngine:
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load MobileNetV2
        self.classifier = models.mobilenet_v2(weights=None)
        self.classifier.classifier[1] = torch.nn.Linear(self.classifier.last_channel, len(CLASSES))
        cls_path = MODELS_DIR / "mobilenetv2_best.pth"
        if cls_path.exists():
            self.classifier.load_state_dict(torch.load(cls_path, map_location=self.device))
        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()
        
        # Load Autoencoder
        self.autoencoder = SimpleAutoencoder()
        ae_path = MODELS_DIR / "autoencoder_best.pth"
        if ae_path.exists():
            self.autoencoder.load_state_dict(torch.load(ae_path, map_location=self.device))
        self.autoencoder = self.autoencoder.to(self.device)
        self.autoencoder.eval()
        
        # Load Thresholds
        self.t_ok = 0.01
        thresh_path = MODELS_DIR / "ae_threshold.txt"
        if thresh_path.exists():
            with open(thresh_path, "r") as f:
                self.t_ok = float(f.read().strip())
                
        self.t_defect = 0.65 # Confidence threshold for classifier
        
        self.cls_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.ae_transforms = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def infer(self, image_path_or_pil):
        if isinstance(image_path_or_pil, (str, Path)):
            img = Image.open(image_path_or_pil).convert("RGB")
        else:
            img = image_path_or_pil
            
        # 1. Anomaly Detection
        ae_input = self.ae_transforms(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            ae_output = self.autoencoder(ae_input)
            mse = torch.mean((ae_output - ae_input)**2).item()
            
        # 2. Classification
        cls_input = self.cls_transforms(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.classifier(cls_input)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)
            conf = conf.item()
            pred_class = CLASSES[pred_idx.item()]
            
        # 3. Decision Logic
        if mse < self.t_ok:
            decision = "OK"
        elif conf > self.t_defect:
            decision = "DEFECT"
        else:
            decision = "CHECK"
            
        return {
            "decision": decision,
            "anomaly_score": float(mse),
            "anomaly_threshold": float(self.t_ok),
            "pred_class": pred_class,
            "confidence": float(conf),
            "class_probs": {CLASSES[i]: float(probs[0][i]) for i in range(len(CLASSES))}
        }

if __name__ == "__main__":
    # Test on a dummy image
    img = Image.new('RGB', (224, 224), color = 'red')
    engine = InspectAIDecisionEngine()
    res = engine.infer(img)
    print(json.dumps(res, indent=2))
