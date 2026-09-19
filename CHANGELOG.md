# Changelog

## [1.0.0] - 2026-09-19

### Added
- **Data Pipeline:** Created a unified dataset with a robust 70/15/15 stratified train/val/test split (fixed seed for reproducibility).
- **Edge ML Pipeline:** Implemented MobileNetV2 (transfer learning) for lightweight supervised classification on Raspberry Pi.
- **Anomaly Detection:** Added a Convolutional Autoencoder architecture to model reconstruction error for novel anomalies.
- **3-Way Decision Engine:** Implemented inference logic to fuse classifier confidence and autoencoder anomaly score into three states: OK, DEFECT, or CHECK.
- **Explainability:** Added Grad-CAM script to generate heatmap overlays identifying the region of defects.
- **Hardware Integration:** Created a GPIO mock script to simulate Raspberry Pi hardware interactions (Servo, Buzzer, LEDs).
- **Dashboard:** Developed a lightweight Flask web dashboard tailored for Edge deployment with SQLite logging for inspection history.
- **Deployment:** Added ONNX/TFLite export script for optimized INT8 quantization deployment.
- **Proposal Content:** Generated final presentation text mapped to the 8 required sections for the Vishwakarma Awards.

### Removed
- Removed simulated evaluation functions that injected fake metrics based on ground-truth data.
- Deprecated Heavy Jetson/GPU dependent multi-model ensemble (YOLOv8 + Faster R-CNN + RT-DETR) in favor of the lightweight Edge architecture.
