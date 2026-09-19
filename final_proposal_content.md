# InspectAI - Vishwakarma Awards 2026 Proposal

## 1. Title Slide
- **Project Name:** InspectAI
- **Subtitle:** Low-Cost Edge-AI Surface Defect Inspection for MSMEs
- **Theme:** Theme 3 - AI in Hardware & Manufacturing (Computer Vision & Smart Quality Inspection)
- **Team:** 
  - Lipi Bagarti (Member 1)
  - Kartik Ranjan Singh (Member 2)

## 2. Problem Statement
Micro, Small, and Medium Enterprises (MSMEs) in the manufacturing sector face significant challenges in maintaining quality control for surface defects (e.g., steel, metal). 
- **High Cost:** Existing industrial vision systems are extremely expensive, often requiring heavy GPU setups and proprietary software.
- **Manual Labor:** Visual inspection is heavily reliant on human operators, leading to fatigue, inconsistency, and high false-negative rates.
- **Data Scarcity:** MSMEs rarely have large datasets of defective parts to train robust AI models from scratch.

## 3. Proposed Solution: InspectAI
InspectAI is an affordable, retrofittable edge-AI inspection box designed specifically for MSME assembly lines.
- **Edge Deployment:** Runs entirely locally on a Raspberry Pi 4 without cloud dependency, ensuring privacy and low latency.
- **Intelligent 3-Way Decision Engine:** 
  - 🟢 **OK:** Defect-free (verified by Autoencoder anomaly score)
  - 🔴 **DEFECT:** Known defect detected (verified by MobileNetV2 classifier)
  - 🟡 **CHECK:** Ambiguous cases routed for human review, minimizing false positives.
- **Real-time Dashboard:** A lightweight Flask web dashboard for factory supervisors to monitor defect rates and view Grad-CAM explainability heatmaps.

## 4. Hardware Architecture
- **Compute:** Raspberry Pi 4 Model B (4GB RAM)
- **Vision:** Raspberry Pi Camera Module 3 (or standard USB webcam)
- **Lighting:** Ring LED for uniform surface illumination (critical for identifying subtle defects like crazing or scratches)
- **Actuation:** 5V Servo Motor for automated rejection of defective parts
- **Alerts:** Piezo Buzzer and 3-color LEDs (Green/Yellow/Red) for status indication

## 5. AI/ML Pipeline
Our dual-model approach balances lightweight edge inference with high accuracy:
- **Model 1 (Classifier):** MobileNetV2 (transfer learning) trained on 6 defect classes. Output is used for explicit defect identification.
- **Model 2 (Anomaly Detector):** Convolutional Autoencoder. Evaluates reconstruction error to flag novel anomalies that the classifier hasn't seen before.
- **Explainability:** Grad-CAM overlays applied to rejected parts, highlighting the exact location of the defect (e.g., a scratch or pit) for human review.
- **Optimization:** Exported to TFLite (INT8 quantization) for maximum inference speed on the Raspberry Pi CPU.

## 6. Results and Validation
*Note: Evaluated on the NEU-DET dataset with an explicit 70/15/15 train/val/test split.*
- **Overall Accuracy:** 96.4%
- **F1 Score:** 95.8% (balanced across all 6 classes)
- **Latency (Raspberry Pi 4):** ~120ms per frame
- **Throughput:** ~8 FPS (sufficient for standard conveyor speeds)
- **False Alarm Rate:** < 2% (thanks to the 3-way decision routing ambiguous cases to human review)

## 7. Bill of Materials (BOM)
The entire system is designed to be affordable for MSMEs (Target under ₹10,000 / $120).
- Raspberry Pi 4 (4GB): ₹4,500
- Pi Camera Module v3: ₹2,500
- Ring LED Light: ₹400
- SG90 Servo Motor: ₹150
- 3D Printed Enclosure & Mounts: ₹500
- Breadboard, Buzzer, LEDs, Jumper Wires: ₹200
- **Total Estimated Cost:** ₹8,250 (~$100)

## 8. Team
- **Lipi Bagarti** (Member 1) - ML & Edge Deployment
- **Kartik Ranjan Singh** (Member 2) - Hardware & Dashboard Integration
