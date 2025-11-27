# DDI-ESPredictor: A Triple-Stage Web-based Predictor for Drug-Drug Interaction Events and Severity Assessment​​
Our model incorporates a unique capsule network architecture with cross-attention fusion mechanisms and introduces innovations in multi-view feature integration and dynamic loss weighting, demonstrating superior performance in predicting drug-drug interaction existence, event types, and severity levels across comprehensive pharmaceutical datasets. 
- **Access CapMolPred at**: (blank)

- [Environment Setup](#install)
- [Quick Start](#quick-start)
- [User Interface](#user-interface)

![DDI-ESPredictor Architecture](Model.jpg)

<a name="install"></a>
## Environment Setup
DDI-ESPredictor requires several dependencies. Here are the main requirements:

### Core Dependencies
- Python 3.10.X
- PyTorch 2.4.0 

### Key Python Packages
- numpy==1.24.0
- scipy==1.15.3
- scikit-learn==1.7.0
- pandas==2.3.1
- matplotlib==3.10.3

### Pip install Packages
- rdkit
- torch-geometric
- torch-scatter


---

## Quick Start

**Notice**  
Make sure you have 3 folder (data, model, output) before running the script.
args --model-name and --drug-descriptor are needed
- data/: Curated raw SMILES corpora employed for model training, hold-out testing, and prospective prediction.
- model/: Pre-trained feature extractors and associated checkpoints utilized in the molecular representation pipeline.
- output/: End-to-end antimicrobial-activity predictions, encompassing extracted molecular descriptors, serialized model weights, and comprehensive training/validation metrics exported as CSV artifacts.
  
**Feature extraction**  
```bash
python drugclassification.py --input-path data/sample.csv --model-name bert_chemmolefusion_capsule --drug-descriptor fusion
```
**Train and Test**  
```bash
python drugclassification.py  --input-path data/ecoli_predict.csv --model-name bert_chemmolefusion_capsule --drug-descripter fusion --train --batch-size 64 -e 100 -dp data -g 0 -sl 1024
```
**Prediction**  
```bash
python drugclassification.py  --input-path data/sample_predict.csv --model-name bert_chemmolefusion_capsule --drug-descripter fusion --predict
```

---

## User Interface(To be implemented)
DDI-ESPredictor offers a user-friendly interface with the following key components:
1. **Input Section**: Submit sequences in SMILES format (manual entry or file upload).
3. **Results Visualization**:
   - **Table View**: Detailed tables of predicted activity and corresponding probabilities.
   - **Structure Displays**: 2D chemicals structure visualizations.
3. **Export Options**: Download results in tab-delimited or XML formats.
---

## Citation
DOI: [to be published].

---

## Contact
For technical support or inquiries, please contact:  
Rui Meng  
School of Data Science, The Chinese University of Hong Kong, Shenzhen.  
Email: ruimeng@cuhk.edu.cn
