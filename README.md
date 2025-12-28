# Genomic WSI MIL Framework

**A PyTorch-based framework for Whole Slide Image analysis with Multiple Instance Learning, Attention Fusion, and Contrastive Learning.**

--------------------------------------------------------------------------------
##  QUICK START & OVERVIEW
--------------------------------------------------------------------------------

###  1. Environment & Core Dependencies
- **Python**: 3.8
- **PyTorch**: **1.18.0** 
- **CUDA**: **12.1** 
- **GPU**: **NVIDIA GeForce RTX 5090** 
# Main Python Packages:

--torch (>=1.10.0)
--torchvision
--numpy
--scikit-learn
--pandas
--tensorboard
--argparse
--json

### 2. Run the Model
```bash
# Basic training with default parameters for CRC-DX project
python run_MIL_model.py --proj CRC-DX --num_epochs 80 --plr 1e-4  --smoothing 0.25

