# Less is More: Clustered Cross-Covariance Control for Offline RL (C4)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Conference](https://img.shields.io/badge/ICLR-2026-blue.svg)](https://openreview.net/forum?id=drOy5wi6Qq)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

---

## 📢 News
- **[Jan 2026]** Our paper has been accepted at **ICLR 2026**. 🚀

---

## 📝 Abstract
**C4** (Clustered Cross-Covariance Control) addresses the fundamental challenge of distributional shift in offline reinforcement learning.  
By identifying and mitigating harmful TD cross-covariance through **partitioned buffer sampling** and **gradient-based corrective penalties**, C4 significantly stabilizes value estimation.  
Our method demonstrates state-of-the-art performance across D4RL locomotion and kitchen tasks, achieving up to **30%** improvement in returns over baseline methods.

---

## 👥 Authors
- Nan Qiao (nanqiao.ai@gmail.com)
- Sheng Yue
- Shuning Wang
- Yongheng Deng
- Ju Ren

---

## 🛠️ Installation

### 1) Clone and Install
```bash
git clone https://github.com/NanMuZ/C4.git
cd C4
pip install -e .
```



## 🚀 Quick Start
```bash
python run/C4.py 
```



🙏 Acknowledgments

This repository is built upon OfflineRL-Kit:
https://github.com/yihaosun1124/OfflineRL-Kit

We sincerely thank the authors for their clean and efficient framework.

C4 is released under the MIT License, consistent with the base repository.

📖 Citation

If you find this work useful, please consider citing:
```
@inproceedings{qiao2026less,
  title     = {Less is More: Clustered Cross-Covariance Control for Offline RL},
  author    = {Qiao, Nan and Yue, Sheng and Wang, Shuning and Deng, Yongheng and Ren, Ju},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```
