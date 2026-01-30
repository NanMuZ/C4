# Less is More: Clustered Cross-Covariance Control for Offline RL

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Conference](https://img.shields.io/badge/ICLR-2026-blue.svg)](https://openreview.net/forum?id=drOy5wi6Qq)
[![Python-3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

### 📢 News

- **[Feb 2026]** Our paper has been accepted for presentation at ICLR 2026! 🚀

------

### 📝 Abstract

**C4** (Clustered Cross-Covariance Control) addresses the fundamental challenge of distributional shift in offline reinforcement learning. By identifying and mitigating harmful TD cross-covariance through **partitioned buffer sampling** and **gradient-based corrective penalties**, C4 significantly stabilizes value estimation. Our method demonstrates state-of-the-art performance across D4RL locomotion and kitchen tasks, achieving up to 30% improvement in returns over baseline methods.

### 👥 Authors

Nan Qiao , Shuning Wang, Yongheng Deng, and Ju Ren.

------

### 🛠️ Installation & Quick Start

1. **Clone and Install:**

   Bash

   ```
   git clone https://github.com/your-username/C4.git
   cd C4
   pip install -e .
   ```

2. **Run Experiments:**

   Bash

   ```
   python run/C4.py 
   ```

------

### 🙏 Acknowledgments

Our codebase is built upon **[OfflineRL-Kit](https://github.com/yihaosun1124/OfflineRL-Kit)**. We sincerely thank the authors for their well-structured and efficient framework.

C4 is released under the **MIT License**, consistent with the base repository.

### 📖 Citation

Code snippet

```
@inproceedings{qiao2026less,
  title={Less is More: Clustered Cross-Covariance Control for Offline RL},
  author={Qiao, Nan and Yue, Sheng and Wang, Shuning and Deng, Yongheng and Ren, Ju},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

------

