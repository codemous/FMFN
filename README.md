This repository provides the **PyTorch implementation** of the paper:
**Learning Fine-Grained Representation with Token-Level Alignment for Multimodal Sentiment Analysis**


> This implementation is reorganized based on the official code repositories of **[ALMT](https://github.com/AIM3-RUC/ALMT)** and **[LNLN](https://github.com/AIM3-RUC/LNLN)**. 
- We **do not truncate audio and visual sequences to length 50**, unlike the original paper.
- Instead of applying different learning rate decay strategies for BERT and other parameters, we adopt the **learning rate schedule strategy from the LNLN  repository**.

> We extend the previous frameworks to support both **complete modality inputs** and **incomplete modality inputs** settings, enabling robust evaluation under missing modality scenarios.


You can download the processed datasets from the **[MMSA](https://github.com/thuiar/MMSA)** repository.

We sincerely thank the authors of the following works for their open-source contributions: 
[**ALMT**](https://github.com/Haoyu-ha/ALMT), 
[**LNLN**](https://github.com/Haoyu-ha/LNLN), and 
[**MMSA**](https://github.com/thuiar/MMSA).

This repository is a reorganized implementation. If you encounter any bugs, please feel free to contact us.

