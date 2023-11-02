---
theme: academic
transition: slide-left
title: "LoRA: Low-Rank Adaptation of Large Language Models"
---

# [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

## Presented by Nima Shoghi

---

# Background: Large Language Models

---

# Background: Transfer-Learning

<img src="/transfer.png" class="h-64 mx-auto" />

<div class="grid grid-cols-2">
<div>

## Feature-based Transfer
- Pre-trained model is used as a **feature extractor**
- Features are fed to a task-specific model
- Only the task-specific model is trained
</div>

<div>

## Fine-Tuning

- Pre-trained model is used as an **initialization**
- **All params** are trained on task-specific data
- Better results compared to feature-based transfer, but more expensive.
</div>
</div>


---

# Existing Solutions

- Freezing the backbone
<!-- - [Adapter Layers](http://arxiv.org/abs/1902.00751) -->
- [Prefix Tuning](https://arxiv.org/abs/2101.00190)


---

# [Adapter Layers: Parameter-Efficient Transfer Learning for NLP](http://arxiv.org/abs/1902.00751)

- Adapters are new modules added **between** layers of a pre-trained network.

---

# [Adapter Layers: Parameter-Efficient Transfer Learning for NLP](http://arxiv.org/abs/1902.00751)

![Adapter Layer](/adapter.png)

---

# Low Intrinsic Dimensionality of Pre-Trained Models

## [Measuring the Intrinsic Dimension of Objective Landscapes](https://arxiv.org/abs/1804.08838)
## [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255)

---
