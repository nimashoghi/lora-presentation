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


<!--
- In the LLM context, the efficiency concern of fine-tuning is even more pronounced.
-->

---

# Existing Solutions

- Freezing the backbone
- [Adapter Layers](http://arxiv.org/abs/1902.00751)
- [Prefix Tuning](https://arxiv.org/abs/2101.00190) and [Prompt Tuning](https://arxiv.org/abs/2104.08691)

---

# Prefix/Prompt Tuning

- Inspired by the success of prompting in GPT-3
- A prefix/prompt is added to the input sequence before feeding it to the LLM
- The prefix/prompt is trained on the task-specific data
- The LLM is fine-tuned on the task-specific data
- Only the prefix/prompt is trained

---

# [Adapter Layers: Parameter-Efficient Transfer Learning for NLP](http://arxiv.org/abs/1902.00751)

- Adapters are new modules added **between** layers of a pre-trained network.
- Adapters are usually **much smaller** than the pre-trained network.
- Adapters are initialized such that, at the beginning of fine-tuning, they **do not change** the pre-trained network's behavior.
- During fine-tuning, only the adapters are trained.

## Main Features
1. Small number of parameters
2. Near-identity initialization

---

# [Adapter Layers: Parameter-Efficient Transfer Learning for NLP](http://arxiv.org/abs/1902.00751)

![Adapter Layer](/adapter.png)

- Bottleneck architecture

---

# [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255)

- An objective function's **intrinsic dimensionality** describes the minimum dimension needed to solve the optimization problem it defines to some precision level.
1. **Connection of Intrinsic Dimensionality and Number of Parameters**: The paper shows that as the number of parameters in a pre-training model increases, the intrinsic dimension (a measure of problem complexity) actually decreases. This means that larger models are more efficient at compressing the information needed to solve a given task.

2. **Connection of Pre-Training and Intrinsic Dimensionality**: The paper proposes that pre-training implicitly reduces the intrinsic dimension. In other words, it reduces the minimal description length needed to fine-tune a task within the framework of the pre-trained model. This is understood as pre-training providing a compression framework for learning NLP tasks.

3. **Connection of Intrinsic Dimensionality and Generalization**: The paper shows that lower intrinsic dimension correlates with better generalization (lower relative generalization gap). This is backed theoretically by applying compression based generalization bounds on the measured intrinsic dimensions, showing that generalization bounds can grow on the order of the intrinsic dimension, not the model's parameter count. This suggests that models with lower intrinsic dimensions are more capable of generalizing across tasks, regardless of their total parameter counts.


<!--
- In the context of pretrained language models, measuring intrinsic dimensional will tell us how many free parameters are required to closely approximate the optimization problem that is solved while fine-tuning for each end task.
- For example, we will show that 200 parameters (randomly projected back into the full parameter space) are enough to represent the problem of tuning a RoBERTa model to within 90% of the performance of the full model.

Papers to read
- [Measuring the Intrinsic Dimension of Objective Landscapes](https://arxiv.org/abs/1804.08838)
- [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255)
 -->
