---
theme: academic
transition: slide-left
title: "LoRA: Low-Rank Adaptation of Large Language Models"
---

# [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

## Presented by Nima Shoghi

<br />

### https://nima.sh/lora-presentation

<img class="absolute top-0 left-0 m-2 b-4" src="https://api.qrserver.com/v1/create-qr-code/?size=125x125&data=https://nima.sh/lora-presentation&format=svg" />

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

# [Prefix Tuning](https://arxiv.org/abs/2101.00190) and [Prompt Tuning](https://arxiv.org/abs/2104.08691)

<div class="grid grid-cols-2">

<div>

- Inspired by the success of prompting and prompt engineering.
- The idea is to **add a prefix** to the input.
    - This prefix isn't a "prompt" in the sense of a natural language prompt. It's a **soft** prefix.
    - The prefix is **learned** during fine-tuning.
- The rest of the backbone model is **frozen**, i.e., not trained.
- In some ways, this approach tries to **learn a prompt** that is good for the task at hand.
</div>

<div>

![Prefix Tuning](/PrefixTuning.png)

</div>

</div>

<!--
A good way to think about the idea of a soft prefix is:
    - In Transformers, all words get converted to a vector representation (embedding).
    - The "prefix" that's tuned here is a vector that gets added to the embedding of the first word.
    - One way to think about this is that this soft prefix is some linear combination of all the words in the model's vocabulary, trained to be a good prefix.
-->


---

# [Adapter Layers: Parameter-Efficient Transfer Learning for NLP](http://arxiv.org/abs/1902.00751)

- Adapters are new modules added **between** layers of a pre-trained network.
- Adapters are usually **much smaller** than the pre-trained network.
- Adapters are initialized such that, at the beginning of fine-tuning, they **do not change** the pre-trained network's behavior.
- During fine-tuning, **only the adapters are trained**.

## Main Features
1. Small number of parameters
2. Near-identity initialization

---

# [Adapter Layers: Parameter-Efficient Transfer Learning for NLP](http://arxiv.org/abs/1902.00751)

![Adapter Layer](/adapter.png)

- Bottleneck architecture

---

# Follow Up Work: [The Versatile Language Model](https://arxiv.org/abs/2004.03829)

<div class="grid grid-cols-2">

<div>

#### Differences
- There is only one adapter layer per transformer block.
- In addition, a **segment embedding** is added to the input sequence.
</div>

<div>
<img src="/VersatileLM.png" alt="Versatile Language Model" class="h-100 mx-auto" />
</div>
</div>

<!-- TODO: Maybe skip this slide -->

---

# Follow Up Work: [AdapterFusion](https://arxiv.org/abs/2005.00247)

<div class="grid grid-cols-2">

<div>

- This tackles the **multi-task** fine-tuning scenario (i.e., we have multiple downstream tasks and we want to fine-tune on all of them at the same time).
- Divides the scenario into two steps:
    - **Knowledge Extraction**: Extracts task-specific knowledge from the pre-trained model.
    - **Knowledge Composition**: Composes the task-specific knowledge (across all tasks) back into the representation of the pre-trained model.

</div>

<div>
<img src="/AdapterFusion.png" alt="AdapterFusion" class="h-80 mx-auto" />
</div>
</div>
<!--
Example Scenario:
    - We have a pre-trained LLM, e.g., GPT3.
    - We want to fine-tune it on sentiment analysis, question answering, and summarization.
    - Each of these tasks has its own separate training data.
    - We fine-tune the LLM on all three tasks at the same time to get a multi-task model.
    - The motivation is that, hopefully, the knowledge from multiple source tasks will help the model learn better and thus improve the performance on each task.
-->


---

# [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255)

- An objective function's **intrinsic dimensionality** describes the minimum dimension needed to solve the optimization problem it defines to some precision level.
- In the context of pretrained language models, measuring intrinsic dimensional will tell us how many free parameters are required to closely approximate the optimization problem that is solved while fine-tuning for each end task.
- The paper demonstrates that:
    - As the number of parameters in a pre-training model increases, the intrinsic dimension decreases.
    - Pre-training implicitly reduces the intrinsic dimension.
    - Lower intrinsic dimension correlates with better generalization.


<!--
- In the context of pretrained language models, measuring intrinsic dimensional will tell us how many free parameters are required to closely approximate the optimization problem that is solved while fine-tuning for each end task.
- For example, we will show that 200 parameters (randomly projected back into the full parameter space) are enough to represent the problem of tuning a RoBERTa model to within 90% of the performance of the full model.

1. **Connection of Intrinsic Dimensionality and Number of Parameters**: The paper shows that as the number of parameters in a pre-training model increases, the intrinsic dimension (a measure of problem complexity) actually decreases. This means that larger models are more efficient at compressing the information needed to solve a given task.

2. **Connection of Pre-Training and Intrinsic Dimensionality**: The paper proposes that pre-training implicitly reduces the intrinsic dimension. In other words, it reduces the minimal description length needed to fine-tune a task within the framework of the pre-trained model. This is understood as pre-training providing a compression framework for learning NLP tasks.

3. **Connection of Intrinsic Dimensionality and Generalization**: The paper shows that lower intrinsic dimension correlates with better generalization (lower relative generalization gap). This is backed theoretically by applying compression based generalization bounds on the measured intrinsic dimensions, showing that generalization bounds can grow on the order of the intrinsic dimension, not the model's parameter count. This suggests that models with lower intrinsic dimensions are more capable of generalizing across tasks, regardless of their total parameter counts.

Papers to read
- [Measuring the Intrinsic Dimension of Objective Landscapes](https://arxiv.org/abs/1804.08838)
- [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255)
 -->

---

# [LoRA: Low-Rank Adaptation of LLMs](https://arxiv.org/abs/2106.09685)

<div class="grid grid-cols-12">

<div class="col-span-9">

- A method for **parameter-efficient** adaptation of LLMs, inspired by:
    1. The success of **adapter layers**.
    2. The idea that over-parametrized models reside on a **low intrinsic dimension**.
- Learns a **low-rank projection** of **updates** to the model's parameters.
- Concretely, for a given dense layer, $h = W_0 x$:
    - **Traditional Adapter**: $h = W_0 x + {\Delta W}_{\text{Adapt}} x$.
    - **LoRA**: $h = W_0 x + B A x$, (i.e., same as above, but ${\Delta W}_{\text{Adapt}} = B A$).
- $A$ uses a random Gaussian initialization, and $B$ is initialized to zero.
    - This means that at the beginning of training, ${\Delta W}_{\text{Adapt}} = B A = 0$ and the adapter layer has **near-identity** behavior.

</div>

<div class="col-span-3">
<img src="/LoRA.png" alt="LoRA" class="h-64 mx-auto" />

<small class="text-xs">

- $B$: Batch size; $S$: Sequence length; $H$: Hidden size, $r$: Low-rank projection size
- $x \in (B, S, H)$: Input
- $W_0 \in (H, H)$: Pre-trained weights
- ${\Delta W}_{\text{Adapt}} \in (H, H)$: Traditional adapter weights
- $B \in (H, r), A \in (r, H)$: LoRA low-rank projection weights

</small>

</div>
</div>


---

# "A Generalization of Full Fine-tuning"

---

# Results:
