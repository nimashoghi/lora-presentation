---
theme: academic
transition: slide-left
title: "LoRA: Low-Rank Adaptation of Large Language Models"
---

# [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

<br />

### https://nima.sh/lora-presentation

<img class="absolute top-0 left-0 m-2 b-4" src="https://api.qrserver.com/v1/create-qr-code/?size=125x125&data=https://nima.sh/lora-presentation&format=svg" />

---

# Agenda

1. Background
    1. Large Language Models (GPT)
    2. Transfer Learning
2. Previous Work
    1. Prefix Tuning
    2. Adapter Layers
    3. Intrinsic Dimensionality
3. LoRA
    1. Motivation
    2. Method
    3. Results
4. Conclusion

---

# BG: Autoregressive Language Modeling

<!-- ![Autoregressive Language Modeling](/AutoregressiveLM.gif) -->
<img src="/AutoregressiveLM.gif" alt="Autoregressive Language Modeling" />

<!--
For the purpose of this talk, I will focus on the autoregressive LMs and GPT architecture as the primary LLM, but the same ideas apply to other LLMs as well.

**An autoregressive model predicts future values based on past values.** The way the autoregressive generative NLP model actually work is that after each token is produced, that token is added to the sequence of inputs. And that new sequence becomes the input to the model in its next step. For example, the user initializes the input as “recite the first law $”, where “$” is a special delimiter token. GPT model will generate the text autoregressively, conditioned on the user input.
- Source: https://medium.com/@YanAIx/step-by-step-into-gpt-70bc4a5d8714

- You can **pre-train** a language model on a large corpus of text, such as Wikipedia. This results in a model that can generate text that is similar to the text in the corpus.
    - To do this, it must learn to **understand the structure of the language**.
- You can also **fine-tune** it on a specific task, such as sentiment analysis. This is called **transfer learning**.
 -->

---

# BG: Large Language Models & GPT

<!-- ![GPT Architecture](/GPT-Architecture.png) -->
<img src="/GPT-Architecture.png" alt="GPT Architecture" class="h-100 mx-auto" />

<!--
The GPT architecture takes the **decoder** part of the Transformer architecture, stacks it on top of itself, and adds a **language modeling head** on top of the last layer.

- We begin with the **word embedding layer**, which converts each word to a vector representation.
- The **positional encoding layer** adds information about the position of each word in the sequence.
- The **transformer blocks**, stacked on top of each other, are the main part of the model which update the representation of each word based on the other words in the sequence.
- The **final prediction layer** converts the output of the last transformer block to a **probability distribution** of the next word in the sequence.
-->

---

# BG: Embedding & Final Prediction Layers
<div class="grid grid-cols-2">

<div class="border-r-2 border-gray-300">

###### Embedding Layer

<!-- ![Word Embedding](/Word-Embedding.png) -->
<img src="/Word-Embedding.png" alt="Word Embedding" class="h-100 mx-auto" />
</div>

<div class="border-l-2 border-gray-300 pl-1">

###### Final Prediction Layer

<!-- ![Final Prediction Layer](/Final-Linear-Layer.png)-->
<img src="/Final-Linear-Layer.png" alt="Final Prediction Layer" class="h-100 mx-auto" />
</div>

</div>

<!-- display the softmax equation on the top right (absolute) -->
<div class="absolute top-16 right-4 m-2">

$$
\sigma(\vec{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}
$$

</div>

<!--

## Embedding Layer:
- The embedding layer converts each word to a vector representation. This is called a **word embedding**.
- It does this by converting each word to a **one-hot vector** and multiplying it by a **learned weight matrix**.
- The operation of multiplying a one-hot vector by a weight matrix is equivalent to **selecting a row** from the weight matrix.

## Final Prediction Layer:
- The final prediction layer is a **linear layer** that converts the output of the last transformer block to a vector of logits.
- The logits are then converted to probabilities using a **softmax**.
    - **Softmax** is a function that converts a vector of **logits** to a vector of probabilities that sum to 1.
- In other words, the final linear + softmax computes the **probability distribution** of the next word in the sequence.
-->

---

# BG: Transfer-Learning

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
- **Transfer learning** is a machine learning technique where a model pre-trained on one task is re-purposed on a second related task.
    - In our case, the pre-trained model is a language model (e.g., GPT-2), and the second task is a downstream task, such as sentiment analysis.
- **Feature-based transfer** is a transfer learning technique where the pre-trained model is used as a **feature extractor**. The features are then fed to a task-specific model, which is trained on the task-specific data.
    - Refer back to the GPT diagram, take the embeddings right before the final prediction layer, and use them to **train a whole new model**. This is the feature-based transfer approach.
- **Fine-tuning** is a transfer learning technique where the pre-trained model is used as an **initialization**. Then, all the parameters are trained on the task-specific data.
    - Refer back to the GPT diagram and **replace the final prediction layer with a new prediction layer**. This is the fine-tuning approach.

- Fine-tuning usually results in better performance compared to feature-based transfer, but it is more expensive.
- In the LLM context, the efficiency concern of fine-tuning is even more pronounced.
-->

---

# Previous Work

- Freezing the backbone and only fine-tuning the final prediction layer (**linear probing**)
- [Prefix Tuning](https://arxiv.org/abs/2101.00190) and [Prompt Tuning](https://arxiv.org/abs/2104.08691)
- [Adapter Layers](http://arxiv.org/abs/1902.00751)
- [LoRA](https://arxiv.org/abs/2106.09685)

<!--
- Linear probing, a technique that freezes the backbone and only fine-tunes the final prediction layer, is a popular approach to fine-tuning LLMs. However, it exhibits poor performance on many tasks.
-->

---

# [Prefix Tuning](https://arxiv.org/abs/2101.00190) and [Prompt Tuning](https://arxiv.org/abs/2104.08691)

<div class="grid grid-cols-2">

<div>

- Inspired by the success of prompting and prompt engineering.
- The idea is to **add a prefix** to the input.
    - This prefix isn't a "prompt" in the sense of a natural language prompt. It's a **soft** prefix.
    - The prefix is **learned** fring fine-tuning.
- The rest of the backbone model is **frozen**, i.e., not trained.
- In a way, this approach tries to **learn a prompt** that is optimal for the task at hand.
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

- Adapters are new mofles added **between** layers of a pre-trained network.
- Adapters are usually **much smaller** than the pre-trained network.
- Adapters are initialized such that, at the beginning of fine-tuning, they **do not change** the pre-trained network's behavior.
- Fring fine-tuning, **only the adapters are trained**.

<br />
<br />

## Main Features
1. Small number of parameters
2. Near-identity initialization

---

# [Adapter Layers: Parameter-Efficient Transfer Learning for NLP](http://arxiv.org/abs/1902.00751)

![Adapter Layer](/adapter.png)

- Bottleneck architecture

<!--
The original paper uses an MLP bottleneck architecture, but other architectures have been proposed since then.
-->

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

<!--
TODO: Maybe skip this slide

- Not that interesting of work.
- Can be thought of as a combination of adapter layers and prefix tuning.
-->

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

<div class="col-span-8">

- **Parameter-efficient** adaptation of LLMs, inspired by:
    1. The success of **adapter layers**.
    2. The idea that over-parametrized models reside on a **low intrinsic dimension**.
- Learns a **low-rank projection** of **updates** to the model's **weights** (i.e., dense layers with no bias).
- Concretely, for a given weights, $h = W_0 x$:
    - **Traditional Adapter**: $h = W_0 x + \psi_{\text{Adapt}}(x)$.
    - **LoRA**: $h = W_0 x + B A x$ (i.e., same as above, but $\psi_{\text{Adapt}}(x) = {\Delta W} x = B A x$).
- $A$ uses a random Gaussian initialization, and $B$ is initialized to zero. This means that at the beginning of training, ${\Delta W}_{\text{Adapt}} = B A = 0$ and the adapter layer has **near-identity** behavior.

</div>

<div class="col-span-4">
<img src="/LoRA.png" alt="LoRA" class="h-64 mx-auto" />

<small class="text-xs">

###### Notation

- $N$: Batch size; $S$: Sequence length; $D$: Hidden size, $r$: Low-rank projection size
- $x \in (N, S, D)$: Input
- $W_0 \in (D, D)$: Pre-trained weights
- $\psi_{\text{Adapt}}(x)$: Adapter layer
- ${\Delta W}_{\text{Adapt}} \in (D, D)$: Adapter weights
- $B \in (D, r), A \in (r, D)$: LoRA low-rank projection weights

</small>

</div>
</div>


<!--
- Note that LoRA restricts the adapter layer's placement to be after some **dense layer**. This is a major core difference between LoRA and adapters in general.
-->

---

# LoRA: A Generalization of Full Fine-tuning
- In **full fine-tuning**, we fine-tune all the parameters of the model.
- One generalization of fine-tuning is **freezing the backbone** and only fine-tuning the final prediction layer (**linear probing**).
- LoRA takes a step further and does not require the accumulated gradient update to have full-rank during adaptation (i.e., the update matrix can be low-rank).
- If the LoRA rank $r$ is equal to the hidden size $D$, then **LoRA is equivalent to full fine-tuning**.

<!--
In other words, as we increase the number of trainable parameters, training LoRA roughly converges to training the original model, while adapter-based methods converges to an MLP and prefix-based methods to a model that cannot take long input sequences.
-->


---

# LoRA: No Additional Inference Latency

- During inference (when deploying the model), we can **explicitly compute** and store $W = W_0 + B A$ and use it instead of $W_0$. This means that we don't need to compute $B A$ at inference time, and thus there is no additional inference latency compared to the original model.
$$
\begin{align}
h &= W_0 x + B A x \\
&= (W_0 + B A) x \\
&= W x
\end{align}
$$

<img src="/inference_latency.png" alt="Inference Latency" class="h-64 mx-auto mt--10" />

---

# LoRA: Differences from Adapter Layers

-  **Placement of the Adaptation Layer:** The original method uses two adapter layers, one after each FFN in the multi-head attention block. LoRA places the adapter "layer" on the query and key projection matrices.
- **Architecture of the Adaptation Component:** LoRA's adapter "layer" consists of a single low-rank matrix ($BA$), whereas the original adapter uses a two-layer MLP with a non-linearity.
- **Integration During Inference:** Because LoRA is just parametrizing the updates to the weight matrix and has no other architectural additions, during inference, we can just update the model to set W = W + BA and run it as usual. This means that there is no additional inference latency compared to the original model.

<!--
1. **Placement of the Adaptation Layer:**
   - Traditional adapters are placed either after the multi-head attention block or after each FFN layer.
   - LoRA modifies the attention mechanism itself by altering the query and key projection matrices within the multi-head attention block.

2. **Architecture of the Adaptation Component:**
   - Traditional adapters use a bottleneck structure consisting of two linear transformations with a non-linearity in between. The downsampling and upsampling with non-linearity introduce additional computational complexity.
   - LoRA uses a simple low-rank update to the pre-existing weight matrices, avoiding the need for non-linear activations and keeping the computational overhead minimal.

3. **Integration During Inference:**
   - With traditional adapters, the architecture during inference still includes the additional adapter layers, and they are an integral part of the forward pass.
   - With LoRA, once training is complete, you can directly update the weight matrices of the original model with the learned low-rank updates. This results in an unchanged inference procedure compared to the original model, which can be advantageous in terms of simplicity and potentially efficiency.

4. **Conceptual View of Weight Updates:**
   - In traditional adapters, the bottleneck architecture can be considered a way to learn residuals or modifications to the network's representations, but it doesn't directly model these as updates to the weight matrices themselves.
   - LoRA is explicitly designed to parameterize updates to the pre-existing model weights, hence the name "Low-Rank Adaptation." It focuses on efficiently learning how to tweak the weights with minimal parameters by enforcing a low-rank structure.

Personal take: LoRA is a much more thoughtful implementation of the same exact underlying motivation: Adapt the original weight matrices using less weights. In the case of LoRA, it's using the low-rank BA matrix formulation, whereas in the original adapter layer, it's a bottleneck architecture (which, if you squint your eyes and ignore bias weights, can be thought of as the same thing, i.e., h = B\sigma(Ax) in the original adapter layer vs h = BAx in LoRA).
-->

---

# Results: GPT-2 on the E2E NLG Challenge

<!-- ![E2E-NLG-GPT2](/E2E-NLG-GPT2.png) -->
<img src="/E2E-NLG-GPT2.png" alt="E2E-NLG-GPT2" class="h-108 mx-auto" />

<!--
The E2E NLG Challenge is a task where the goal is to generate a natural language description of a restaurant based on a set of attributes. For example, given the attributes "name: The Eagle", "food: English", "area: riverside", "familyFriendly: yes", "near: The Rice Boat", the goal is to generate the sentence "The Eagle is a family-friendly English restaurant near The Rice Boat in the riverside area.".
-->

----
# Results: GPT-3

<div class="grid grid-rows-2">
<div>
<!-- ![GPT-3](/GPT-3.png) -->
<img src="/GPT-3.png" alt="GPT-3" class="h-64 mx-auto" />
</div>
<div>
<!-- ![GPT3-Plot](/GPT3-Plot.png) -->
<img src="/GPT3-Plot.png" alt="GPT3-Plot" class="h-64 mx-auto" />
</div>
</div>

<!--
## WikiSQL
**WikiSQL** is a task where the goal is to answer a SQL query based on a table. For example, given the table below, the goal is to answer the question "What is the name of the team that plays in the city that the team named 'The Eagles' plays in?". The answer is "SELECT teamName FROM table WHERE city = (SELECT city FROM table WHERE teamName = 'The Eagles')".

| teamName | city | state | stadiumName | capacity |
|----------|------|-------|-------------|----------|
| The Eagles | Philadelphia | Pennsylvania | Lincoln Financial Field | 68532 |
| The Steelers | Pittsburgh | Pennsylvania | Heinz Field | 65050 |
| The Patriots | Foxborough | Massachusetts | Gillette Stadium | 68756 |
| The Giants | East Rutherford | New Jersey | MetLife Stadium | 82500 |


## MNLI-m
**MNLI-m** is a task where the goal is to predict whether a sentence is an entailment, contradiction, or neutral given a premise. For example, given the premise "A person on a horse jumps over a broken down airplane.", the goal is to predict that the sentence "A person is outdoors, on a horse." is an entailment, because it is likely that a person on a horse is outdoors.

## SAMSum
**SAMSum** is a task where the goal is to generate a summary of a conversation. For example, given the conversation:

- A: i have a doctor appointment tomorrow at 4pm
- B: ok
- A: i need to go to the doctor
- B: i have a doctor appointment tomorrow at 3pm

The goal is to generate the summary "A has a doctor appointment tomorrow at 4pm. B has a doctor appointment tomorrow at 3pm.".
-->

---

# Which Weight Matrices In Transformer Should We Apply LoRA To?

<!-- ![GPT-3 Weight Matrix](/GPT-3-Weight-Matrix.png) -->
<img src="/GPT-3-Weight-Matrix.png" alt="GPT-3 Weight Matrix" class="h-64 mx-auto" />

###### Takeaways
- Note that putting all the parameters in $\Delta W_q$ or $\Delta W_k$ results in significantly lower performance, while adapting both $W_q$ and $W_v$ yields the best result.
- This suggests that even $r = 4$ captures enough information in $\Delta W$ such that it is preferable to adapt more weight matrices than adapting a single type of weights with a larger rank.

---

# What Is The Optimal Rank R For LoRA?

![GPT-3 Rank](/GPT-3-Rank.png)

###### Takeaways
- LoRA already performs competitively with a very small $r$ (more so for ${W_q, W_v}$ than just $W_q$). This suggests the update matrix $\Delta W$ could have a very small "intrinsic rank".

---

# How Does $\Delta W$ Compare To $W$?

<!-- ![GPT-3 Delta W](/GPT-3-Delta-W.png) -->
<img src="/GPT-3-Delta-W.png" alt="GPT-3 Delta W" class="h-48 mx-auto" />

###### Takeaways
- $\Delta W$ has a stronger correlation with $W$ compared to a random matrix, indicating that $\Delta W$ amplifies some features that are already in $W$.
- Instead of repeating the top singular directions of $W$, $\Delta W$ only amplifies directions that are not emphasized in $W$.
- The amplification factor is rather huge: $21.5 \approx  6.91/0.32$ for $r = 4$. This suggests that the low-rank adaptation matrix potentially amplifies the important features for specific downstream tasks that were learned but not emphasized in the general pre-training model.

---

# Conclusion

- LoRA is a parameter-efficient adaptation of LLMs, inspired by the **success of adapter layers** and the idea that **over-parametrized models reside on a low intrinsic dimension**.
- LoRA learns a **low-rank projection** of updates to the model's weights.
- LoRA **outperforms prefix/prompt-tuning and adapter layers** and is much more **computationally efficient** than full fine-tuning whie achieving competitive performance on many tasks.
- Questions? Comments?

<!--
From the paper:
> Fine-tuning enormous language models is prohibitively expensive in terms of the hardware required and the storage/switching cost for hosting independent instances for different tasks. We propose LoRA, an efficient adaptation strategy that neither  ntroduces inference latency nor reduces input sequence length while retaining high model quality. Importantly, it allows for quick task-switching when deployed as a service by sharing the vast majority of the model parameters. While we focused on Transformer language models, the proposed principles are generally applicable to any neural networks with dense layers.
-->
