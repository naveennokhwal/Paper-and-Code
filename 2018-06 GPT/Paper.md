# GPT: Generative Pre-Training (Paper Summary & Code)

This repository provides a summary of the influential paper "Improving Language Understanding by Generative Pre-Training" and accompanies it with a PyTorch implementation focused on the pre-training phase of a Generative Pre-trained Transformer (GPT) model.

## Introduction to GPT

The paper "Improving Language Understanding by Generative Pre-Training" addresses the challenge of scarcity of labeled data for specific language understanding tasks, despite the abundance of unlabeled text corpora. It proposes a semi-supervised approach where a language model is pre-trained on a diverse corpus of unlabeled text. This pre-training allows the model to leverage rich linguistic information from the unlabeled data, offering a valuable and less resource-intensive alternative to gathering extensive annotations.

### Challenges Addressed:

The authors identify two key challenges in effectively leveraging linguistic information from unlabeled text data:

1.  **Optimization Objectives:** Determining the most effective optimization objectives for learning text representations that are useful for transfer. Various objectives exist, such as language modeling, machine translation, and discourse coherence.
2.  **Transfer Mechanisms:** The lack of consensus on the most effective way to transfer these learned representations to target tasks. Existing techniques often involve task-specific architectural changes, intricate learning schemes, and additional auxiliary learning objectives, which can be complex.

### Semi-supervised Approach

To overcome these challenges, the paper introduces a semi-supervised approach combining "unsupervised pre-training and supervised fine-tuning." The core idea is to learn a universal representation that transfers with minimal adaptation to a wide range of tasks. They utilize a "Language Modeling Objective" on unlabeled data to learn the initial parameters of a neural network model. Subsequently, these pre-trained parameters are adapted to a target task using its corresponding supervised objective. This setup is flexible and does not require the target tasks to be in the same domain as the unlabeled corpus.

## Training Procedure

The training procedure consists of two main steps:

1.  **Pre-training:** This first stage involves learning a high-capacity language model on a large corpus of text.
2.  **Fine-tuning:** In the fine-tuning stage, the pre-trained model is adapted to a discriminative task using labeled data.

**Note on our code:** In this repository, we focus specifically on the **Pre-training** phase of the model.

## Model Architecture

The authors of the GPT paper chose a **Decoder-only Transformer architecture**. This choice provides a more structured memory, particularly effective for handling long-term dependencies within text.

### Architecture Details:

* **Number of Decoder-only Transformer Blocks (L):** 12
* **Number of Dimensions of Embedding Vector:** 768
* **Hidden Dimensions of Feed-Forward Network:** 3072
* **Number of Self-Attention Heads (A):** 12
* **Optimizer:** Adam
* **Activation Function:** Gaussian Error Linear Unit (GELU)
* **Max Learning Rate:** 2.5e-4 (Learning rate was increased linearly from zero over the first 2000 updates and then annealed to 0 using a cosine schedule.)
* **Positional Embeddings:** Learned position embeddings are used instead of the sinusoidal version proposed in the original Transformer work.

### Our Implemented Model:
* **Number of Decoder-only Transformer Blocks (L):** 
* **Number of Dimensions of Embedding Vector:** 
* **Hidden Dimensions of Feed-Forward Network:** 
* **Number of Self-Attention Heads (A):** 
* **Optimizer:** 
* **Activation Function:** 
* **Max Learning Rate:** 
* **Positional Embeddings:** 

## Input Representation

Raw text is converted into tokens using **Byte-pair Encoding (BPE)**. These tokens are then converted into embeddings, and positional embeddings are added to them to form the final input representation for the decoder-only blocks.

The input representation for the first decoder block ($h_0$) is formulated as:

$h_0 = U \cdot W_e + W_p$

Where:
* $U = (u_{-k}, \dots, u_{-1})$ is the context vector of tokens.
* $W_e$ is the token embedding matrix.
* $W_p$ is the position embedding matrix.
* $h_0$ is the final embedding entering the first decoder block.

## Unsupervised Pre-training Dataset

The GPT model was pre-trained on the **BooksCorpus dataset**.

## Getting Started

[Add instructions on how to run your code, e.g., installation, data preparation, training, etc.]

## References

* **Improving Language Understanding by Generative Pre-Training**
    * Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever
    * OpenAI
    * [Paper Link (OpenAI Blog Post)](https://openai.com/research/language-unsupervised) (Note: This paper was initially released as a blog post by OpenAI. The closest widely cited "paper" is the one often associated with the model.)
