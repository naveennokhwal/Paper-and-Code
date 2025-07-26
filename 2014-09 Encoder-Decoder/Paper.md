# Sequence to Sequence Learning with Neural Networks (Paper Summary & Code)

This repository provides a summary of the seminal paper "Sequence to Sequence Learning with Neural Networks" by Sutskever, Vinyals, and Le, which introduced a powerful end-to-end approach for sequence transformation using recurrent neural networks. It also aims to accompany this with a code implementation (focused on relevant aspects).

## Introduction

Many important problems in machine learning and artificial intelligence, such as machine translation, speech recognition, and question answering, involve mapping an input sequence to an output sequence, where the lengths of these sequences are not known a-priori and the relationship between them can be complex and non-monotonic. While Deep Neural Networks (DNNs) have shown remarkable success in areas like speech recognition and visual object recognition, their direct application is often limited to problems with fixed-dimensionality inputs and targets.

This paper addresses this limitation by demonstrating that a straightforward application of the Long Short-Term Memory (LSTM) architecture can effectively solve general sequence-to-sequence problems. The core idea is to use one LSTM to read the input sequence, encoding it into a large, fixed-dimensional vector representation. A second LSTM then decodes this vector to generate the target output sequence.

The authors emphasize the power of DNNs, stating that if a good solution (i.e., a set of parameters, weights, and biases) exists for a DNN, then supervised backpropagation—given sufficient data and gradient signal—can reliably discover it.

## Key Innovations & Insights

### Overcoming Long-Term Dependencies

A significant challenge in training RNNs for sequence-to-sequence tasks, particularly with very long sequences, is the difficulty in learning long-range temporal dependencies. The authors made a crucial observation and introduced a key technical contribution to mitigate this:

* **Reversing Source Sentences:** They reversed the order of words in all source sentences (but not target sentences) during training and testing. This simple trick dramatically improved the LSTM's performance by introducing many short-term dependencies between corresponding source and target words. For example, the first word of the source is now directly "aligned" with the first word of the target, simplifying the optimization problem and enabling SGD to learn LSTMs that handle long sentences effectively.

### Learning Sentence Representations

The paper also provides qualitative evidence that the LSTM model learns meaningful sentence representations. Sentences with similar meanings are represented by vectors that are "close" to each other in the learned embedding space, while sentences with different meanings are "far" apart. This indicates that the model is not merely memorizing sentences but is capable of understanding word order and is fairly invariant to variations like active and passive voice.

## Model Architecture

The proposed approach uses a pair of LSTMs: an **encoder** and a **decoder**.

### Theoretical Model:

The goal is to estimate the conditional probability $p(y_1, \dots, y_{T'} | x_1, \dots, x_T)$, where $(x_1, \dots, x_T)$ is an input sequence of length $T$, and $y_1, \dots, y_{T'}$ is its corresponding output sequence of length $T'$ (which may differ from $T$).

The LSTM computes this conditional probability by:
1.  First obtaining a fixed-dimensional vector representation $v$ of the input sequence $(x_1, \dots, x_T)$. This vector $v$ is given by the last hidden state of the **encoder LSTM**.
2.  Then computing the probability of $y_1, \dots, y_{T'}$ with a standard LSTM-based language model formulation, where its initial hidden state is set to the representation $v$ of the input sequence:
    $$p(y_1, \dots, y_{T'} | x_1, \dots, x_T) = \prod_{t=1}^{T'} p(y_t | v, y_1, \dots, y_{t-1})$$

**Note:** Each sentence ends with a special token `<EOS>`, which enables the model to define a distribution over sequences of all possible lengths.

### Implemented Model Variations (as described in the paper):

The model implemented and evaluated in the paper varies from the above theoretical model in several key ways:

1.  **Two Different LSTMs (Encoder-Decoder):** They explicitly used two separate LSTMs—one for encoding the input sequence and another for decoding the output sequence. This design choice increases the model's parameter capacity and naturally facilitates training on multiple language pairs simultaneously.
2.  **Deep LSTM:** Instead of shallow LSTMs, they used **deep LSTMs (four layers)**. Deep neural networks are known for their ability to learn complex hierarchical representations.
3.  **Reversed Input Sentence Order:** As discussed, they reversed the order of words in the input sentences to simplify the optimization problem and improve performance with long sequences.

## Training Objective

The authors trained their model on an English-to-French Machine Translation task. The objective was to maximize the log probability of a correct translation $T$ given the source sentence $S$. The training objective function is defined as:

$$\frac{1}{|\mathcal{S}|} \sum_{(T,S) \in \mathcal{S}} \log p(T|S)$$

where $\mathcal{S}$ represents the training set.

## Dataset

The models were trained on a subset of the **WMT'14 English to French dataset**. Specifically, they used 12 million sentences, consisting of 348 million French words and 304 million English words. This particular dataset was chosen due to the public availability of tokenized training and test sets, along with 1000-best lists from baseline Statistical Machine Translation (SMT) systems, allowing for robust comparison.

### Tokenization Details:

* A fixed vocabulary was used for both languages.
* **Source Language (English):** 160,000 most frequent words.
* **Target Language (French):** 80,000 most frequent words.
* Any out-of-vocabulary word was replaced with a special `<UNK>` token.

## Model Architecture and Training Details

* **Number of LSTM Layers:** 4
* **Number of Cells at Each Layer:** 1000 cells per layer
* **Dimensions of Word Embeddings:** 1000 dimensions
* **Input Vocabulary Size:** 160,000
* **Output Vocabulary Size:** 80,000
* **Total Number of Parameters:** Approximately 384 million (32 million for the "encoder" LSTM and 32 million for the "decoder" LSTM; note: original paper states 384M in total including embeddings and output layer weights, 64M for LSTMs alone).
* **Parameter Initialization:** Uniform distribution between -0.08 and 0.08.
* **Optimizer:** Stochastic Gradient Descent (SGD) without momentum.
* **Learning Rate:** Started at 0.7. After 5 epochs, it was halved every half epoch. Models were trained for a total of 7.5 epochs.
* **Batch Size:** 128 sentence sequences.
* **Gradient Clipping:** To address exploding gradients (a potential issue even with LSTMs, though less severe than vanishing gradients), a hard constraint on the norm of the gradient was enforced by scaling it when its norm exceeded a threshold.
* **Minibatch Length Handling:** To optimize training efficiency due to varying sequence lengths, minibatches were constructed such that all sentences within a minibatch were of roughly the same length.
* **Sentence Representation:** The deep LSTM effectively uses 8000 real numbers to represent a sentence (1000 cells per layer * 4 layers * 2 directions for encoder output).

## Getting Started

[Add instructions on how to run your code, e.g., installation, data preparation, training, etc.]

## References

* **Sequence to Sequence Learning with Neural Networks**
    * Ilya Sutskever, Oriol Vinyals, Quoc V. Le
    * Google
    * [arXiv:1409.3215](https://arxiv.org/abs/1409.3215)
    * [Published in NeurIPS 2014](https://proceedings.neurips.cc/paper_files/paper/2014/file/a14ac55a4f27472c85028eb1c78417dd-Paper.pdf)

```