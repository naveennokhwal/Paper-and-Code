# BERT: Bidirectional Encoder Representations from Transformers (Paper Summary & Code)

This repository provides a summary of the groundbreaking paper "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" and accompanies it with a PyTorch implementation of a small BERT model.

## Introduction to BERT

BERT, which stands for Bidirectional Encoder Representations from Transformers, is a powerful language representation model. It is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.

The authors of the BERT paper highlight that traditional language models (e.g., GPT) are unidirectional, meaning they only consider context from one side (left-to-right or right-to-left). This unidirectionality restricts the power of the pre-trained representations, especially for fine-tuning approaches, and limits the choice of architectures that can be used during pre-training. BERT overcomes this limitation by being truly bidirectional.

## Model Architecture

BERT's core architecture is a multi-layer bidirectional Transformer Encoder. The original paper introduced two main model sizes:

### Original BERT Models

* **BERT (BASE):**
    * Number of Transformer Blocks (L): 12
    * Hidden Size (H): 768
    * Number of Self-Attention Heads (A): 12
    * Total Parameters: 110M

* **BERT (LARGE):**
    * Number of Transformer Blocks (L): 24
    * Hidden Size (H): 1024
    * Number of Self-Attention Heads (A): 16
    * Total Parameters: 340M

### My Implemented Model

For demonstration and learning purposes, my code implements and trains a much smaller version of BERT:

* Number of Transformer Blocks (L): 
* Hidden Size (H): 
* Number of Self-Attention Heads (A): 
* Total Parameters: 

## Key Concepts and Input Representation

**Note:**
* A "sentence" in the context of BERT can be an arbitrary span of contiguous text, not necessarily a linguistic sentence.
* A "sequence" refers to the input token sequence to BERT, which may be a single sentence or two sentences packed together.

### Input Representation

For a given token, its input representation is constructed by summing three types of embeddings: Token Embeddings, Segment Embeddings, and Positional Embeddings. WordPiece embeddings are used with a 30k token vocabulary.

#### Special Tokens:

* **`[CLS]`:** This is a special classification token appended to the beginning of every input sequence. The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks.
* **`[SEP]`:** Used to separate sentence pairs when they are packed together into a single sequence. A learned embedding is also added to every token indicating whether it belongs to sentence A or sentence B.

**Note on my code:** In my implementation, I primarily use single sentences as input, so the `[SEP]` token and segment embeddings for differentiating sentences are not utilized.

## Pre-training Objectives

BERT is pre-trained using two novel unsupervised tasks:

### 1. Masked Language Model (MLM)

The MLM objective involves randomly masking some of the tokens from the input, and the model's task is to predict the original vocabulary ID of the masked word based only on its context. The authors mask 15% of all WordPiece tokens in each sequence at random, predicting only the masked words rather than reconstructing the entire input.

**Addressing the `[MASK]` token mismatch:**
To mitigate the pre-training/fine-tuning mismatch (where `[MASK]` tokens do not appear during fine-tuning), a clever strategy is employed:
When a token position is chosen for masking (15% of tokens):
* **80% of the time:** The chosen token is replaced with the `[MASK]` token.
* **10% of the time:** The chosen token is replaced with a random token.
* **10% of the time:** The chosen token remains unchanged.

### 2. Next Sentence Prediction (NSP)

NSP is a binarized next sentence prediction task. During pre-training, for each example, 50% of the time sentence B is the actual next sentence that follows sentence A (labeled as `IsNext`), and 50% of the time it is a random sentence from the corpus (labeled as `NotNext`). The `[CLS]` token's final hidden state is then used to predict whether the sentences are `IsNext` or `NotNext`.

**Note on my code:** In my implementation, I primarily focus on the Masked Language Model (MLM) as the objective for pre-training.

## Pre-training Data

The original BERT models were pre-trained on a massive amount of text data:

* BooksCorpus (800M words)
* English Wikipedia (2,500M words)

**Note on my code:** In my implementation, I used [---].

## Getting Started

[Add instructions on how to run your code, e.g., installation, data preparation, training, etc.]

## References
* **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**
    * Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
    * Google AI Language
    * [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
