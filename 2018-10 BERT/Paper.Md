BERT which stands for Bidirectional Encoder Representations from Transformers is a language representation model.

BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. 

In this paper authors mentions that standard language models are unidirectional(eg. GPT). They argue this restrict the power of the pre-trained representations, especially for the fine-tuning approaches. Also unidirectional models limits the choice of architectures that can be used during pre-training. 

Model Architecture: BERT’s model architecture is a multi-layer bidirectional Transformer Encoder.
    In this paper, Authors trained two models:
        BERT(BASE): 
            Number of transformer blocks(L) = 12
            Hidden Size(H) = 768
            Number of self attention heads(A) = 12
            Total Paramters = 110M

        BERT(LARGE): 
            Number of transformer blocks(L) = 24
            Hidden Size(H) = 1024
            Number of self attention heads(A) = 16
            Total Paramters = 340M

    In our code, we train a very small model:
            Number of transformer blocks(L) = 
            Hidden Size(H) = 
            Number of self attention heads(A) = 
            Total Paramters = 

Note:
    A “sentence” can be an arbitrary span of contiguous text, rather than an actual linguistic sentence. 
    A “sequence” refers to the input token sequence to BERT, which may be a single sentence or two sentences packed together.

Input Representation: 
    Special tokens:
        1. [CLS]: This is a special classification token of every input sequence. The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks. 
        2. [SEP]: Sentence pairs are packed together into a single sequence. To differentiate the sentences they are separate with this special token. Also, a learned embedding added to every token indicating whether it belongs to sentence A or sentence B.

        For a given token, its input representation is constructed by summing the corresponding Token Embeddings( WordPiece embeddings is used with 30k token vocabulary), Segment Embeddings, and Positional Embeddings.

    In our code, we only use single sentence as a input so we do not use [SEP] token.

Objective for pre-training of BERT: 
    1. Masked Language Model (MLM): The masked language model randomly masks some of the tokens from the input, and the objective is to predict the original vocabulary id of the masked word based only on its context. They mask 15% of all WordPiece tokens in each sequence at random. They only predict the masked words rather than reconstructing the entire input.

    Problem with MLM: This objective creates a mismatch between pre-training and fine-tuning, since the [MASK] token does not appear during fine-tuning.
    Clever strategy:
     To mitigate this, we do not always replace “masked” words with the actual [MASK] token. The training data generator chooses 15% of the token positions at random for prediction. If the i-th token is chosen, we replace the i-th token with   
        (1) the [MASK] token 80% of the time 
        (2) a random token 10% of the time 
        (3) the unchanged i-th token 10% of the time.

    2.  Next Sentence Prediction (NSP): NSP is a binarized next sentence prediction task that can be trivially generated from any monolingual corpus. Specifically, when choosing the sentences A and B for each pre-training example, 50% of the time B is the actual next sentence that follows A (labeled as IsNext), and 50% of the time it is a random sentence from the corpus (labeled as NotNext). Then, C is used for next sentence prediction (NSP)

    In our code, only MLM is used as an objective for pre-training the model.

Pre-training data:
    BooksCorpus (800M words) 
    English Wikipedia (2,500M words).

    In our code, we used 