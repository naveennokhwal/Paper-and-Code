Unlabeled text corpora are abundant, labeled data for learning these specific tasks is scarce, making it challenging for discriminatively trained models to perform adequately. Generative Pre-Training of a language model on a diverse corpus of unlabeled text allows model to leverage linguistic information from unlabeled data that provide a valuable alternative to gathering more annotation, which can be time-consuming and expensive. 

Challenges in leverage linguistic information from unlabeled text data:
    1. what type of optimization objectives are most effective at learning text representations that are useful for transfer?
        There are different optimization objectives exists: language modeling, machine translation, and discourse coherence.
    
    2. There is no consensus on the most effective way to transfer these learned representations to the target task.
        Existing techniques involve a combination of making task-specific changes to the model architecture, using intricate learning schemes and adding auxiliary learning objectives.

In this paper, Author introduced semi-supervised approach for language understanding tasks using a combination of "unsupervised pre-training and supervised fine-tuning". The goal is to learn a universal representation that transfers with little adaptation to a wide range of tasks. they use a "Language Modeling Objective" on the unlabeled data to learn the initial parameters of a neural network model. Subsequently, they adapt these parameters to a target task using the corresponding supervised objective. This setup does not require these target tasks to be in the same domain as the unlabeled corpus. 

Training procedure consist two steps: 
    1. Pre-training:  This first stage is learning a high-capacity language model on a large corpus of text
    2. Fine-tuning:  fine-tuning stage, where we adapt the model to a discriminative task with labeled data\

Model Architecture: Authors choose Decoder-only Transformer architecture. This model choice provides us with a more structured memory for handling long-term dependencies in text.
            Number of Decoder-only transformer blocks(L) = 12
            Number of dimensions of embedding vector = 768
            Hidden dimensions of feed-forward network = 3072
            Number of self attention heads(A) = 12
            Optimizer: Adam
            Activation function: Gaussian Error Linear Unit (GELU)
            max learning rate = 2.5e-4 (The learning rate was increased linearly from zero over the first 2000 updates and annealed to 0 using a cosine schedule.)

Input Representation:
    Using Byte-pair Encoding (BPE), raw text is converted into tokens. These tokens are then converted into embeddings and postional embedding is also added to these embedding.
            h0 = U*W_e + W_p
            where, U = (u_-k, ..., u_-1) is the context vector of tokens, W_e is the token embedding matrix, and W_p is the position embedding matrix.

Unsupervise pre-training dataset: BooksCorpus dataset