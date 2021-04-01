# NLP assignment 2 

This is the implementation in the assignment of CSI5386[W] Natural Language Processing 2021"[
INGREDIENTS FOR HAPPINESS](https://www.site.uottawa.ca/~diana/csi5386/A2_2021/A2_2021.htm)" .


### 1. Step1: Baseline
- **Execution**: Get the same results by below command.
    ```bash
    python train.py --cuda --bid --epochs 50
    ```
    - `--traindata` The path to the training data.
    - `--testdata` The path to test data.
    - `--epochs` Epochs to train.
    - `--dropout` Dropout rate of RNN.
    - `--lr` Learning rate.
    - `--weight_decay` Factor for L2 regularization.
    - `--seed` Manual seed.
    - `--cuda` Use cuda or not.
    - `--embed_dim` The dimension of embedding.
    - `--hidden_dim` The dimension of hidden layer.
    - `--pretrain` Use pretrain models or not.
    - `--layer` The number of RNN layers.
    - `--bid` RNN is bidirectional or not.
    - `--test` Test or not.
 ### 2. Step2: Fine-tuning BERT
 - **Execution**: open ipynb file in google colab and run each code cell in order.
 - As the model trained is random, running the code may not get exactly the same result, the score of precision, recall and F1 may vary but the accuracy will be the same.
 - Agency label parameter: 
   - sequence length -- 40, 
   - batch size -- 32, 
   - epoch -- 50
 - Social label parameter:
   - sequence length -- 40, 
   - batch size -- 32, 
   - epoch -- 100
