# Intention-Mining-TSE
The dataset and source code for paper "Automating Intention Mining"

The code is based on [dennybritz's implementation](https://github.com/dennybritz/cnn-text-classification-tf) of Yoon Kim's paper [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

Running 'online_prediction.py', you can input any sentence and check the classification result produced by a pre-trained CNN model. The model uses all sentences of the four Github projects as training data.

Running 'play.py', you can get the evaluation result of cross-project prediction. Please check the code for more details of the configuration. By default, it will use the four Github projects as training data to predict the sentences in DECA dataset, and in this setting, the category 'aspect evaluation' and 'others' are dropped since DECA dataset does not contain these two categories.
