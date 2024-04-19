# Rating-oriented Explicit Disentangling Graph Convolution Network for Review-aware Recommendation

## Requirements

```
python == 3.8.3
transformers == 3.1.0
dgl == 0.7.2
pytorch == 1.10.2
```

## Data prepration

1. Run word2vector.py for word embedding. Glove pretraining weight is required. The word embedding is insufficent for REDC but legacy for other review-based recommendation models. 
2. Make sure can run load_sentiment_data in load_data.py 
3. Run BERT/bert_whitening.py for obtaining the feature vector for each review.
4. If previous steps successfully run, then you can run REDC.py. 

### A processed data: Digital_Music
Dowload from [here](https://drive.google.com/drive/folders/1OPkb_XLlxDp4otLy5-WKX4j_RvxODPuj?usp=sharing).


