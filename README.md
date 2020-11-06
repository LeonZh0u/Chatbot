# Chatbot
A dialogue system that can handle casual conversation and task-oriented queries. 

Fasttext is used to classify input as either casual or task-oriented.
The query system uses Hierarchical Navigable Small World graphs (facebook lib) to recall top k best matches. In order to rank the top k entries, multiple features are used: Longest Common Sequence, Jaccard, Cosine, Sine distance... as well as Bert classfication result, all of these features are inputs to lightBGM to produce the final ranking.

The generative system is an implementation of Unified Language Model Pre-training for Natural Language Understanding and Generation https://arxiv.org/abs/1905.03197.
TextBrew is used to compress the model size.

Task.py integrates both models and works as a demo.
