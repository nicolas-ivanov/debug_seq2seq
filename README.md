# debug seq2seq
Make [seq2seq for keras](https://github.com/farizrahman4u/seq2seq) work.

The code includes:

* small dataset of movie scripts to train your models on
* preprocessor function to properly tokenize the data
* word2vec helpers to make use of gensim word2vec lib for extra flexibility
* train and predict function to harness the power of seq2seq
 

**Warning**

* The code has bugs, undoubtedly. Feel free to fix them and pull-request.


**Papers**

* [Sequence to Sequence Learning with Neural Networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
* [A Neural Conversational Model](http://arxiv.org/pdf/1506.05869v1.pdf)

**Nice picture**

[![seq2seq](https://4.bp.blogspot.com/-aArS0l1pjHQ/Vjj71pKAaEI/AAAAAAAAAxE/Nvy1FSbD_Vs/s640/2TFstaticgraphic_alt-01.png)](http://4.bp.blogspot.com/-aArS0l1pjHQ/Vjj71pKAaEI/AAAAAAAAAxE/Nvy1FSbD_Vs/s1600/2TFstaticgraphic_alt-01.png)

**Setup&Run**

    git clone https://github.com/nicolas-ivanov/debug_seq2seq
    cd debug_seq2seq
    bash bin/setup.sh
    python bin/train.py
or

    python bin/test.py
    