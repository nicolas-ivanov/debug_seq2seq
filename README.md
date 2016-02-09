# debug seq2seq
Make [seq2seq for keras](https://github.com/farizrahman4u/seq2seq) work.

The code includes:

* small dataset of movie scripts to train your models on
* preprocessor function to properly tokenize the data
* word2vec helpers to make use of gensim word2vec lib for extra flexibility
* train and predict function to harness the power of seq2seq
 
**Warning**

* The code has bugs, undoubtedly. Feel free to fix them and pull-request.
* No good results were achieved with this architecture yet. See 'Results' section below for details. 

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

and then

    python bin/test.py


**Results**

No good results were achieved so far:

    [why ?] -> [i ' . . $$$ . $$$ $$$ $$$ $$$ as as as as i i]
    [who ?] -> [i ' . . $$$ . $$$ $$$ $$$ $$$ as as as as i i]
    [yeah ?] -> [i ' . . $$$ . $$$ $$$ $$$ $$$ as as as as i i]
    [what is it ?] -> [i ' . . $$$ . $$$ $$$ $$$ $$$ as as as as as i]
    [why not ?] -> [i ' . . $$$ . $$$ $$$ $$$ $$$ as as as as i i]
    [really ?] -> [i ' . . $$$ . $$$ $$$ $$$ $$$ as as as as i i]

My guess is that there are some foundational problems in this approach:

* Since word2vec vectors are used for words representations and the model returns an approximate vector for every next word, this error is accumulated from one word to another and thus starting from the third word the model fails to predict anything meaningful...
This problem might be overcome if we replace our approximate word2vec vector every thimestamp with a "correct" vector, i.e. the one that corresponds to an actual word from the dictionary. Does it make sence?
However you need to dig into seq2seq code to do that.

* The second problem relates to word sampling: even if you manage to solve the aforementioned issue, in case you stick to using argmax() for picking the most probable word every time stamps, the answers gonna be too simple and not interesting, like:

```
are you a human?			-- no .
are you a robot or human?	-- no .
are you a robot?			-- no .
are you better than siri?  		-- yes .
are you here ?				-- yes .
are you human?			-- no .
are you really better than siri?	-- yes .
are you there 				-- you ' re not going to be
are you there?!?!			-- yes .
```

Not to mislead you: these results were achieved on a different seq2seq architecture, based on tensorflow.

Sampling with temperature could be used in order to diversify the output results, however that's again should be done inside seq2seq library.
