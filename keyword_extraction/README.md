**Twitter keyword extraction**
---
As an NLP Specialist, I was working on keyword extraction by rule-based and machine learning methods on informal Persian tweets for a while. After experimental results on different rule-based methods, I found that using just rule-based methods used POS-Tagging as a central part in identifying noun phrases does not work well on an informal text like tweets as formal texts. I found the reason is some parts of the speech are removed in informal text. So the main problem with rule-based methods in informal tweets keyword extraction is the lack of ability of these methods in identifying the right part of speech. so I decide to use machine learning methods to learn the right position of keywords in tweets. This method is as follow :

1) A database was created as training data from tweets with keywords. For this purpose, 400 tweets were tagged by 3 people. Each word was assigned one of three tags (part key-full key-none) depending on whether it was part of a keyword the phrase, or whether it was a keyword or whether it was not a keyword. based on my intuition, 12 features including some rule-based features (that I know have low confidence but not zero!) and some innovative features were extracted for each of these words.
2) build a classical model on the provided dataset
3) predict the keywords using the model


 **Feature extraction**
 
The basic features I used in this model is as follow:
 
    is_np_part: Is this word part of a noun phrase (which is identified utilizing external pos tagger)

    is_after_StopWord: Is this word after a stop word?

    is_after_verb: Is this word after a verb? (we have a list of a frequent verb in Persian)

    is_before_StopWord:  Is this word before a stop word?

    is_before_verb: Is this word before a verb?

    np_index: The index of word in a noun phrase (if this word is part of np otherwise this would be -1)

    tag_before: The tag of the word before this word

    tag_after: The tag of the word after this word

    is_after_StartStops: Is this word after start-stoplist? (start-stoplist is the words are the most possible words  before keywords .this list is built manually and based on exploring tweets )

    token_tf: The frequency of the word in the tweet it belongs

    normalized_tf: The frequency of word divide by the total number of the word in the tweet it belongs to

    token_sent_index_feature: The index of token in the tweet it belongs to 


 **runing the code**
 -
 **Train**
 
To get started, run exmpale train with my example dataset provided in 'ml/data/labeleddata/tweets.csv: 

    python train.py tweets.csv rf
    
'rf' is the method we train model with in training step.(rf: RandomForest |nb: GaussianNB| svm: SVM)

**Predict** 

To predict a sentence based on model build you have 2 option:

option1: pass the sentence as input:

    python predict.py rf -i 'امروز 22 بهمن است . روز پیروزی انقلاب اسلامی ایران است'

'rf' is the method we train model with in training step.

option2: pass the sentence as text file :

    python predict.py rf  -f testinput.txt

**TODO**:

  add model evaluation
