import get_data
from tensorflow.python.keras import Sequential, regularizers
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy as np
from sklearn import preprocessing
import tensorflow as tf


from tensorflow.python.keras import backend as K
K._get_available_gpus()



seed = 8

seq_embeding_method='avg'
use_pca=True

cbow_vectores, skip_vectores, y_enc, emsize, x_enc=get_data.getfasttextdatset()
x=cbow_vectores.as_matrix()
y=y_enc
x=[a.tolist() for a in x]
y=[a.tolist() for a in y]
x=np.array(x)
y=np.array(y)
y_cat=[int(np.nonzero(yy)[0]) for yy in y ]

#0: culture |1: economy|2:IT |3: LiteralAndArt |4:politics |5: social |6:sport | 7:dailynotes
mycats=[0,1,4,3,6]


cat_number=len(mycats)
cat_indexes = [k for k, j in enumerate(y_cat) if j in mycats]
x=x[cat_indexes]
y_cat=np.array(y_cat)[cat_indexes]


def encode(le, labels):
    enc = le.transform(labels)
    return tf.keras.utils.to_categorical(enc)
le = preprocessing.LabelEncoder()
le.fit(y_cat)
y = encode(le, y_cat)




def base_line_model():
    model = Sequential()
    model.add(Dense(20, input_dim=emsize, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(20, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(cat_number, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=base_line_model,
                            epochs=1000, verbose=1)

kfold = KFold(n_splits=3, shuffle=True, random_state=seed)
results = cross_val_score(estimator, x, y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))



