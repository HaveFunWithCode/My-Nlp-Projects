__author__ = "Marzieh sepehr"
import sys

import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB as GNB
import pickle
from sklearn import neural_network
import util as util
from sklearn.model_selection import train_test_split


def evaluate_model(features, target, method,test_part):
    """ param method: nb |svm|nn """
    X_train, X_test, Y_train, Y_test = train_test_split(features,
                                                                        target, 
                                                                        test_size=text_part,
                                                                        random_state=100)
    if method == 'nb':
        classifier = GNB()
    elif method == 'svm':
        classifier = svm.SVC(kernel='rbf', gamma=0.1, C=1)
    elif method == 'nn':
        classifier = neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    elif method== 'rf':
        classifier = RandomForestClassifier(n_estimators=10, max_depth=10)
    else:
        classifier = GNB()
    classifier.fit(X_train, Y_train)
    result = classifier.score(X_test, Y_test)

    print("Accuracy: %.2f%%" % (result*100.0))


def evaluate(dataset, method, loadfeature=False,test_part=0.3):
    all_feature_vectors = []
    all_targets = []
    tokens_all = []
    c = 0
    if not loadfeature:
        for i, row in dataset.iterrows():
            c = c + 1
            print(c)
            # print(i)
            maintext = row['text']
            target_keys = row['keys']
            feature_vectors, target_vector, tokens = util.extract_key_target_vector(maintext, target_keys)
            tokens_all.extend(tokens)

            all_targets.extend(target_vector)
            all_feature_vectors.extend(feature_vectors)

        db_df = pd.DataFrame({
            'token': tokens_all,
            'target': all_targets
        })
        for i, column in enumerate(util.features):
            db_df[column] = [a[i] for a in all_feature_vectors]

        # convert features to categorical feature
        converted_df = util.convert_feature_to_categorical(db_df)
        # train by learning methods and create model
        converted_df = converted_df.drop(columns=['np_pattern'])

        features = converted_df.iloc[:, 2:-1]
        target = converted_df.iloc[:, 1]
        features.to_csv('temp/features.csv')
        target.to_csv('temp/target.csv', header=['target'])

    else:
        features = pd.DataFrame.from_csv('temp/features.csv')
        target = pd.DataFrame.from_csv('temp/target.csv')

    evaluate_model(features=features, target=target, method=method,test_part=test_part)



def main():
    if len(sys.argv) == 1:
        raise ValueError('Must specify input  dataset file.')
    if len(sys.argv) == 2:
        raise ValueError('Must specify input taining method (nb|svm|rf)')

    else:
        df = pd.read_csv(sys.argv[1])
        method=sys.argv[2]
        method = sys.argv[3]
        # df = pd.read_csv('data/lableddata/tweets.csv')
        df.columns = ['text', 'keys', 'isformal', 'annotator']
        df = df[df['annotator'] == 'annotator1']
        informaldf = df[df['isformal'] == False].copy()
        informaldf = informaldf[informaldf['keys'].apply(lambda x: len(x) > 2)]

        evaluate(informaldf.iloc[1:100, 0:2], method=method, loadfeature=False,test_part=0.3)



if __name__ == '__main__':
    main()

