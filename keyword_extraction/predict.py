import argparse
import pickle
import pandas as pd
from util import (feature_extraction,
                   convert_feature_to_categorical,
                   features)


def predict_text(test_text, keymodel):
    feature_vectores, tokens, _tokens_with_tags = feature_extraction(test_text)

    # convert feature to categorial representation
    columns = features

    db_df = pd.DataFrame({
        'token': tokens,

    })
    for i, column in enumerate(columns):
        db_df[column] = [a[i] for a in feature_vectores]

    # convert features to categorical feature
    converted_df = convert_feature_to_categorical(db_df)
    converted_df = converted_df.drop(columns=['np_pattern'])

    predict_targets = []

    for i, token in enumerate(_tokens_with_tags):
        predicted = keymodel.predict([list(converted_df.iloc[i, 1:-1])])
        predict_targets.append(predicted)

    return predict_targets, tokens

def modeltest(model):
    df = pd.read_csv('data/lableddata/tweets.csv')
    df.columns = ['text', 'keys', 'isformal', 'annotator']
    informaldf = df.iloc[100:110, :]

    texts = list(informaldf['text'])
    # preprocessing
    for test_text in texts:
        predict_targets, tokens = predict_text(test_text=test_text, keymodel=model)
        print(60 * '^^')
        print('original text:')
        print(test_text)
        print('clean text:')
        print(' '.join(tokens))
        print('key by supervisd method:')
        print([a for a in list(zip([a[0] for a in predict_targets], tokens)) if a[0] != 'nonkey'])



def main():
    parser=argparse.ArgumentParser(
        description="extrcat keyword from tweets"
    )
    parser.add_argument('method', help="(rf: RandomForest |nb: GaussianNB| svm: SVM)",type=str)
    parser.add_argument('-i','--inputtext', help= "The input tweet",type=str)
    parser.add_argument('-f','--inputfile', help= "The input tweet file as txt file",type=str)
    args = parser.parse_args()
    if args.method is None:
        raise ValueError('Must specify the method that model is built by that model (nb|svm|rf).')
    
    if args.inputfile is None and args.inputtext is None:
        print('You should Enter the name of text input file as argument or the input text ')
    if args.inputfile is not None:
        with open(args.inputfile,'r') as f:
            inputtext=f.read()
    else:
        inputtext = args.inputtext
    print(inputtext)
    pkl_filename = str.format("models/{0}keymodel.pkl", args.method)
    with open(pkl_filename, 'rb') as file:
        keymodel = pickle.load(file)
    predict_targets, tokens = predict_text(inputtext, keymodel=keymodel)
    print(tokens)
    print([a for a in list(zip([a[0] for a in predict_targets], tokens)) if a[0] != 'nonkey'])
    key_phraces = []
    temp = ''
    for i, token in enumerate(tokens):

        if predict_targets[i][0] == 'fullkey':
            if len(temp) > 0:
                key_phraces.append(temp)
                temp = ''
            key_phraces.append(token)
        elif predict_targets[i][0] == 'partkey':
            temp = temp + ' ' + token
        elif predict_targets[i][0] == 'nonkey':
            if len(temp) > 0:
                key_phraces.append(temp)
                temp = ''
    print('keyphraces are:', key_phraces)
    keys='\n'.join(key_phraces)
    with open('keys.txt','wt') as f:
        f.write(keys)

if __name__ == '__main__':
    main()

