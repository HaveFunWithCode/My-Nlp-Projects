import itertools
from nltk import Tree
from nltk import RegexpParser
import ast
from sklearn import preprocessing

from pertopic import PersianPreprocess as per_pre
from pertopic import TwitterPreprocess as tw_pre
from pertopic.Preprocess import Clean_tweet_actions, Stemmer_method
from pertopic.utils import (StopLists,
                            StopListReader,
                            getPersianVerbs_dataset)
from external import Virastar as vira_pre

features = ['tag',
            'is_np_part',
            'is_after_StopWord',
            'is_after_verb',
            'is_before_StopWord',
            'is_before_verb',
            'np_index',
            'np_pattern',
            'tag_before',
            'tag_after',
            'is_after_StartStops',
            'is_before_key_stop',
            'token_tf',
            'normalized_tf'
            'num_set',
            'token_sent_index_feature'
            ]

# tags from Bijankhan, M., Sheykhzadegan, J., Bahrani, M., & Ghayoomi, M. (2011). Lessons from building a Persian written corpus: Peykare. Language Resources and Evaluation, 45, 143–164.
tags = ['A0Empty', 'ADV', 'ADVe', 'AJ', 'AJe', 'CONJ', 'CONJe', 'DET', 'N', 'NUM', 'NUMe', 'Ne', 'P', 'POSTP', 'POSTPe',
        'PRO', 'PROe',
        'PUNC', 'PUNCe', 'V', 'Ve', 'RES', 'CL', 'CLe', 'DETe', 'INT', 'INTe', 'Pe', 'RESe']


def get_continuous_chunks(chunk_func, posttags):
    chunked = chunk_func(posttags)
    continuous_chunk = []
    current_chunk = []
    continuous_chunk_taged = []

    for subtree in chunked:
        if type(subtree) == Tree:
            current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
            continuous_chunk_taged.append(subtree.leaves())
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue

    return continuous_chunk, continuous_chunk_taged


def get_all_NPs(_token_with_tags):
    NP_grammar = r"""NP:{(<Ne>+<AJe>*<AJ>?(<N>?|<Ne>+N)(P|Pe|CONJ))?<Ne>+<AJe>*<AJ>?(<N>?|<Ne>+N)}"""
    chunker = RegexpParser(NP_grammar)
    continuous_chunk, continuous_chunk_taged = get_continuous_chunks(chunker.parse, _token_with_tags)
    NPs = continuous_chunk_taged
    return NPs


def is_part_of_np(_all_NPs, token):
    for np_list in _all_NPs:
        if token in np_list:
            return True


def encode_string_to_integer(strfeature):
    return int(''.join(str(ord(c)) for c in strfeature))


def decode_patern(paterncode):
    tag_codes = [paterncode[i:i + 2] for i in range(0, len(paterncode), 2)]
    tags = [decode_teg(code) for code in tag_codes]
    return '-'.join(tags)


def encode_tag(tag):
    return str(tags.index(tag)).zfill(2)


def decode_teg(index):
    return tags[int(index)]


def make_text_ready(inputtext):
    # preprocess
    # remove retweeted
    tw_pre_obj = tw_pre(input_text=inputtext)
    tw_pre_obj.remove_retweeted_part()
    # remove emoji
    tw_pre_obj.remove_emoji()
    # clean by tweeter util
    text_ct = tw_pre_obj.clean_tweets(Clean_tweet_actions.removeMention,
                                      Clean_tweet_actions.removeUrl)
    # clean by virastar
    vira_pre_obj = vira_pre.PersianEditor()
    text = vira_pre_obj.cleanup(text_ct)

    # tokenize

    prepobj = per_pre(input_text=text,
                      stemmer_method=Stemmer_method.HTaghizadeh,
                      remove_stopword=False)
    list_alltokensWithTags = prepobj.get_original_token_tagged()

    tokens = [a[0] for a in list_alltokensWithTags]
    return tokens, list_alltokensWithTags, prepobj.num_setns, prepobj.token_sent_index


def feature_extraction(inputtext):
    tokens, _tokens_with_tags, num_setns, token_sent_index = make_text_ready(inputtext)
    feature_vectores = []

    Stop_words = StopListReader(StopLists.stop_hazm).get()
    PersianVerbs = getPersianVerbs_dataset()
    all_NPs = get_all_NPs(_tokens_with_tags)
    Start_Stops = StopListReader(StopLists.stop_start_words).get()
    After_Stops = StopListReader(StopLists.stop_after_words).get()

    for i, token_with_tag in enumerate(_tokens_with_tags):
        num_set = num_setns
        token_sent_index_feature = token_sent_index[i]

        token = token_with_tag[0]
        # token_tf
        token_tf = [a[0] for a in _tokens_with_tags].count(token)
        # normalized_tf
        normalized_tf = token_tf / len(_tokens_with_tags)

        tag = token_with_tag[1]
        tag_after = 'A0Empty'
        tag_before = 'A0Empty'

        token_before = '-1'
        token_after = '-1'
        is_after_StartStops = False
        is_before_key_stop = False

        # _is_part_of_NP
        is_np_part = token_with_tag in list(itertools.chain.from_iterable(all_NPs))
        # is after stop word
        # is after verb
        if i > 0:
            is_after_StopWord = _tokens_with_tags[i - 1][0] in Stop_words
            is_after_verb = _tokens_with_tags[i - 1][0] in PersianVerbs
            tag_before = _tokens_with_tags[i - 1][1]
            token_before = _tokens_with_tags[i - 1][0]
            if token_before.strip() in Start_Stops:
                is_after_StartStops = True


        else:
            is_after_StopWord = False
            is_after_verb = False

        # is before Stop word
        # is before verb
        # is before quote
        if i < len(_tokens_with_tags) - 1:
            is_before_StopWord = _tokens_with_tags[i + 1][0] in Stop_words
            is_before_verb = _tokens_with_tags[i + 1][0] in PersianVerbs
            is_before_quote_punc = _tokens_with_tags[i + 1][0] == ':'
            tag_after = _tokens_with_tags[i + 1][1]
            token_after = _tokens_with_tags[i + 1][0]
            if token_after.strip() in After_Stops:
                is_before_key_stop = True
        else:
            is_before_StopWord = False
            is_before_verb = False
            is_before_quote_punc = False
        np_index = -1
        np_pattern = 'A0Empty'

        for i, np in enumerate(all_NPs):

            if token in [a[0] for a in np]:
                np_index = i
                np_pattern = '-'.join([a[1] for a in np])
                break
            else:
                np_index = -1

        feature_vectores.append([tag,
                                 is_np_part,
                                 is_after_StopWord,
                                 is_after_verb,
                                 is_before_StopWord,
                                 is_before_verb,
                                 np_index,
                                 np_pattern,
                                 tag_before,
                                 tag_after,
                                 is_after_StartStops,
                                 is_before_key_stop,
                                 token_tf,
                                 normalized_tf,
                                 num_set,
                                 token_sent_index_feature])

        if (feature_vectores[-1][6]):
            pass
            # print(feature_vectores[-1])
            # print(token)
            # print(' '.join([a[0] for a in _tokens_with_tags]))

    return feature_vectores, tokens, _tokens_with_tags


def target_extraction(_tokens_with_tags, _target_keys):
    _target_keys = ast.literal_eval(_target_keys)
    keys = _target_keys
    # keys=[a.strip() for a in _target_keys.split('،')]
    keyparts = []
    for key in keys:
        keyparts.extend(key.strip().split(' '))
    vira_pre_obj = vira_pre.PersianEditor()
    keyparts = [vira_pre_obj.cleanup(a) for a in keyparts]
    targets = []  # fullkey-nonkey-partkey
    for token_with_tag in _tokens_with_tags:
        token = token_with_tag[0].strip()

        if token in keys:
            targets.append('fullkey')
        elif token in keyparts:
            targets.append('partkey')
            if token == 'به':
                print(token)
        else:
            targets.append('nonkey')

    return targets


def convert_feature_to_categorical(db_df):
    le = preprocessing.LabelEncoder()
    le.fit(tags)

    # binary_tags=[bin(a)[2:].zfill(5) for a in le.transform(tags)]
    new_patter_tags = []
    np_patters = list(db_df['np_pattern'])
    for pattern in np_patters:
        partn_tags = pattern.split('-')
        pattern_bin_rep = ''.join([encode_tag(a) for a in partn_tags])
        new_patter_tags.append(pattern_bin_rep)

    encoded_features = db_df.copy()
    encoded_features['tag'] = le.transform(db_df['tag'])
    encoded_features['tag_before'] = le.transform(db_df['tag_before'])
    encoded_features['tag_after'] = le.transform(db_df['tag_after'])
    encoded_features['np_pattern'] = new_patter_tags
    encoded_features[['is_np_part',
                      'is_after_StopWord',
                      'is_after_verb',
                      'is_before_StopWord',
                      'is_before_verb',
                      'is_after_StartStops',
                      'is_before_key_stop'
                      ]] = \
        (encoded_features[['is_np_part',
                           'is_after_StopWord',
                           'is_after_verb',
                           'is_before_StopWord',
                           'is_before_verb',
                           'is_after_StartStops',
                           'is_before_key_stop'
                           ]] == True).astype(int)

    return encoded_features


def extract_key_target_vector(maintext, target_keys):
    # feature extraction

    feature_vectores, tokens, _tokens_with_tags = feature_extraction(maintext)
    target_vector = target_extraction(_tokens_with_tags=_tokens_with_tags, _target_keys=target_keys)
    return feature_vectores, target_vector, tokens
