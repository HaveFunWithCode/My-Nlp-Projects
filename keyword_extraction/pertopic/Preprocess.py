import re
from enum import Enum
from hazm import (Normalizer,
                  sent_tokenize,
                  word_tokenize,
                  Stemmer)
import numpy as np
import PersianStemmer as ps
from pertopic.utils import (getPosTagger,
                            StopListReader,
                            StopLists,
                            ExternalListReader,
                            ExternalList)
import urllib.parse as uparse

class Stemmer_method(Enum):
    Hazm = 1
    HTaghizadeh = 2


class Clean_tweet_actions(Enum):
    # xy: x:0(remove)|1(replace)    y:1=hashtag|2=url|3= mention|
    # removePunc=4 rempoveStop=5
    replaceHashtag = '01'
    removeHashtag = '11'

    replaceUrl = '02'
    removeUrl = '12'

    replaceMention = '03'
    removeMention = '13'

    removePunc = '4'
    removeStop = '5'


class PersianPreprocess():
    def __init__(self, input_text,
                 stemmer_method,
                 remove_stopword):
        self.input_text = input_text
        self.stemmer_method = stemmer_method
        self.remove_stoped = remove_stopword

        self.normalizer = Normalizer()
        self.stop_words = StopListReader(StopLists.stop_hazm).get()
        self.tagger = getPosTagger()
        self.token_sent_index = []

        # sentense tokenize and postagging
        normalized_text = self.normalizer.normalize(self.input_text)
        sentences = sent_tokenize(normalized_text)
        self.num_setns = len(sentences)

        _list_allTokens = []
        _list_alltokensWithTags = []

        # specify three main Part Of Speach Tages
        for i, sent in enumerate(sentences):
            sent_tokens = word_tokenize(sent)
            taged_tokens = self.tagger.tag(sent_tokens)

            _list_alltokensWithTags.extend(taged_tokens)
            _list_allTokens.extend(sent_tokens)
            self.token_sent_index.extend(len(_list_alltokensWithTags) * [i])

        if self.remove_stoped:
            stopword_indexes = [i for i, x in enumerate(_list_allTokens) if x in set(self.stop_words)]
            self.clean_tokens = np.delete(_list_allTokens, stopword_indexes)
            for a in sorted(stopword_indexes)[::-1]:
                _list_alltokensWithTags.pop(a)
            self.clean_tokens_taged = _list_alltokensWithTags
        else:
            self.clean_tokens_taged = _list_alltokensWithTags
            self.clean_tokens = _list_allTokens

    def get_stemmed_tokens(self):
        stemmed_tokens = []
        for cleaned_token in self.clean_tokens:
            if (self.stemmer_method == Stemmer_method.HTaghizadeh):
                pps = ps.PersianStemmer()
                stemmed_tokens.append(pps.run(cleaned_token))
            elif (self.stemmer_method == Stemmer_method.Hazm):
                pps = Stemmer()
                stemmed_tokens.append(pps.stem(cleaned_token))
            else:
                pps = ps.PersianStemmer()
                stemmed_tokens.append(pps.run(cleaned_token))
        return stemmed_tokens

    def get_original_token(self):
        return self.clean_tokens

    def get_original_token_tagged(self):
        return self.clean_tokens_taged


class TwitterPreprocess():
    def __init__(self, input_text):

        self.input_text = input_text

        self._mycompile = lambda pat: re.compile(pat, re.UNICODE)
        self._EdgePunct = r"""[  ' " “ ” ‘ ’ < > « » { } ( \) [ \]  ]""".replace(' ', '')
        # _NotEdgePunct = r"""[^'"([\)\]]"""  # alignment failures?
        self._NotEdgePunct = r"""[a-zA-Z0-9]"""
        self._EdgePunctLeft = r"""(\s|^)(%s+)(%s)""" % (self._EdgePunct, self._NotEdgePunct)
        self._EdgePunctRight = r"""(%s)(%s+)(\s|$)""" % (self._NotEdgePunct, self._EdgePunct)
        self._EdgePunctLeft_RE = self._mycompile(self._EdgePunctLeft)
        self._EdgePunctRight_RE = self._mycompile(self._EdgePunctRight)

    def _edge_punct_munge(self, s):
        s = self._EdgePunctLeft_RE.sub(r"\1\2 \3", s)
        s = self._EdgePunctRight_RE.sub(r"\1 \2\3", s)
        return s

    def _regex_or(self, *items):
        r = '|'.join(items)
        r = '(' + r + ')'
        return r

    def _pos_lookahead(self, r):
        return '(?=' + r + ')'

    def _optional(self, r):
        return '(%s)?' % r

    def _depart_sticked_punc_fromString(self, tokens):
        """ check token have persian character ,if it has no character so maybe this is emotion so ignore it  """

        new_tokens = []
        for token in tokens:

            puncs = re.findall(r"[.,!?؟;:،/\\{}()]", token)
            chars = re.findall(r"[\w']+|[ا آ ب پ ت ج چ ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ک گ ل م ن و ه ی ژ]", token)

            if len(puncs) > 0 and len(chars) > 0:
                # if punc is between two word split tokens based on punc as delimitter
                for punc in puncs:
                    token = token.replace(punc, ' ' + punc)
                    parts = token.split()
                    new_tokens.extend(parts)

            else:
                new_tokens.append(token)

        return new_tokens

    def _remove_sticked_punc_fromString(self, tokens):
        """ check token have persian character ,if it has no character so maybe this is emotion so ignore it..."""

        new_tokens = []
        for token in tokens:

            puncs = re.findall(r"[.,!?؟;:،/\\{}()]", token)
            chars = re.findall(r"[\w']+|[ا آ ب پ ت ج چ ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ک گ ل م ن و ه ی ژ]", token)

            if len(puncs) > 0 and len(chars) > 0:
                subtokens = []
                # if punc is between two word split tokens based on punc as delimitter
                for punc in puncs:
                    parts = token.split(punc)
                    # it means punctuation is between two word like علی گفت:ما.... :
                    if '' not in parts:
                        new_tokens.extend(parts)
                    # it means punctuation is end or start of the token
                    else:
                        exclude = set(text.punctuation)
                        token_clean = ''.join(ch for ch in token if ch not in exclude)
                        new_tokens.append(token)
            else:
                new_tokens.append(token)

        return new_tokens

    def _remove_stop_words(self, tokens):
        """ use this function just if you dont wnat to preprocess after tweet preprocess"""
        STOPWORDS = StopListReader(StopLists.stop_hazm).get()
        stopword_indexes = [i for i, x in enumerate(tokens) if x in set(STOPWORDS)]
        clean_tokens = np.delete(tokens, stopword_indexes)
        return list(clean_tokens)

    def _remove_punctuations(self, tokens):

        """ use this just if you dont wnat to preprocess after tweet preprocess """
        puncs = ExternalListReader(ExternalList.punctuations).get()
        stopword_indexes = [i for i, x in enumerate(tokens) if x in set(puncs)]
        clean_tokens = np.delete(tokens, stopword_indexes)
        return list(clean_tokens)

    def _clean_hashtag_from_text(self, text):
        text = text.replace('#', '').replace('_', ' ')
        sent_token_parts = text.split(' ')
        return ' '.join(sent_token_parts)

    def remove_retweeted_part(self):
        if (str(self.input_text).find('Retweeted') != -1):
            retweetindex = str(self.input_text).index('Retweeted')
            if retweetindex != -1:
                puncindex = str(self.input_text).index(':')
                self.input_text=str(self.input_text)[puncindex + 1:-1]

    def remove_emoji(self):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        try:
            self.input_text = emoji_pattern.sub(r'', self.input_text)
        except:
            print(emoji_pattern.sub(r'', self.input_text))


    def _depart_sticked_hashtags_fromtokens(self, tokens):
        # check have sticked hashtags like: #اسلامی#وطن or  ایران سرفراز باد#ایران#طن
        final_tokens = []
        for token in tokens:
            partslen = len(token.split('#'))
            if partslen <= 2:
                final_tokens.append(token)
            else:
                final_tokens.extend(token.replace('#', ' #').split())
        return final_tokens

    def clean_tweets(self, *params):
        """ remove or replace hashtag - mention - link """


        # remove repeted chars from text like ... or !!!
        # text = ''.join(ch for ch, _ in itertools.groupby(text))

        # modify tweet with problems
        text = self.input_text.replace('// ', '//').replace(' //', '//')
        text = text.replace('https://www. ', 'https://www.').replace('/ ?', '/?')
        text = self._edge_punct_munge(text)
        Entity = '&(amp|lt|gt|quot);'
        PunctChars = r'''['“".?!,:;]'''
        should_clean = ['RT']
        listS = []
        tweet = dict()
        processedSentencees = sent_tokenize(text)

        for sent in processedSentencees:

            sent_tokens = sent.split(' ')
            # check have sticked hashtags like: #اسلامی#وطن or  ایران سرفراز باد#ایران#طن
            sent_tokens = self._depart_sticked_hashtags_fromtokens(sent_tokens)
            #
            for word in sent_tokens:

                ##########################################  match URLs   ###############################
                # one-liner URL recognition:
                # Url = r'''https?://\S+'''
                UrlStart1 = self._regex_or('https?://', r'www\.')

                CommonTLDs = self._regex_or('com', 'co\\.uk', 'org', 'net', 'info', 'ca')
                UrlStart2 = r'[a-z0-9\.-]+?' + r'\.' + CommonTLDs + self._pos_lookahead(r'[/ \W\b]')
                UrlBody = r'[^ \t\r\n<>]*?'  # * not + for case of:  "go to bla.com." -- don't want period
                UrlExtraCrapBeforeEnd = '%s+?' % self._regex_or(PunctChars, Entity)
                UrlEnd = self._regex_or(r'\.\.+', r'[<>]', r'\s', '$')
                Url = (r'\b' +
                       self._regex_or(UrlStart1, UrlStart2) +
                       UrlBody +
                       self._pos_lookahead(self._optional(UrlExtraCrapBeforeEnd) + UrlEnd))

                Url_RE = re.compile("(%s)" % Url, re.U | re.I)
                match_url = re.match(Url_RE, word)
                s, n, p, pa, q, f = uparse.urlparse(word)

                ############################################# match mentions ########################

                # Twitter username:
                handle_regex = r"""(?:@)"""
                handle_re = re.compile(handle_regex, re.VERBOSE | re.I | re.UNICODE)

                # def handle_repl(match):
                #   return ' HANDLE '
                # print(w)
                match_mention = re.match(handle_re, word)
                # print(match_mention)

                #############################################  match Hashtags  ######################
                # Twitter hashtags:

                hashtag_regex = r"""(?:\#)"""
                hashtag_re = re.compile(hashtag_regex, re.VERBOSE | re.I | re.UNICODE)

                match_hashtag = re.match(hashtag_re, word)
                ######################################################################################
                paramlist = []
                for param in params:
                    paramlist.append(param)

                if match_url:
                    # if 'urls' in tweet and len(tweet['urls']):
                    #     temp = list(tweet['urls'])
                    #     tweet['urls'] = temp.append(word)
                    # else:
                    #     tweet['urls'] = [word]
                    #
                    # if listS[len(listS) - 1] == '/':
                    #     listS.pop()
                    # if listS[len(listS) - 1] == ':/':
                    #     listS.pop()
                    # # print(listS[len(listS) - 1])
                    # # print(listS[len(listS)-1].replace('https', ''))
                    # temp = listS[len(listS) - 1].replace('https', '')
                    # listS.pop()
                    # listS.append(temp)

                    listS.append('لینک')


                elif '.twitter' in word:
                    temp = listS[len(listS) - 1].replace('pic', '')
                    listS.pop()
                    listS.append(temp)
                    listS.append('لینک')

                elif match_mention:
                    listS.append('منشن')
                elif match_hashtag:
                    if Clean_tweet_actions.removeHashtag not in paramlist and \
                            Clean_tweet_actions.replaceHashtag not in paramlist:
                        temp = self._clean_hashtag_from_text(word)
                        listS.append(temp)
                    else:
                        listS.append('هشتگ')
                else:
                    listS.append(word)

            tweet['clean'] = self._depart_sticked_punc_fromString(listS)

            tweet['s'] = " ".join(listS)
            for param in params:
                if param == Clean_tweet_actions.removeMention:
                    should_clean.append('منشن')
                if param == Clean_tweet_actions.removeHashtag:
                    should_clean.append('هشتگ')
                if param == Clean_tweet_actions.removeUrl:
                    should_clean.append('لینک')

            _tweet_clean = ' '.join([word for word in tweet['clean'] if word not in should_clean])

        if Clean_tweet_actions.removeStop in params:
            _tweet_clean = ' '.join(self._remove_stop_words(_tweet_clean.split(' ')))
        if Clean_tweet_actions.removePunc in params:
            # _tweet_clean=' '.join(self._remove_punctuations(_tweet_clean.split(' ')))
            _tweet_clean = ' '.join(self._depart_sticked_punc_fromString(_tweet_clean.split(' ')))

        return _tweet_clean


if __name__ == "__main__":
    tt = 'اینجا ایران است'

    prepobj = PersianPreprocess(input_text=tt, stemmer_method=Stemmer_method.HTaghizadeh, remove_stopword=False)

    print(prepobj.get_original_token())
    print(prepobj.get_original_token_tagged())
    print(prepobj.get_stemmed_tokens())
    print(prepobj.num_setns)


    # ----------------------------------------
    test = '#پمپئو سیاست‌های تازه #آمریکا علیه #ایران را اعلام کرد. نفوذ منطقه‌ای #ایران و #برنامه_هسته‌ای باید متوقف شود. حقوق مدنی مردم #ایران هم مورد توجه قرار گرفت. https://t.co/9Hwklb7So3 https://t.co/74QBOgNysJ'
    cc = TwitterPreprocess(input_text=test)
    text_ct = cc.clean_tweets(Clean_tweet_actions.removeMention,
                              Clean_tweet_actions.removeUrl)
    print(text_ct)