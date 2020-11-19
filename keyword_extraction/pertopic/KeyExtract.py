from abc import ABC, abstractmethod
from itertools import groupby

from hazm import Stemmer, Lemmatizer
from nltk import Tree, RegexpParser
from pertopic.Preprocess import PersianPreprocess, Stemmer_method
from pertopic.utils import (StopLists,
                    StopListReader,
                    ExternalList,
                    ExternalListReader,
                    getPersianVerbs_dataset)


class KeyExtraction(ABC):

    @abstractmethod
    def get_candidates(self):
        pass

    @abstractmethod
    def get_keys(self):
        pass


class RuleBasedKeyExtraction(KeyExtraction):

    def __init__(self, input_text):
        self._input_text = input_text

    def _get_continuous_chunks(self, chunk_func, posttags):

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

    def get_candidates(self):
        tokens_with_tags = PersianPreprocess(self._input_text,
                                             stemmer_method=Stemmer_method.Hazm,
                                             remove_stopword=False).get_original_token_tagged()
        grammar = r"""  NP:{(<Ne>+<AJe>*<AJ>?(<N>?|<Ne>+N)(P|Pe|CONJ))?<Ne>+<AJe>*<AJ>?(<N>?|<Ne>+N)}"""

        final_candidate = []
        unigram_parts = []
        chunker = RegexpParser(grammar)
        continuous_chunk, continuous_chunk_taged = self._get_continuous_chunks(chunker.parse, tokens_with_tags)
        self._all_candidate = continuous_chunk_taged
        # -----------------------------------------------------------------------------------------------------
        # rule 1 remove the first part of dependent phrase from candidate
        multipartdependent = r"NP:{<Ne><AJ><P><Ne><AJ>|<Ne><Ne><AJ><P><Ne><AJ>}"
        chunker_multipartdependent = RegexpParser(multipartdependent)
        multipartdependent_phrase = self._get_continuous_chunks(chunker_multipartdependent.parse, tokens_with_tags)
        self._low_weight_parts = []
        for candid in multipartdependent_phrase[1]:
            parts = [list(group) for key, group in
                     groupby(candid, key=lambda t: t[1] != 'P') if key]
            self._low_weight_parts.append(parts[0])
        # final_candidate.append([a for a in candidate_phrase1 if a not in self._low_weight_parts])

        # -----------------------------------------------------------------------------------------------------
        # rule 2 remove the 'Ne AJ's where AJ is shifter adj like : بیشتر
        smalAjPhrase = r"NP:{<Ne><N>|<Ne><AJ>}"
        chunker_AjPhrases = RegexpParser(smalAjPhrase)
        smalAjPhrase_phrases = self._get_continuous_chunks(chunker_AjPhrases.parse, tokens_with_tags)
        self._low_weight_Ajphrase = []
        shifters = ExternalListReader(ExternalList.shifters).get()
        stops = StopListReader(StopLists.stop_hazm).get()
        persian_verbs = getPersianVerbs_dataset()
        # stops2=getdata.getMaherStopWords()
        stemmer = Stemmer()
        lematizer = Lemmatizer()
        for candid in smalAjPhrase_phrases[1]:

            nepart = candid[0][0]
            adjorN = candid[1][0]
            if adjorN in shifters \
                    or stemmer.stem(adjorN) in shifters \
                    or lematizer.lemmatize(adjorN) in shifters \
                    or nepart in (stops + persian_verbs) \
                    or adjorN in (stops + persian_verbs):
                self._low_weight_Ajphrase.append(candid)
            else:
                unigram_parts.append(nepart)

        # -----------------------------------------------------------------------------------------------------
        # rule 6 remove adj from long adj phrase with shifter adj
        # remove low_weight_long and add low_weight_long_alternative
        longAjPhrase = r"NP:{<Ne><Ne><AJ>}"
        chunker_longAdj = RegexpParser(longAjPhrase)
        longAjPhrase_phrases = self._get_continuous_chunks(chunker_longAdj.parse, tokens_with_tags)
        self._low_weight_long = []
        self._low_weight_long_alternative = []
        for candid in longAjPhrase_phrases[1]:
            ajpart = candid[2][0]
            if ajpart in shifters \
                    or stemmer.stem(ajpart) in shifters in shifters:
                self._low_weight_long.append(candid)
                self._low_weight_long_alternative.append(candid[0:2])
        # -----------------------------------------------------------------------------------------------------
        # rule 5
        multipartgrammer = r"NP:{<N><Pe><Ne><AJ><CONJ><AJ>}"
        chunker_multipartgrammer = RegexpParser(multipartgrammer)
        multipartgrammer_phrase = self._get_continuous_chunks(chunker_multipartgrammer.parse, tokens_with_tags)
        self._candidate_phrase2 = []
        if len(multipartgrammer_phrase[1]) > 0:
            self._candidate_phrase2 = [list(group) for key, group in
                                       groupby(multipartgrammer_phrase[1][0], key=lambda t: t[1] != 'Pe') if key]

        final_candidate = []
        print(50 * '-' + 'allcandid after rule 1' + 50 * '-')
        final_candidate = [a for a in self._all_candidate if a not in self._low_weight_parts]
        self.get_candids(final_candidate)

        print(50 * '-' + 'allcandid after rule 2' + 50 * '-')
        final_candidate = [a for a in final_candidate if a not in self._low_weight_Ajphrase]
        self.get_candids(final_candidate)

        print(50 * '-' + 'allcandid after rule 6' + 50 * '-')
        final_candidate = [a for a in final_candidate if a not in  self._low_weight_long]
        final_candidate.extend( self._low_weight_long_alternative)
        self.get_candids(final_candidate)

        print(50 * '-' + 'allcandid after rule 5' + 50 * '-')
        final_candidate.extend( self._candidate_phrase2)
        self.get_candids(final_candidate)
        # ----------------------------------------------------post process--------------------------------------------------

        finall = self.get_candids(final_candidate, False)

        # post process
        for i, token in enumerate(finall):
            if str(token).endswith('\u200cهای'):
                finall[i] = token[0:-4]
            elif str(token).endswith('\u200cها'):
                finall[i] = token[0:-3]
            elif str(token).endswith('\u200cی'):
                finall[i] = token[0:-2]
        # remove non sense from first
        shouldremove = []
        persian_verbs = getPersianVerbs_dataset()
        strstops = StopListReader(StopLists.stop_start_words).get()
        for token in finall:

            for st in strstops:
                if token.startswith(st + ' '):
                    shouldremove.append(token)
                    break
            for verb in persian_verbs:
                if token.startswith(verb + ' ') or token.endswith(' ' + verb):
                    shouldremove.append(token)
        finall = [a for a in finall if a not in shouldremove]

        # remove my stoplist
        unigrams = [a for a in finall if len(a.split()) == 1]
        bigrams = [a for a in finall if len(a.split()) == 2]
        others = [a for a in finall if len(a.split()) > 2]

        myfinall = unigrams + bigrams + others

        # remove #
        finall = [a.replace('#', ' ').replace('_', ' ') for a in myfinall]
        finall = [a.replace('\u200c', ' ') for a in finall]

        # remove verb which did not seen by POSTagger
        final_candidate = [a for a in finall if a not in persian_verbs + stops and len(a) > 1]
        print(50 * '---')
        for f in final_candidate:
            print(f)

        return final_candidate

    def get_all_candidate(self):
        return self._all_candidate

    def get_candidate_after_rule_1(self):
        return [a for a in self._all_candidate if a not in self._low_weight_parts]

    def get_candidate_after_rule_2(self):
        return [a for a in self.get_candidate_after_rule_1() if a not in self._low_weight_Ajphrase]

    def get_candidate_after_rule_6(self):
        final_candidate = [a for a in self.get_candidate_after_rule_2() if a not in self._low_weight_long]
        final_candidate.extend(self._low_weight_long_alternative)
        return final_candidate

    def get_candidate_after_rule_5(self):
        return self.get_candidate_after_rule_6() + self._candidate_phrase2

    @staticmethod
    def get_candids(finall_candids, doprint=True):
        cc = [' '.join([b[0] for b in a]) for a in finall_candids]
        if doprint:
            for c in cc:
                print(c)
        return cc

    def get_keys(self):
        pass


if __name__=='__main__':
    text='پيام تشكر رهبر انقلاب اسلامی در پی حضور پرشور مردم در انتخابات رياست جمهورى‌ و انتخاب روحانی'

    obj=RuleBasedKeyExtraction(text)
    print(obj.get_candidates())
    print(RuleBasedKeyExtraction.get_candids(obj.get_candidate_after_rule_1()))
    print(RuleBasedKeyExtraction.get_candids(obj.get_candidate_after_rule_2()))
    print(RuleBasedKeyExtraction.get_candids(obj.get_candidate_after_rule_6()))


