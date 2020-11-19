from pertopic.utils import getPersianVerbs_dataset,StopLists,StopListReader

class PostProcess():
    def __init__(self,keyphrase):
        self._keyphrase= keyphrase

    def remove_sticked_verbs(self):
        '''
        this method remove sticked verbs which could not detected in preprocess and PosTagging
        :param key phrase: keyphrase
        :return: key phrase without sticked verb
        '''
        verbsall = getPersianVerbs_dataset()
        verbs = StopListReader(StopLists.stop_post_verbs) + verbsall

        phrase_parts = self._keyphrase.split(' ')

        if phrase_parts[0] in verbs:
            del phrase_parts[0]
        if phrase_parts[-1] in verbs:
            del phrase_parts[-1]

        return ' '.join(phrase_parts)
