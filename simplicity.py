import spacy
def score_sentence(sentence, word):
    """ Scores based on the proportion of the sentence in the range of words related to the target """ 
    # print(sentence, word, type(sentence), type(word))
    ancestors_min = word if (len(list(word.ancestors)) == 0) else min(word.ancestors, key=ind)
    children_max = word if (len(list(word.children)) == 0) else max(word.children, key=ind)
    children_max = max(children_max, word, key=ind) 

    coverage_range = children_max.i - ancestors_min.i
    sentence_len = len(sentence)
    # coverage = (len(ancestors)+len(children))/sentence_len
    coverage = (coverage_range/sentence_len)
    return coverage

def score_sentence2(sentence, word):
    """ Scores based on the proportion of the words related to the target word in the sentence """
    ancestors = list(word.ancestors)
    children = list(word.children)
    related_word_count = len(ancestors) + len(children) + 1
    coverage = related_word_count/len(sentence)
    return coverage


def ind(token):
    return token.i

