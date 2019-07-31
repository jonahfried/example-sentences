import json
import pandas

with open("../example_sentences.json") as file:
    sentences_from_json = json.load(file)
    
example_sentences = {
    word:[sentence_data["sentence"]["string"].replace("\t", " ") for sentence_data in sentences]
    for (word, sentences) 
    in sentences_from_json.items()
}

word_sentence_pairs = []
for word, lst in example_sentences.items():
    for sentence in lst:
        word_sentence_pairs.append((word, sentence))
        
data = pd.DataFrame(word_sentence_pairs)
data.to_csv(path_or_buf="train.tsv", sep="\t", index=False)