# Example Sentence Generator

The purpose of this project is to develope a more effective way of generating a set of example sentences for given of words.

We loosely define what makes an effective example sentence as a sentence with the properties:

1)  **Simplicity**:    As Einstein said (kind of), an example sentence should be as as simple as possible, but no simpler.  There should be enough words in the sentence to elucidate the meaning of the word, but not so many that the reader gets derailed by superfluous nonsense. 

2)  **Precision**:    How well does the sentence distinguish this word from semantically adjacent words?   If the word is "largesse", does the example sufficiently demonstrate how it is different from mere "generosity"?  

3)  **Topicality**:   How well does the sentence reflect something that is of current interest to people? While it's still an untested hypothesis, we believe that a factual sentence reinforces the meaning of the word more durably than the kind of "synthetic" sentences you often find in dictionaries, and we further think that topics in the news connect well with people.

4)  **Interestingness**:   How engaging and memorable is the sentence?   Some sentences are dry, others are fun. Similar to topicality, we hope that a more interesing sentence is more more memorable, and therefore more effective teaching users the meaning of a word


# Usage
For now, the code here is tailored towards running smaller examples fit for running locally. We begin by pulling down a corpus of wikipedia articles, and then parsing them for sentences that contain a word that we are interested in, then writing those sentences and some metadata (grouped by word) to a json file. To do this, run `python process_wikipedia_data.py`. We can now use [BERT](https://github.com/google-research/bert) to extract bidirectional contextual embeddings for a specific word in each of these sentences. To extract the features for a word of interest from all of the sentences associated with that word in our json file, we can call `./run_extractions.sh [word1-word2-word3...]`. The argument takes a string of words to analyze seperated by hyphans. This is just a shortcut for calling a python script with some default arguments set. After this script extracts the features, we can run `python graph_usage_distributions.py --words [word1-word2-word3...]` which outputs an html file with the graph to `./graphs/`.