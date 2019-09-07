#!/usr/local/bin/python3

"""
Extracts example sentences for every word in a vocabulary from a corpus of
Wikipedia articles and produces a JSON file (to OUTPUT_FILE, below)
with the words, sentences, and suitability scores.

The corpus of articles to use is in CORPUS_FILE (on the AWS s3 bucket
given by S3_BUCKET).  Each line in this file is an S3 file pointing to a
document containing article text that was produced by the wikiextractor tool
(https://github.com/attardi/wikiextractor).
"""

import heapq
import json
import logging
import re

import boto3
import spacy   # Used to split the Wikipedia articles into sentences

import simplicity

# S3 bucket and region in which wikipedia data resides
S3_BUCKET = "dougb_wikipedia"
S3_REGION = "us-east-1"

# For each word, output this many example sentences
OUTPUTS_PER_WORD = 5

# File which contains a list of wikipedia data files (as S3 keys) to process
CORPUS_FILE = "enwiki.full"

# File which contains the vocabulary of words to seek example sentences for
VOCAB_FILE = "enwiki.tiny_vocab"
# Exclude examples sentences for these very common words
STOPWORDS_FILE = "enwiki.stopwords"

# File which maps the wikipedia titles to the number of recent pageviews, used as a
# measure of popularity
TITLE_WEIGHTS_FILE = "enwiki.pageviews"
# Only consider articles that are in the top TOPN_TITLES of articles by pageview count
TOPN_TITLES = 100000

# File to write out example sentences
OUTPUT_FILE = "example_sentences.json"

class WikiSentenceExtractor():
    """Extracts example sentences from a corpus of wikipedia documents."""
    def __init__(self, logger=None):
        if not logger:
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger("WikiSentenceExtractor")
        self.logger = logger
        self.s3 = boto3.resource('s3', region_name=S3_REGION)
        self.title_weights = {}  # title -> pageview count
        self.vocab = set()  # Set of words for which to extract example sentences

        # Dictionary of word -> heap of OUTPUTS_PER_WORD example sentences,
        # valued by the suitability score
        self.example_sentences = {}

        self.logger.info("Loading Spacy data...")
        self.nlp = spacy.load('en_core_web_sm')
        # self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

    def _sentence_suitability_score(self, sentence, word):
        """Returns a measure of how well the sentence "sentence" exemplifies
        the word "word".  For now it's just a stub that returns a larger number
        for shorter sentences.  TODO:  Include measures of precision, simplicity,
        topicality, and interestingness into how the score is computed.
        """
        return 1000*simplicity.score_sentence2(sentence, word)

    def _read_s3_file(self, key):
        """Read a text file from s3."""
        self.logger.info("Reading s3 key: s3://%s/%s", S3_BUCKET, key)
        return self.s3.Object(S3_BUCKET, key).get()["Body"].read().decode("utf-8").strip()

    def run(self):
        """Iterate over the corpus, populating example_sentences."""
        # Read vocabulary
        self.vocab = {x.split()[0].lower() for x in
                      self._read_s3_file(VOCAB_FILE).split("\n")}
        # Remove stopwords from vocabulary
        self.vocab = self.vocab.difference(
            {x.split()[0].lower() for x in self._read_s3_file(STOPWORDS_FILE).split("\n")})

        # Read weights for each title
        title_pageviews = self._read_s3_file(TITLE_WEIGHTS_FILE).split("\n")
        for row in title_pageviews:
            if row:
                title, pageview_count = row.split()
                self.title_weights[title.replace("_", " ")] = int(pageview_count)
                if len(self.title_weights) >= TOPN_TITLES:
                    break

        # Read corpus file and process each file
        doc_keys_to_process = self._read_s3_file(CORPUS_FILE).split("\n")
        for doc_key in doc_keys_to_process:
            self._process_corpus_file(doc_key)

    def _process_wiki_doc(self, title, article_content):
        """Process a single wikipedia article."""
        # nlp = spacy.load("en_core_web_sm")
        sentences = list(self.nlp(article_content).sents)        

        self.logger.info("Reading %d sentences in %s", len(sentences), title)

        for sentence in sentences:
            # For each word, score the sentence as an example for the word, and
            # push the results into the heap of example sentences for the word.
            sentence_str = str(sentence)
            sentence_obj = json.dumps({"string": sentence_str, "wiki_title": title})
            for token in sentence:
                str_token = str(token).lower()
                if str_token in self.vocab:
                    score = self._sentence_suitability_score(sentence, token)
                    if str_token not in self.example_sentences:
                        self.example_sentences[str_token] = []
                    current_examples = self.example_sentences[str_token]
                    if len(current_examples) < OUTPUTS_PER_WORD:
                        heapq.heappush(current_examples, (score, sentence_obj))
                    elif score > current_examples[0][0]:
                        heapq.heappushpop(current_examples, (score, sentence_obj))

    def example_sentences_to_json(self):
        """Render the whole set of example sentences as a big JSON document."""
        result = {}
        for key, heap in self.example_sentences.items():
            heap.sort(reverse=True)
            result[key] = [{"score": x[0], "sentence": json.loads(x[1])} for x in heap]
        return json.dumps(result, indent=4)

    def _process_corpus_file(self, s3_key):
        """A corpus file contain multiple wikipedia articles in the XML format
        produced by wikiextractor."""
        article_lines = []
        lines = self._read_s3_file(s3_key).split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("<doc"):
                title = re.sub(".*title=\"", "", line)
                title = re.sub("[\"].*", "", title)
                article_lines = []
                continue
            elif line.startswith("</doc>"):
                if title in self.title_weights:
                    self._process_wiki_doc(title, " ".join(article_lines))
            else:
                article_lines.append(line)

if __name__ == "__main__":
    wse = WikiSentenceExtractor()
    wse.run()
    with open(OUTPUT_FILE, "w") as out_fs:
        print(wse.example_sentences_to_json(), file=out_fs)