import findspark
findspark.init("/opt/spark")

import pyspark
from pyspark.sql.functions import *
from pyspark.sql.types import *

import heapq
import json
import logging
import re

import pandas as pd

import boto3
import spacy   # Used to split the Wikipedia articles into sentences

import text_to_cluster as cluster


# ——————— CONSTANTS —————————

# S3 bucket and region in which wikipedia data resides
S3_BUCKET = "datamuse-misc"
S3_REGION = "us-east-1"

# For each word, output this many example sentences
OUTPUTS_PER_WORD = 3000 #5

# File which contains a list of wikipedia data files (as S3 keys) to process
CORPUS_FILE = "enwiki.full"

# File which contains the vocabulary of words to seek example sentences for
VOCAB_FILE = "enwiki.vocab"

# Exclude examples sentences for these very common words
STOPWORDS_FILE = "enwiki.stopwords"

# File which maps the wikipedia titles to the number of recent pageviews, used as a
# measure of popularity
TITLE_WEIGHTS_FILE = "enwiki.pageviews"
# Only consider articles that are in the top TOPN_TITLES of articles by pageview count
TOPN_TITLES = 100000
CORPUSES_TO_READ = 1

# ______ END OF CONSTANTS ______

s3 = boto3.resource('s3', region_name=S3_REGION)

def read_s3_file(key):
    return s3.Object(S3_BUCKET, f"wikipedia/{key}").get()["Body"].read().decode("utf-8").strip()

title_weights = {}  # title -> pageview count
vocab = {x.split()[0].lower() for x in read_s3_file(VOCAB_FILE).split("\n")} # Set of words for which to extract example sentences
vocab = vocab.difference(
    {x.split()[0].lower() for x in read_s3_file(STOPWORDS_FILE).split("\n")}
)

# Dictionary of word -> heap of OUTPUTS_PER_WORD example sentences,
# valued by the suitability score
example_sentences = {}

nlp = spacy.load('en_core_web_sm')

def _process_corpus_file(s3_key):
    """A corpus file contain multiple wikipedia articles in the XML format
    produced by wikiextractor."""
    article_lines = []
    lines = read_s3_file(s3_key).split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("<doc"):
            title = re.sub(".*title=\"", "", line)
            title = re.sub("[\"].*", "", title)
            article_lines = []
            continue
        elif line.startswith("</doc>"):
            return {"title": title, "article": " ".join(article_lines)}
        else:
            article_lines.append(line)
    return {"title": title, "article": " ".join(article_lines)}

def score_sentence(sentence, word):
    """ Scores based on the proportion of the words related to the target word in the sentence """
    ancestors = list(word.ancestors)
    children = list(word.children)
    related_word_count = len(ancestors) + len(children) + 1
    coverage = related_word_count/len(sentence)
    return coverage

def _process_wiki_doc(article_content):
    """Process a single wikipedia article."""
    sentences = list(nlp(article_content).sents)
    
    sentence_word_pairs = []
    for sentence in sentences:
        # For each word, score the sentence as an example for the word, and
        # push the results into the heap of example sentences for the word.
        sentence_str = str(sentence)
        
        tokens = []
        scores = []
        for token in sentence:
            str_token = str(token).lower()
            if str_token in vocab:
                tokens.append((str_token))
                scores.append(str(score_sentence(sentence, token)))
        sentence_word_pairs.append({"sentence":[sentence_str], "word":tokens, "score":scores})
    return sentence_word_pairs
 
# Would it be cleaner to use a StructType() here?
process_article = udf(
    _process_wiki_doc, 
    ArrayType(MapType(StringType(), ArrayType(StringType())))
)

def sentence_word_bundle_to_words(col):
    return col["word"]

def sentence_word_bundle_to_sentence(col):
    return col["sentence"][0]

bundle_to_words = udf(sentence_word_bundle_to_words, ArrayType(StringType()))
bundle_to_sentence = udf(sentence_word_bundle_to_sentence, StringType())

if __name__ == "__main__":
    spark = pyspark.sql.SparkSession.builder.master("local").appName("spark_testing").getOrCreate()
    # Can't use boto3 with rdd's (as far as I can tell)
    # So we must load all of the articles manually.
    files = []
    for i, key in enumerate(read_s3_file(CORPUS_FILE).split("\n")):
        files.append(_process_corpus_file(key))
        if i > 1:
            break
    
    # We can load the articles into a Dataframe
    # NOTE: This uses a depricated feature (interpretting dict type).
    #       We should be using Rows with a schema
    #       that we define beforehand
    rdd_files = spark.createDataFrame(files).select("title", "article")
    # break into {sentences, words} for each sentence in each article
    expanded = rdd_files.select(
        "title",
        explode(process_article(rdd_files.article)).alias("sentence_word_bundle")
    )
    # Remove the article column from expanded now? It stores a lot of data and I don't think we need it again after here.

    words_by_sentence = expanded.select(
        "*",
        col("sentence_word_bundle.sentence"),
        arrays_zip(col("sentence_word_bundle.word"), col("sentence_word_bundle.score")).alias("tmp_zip")
        
    )
    words_by_sentence = words_by_sentence.withColumn("tmp_zip", explode("tmp_zip"))
    words_by_sentence = words_by_sentence.select(
        "title",
        explode("sentence").alias("sentence"),
        col("tmp_zip.0").alias("word"),
        col("tmp_zip.1").alias("score")
        
    )

    # words_by_sentence.select(
    #     "title", "sentence", "word", "score"
    # ).show() #toPandas().to_csv("words_by_sentence.csv")

    package = udf(
        lambda title, word, sentence, score: [title, word, sentence, float(score)], #{"title" : title, "word": word, "sentence":sentence, "simplicity":score},
        StructType([
            StructField("title", StringType(), True),
            StructField("word", StringType(), True),
            StructField("sentence", StringType(), True),
            StructField("simplicity", FloatType(), True)
        ])
    )

    words_by_sentence = words_by_sentence.select(
        "word",
        package(
            words_by_sentence.title, 
            words_by_sentence.word, 
            words_by_sentence.sentence, 
            words_by_sentence.score
        ).alias("packaged_parse")
    )


    grouped = words_by_sentence.groupBy("word").agg(collect_list("packaged_parse"))
    

    cluster_spark = udf(
        cluster.main,
        ArrayType(
            StructType([
                StructField("title", StringType(), True),
                StructField("word", StringType(), True),
                StructField("sentence", StringType(), True),
                StructField("simplicity", FloatType(), True),
                StructField("cluster", IntegerType(), True),
            ])
        )
    )

    grouped = grouped.where(col("word") == "1970s").select(
        "*",
        explode(cluster_spark(grouped.word, col("collect_list(packaged_parse)"))).alias("packaged_parse")
    )

    grouped.select(
        "packaged_parse.title",
        "packaged_parse.word",
        "packaged_parse.sentence",
        "packaged_parse.simplicity",
        "packaged_parse.cluster"
    ).show()

    spark.stop()