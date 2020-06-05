import re
import collections

import tensorflow as tf
from bert import modeling
from bert import tokenization

import umap
import hdbscan 

import extract_features_json as extract

def read_examples(word, packaged_parse_array):
    """
        takes in a a word and an array of 'packed_parse' objects.
        (a 'packaged_parse' is a dict with fields
        "title", "word", "sentence", "simplicity").

        returns InputExample's
    """
    examples = []
    guid_to_packaged_parse = {}
    id_int = 0
    for package in packaged_parse_array:
        sentence = package["sentence"]
        text_a = None
        text_b = None
        m = re.match(r"^(.*) \|\|\| (.*)$", sentence)
        if m is None:
            text_a = sentence 
        else:
            text_a = m.group(1)
            text_b = m.group(2)
        unique_id = ("%s-%d" % (word, id_int))
        guid_to_packaged_parse[unique_id] = package
        examples.append(extract.InputExample(unique_id=(unique_id), text_a=text_a, text_b=text_b))
        id_int += 1
    return examples, guid_to_packaged_parse



def get_feature_data_pairs(extracted_features):
  features = []
  sentence_data = []
  for sentence in extracted_features:
    for sentence_feature in sentence["features"]:
      features.append(sentence_feature["layers"][0]["values"]) 
      sentence_data.append(sentence["original_sentence"].asDict())

  return (features, sentence_data)

def cluster(extracted_features):
  features, sentence_data = get_feature_data_pairs(extracted_features)
  reducer = umap.UMAP(n_neighbors=80, min_dist=0, init='random', n_epochs=30)
  embedding = reducer.fit_transform(features)
  clusters = hdbscan.HDBSCAN().fit_predict(embedding)
  for i, cluster in enumerate(clusters):
    sentence_data[i]["cluster"] = int(cluster)
  return sentence_data





def main(word, packaged_parse_array):
  tf.logging.set_verbosity(tf.logging.INFO)

  layer_indexes = [int(x) for x in LAYERS.split(",")]

  bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG_FILE)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      master=MASTER,
      tpu_config=tf.contrib.tpu.TPUConfig(
          num_shards=NUM_TPU_CORES,
          per_host_input_for_training=is_per_host))

  examples, guid_to_packaged_parse = read_examples(word, packaged_parse_array)

  features = extract.convert_examples_to_features(
      examples=examples, seq_length=MAX_SEQ_LENGTH, tokenizer=tokenizer)

  unique_id_to_feature = {}
  for feature in features:
    unique_id_to_feature[feature.unique_id] = feature

  model_fn = extract.model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=INIT_CHECKPOINT,
      layer_indexes=layer_indexes,
      use_tpu=USE_TPU,
      use_one_hot_embeddings=USE_ONE_HOT_EMBEDDINGS)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=USE_TPU,
      model_fn=model_fn,
      config=run_config,
      predict_batch_size=BATCH_SIZE)

  input_fn = extract.input_fn_builder(
      features=features, seq_length=MAX_SEQ_LENGTH)

  extracted_features = []

  for result in estimator.predict(input_fn, yield_single_examples=True):
    unique_id = result["unique_id"].decode("utf-8")
    feature = unique_id_to_feature[unique_id]
    
    output_json = collections.OrderedDict()
    output_json["linex_index"] = unique_id
    output_json["original_sentence"] = guid_to_packaged_parse[unique_id]
    all_features = []
    for (i, token) in enumerate(feature.tokens):
      if (token == word) or FULL_OUTPUT:
        all_layers = []
        for (j, layer_index) in enumerate(layer_indexes):
          layer_output = result["layer_output_%d" % j]
          layers = collections.OrderedDict()
          layers["index"] = layer_index
          layers["values"] = [
              round(float(x), 6) for x in layer_output[i:(i + 1)].flat
          ]
          all_layers.append(layers)

        features = collections.OrderedDict()
        features["token"] = token
        features["layers"] = all_layers if FULL_OUTPUT else [all_layers[0]]
        all_features.append(features)
    output_json["features"] = all_features
    extracted_features.append(output_json)
  
  return cluster(extracted_features)


LAYERS =  "-1,-2,-3,-4"

BERT_CONFIG_FILE = "./bert-base-uncased/bert_config.json"

MAX_SEQ_LENGTH = 128

INIT_CHECKPOINT = "./bert-base-uncased/bert_model.ckpt"

VOCAB_FILE = "./bert-base-uncased/vocab.txt"

DO_LOWER_CASE = True

BATCH_SIZE = 8 # maybe want this bigger on more cores?

USE_TPU = False

MASTER = None

NUM_TPU_CORES = 8

USE_ONE_HOT_EMBEDDINGS = False

FULL_OUTPUT = False

