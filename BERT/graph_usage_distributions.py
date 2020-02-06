import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import umap
import json
import hdbscan
import numpy as np
from collections import defaultdict
import argparse 
import re
from tqdm import tqdm

def get_data(filename): #Give a use default location option (could do a str comp?)
    ''' loads in data from f"./output/{filename}.jsonl", breaks it up into a list of '''

    with open(f"./output/{filename}.jsonl") as file: 
        data = [json.loads(line) for line in file]
    data_features = []
    labels = []
    text = []
    for sentence in tqdm(data):
        word = sentence["linex_index"].split("-")[0] # the unique index consists of the word chosen for analysis and and incremental variable (ex word-9)
        features = sentence['features']
        sentence_text = sentence['original_sentence']
        for feature in features:
            if feature["token"] == word: #possible error from change due to tokenization
                labels.append(word)
                data_features.append(feature["layers"][0]["values"])
                text.append(sentence_text)
    #             break # only one context per sentence
    return (data_features, labels, text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Define what you would like graphed, and with what parameters')
    parser.add_argument('--words', type=str,help='Words seperated by hyphens (ex: "queen-king-peasent")')


    args = parser.parse_args()
    features, labels, text = get_data(args.words)

    word_data_groups = [] # Package the vector, labels, and text for each sentence together by word

    current_label = labels[0] 
    features_by_word = [] 
    labels_by_word = [] 
    text_by_word = []
    for (i, label) in enumerate(labels):
        if label == current_label:
            features_by_word.append(features[i])
            labels_by_word.append(labels[i])
            text_by_word.append(text[i])
        else:
            current_label = labels[i]
            word_data_groups.append((features_by_word, labels_by_word, text_by_word)) 
            features_by_word = [features[i]] 
            labels_by_word = [labels[i]] 
            text_by_word = [text[i]]
    word_data_groups.append((features_by_word, labels_by_word, text_by_word))

    # MAP DOWN DATA. (Make Function?)
    reducer = umap.UMAP(n_neighbors=80, min_dist=0, init='random', n_epochs=30)
    embeddings = [reducer.fit_transform(group[0]) for group in word_data_groups]
    # if len(word_data_groups) > 1 : # If there is more than one word, prep a graph with all words together
    #     embeddings.append(reducer.fit_transform([group[0] for group in word_data_groups])) 
    clusters_by_embedding = [hdbscan.HDBSCAN().fit_predict(embedding) for embedding in embeddings]
    clustered_by_embedding = [(cluster >= 0) for cluster in clusters_by_embedding] # a list of a lists, each marking whether a case was clustered in paired embedding 
    
    for (i, embedding) in tqdm(enumerate(embeddings)):
        clustered = clustered_by_embedding[i] # The array defining whether they were grouped (could just set to all yes if user tags doesn't care?)
        title = word_data_groups[i][1][0] #a label from the current group
        fig = go.Figure(
            layout = {"title" : {"text": title}},
            data=go.Scatter(
                x=embedding[clustered, 0],
                y=embedding[clustered, 1],
                hovertext=(np.array(word_data_groups[i][2]))[clustered],
                mode='markers',
                marker=dict(
                    color=clusters_by_embedding[i][clustered],#[word_to_id[label] for label in labels[:500]],
                    line=dict(
                        width=1,
                        color='DarkSlateGrey'
                    ),
                )
            )
        )
        plot(fig, filename=f"./output/graphs/{title}.html")