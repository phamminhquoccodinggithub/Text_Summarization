"""
    Contain some util functions
"""
import os
import numpy as np


def convert_sentences_to_vector(first_sentence, second_sentence):
    """
        Vertorize sentences based on existence of words strategy
        Ex: (1) John(1), likes(1), watch(1), movies(1), Mary(1), too(1)
            (2) likes(1), watch(1), Mary(1), also(1), football(1), games(1)

        ->  (1) [1, 1, 1, 1, 1, 1, 0, 0, 0]
            (2) [0, 1, 1, 0, 1, 0, 1, 1, 1]

        @param first_sentence: sentence 1
        @param second_sentence: sentence 2
    """
    words = set(first_sentence).union(set(second_sentence))

    first_sent_vec = []
    second_sent_vec = []

    for word in words:
        if word in set(first_sentence):
            first_sent_vec.append(1)
        else:
            first_sent_vec.append(0)
        if word in set(second_sentence):
            second_sent_vec.append(1)
        else:
            second_sent_vec.append(0)

    return np.array(first_sent_vec), np.array(second_sent_vec)


def get_sentence_similarity(vec_st, vec_nd):
    """
        Return a similarity score between 2 vectors based on cosine similarity

        @param vec_st: first sentence vector
        @param vec_nd: second sentence vector
    """
    norm = np.linalg.norm(vec_st) * np.linalg.norm(vec_nd)
    cosine = 0
    if norm != 0:
        cosine = np.dot(vec_st, vec_nd) / norm
    return cosine


def get_similarity_matrix(sentences):
    """
        Return a similarity matrix of sentences

        @param sentences: a list input sentences
    """
    similarity_matrix = np.zeros([len(sentences), len(sentences)]) # Make a zero matrix
    for i,_ in enumerate(sentences):
        for j,_ in enumerate(sentences):
            if i != j:
                # Vectorize sentences
                veci, vecj = convert_sentences_to_vector(sentences[i], sentences[j])
                # Find similarities between the sentences and make 2D matrix
                similarity_matrix[i][j] = get_sentence_similarity(veci, vecj)
    return similarity_matrix


def get_text_summarization(page_rank, sentences, num_sum=18):
    """
        Return text summarization

        @param page_rank: a list scores of each sentence using page rank algorithm
        @param sentences: a list input sentences
        @param num_sum: number of sentences to form the summary
    """
    ranked_sentences = sorted(((page_rank[i],s) for i,s in enumerate(sentences)), reverse=True)

    text = ""
    # Generate summary
    for i in range(num_sum):
        text += "".join(ranked_sentences[i][1]) + "<NewLine>"
    sentences = text.split("<NewLine>")

    return '\n'.join(sentences)


def write_file(contents, file_name, sum_dir = "OUTPUT/"):
    """
        Write file

        @param contents: contents used to write into file
        @param file_name: name of the file to save contents
        @param sum_dir: directory to save files
    """
    if not os.path.exists(sum_dir):
        os.makedirs(sum_dir)
    with open(sum_dir + file_name, "w", encoding="utf-8") as f:
        f.write(contents)


def get_num_of_sent_in_summary(file_name, sum_dir="DUC_SUM"):
    """
        Return the number of sentences in summary

        @param sum_dir: directory to DUC_SUM
        @param file_name
    """
    with open(sum_dir + "/" + file_name, encoding="utf-8") as f:
        text = f.readlines()
    return len(text)
