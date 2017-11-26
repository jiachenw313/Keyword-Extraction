import numpy as np
import math
import re
import operator
import nltk
import glob, os
from scipy.stats import norm
np.set_printoptions(threshold=np.nan)

def main():
    filename_list = read_filename()

    # evaluation list
    recall_list = []
    precision_list = []
    fmeasure_list = []

    # hyperparameter
    half_sliding_window = 2
    gaussian_filter = False
    stepwise_weight = False
    damping_factor = 0.85
    convergence_threshold = 0.0001
    # num_keywords = 10


    for filename in filename_list:
        print(filename)
        content, title, text = read_file(filename)
        tags = extract_content(text)
        # print(tags)


        # find keywords from tile directly
        keywords_from_title = extract_keywords_from_title(title)
        # print(keywords_from_title)

        # find vertices for textrank
        candidates, reverse_candidates = find_candidates(tags)
        # print(candidates)
        assert(len(candidates) == len(reverse_candidates))
        num_keywords = int(len(candidates)/3)

        # candidate vectors, initialize all the score to be 1
        word_vector = np.full((len(candidates), 1), 1.0)
        # calcualte all the edge weights
        edge_weights = calculate_edge_weight(text, candidates, half_sliding_window, gaussian_filter, stepwise_weight)
        word_vector, num_iter = run_pagerank(word_vector, edge_weights, damping_factor, convergence_threshold)
        keywords_from_text = convert_to_keywords(word_vector, reverse_candidates, num_keywords)
        # print(keywords_from_text)
        keywords = (set(keywords_from_title + keywords_from_text))
        # print("**************************")
        # print(keywords)
        # collapsing
        extracted_keywords = collapsing(keywords, content)

        # evaluation
        test_filename = filename.split(".")[0] + ".uncontr"
        true_keywords = read_true_keywords(test_filename, content)
        if len(true_keywords) == 0:
            continue
        # print(true_keywords)
        # print("***********************")
        # print(extracted_keywords)

        recall, precision, fmeasure = evaluation(extracted_keywords, true_keywords)
        recall_list.append(recall)
        precision_list.append(precision)
        fmeasure_list.append(fmeasure)
        print("recall: " + str(recall) + " precision: " + str(precision) + " fmeasure: " + str(fmeasure) + " num_iter: " + str(num_iter))
    
        # break
    print("Total recall: ", sum(recall_list) / len(recall_list))
    print("Total precision: ", sum(precision_list) / len(precision_list))
    print("Total f-measure: ", sum(fmeasure_list) / len(fmeasure_list))
    # print(len(recall_list))

    
def evaluation(extracted_keywords, true_keywords):
    '''
    INPUT:
        extracted_keywords, set of key phrases
        true_keywords, set of tru key phrases
    OUTPUT:
        recall, float
        precision, float
        f_measure, float
    '''
    num_correct = 0
    for word in extracted_keywords:
        if word in true_keywords:
            num_correct += 1
    recall = float(num_correct) / len(true_keywords)
    precision = float(num_correct) / len(extracted_keywords)
    if precision + recall == 0:
        f_measure = 0
    else:
        f_measure = 2 * precision * recall / (precision + recall)

    return recall, precision, f_measure

def read_true_keywords(test_filename, test_content):
    '''
    INPUT:
        test_filename, string, name of the uncontrolled file
        test_content, string
    OUTPUT:
        true_keywords, list of keywords that appear in abstract
    '''
    with open(test_filename) as f:
        content = f.readlines()

    content_str = ""
    for line in content:
        content_str = content_str + " " + line.strip()

    true_keywords = []
    for word in content_str.split(";"):
        if word.strip() != "" and word in test_content:
            true_keywords.append(word.strip())

    # how many uncontrolled keywords appear in content
    # print(test_content)
    # print(true_keywords)
    # for line in content:
    # 	for word in line.split(";"):
    # 		if word.strip() != "":
    # 			true_keywords.append(word.strip())
    return true_keywords

def collapsing(keywords, content):
    '''
    INPUT:
        keywords, list of words
        content, string
    '''
    # tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    # content_token = tokenizer.tokenize(content)
    content_token = nltk.word_tokenize(content)
    keyword_phrase = set()

    # print(keywords)
    current_keyword = ""
    for word in content_token:
        if word in keywords:
            if current_keyword == "":
                current_keyword = word
            else:
                current_keyword = current_keyword + " " + word
        else:
            if current_keyword != "":
                if len(current_keyword.split(" ")) > 1:
                    keyword_phrase.add(current_keyword)
            current_keyword = ""

    for phrase in keyword_phrase:
        words = phrase.split(" ")
        for word in words:
            if word in keywords:
                keywords.remove(word)
    keyword_phrase = keyword_phrase.union(keywords)
    return keyword_phrase


def extract_keywords_from_title(title):
    '''
    INPUT: 
        title, string, title of abstract
    OUTPUT:
        keywords_from_title, list of keywords
    '''
    # tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    # title_tokens = tokenizer.tokenize(title)
    title_tokens = nltk.word_tokenize(title)
    title_tags = nltk.pos_tag(title_tokens)

    keywords_from_title = []

    for tag in title_tags:
        if syntactic_filter(tag[1]):
            keywords_from_title.append(tag[0])

    return keywords_from_title

def convert_to_keywords(word_vector, reverse_candidates, num_keywords):
    '''
    INPUT:
        word_vector, len(candidates) x 1, the score for all the candidates
        reverse_candidates, dict, {<id> : <word>}
        num_keywords, integer
    OUTPUT:
        keywords_from_text, list of keywords
    '''
    keywords_from_text = []
    top_index = np.argsort(word_vector.flatten())[::-1][:num_keywords]
    # print(top_index)
    for idx in top_index:
        keywords_from_text.append(reverse_candidates[idx])
    return keywords_from_text

def run_pagerank(word_vector, edge_weights, damping_factor, convergence_threshold):
    '''
    INPUT:
        word_vector, len(candidates) x 1 vector
        edge_weights, len(candidates) x len(candidates) matrix
        damping_factor, float beteen 0 and 1
        convergence_threshold, float
    '''
    prev_vector = np.zeros(word_vector.shape)
    num_iter = 0
    while(not_converge(prev_vector, word_vector, convergence_threshold)):
        num_iter += 1
        prev_vector = word_vector
        # print(np.dot(edge_weights, word_vector))
        word_vector = 1 - damping_factor + damping_factor * np.dot(edge_weights, word_vector)
        # print(word_vector)
    return word_vector, num_iter


def not_converge(prev_vector, word_vector, convergence_threshold):
    '''
    INPUT: 
        prev_vector, the word vector from last iteration
        word_vector, the word vector for current iteration
        convergence_threshold, float
    OUTPUT:
        True, if the distance between two vectors is greater than convergence_threshold
        False, otherwise
    '''
    norm = np.linalg.norm(word_vector - prev_vector)
    if norm <= convergence_threshold:
        return False
    return True


def calculate_edge_weight(text, candidates, half_sliding_window, gaussian_filter = False,  stepwise_weight = False):
    '''
    INPUT: 
        text, string, the text of abstract
        candidates, dict, {<word>, id}
        half_sliding_window, integer
        gaussian_filter, bool, if true, use gaussian filter instead of sliding window, DEFAULT = False
    OUTPUT:
        edge_weights, len(candidates) x len(candidates) matrix, representing the weights of edges in graph
    '''
    edge_weights = np.zeros((len(candidates), len(candidates)))
    token_list = nltk.word_tokenize(text)

    for i in range(len(token_list)):
        if gaussian_filter:
            if token_list[i] in candidates:
                for j in range(len(token_list)):
                    if token_list[j] in candidates:
                        edge_weights[candidates[token_list[i]], candidates[token_list[j]]] += norm.pdf(abs(j - i))
        else:
            if token_list[i] in candidates:
                for j in range(max([0, i - half_sliding_window]), min([i + half_sliding_window + 1, len(token_list)])):
                    if token_list[j] in candidates:
                        if stepwise_weight:
                            edge_weights[candidates[token_list[i]], candidates[token_list[j]]] += 2 - abs(j - i) * float(2) / (half_sliding_window + 1)
                        else:
                            edge_weights[candidates[token_list[i]], candidates[token_list[j]]] += 1
    # print(edge_weights)
    assert(check_symmetric(edge_weights))

    # normailize
    for i in range(len(candidates)):
        if np.sum(edge_weights[:, i]) != 0:
            edge_weights[:, i] = edge_weights[:, i] / np.sum(edge_weights[:, i])
    return edge_weights


def check_symmetric(matrix, tol = 1e-8):
    return np.allclose(matrix, matrix.T, atol=tol)


def find_candidates(tags):
    '''
    INPUT: tags, a list of tuples, [(<word>, <tag>), ...]
    OUTPUT:
        candidates, dict, KEY = candidate word, VALUE = id
        reverse_candidates, dict, KEY = id, VALUE = candidate word
    '''
    candidates = {}
    reverse_candidates = {}
    for tag in tags:
        if syntactic_filter(tag[1]):
            if tag[0] not in candidates:
                candidates[tag[0]] = len(candidates)
                reverse_candidates[candidates[tag[0]]] = tag[0]
    return candidates, reverse_candidates


def syntactic_filter(tag):
    '''
    INPUT: tag, string, part of speech tag
    OUTPUT:
        True, if tag is a noun or noun phrase, or adjective
        False, otherwise
    '''
    if tag.startswith('N') or tag.startswith('J'):
        return True
    return False



def extract_content(text):
    '''
    INPUT: text, string
    OUTPUT:
        tags, a list of tuples, [(<word>, <tag>), ...]
    '''
    # tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    # tokens = tokenizer.tokenize(text)
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    return tags

def read_file(filename):
    '''
    INPUT: filename, string
    OUTPUT: 
        content, string, title + text
        title, string
        text, string
    '''
    with open(filename) as f:
        content = f.readlines()

    title = content[0].strip()
    text = ""
    for i in range(1, len(content)):
        text += content[i].strip()
        text += " "
    content_str = title + " " + text

    return content_str, title, text


def read_filename():
    '''
    OUTPUT: filename_list, list of file names under Data folder
    '''
    filename_list = []
    for abstract in glob.glob("Hulth2003/Test/*.abstr"):
        filename_list.append(abstract)
    return filename_list


if __name__ == '__main__':
    main()
