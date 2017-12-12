from __future__ import division
import textrank_
import numpy as np
import nltk
import scipy

class hits:

    def __init__(self):
        filename_list = textrank_.read_filename()
        recall_list = []
        precision_list = []
        fmeasure_list = []
        for filename in filename_list:
            self.content, self.title, self.text = textrank_.read_file(filename)
            tags = textrank_.extract_content(self.content)
            self.candidates, self.reverse_candidates = textrank_.find_candidates(tags)
            # ----- 0/1 -----
            # self.edge_weights = self.get_weights(self.content, 5)
            # ----- stepwise -----
            # self.edge_weights = self.get_weighted_weights(self.content, 5, gaussian_filter=False)
            # ----- Gaussian -----
            self.edge_weights = self.get_weighted_weights(self.content, 5, gaussian_filter=True)
            self.h = np.ones((len(self.candidates), 1), dtype=float)
            self.h /= np.sqrt(len(self.candidates))
            self.a = np.ones((len(self.candidates), 1), dtype=float)
            self.a /= np.sqrt(len(self.candidates))
            for i in range(100):
                self.update()
            self.h_index = sorted(range(len(self.h)), key=lambda x: self.h[x], reverse=True)
            self.a_index = sorted(range(len(self.a)), key=lambda x: self.a[x], reverse=True)
            # ----- with title -----
            self.get_keywords(ahtype="ah", plus_title=True)
            # ----- without title -----
            # self.get_keywords(ahtype="ah", plus_title=False)
            test_filename = filename.split(".")[0] + ".uncontr"
            true_keywords = textrank_.read_true_keywords(test_filename, self.content)
            if len(true_keywords) == 0:
                continue
            # ----- PoS = noun for single keywords -----
            self.extracted_keywords = self.filter_single(self.extracted_keywords)
            recall, precision, fmeasure = textrank_.evaluation(self.extracted_keywords, true_keywords)
            recall_list.append(recall)
            precision_list.append(precision)
            fmeasure_list.append(fmeasure)
            print("recall: " + str(recall) + " precision: " + str(precision) + " fmeasure: " + str(
                fmeasure))

        print("Total recall: " + str(sum(recall_list) / len(recall_list)))
        print("Total precision: " + str(sum(precision_list) / len(precision_list)))
        print("Total f-measure: " + str(sum(fmeasure_list) / len(fmeasure_list)))

    def filter_single(self, set):
        temp_set = set.copy()
        for word in set:
            list = word.strip().split(' ')
            if len(list) == 1:
                if nltk.pos_tag(list)[0][1][0] != 'N':
                    temp_set.remove(word)
        return temp_set

    def get_keywords(self, ahtype="ah", plus_title=False):
        if ahtype == "ah":
            ah_index = self.h_index + self.a_index
        if ahtype == "a":
            ah_index = self.a_index
        if ahtype == "h":
            ah_index = self.h_index
        self.single_keyword = []
        for i in range(int(len(ah_index) / 3)):
            self.single_keyword.append(self.reverse_candidates[ah_index[i]])
        if plus_title:
            temp = textrank_.extract_keywords_from_title(self.title)
            self.single_keyword += temp
        self.extracted_keywords = textrank_.collapsing(self.single_keyword, self.content)

    def get_weights(self, intext, window_size):
        weights = np.zeros((len(self.candidates), len(self.candidates)), dtype=float)
        token_list = nltk.word_tokenize(intext)
        for i1 in range(len(token_list)):
            if token_list[i1] in self.candidates:
                for i2 in range(i1 - window_size, i1):
                    if token_list[i2] in self.candidates:
                        weights[self.candidates[token_list[i2]], self.candidates[token_list[i1]]] += 1.0
        return weights

    def get_weighted_weights(self, text, half_sliding_window, gaussian_filter=False, stepwise_weight=False):
        edge_weights = np.zeros((len(self.candidates), len(self.candidates)))
        token_list = nltk.word_tokenize(text)

        for i in range(len(token_list)):
            if gaussian_filter:
                if token_list[i] in self.candidates:
                    for j in range(i):
                        if token_list[j] in self.candidates:
                            edge_weights[self.candidates[token_list[i]], self.candidates[token_list[j]]] += scipy.stats.norm.pdf(abs(j - i))
            else:
                if token_list[i] in self.candidates:
                    for j in range(max([0, i - half_sliding_window]), i):
                        if token_list[j] in self.candidates:
                            if stepwise_weight:
                                edge_weights[self.candidates[token_list[i]], self.candidates[token_list[j]]] += 2 - abs(
                                    j - i) * float(2) / (half_sliding_window + 1)
                            else:
                                edge_weights[self.candidates[token_list[i]], self.candidates[token_list[j]]] += 1
        for i in range(len(self.candidates)):
            if np.sum(edge_weights[:, i]) != 0:
                edge_weights[:, i] = edge_weights[:, i] / np.sum(edge_weights[:, i])
        return edge_weights

    def update(self):
        self.a = np.dot(self.edge_weights.T, self.h)
        sum_a = np.sum(np.square(self.a))
        self.a /= np.sqrt(sum_a)
        self.h = np.dot(self.edge_weights, self.a)
        sum_h = np.sum(np.square(self.h))
        self.h /= np.sqrt(sum_h)

if __name__ == "__main__":
    hits()
