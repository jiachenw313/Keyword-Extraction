from __future__ import division
import textrank_
import numpy as np
import nltk

class hits:

    def __init__(self):
        filename_list = textrank_.read_filename()
        recall_list = []
        precision_list = []
        fmeasure_list = []
        for filename in filename_list:
            self.content, title, text = textrank_.read_file(filename)
            tags = textrank_.extract_content(self.content)
            self.candidates, self.reverse_candidates = textrank_.find_candidates(tags)
            self.edge_weights = self.get_weights(self.content, 5)
            self.h = np.ones((len(self.candidates), 1), dtype=float)
            self.h /= np.sqrt(len(self.candidates))
            self.a = np.ones((len(self.candidates), 1), dtype=float)
            self.a /= np.sqrt(len(self.candidates))
            for i in range(100):
                self.update()
            self.h_index = sorted(range(len(self.h)), key=lambda x: self.h[x], reverse=True)
            self.a_index = sorted(range(len(self.a)), key=lambda x: self.a[x], reverse=True)
            self.get_keywords("ah")
            test_filename = filename.split(".")[0] + ".uncontr"
            true_keywords = textrank_.read_true_keywords(test_filename, self.content)
            if len(true_keywords) == 0:
                continue
            recall, precision, fmeasure = textrank_.evaluation(self.extracted_keywords, true_keywords)
            recall_list.append(recall)
            precision_list.append(precision)
            fmeasure_list.append(fmeasure)
            print("recall: " + str(recall) + " precision: " + str(precision) + " fmeasure: " + str(
                fmeasure))
                  #+ " num_iter: " + str(num_iter))
            # break

        print("Total recall: ", sum(recall_list) / len(recall_list))
        print("Total precision: ", sum(precision_list) / len(precision_list))
        print("Total f-measure: ", sum(fmeasure_list) / len(fmeasure_list))

    def get_keywords(self, ahtype="ah"):
        if ahtype == "ah":
            ah_index = self.h_index + self.a_index
        if ahtype == "a":
            ah_index = self.a_index
        if ahtype == "h":
            ah_index = self.h_index
        single_keyword = []
        for i in range(int(len(ah_index) / 3)):
            # print(self.reverse_candidates[ah_index[i]])
            single_keyword.append(self.reverse_candidates[ah_index[i]])
        # print(ah_index)
        self.extracted_keywords = textrank_.collapsing(single_keyword, self.content)

    def get_weights(self, intext, window_size):
        weights = np.zeros((len(self.candidates), len(self.candidates)), dtype=float)
        token_list = nltk.word_tokenize(intext)
        for i1 in range(len(token_list)):
            if token_list[i1] in self.candidates:
                for i2 in range(i1 - window_size, i1):
                    if token_list[i2] in self.candidates:
                        weights[self.candidates[token_list[i2]], self.candidates[token_list[i1]]] += 1.0
        return weights

    def update(self):
        self.a = np.dot(self.edge_weights.T, self.h)
        sum_a = np.sum(self.a)
        self.a /= sum_a
        self.h = np.dot(self.edge_weights, self.a)
        sum_h = np.sum(self.h)
        self.h /= sum_h

if __name__ == "__main__":
    hits()
