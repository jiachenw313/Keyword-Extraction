import rake
from textrank_ import *

if __name__ == '__main__':
    rake_object = rake.Rake("SmartStoplist.txt")
    filename_list = read_filename()

    # evaluation list
    recall_list = []
    precision_list = []
    fmeasure_list = []

    for filename in filename_list:
        content, title, text = read_file(filename)
        keywords_from_title = extract_keywords_from_title(title)
        keywords_from_text = rake_object.run(content)
        keywords_from_text = [tup[0] for tup in keywords_from_text]

        keywords = (set(keywords_from_title + keywords_from_text))
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
        # print("recall: " + str(recall) + " precision: " + str(precision) + " fmeasure: " + str(fmeasure))

    print("Total recall: ", sum(recall_list) / len(recall_list))
    print("Total precision: ", sum(precision_list) / len(precision_list))
    print("Total f-measure: ", sum(fmeasure_list) / len(fmeasure_list))
    # print(len(recall_list))
