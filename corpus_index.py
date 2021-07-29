import json
import tokenizer
import math
import vectors
from lxml import etree

FILE_NAMES = ["74", "75", "76", "77", "78", "79"]
TEXT_FROM = ["TITLE", "EXTRACT", "ABSTRACT"]
RECORD_NUM = "RECORDNUM"


def build_index(path):
    docs = []
    doc_ids = []
    dictionary = {}
    idf = {}
    # extract data and produce dictionary
    for file in FILE_NAMES:
        with open(path+"/cf"+file+".xml") as f:
            xml = f.read()
        root = etree.fromstring(xml)
        records = root.getchildren()
        # go over records
        for r in records:
            doc = {}
            attributes = r.getchildren()
            words = []
            for a in attributes:
                if a.tag == RECORD_NUM:
                    doc["num"] = a.text.strip()
                    doc_ids.append(doc["num"])
                if a.tag in TEXT_FROM:
                    words.extend(tokenizer.tokenize(a.text))
            words = tokenizer.clean_token_list(words)
            doc["words"] = words  # list of words that appear in the record
            doc["vocabulary"] = set(words)  # set of words that appear in the record
            for word in doc["vocabulary"]:
                if word not in dictionary:
                    dictionary[word] = {"appearances": [], "tf_idf": []}
                dictionary[word]["appearances"].append(doc["num"])
            docs.append(doc)
    # calc idf scores
    for w in dictionary:
        idf[w] = math.log2(len(doc_ids) / len(dictionary[w]["appearances"]))
    # calc tf scores
    for d in docs:
        d["tf_idf"] = dict()
        term_freq_unnormalized = {w: 0 for w in d["vocabulary"]}
        for w in d["words"]:
            term_freq_unnormalized[w] += 1
        most_common_word_freq = max(term_freq_unnormalized.values()) if len(term_freq_unnormalized.values()) else 1
        for w in term_freq_unnormalized:
            term_freq_unnormalized[w] = term_freq_unnormalized[w] / most_common_word_freq
            d["tf_idf"][w] = term_freq_unnormalized[w] * idf[w]
            dictionary[w]["tf_idf"].append({"doc": d["num"], "tf_idf": d["tf_idf"][w]})
        d["length"] = vectors.norm(list(d["tf_idf"].values()))
    # calc document lengths (stored for use during
    doc_lengths = {d["num"]: d["length"] for d in docs}
    for w in dictionary:
        dictionary[w] = {x["doc"]: x["tf_idf"] for x in dictionary[w]["tf_idf"]}
    result = {"tf_idf": dictionary, "documents": doc_lengths, "idf": idf}
    with open('vsm_inverted_index.json', 'w') as f:
        json.dump(result, f)
    print("index done")
    return dictionary
