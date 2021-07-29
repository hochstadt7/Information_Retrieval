import json
import tokenizer
import vectors

data = None
idf = None
tf_idf_index = None
docs = None
words = None


def query(index_path, search_query):
    global data, idf, tf_idf_index, docs, words
    if not data:
        # load inverse index
        with open(index_path) as f:
            data = f.read()
        data = json.loads(data)
        tf_idf_index = data["tf_idf"]
        idf = data["idf"]
        docs = data["documents"]
        words = list(tf_idf_index.keys())
    query_tokens = tokenizer.keep_vocabulary(tokenizer.clean_token_list(tokenizer.tokenize(search_query)), words)
    relevant_docs = []
    # tokenize query
    for t in query_tokens:
        relevant_docs.extend(list(tf_idf_index[t].keys()))
    # calc tf-idf scores for the query itself
    query_token_freq = {}
    query_tf_idf = {}
    for t in query_tokens:
        if t not in query_token_freq:
            query_token_freq[t] = 0
        query_token_freq[t] += 1
    max_term_freq = max(query_token_freq.values())
    for t in query_token_freq.keys():
        query_tf_idf[t] = query_token_freq[t] / max_term_freq * idf[t]
    query_length = vectors.norm(query_token_freq.values())
    # focus on documents in which at least one of the search terms appear
    relevant_docs = set(relevant_docs)
    # score each doc according to cosine similarity
    doc_scores = []
    for d in relevant_docs:
        score = 0
        for t in query_tokens:
            if d in tf_idf_index[t]:
                score += tf_idf_index[t][d] * query_tf_idf[t]
        score /= (docs[d] * query_length)
        doc_scores.append({"doc": d, "relevance": score})
    doc_scores.sort(key=lambda x: x["relevance"], reverse=True)
    median_score = doc_scores[int(len(doc_scores) / 4)]["relevance"]
    search_results = [str(int(x["doc"])) for x in doc_scores if x["relevance"] > (0.4 - (median_score * 0.6))]
    print_results = "\n".join(search_results)
    with open('ranked_query_docs.txt', 'w') as f:
        f.write(print_results)
        f.close()
    print("query done")
    return print_results

