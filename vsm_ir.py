import sys
from corpus_index import build_index
from query import query


# allows control of the program mode (create_index / query) directly from the code
controller = "production"

if __name__ == "__main__":
    if controller == "create":
        mode = "create_index"
        args = ["./cfc-xml_corrected"]
    elif controller == "query":
        mode = "query"
        args = ["./vsm_inverted_index.json", '''What histochemical differences have been described between normal and
   CF respiratory epithelia?''']
    else:
        mode = sys.argv[1]
        args = sys.argv[2:]

    if mode == 'query':
        query(args[0], args[1])
    elif mode == 'create_index':
        build_index(args[0])
