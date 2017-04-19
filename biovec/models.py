from Bio import SeqIO
import sys
from gensim.models import word2vec
import csv


def split_ngrams(seq, n):
    """
    'AGAMQSASM' => [['AGA', 'MQS', 'ASM'], ['GAM','QSA'], ['AMQ', 'SAS']]
    """
    a, b, c = zip(*[iter(seq)]*n), zip(*[iter(seq[1:])]*n), zip(*[iter(seq[2:])]*n)
    str_ngrams = []
    for ngrams in [a,b,c]:
        x = ["".join(ngram) for ngram in ngrams]
        str_ngrams.append(x)
    return str_ngrams


def generate_corpusfile(corpus_fname, n, out):
    '''
    Args:
        corpus_fname: corpus file name
        n: the number of chunks to split. In other words, "n" for "n-gram"
        out: output corpus file path
    Description:
        Protvec uses word2vec inside, and it requires to load corpus file
        to generate corpus.
    '''
    f = open(out, "w")
    for r in SeqIO.parse(corpus_fname, "fasta"):
        ngram_patterns = split_ngrams(r.seq, n)
        for words in ngram_patterns:
            f.write(" ".join(words) + "\n")
        sys.stdout.write(".")

    f.close()

def generate_corpusfile_csv(corpus_fname, n, out, sequence_column, **kwargs):
    '''
    Args:
        corpus_fname: corpus file name
        n: the number of chunks to split. In other words, "n" for "n-gram"
        out: output corpus file path
        sequence_column: column in csv containing sequence
        **kwargs: arguments for csv reader, e.g. delimiter
    Description:
        Protvec uses word2vec inside, and it requires to load corpus file
        to generate corpus.
    '''
    f = open(out, "w")
    reader = csv.reader(open(corpus_fname, "rb"), **kwargs)
    header = reader.next()
    sequence_column_index = header.index(sequence_column)

    sequences = [record[sequence_column_index] for record in reader]
    all_ngram_patterns = [split_ngrams(x, n) for x in sequences]
    sentences = [" ".join(words) for ngram_patterns in all_ngram_patterns for words in ngram_patterns]
    [f.write(x + "\n") for x in sentences]
    sys.stdout.write(".")
    f.close()


def read_corpusfile(file):
    corpus = word2vec.Text8Corpus(file)
    return corpus


def load_protvec(model_fname):
    return word2vec.Word2Vec.load(model_fname)


class ProtVec(word2vec.Word2Vec):

    def __init__(self, corpus_fname=None, corpus=None, n=3, size=100, out="corpus.txt",  sg=1, window=25, min_count=2, workers=3):
        """
        Either fname or corpus is required.

        corpus_fname: fasta file for corpus
        corpus: corpus object implemented by gensim
        n: n of n-gram
        out: corpus output file path
        min_count: least appearance count in corpus. if the n-gram appear k times which is below min_count, the model does not remember the n-gram
        """

        self.n = n
        self.size = size
        self.corpus_fname = corpus_fname

        if corpus is None and corpus_fname is None:
            raise Exception("Either corpus_fname or corpus is needed!")

        if corpus_fname is not None:
            print('Generate Corpus file from fasta file...')
            generate_corpusfile(corpus_fname, n, out)
            corpus = read_corpusfile(out)

        word2vec.Word2Vec.__init__(self, corpus, size=size, sg=sg, window=window, min_count=min_count, workers=workers)

    def to_vecs(self, seq):
        """
        convert sequence to three n-length vectors
        e.g. 'AGAMQSASM' => [ array([  ... * 100 ], array([  ... * 100 ], array([  ... * 100 ] ]
        """
        ngram_patterns = split_ngrams(seq, self.n)

        protvecs = []
        for ngrams in ngram_patterns:
            ngram_vecs = []
            for ngram in ngrams:
                try:
                    ngram_vecs.append(self[ngram])
                except:
                    raise Exception("Model has never trained this n-gram: " + ngram)
            protvecs.append(sum(ngram_vecs))
        return protvecs
