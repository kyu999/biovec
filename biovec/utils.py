from pyfaidx import Fasta
from tqdm import tqdm


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


def generate_corpusfile(fasta_fname, n, corpus_fname):
    '''
    Args:
        fasta_fname: corpus file name
        n: the number of chunks to split. In other words, "n" for "n-gram"
        corpus_fname: corpus_fnameput corpus file path
    Description:
        Protvec uses word2vec inside, and it requires to load corpus file
        to generate corpus.
    '''
    f = open(corpus_fname, "w")
    fasta = Fasta(fasta_fname)
    for record_id in tqdm(fasta.keys(), desc='corpus generation progress'):
        r = fasta[record_id]
        seq = str(r)
        ngram_patterns = split_ngrams(seq, n)
        for ngram_pattern in ngram_patterns:
            f.write(" ".join(ngram_pattern) + "\n")
    f.close()
