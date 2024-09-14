BioVec
--------------------------------------

A new approaches for representing biological sequences.
The algorithm comes from this paper: "[ProtVec: A Continuous Distributed Representation of
Biological Sequences](http://arxiv.org/pdf/1503.05140v1.pdf)"

## ⚠️ **Repository Archived – No Longer Maintained**

This repository is no longer actively maintained and will not receive updates or bug fixes. The project may be outdated, and some components may not work with current software versions. Feel free to explore the code, but please be aware that it may not function as expected in modern environments.

Thank you for your understanding.

### Installation
```
pip install biovec
```

### Quick Start

```
import biovec

pv = biovec.models.ProtVec("some_fasta_file.fasta", corpus_fname="output_corpusfile_path.txt", n=3)

# The n-gram "QAT" should be trained in advance
pv["QAT"]

# convert whole amino acid sequence into vector
pv.to_vecs("ATATQSQSMTEEL")

# convert multi-sequence FASTA file to a numpy array of protvec vectors (axis 0 = sample axis)
# seqtype - 'amino acid' [default] or 'nucleotide'
pv.multiseq_to_vecs('sample_aa_file.fasta','output_filename', seqtype = 'amino acid')

# save trained model into file
pv.save('model_file_path')

# load trained model from file
pv2 = biovec.models.load_protvec('model_file_path')

```

### Trained Model

This package includes already trained model in '/trained_models/swissprot-reviewed.model'.
It is a protvec model fed all Swiss-Prot reviewed proteins(551,754 proteins as of 2018/09/13) as the training data.

### Source
Paper: [ProtVec: A Continuous Distributed Representation of
Biological Sequences](http://arxiv.org/pdf/1503.05140v1.pdf)

#### Abstract of the paper

> We propose a new approach for representing biological sequences. This method, named protein-vectors or ProtVec for short, can be utilized in bioinformatics applications such as family classification, protein visualization, structure prediction, disordered protein identification, and protein-protein interaction prediction. Using the Skip-gram neural networks, protein sequences are represented with a single dense n-dimensional vector. This method was evaluated by classifying protein sequences obtained from Swiss-Prot belonging to 7,027 protein families where an average family classification accuracy of 94%±0.03% was obtained, outperforming existing family classification methods. In addition, our model was used to predict disordered proteins from structured proteins. Two databases of disordered sequences were used: the DisProt database as well as a database featuring the disordered regions of nucleoporins rich with phenylalanine-glycine repeats (FG-Nups). Using support vector machine classifiers, FG-Nup sequences were distinguished from structured Protein Data Bank (PDB) sequences with 99.81\% accuracy, and unstructured DisProt sequences from structured DisProt sequences with 100.0\% accuracy. These results indicate that by only providing sequence data for various proteins into this model, information about protein structure can be determined with high accuracy. This so-called embedding model needs to be trained only once and can then be used to ascertain a diverse set of information regarding the proteins of interest. In addition, this representation can be considered as pre-training for various applications of deep learning in bioinformatics.

### References
1. [Disordered Proteins](https://en.wikipedia.org/wiki/Intrinsically_disordered_proteins)
2. [DisProt](http://www.disprot.org/)
3. [gemsim word2vec](https://radimrehurek.com/gensim/models/word2vec.html)
4. [NIPS2013読み会: Distributed Representations of Words and Phrases and their Compositionality](http://www.slideshare.net/unnonouno/nips2013-distributed-representations-of-words-and-phrases-and-their-compositionality)
5. [Skip gram shirakawa_20141121
](http://www.slideshare.net/nttdata-msi/skip-gram-shirakawa20141121-41833306)
6. [論文紹介「Distributed Representations of Words and Phrases and their Compositionality」](http://qiita.com/nishio/items/3860fe198d65d173af6b)
7. [sklearn.manifold.TSNE](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
8. [Lipschitz continuity](http://izumi-math.jp/F_Wada/fixpoint_theorem.pdf)

---

# Japanese Explanation

### Abstract
通常生物情報は文字の配列で表現されるが、それをベクトルとして表現することによってより分析しやすく情報を収納することができるのではないかと提案されている。具体的な適用範囲としては、

1. family classification
2. protein visualization
3. structure prediction
4. disordered protein identification
5. protein-protein interaction prediction.

など。
classificationやpredictionはわかりやすい使い方だが、個人的にはprotein visualizationもまた面白いアプリケーションであると感じた。短い配列や、構造が既知の配列でない限り、現状簡単にタンパク質の全容を掴む方法は確立していないので、このような表現方法は一定の有用性があると考える。

### ProtVec実装

[前処理]
* uniprotのswis-protの全データを収集する
* 各配列を3つのn-gramのリストに変換する

```
'AGAMQSASM' => [['AGA', 'MQS', 'ASM'], ['GAM','QSA'], ['AMQ', 'SAS']]
```

* word2vecに読み込ませるために、変換した配列をテキストファイル形式に書き出す

[モデル構築]

word2vecのライブラリを用いれば基本大丈夫そう。
いろいろあるけど、gensimをここでは使う。ただしSkip-gramを論文では採用しているので注意。
gensimではsgパラメータを1に設定するとskip-gramになる。
> sg defines the training algorithm. By default (sg=0), CBOW is used. Otherwise (sg=1), skip-gram is employed.

前処理では上記のように単語をn-gramのリストへ変換するが、sequenceをqueryとしてモデルに投げるときは逆に再変換する必要がある。訓練済みモデルにn-gramを指定することで、対応するベクトルを得ることができる。論文ではそのベクトルの和をその配列に対応するベクトルとして扱っている。

```
seq = 'AGAMQSASM'
n_grams = split_to_grams(seq)
gram_vecs = [to_gram_vec(n_gram) for n_gram in n_grams]
seq_vec = sum(gram_vecs)
```

論文での、ベクトルの次元数は100。
元の配列に対応するベクトルはn-gramベクトルの和なので次元数は変わらず、100次元。
negative samplingも行っているので忘れずに。

### Visualization of ProtVec
[sklearn.manifold.TSNE](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
t-SNEによって2次元もしくは3次元に次元圧縮を行った後、可視化している。しかしこの圧縮方法が適しているのか、なぜこの方法を取ったのかは言及されていない。scikit-learnではt-SNEは非推奨されており、密なデータであればPCA、疎なデータであればTruncatedSVDが勧められている。圧縮後の次元数がとても低いためだろうか。

> It is highly recommended to use another dimensionality reduction method (e.g. PCA for dense data or TruncatedSVD for sparse data) to reduce the number of dimensions to a reasonable amount (e.g. 50) if the number of features is very high. This will suppress some noise and speed up the computation of pairwise distances between samples. For more tips see Laurens van der Maaten’s FAQ.

[Lipschitz continuity](http://izumi-math.jp/F_Wada/fixpoint_theorem.pdf)
