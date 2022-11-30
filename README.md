<h1> TRANS-KBLSTM : An External Knowledge Enhance Transformer BiLSTM for Tabular Reasoning </h1>
<img src="./trans-kblstm.png"/>

Implementation of TRANS-KBLSTM, winner of <strong> Best Paper award </strong> at DeeLIO@ACL 2022 paper: 
<a href="https://vgupta123.github.io/docs/TransKBLSTM.pdf"> TRANS-KBLSTM : An External Knowledge Enhance Transformer BiLSTM for Tabular Reasoning </a>



# Setup
Create a conda environment for the project and install all requirements.

```
conda env create --name myenv python=3.7
conda activate myenv
pip install -r requirements.txt
```

If you want to generate relations on your data, you will needing Conceptnet and Wordnet knowledge base. \
Download `Conceptnet_wordnet_full.csv` and `glove.6B.100.txt` from this [link](https://drive.google.com/drive/folders/1kA44dYJ6wEA4MzexD_-KhR-PiDw6lYKo) and place them `data` folder.

<br>

# Data
Our work was mainly based on previous datasets by [Knowledge-Infotabs](https://knowledge-infotabs.github.io/). Their datasets are already present in data section.

If you want to try out on your own dataset, place them here.

<br>

# Reproducing Results

The `Relation_generator.py` script can be found in `scripts/preprocess` folder. This takes the data file as an Input and finds all relational connections between premise and hypothesis. More details are found in [paper](https://trans-kblstm.github.io/).

```
cd scripts/preprocess
python Relation_generator.py --path ../../data/kinfotabs_original/test_alpha1.tsv --savename ../../data/kinfotabs_withrels/a1_with_lstmrels.csv
```

This is a time taking process.. but the files are already generated in the folder for your use!
<br>

The `train.py` supports multiple training formats, some of which are described below.

### <u> Original TransKBLSTM </u>
This is the basic transkblstm model

```
cd scripts/train
python train.py --addkb \
                --batchsize 5
```

### <u> Gold Infotabs </u>
Gold set of infotabs refers to a perfectly relevant set of sentence that describes the premise. It is expected to achieve maximum possible performance for any task. /
To train on this task, you can get the `Gold_lstm_bert_zipped.zip` from the same [dataset link](https://drive.google.com/drive/folders/1kA44dYJ6wEA4MzexD_-KhR-PiDw6lYKo) and extract it to data folder.

```
cd scripts/train
python train.py --addkb \
                --batchsize 5 \
                --gold
```

### <u> KG Explicit </u>
This is our main comparison model introduced by <strong>Neeraja et al., 2021</strong>. The data for this can also be found in `data` section.

```
cd scripts/train
python train.py --kg_exp \
                --batchsize 5 \
                --gold
```
<br>


All other commands are different hyper parameters to play with. For example, <br>
<b> Q. Does Gold data with random noise perform better or worse than 75% Normal data? </b>

```
cd scripts/train
python train.py --addkb \
                --batchsize 5 \
                --gold \
                --makekbrandom 

python train.py --addkb \
                --batchsize 5 \
                --data_percent 75
```
can help us answer the question.


# Custom Dataset

It is very easy to generate relations on a custom dataset. Just save the files in `data` folder and give correct paths in the following commands.

1. First Generate relations.

```
cd scripts/preprocess
python Relation_generator.py \
--path ../../data/kinfotabs_original/test_alpha1.tsv \
--savename ../../data/kinfotabs_withrels/a1_with_lstmrels.csv
```

2. Train using `train_custom.py`

```
cd scripts/train
python train_custom.py \
--train_path ../../data/ kinfotabs_withrels/train_with_lstmrels.csv \
--val_path ../../data/kinfotabs_withrels/dev_with_lstmrels.csv \
--addkb \
--batchsize 5 \
--glove_dim 100
```


## Recommended Citations

Kindly cite our work, if you find it helpful.

```

@inproceedings{varun-etal-2022-trans,
    title = "Trans-{KBLSTM}: An External Knowledge Enhanced Transformer {B}i{LSTM} Model for Tabular Reasoning",
    author = "Varun, Yerram  and
      Sharma, Aayush  and
      Gupta, Vivek",
    booktitle = "Proceedings of Deep Learning Inside Out (DeeLIO 2022): The 3rd Workshop on Knowledge Extraction and Integration for Deep Learning Architectures",
    month = may,
    year = "2022",
    address = "Dublin, Ireland and Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.deelio-1.7",
    pages = "62--78",
    abstract = "Natural language inference on tabular data is a challenging task. Existing approaches lack the world and common sense knowledge required to perform at a human level. While massive amounts of KG data exist, approaches to integrate them with deep learning models to enhance tabular reasoning are uncommon. In this paper, we investigate a new approach using BiLSTMs to incorporate knowledge effectively into language models. Through extensive analysis, we show that our proposed architecture, Trans-KBLSTM improves the benchmark performance on InfoTabS, a tabular NLI dataset.",
}

@inproceedings{neeraja-etal-2021-incorporating,
    title = "Incorporating External Knowledge to Enhance Tabular Reasoning",
    author = "Neeraja, J.  and
      Gupta, Vivek  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.224",
    pages = "2799--2809",
    abstract = "Reasoning about tabular information presents unique challenges to modern NLP approaches which largely rely on pre-trained contextualized embeddings of text. In this paper, we study these challenges through the problem of tabular natural language inference. We propose easy and effective modifications to how information is presented to a model for this task. We show via systematic experiments that these strategies substantially improve tabular inference performance.",
}

```

