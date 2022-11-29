<h1> TRANS-KBLSTM : An External Knowledge Enhance Transformer BiLSTM for Tabular Reasoning </h1>
<img src="./trans-kblstm.png"/>

Implementation of TRANS-KBLSTM, our DeeLIO@ACL 2022 paper: 
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
--val_path ../../data/kinfotabs_withrels/dev_with_lstmrels.csv
```


