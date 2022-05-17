# Dataset download

import os


def download_drive(id, name):
    os.system(f"sudo wget --load-cookies /tmp/cookies.txt 'https://docs.google.com/uc?export=download&confirm=t&id={id}' -O {name} && rm -rf /tmp/cookies.txt")


if not os.path.exists("./glove.6B.100d.txt"):
    download_drive("1wu6kUYMuRAOkS1O7et0XC_IQV67lnwCc", "Glove.zip")
    # !unzip /content/Glove.zip

if not os.path.exists("./Lstm_bert"):
    download_drive("1DUaUjHftw3zvxIhVisq5TmovqcDgl3WO", "lstm-bert_zipped.zip")
    # !unzip lstm-bert_zipped.zip

if not os.path.exists("./Gold_Lstm_bert"):
    download_drive("1aoquiMynXY49QYgohPT9_wM8RDiZ_swG", "Gold_lstm_bert_zipped.zip")
    # !unzip Gold_lstm_bert_zipped.zip

if not os.path.exists("./kg_explicit"):
    download_drive("1Kp3-XhBwkdea0P-fGxclaahhK4pq7aNf", "kg_explicit.zip")
    # !unzip kg_explicit.zip

if not os.path.exists("./pair_features.txt"):
    download_drive("153W-5aEOK_YPGf58axpZ9xluiDl_LnAF", "pair_features.txt")