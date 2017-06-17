# coding: utf-8
"""
gensimでword2vecを使うためのサンプルコード
"""
import numpy as np
from gensim.models import word2vec
import glob
from filer2 import Filer


# 定数
TXT_FILE_PATH = ''
SIZE = 150
WINDOW = 5
MIN_COUNT = 5
WORKERS = 4
NUM_EPOCH = 5
OUTPUT_MODEL_PATH = 'word2vec_epoch%s.model'


def train():
    # 形態素解析済みテキストファイルの読み込み
    sentences = word2vec.LineSentence(TXT_FILE_PATH)
    # modelのセット
    model = word2vec.Word2Vec(
                sentences,
                size=SIZE,
                window=WINDOW,
                min_count=MIN_COUNT,
                workers=WORKERS
            )

    # 学習部分
    for epoch in range(1, NUM_EPOCH+1):
        model.train(sentences)
        model.save(OUTPUT_MODEL_PATH%epoch)


def show_similar_words(word, model):
    for word, sim in model.most_similar(positive=[word.decode('utf-8')]):
        print word, sim


def save_word_vec_dictionary(input_model_path, output_dict_path):
    # modelのインスタンス化
    model = word2vec.Word2Vec(
                size=SIZE,
                window=WINDOW,
                min_count=MIN_COUNT,
                workers=WORKERS
            )

    # modelのロード
    model.load(input_model_path)

    words = [word for word in model.vocab.keys()]
    word_to_vec = {word.encode('utf-8'): model[word] for word in words}

    # save
    Filer.write_pkl(word_to_vec, output_model_path)
