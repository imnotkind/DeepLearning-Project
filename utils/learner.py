import pandas as pd
from annot2vec import annot_to_words_fun
from sklearn.model_selection import train_test_split
from model.convFunChess import Net, FCNet
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch
import random
from gensim.models import Word2Vec
import json
import numpy as np
from collections import defaultdict

BOARD_SIZE = (7, 8, 8)
PIECE_TO_INDEX = {'P' : 0, 'R' : 1, 'N' : 2, 'B' : 3, 'Q' : 4, 'K' : 5}
MOVE_INDEX = 6
################################## Parameters ##################################
epoches = 100
batch_size= 50
patience = 50
learning_rate = 0.000001
test_size=0.2
################################################################################

def board_to_mat(fen, move):
    fen, color = fen.split()[0:2]
    board_mat = np.zeros(BOARD_SIZE)
    white_move = 1 if color == 'b' else -1
    for i, row in enumerate(fen.split('/')):
        j = 0
        for col in row:
            if col.isdecimal():
                j += int(col)
            elif col.isupper():
                board_mat[PIECE_TO_INDEX[col], i, j] = white_move
                j += 1
            else:
                board_mat[PIECE_TO_INDEX[col.upper()], i, j] = - white_move
                j += 1
    board_mat[MOVE_INDEX, 8 - int(move[1]), ord(move[0]) - ord('a')] = -1
    board_mat[MOVE_INDEX, 8 - int(move[3]), ord(move[2]) - ord('a')] = 1
    return board_mat

def sentence_to_vec(sentence, w2v, w2w):
    if len(sentence) == 0:
        return np.nan
    wsum = 0.0
    wnum = 0.0
    for i, w in enumerate(sentence):
        if w2v.get(w) is None:
            continue
        wsum += w2v[w] * w2w[w]
        wnum += w2w[w]
    if wnum == 0.0:
        return np.nan
    return wsum / wnum

def vectorize_sentence_fun(w2v, w2w):
    annot_to_words = annot_to_words_fun()
    return (lambda x: sentence_to_vec(annot_to_words(x), w2v, w2w))

def sentence_to_word(sentence, w2v_model):
    return w2v_model.similar_by_vector(sentence_to_vec(sentence))[0]

if __name__ == "__main__":
    # word2vec load
    w2v_model = Word2Vec.load("bestAnnotWords.word2vec.model")
    w2v = dict(zip(w2v_model.wv.index2word, w2v_model.wv.vectors))
    with open('word2weights.json.dict', 'r') as f:
        w2w = json.load(f)
    max_idf = max(w2w.values())
    w2w = defaultdict(lambda:max_idf, w2w)
    vectorize_sentence = vectorize_sentence_fun(w2v, w2w)
    num_features = w2v_model.vector_size
    
    # data load
    df = pd.read_csv('gameknot.csv')
    X = df.apply(lambda x:board_to_mat(x['fen'], x['move']), axis=1)
    y= df.apply(lambda x:vectorize_sentence(x['annotation']),
                axis=1)

    # check nan
    where = [~np.isnan(np.sum(X[i])) and ~np.isnan(np.sum(y[i]))
             for i in range(len(X))]
    X = np.stack(X[where])
    y = np.stack(y[where])

    # split data
    train, test = train_test_split(range(len(X)), test_size=test_size, shuffle=True)

    # net = Net(input_shape=BOARD_SIZE, output_shape=num_features, num_filters=(32, 256, 128)).cuda()
    net = FCNet(input_shape=BOARD_SIZE, output_shape=num_features,
                num_layers=(256, 128, 50)).cuda()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    best_score = -np.infty
    best_state = None
    patience_cnt = 0
    for epoch in range(epoches):
        running_loss = 0.0
        random.shuffle(train)
        for i in range(int(len(train) / batch_size)):
            xs, ys = X[i * batch_size: (i+1) * batch_size], y[i * batch_size:(i+1) * batch_size]
            inputs = Variable(torch.tensor(xs), requires_grad=True).float().cuda()
            labels = Variable(torch.tensor(ys), requires_grad=True).float().cuda()
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


        inputs = Variable(torch.tensor(X[test])).float().cuda()
        labels = Variable(torch.tensor(y[test])).float().cuda()

        outputs = net(inputs)
        test_loss = criterion(outputs, labels)
        score = - test_loss.item()
        if score > best_score:
            best_score = score
            best_state = net.state_dict()
            patience_cnt = 0

        if epoch % 10 == 9:
            print('[%04d/%d] loss: %.8f, test: %.8f' % (epoch+1, epoches, running_loss / (i + 1), test_loss.item()))

        patience_cnt += 1
        if patience_cnt > patience:
            break

    net.load_state_dict(best_state)

    # calculate test RMSE
    inputs = Variable(torch.tensor(X[test])).float().cuda()
    labels = Variable(torch.tensor(y[test])).float().cuda()

    outputs = net(inputs)
    loss = criterion(outputs, labels)
    print('test RMSE: %.6f' % np.sqrt(loss.item()))
    
    torch.save(net, 'FCNnet.torch.model')
