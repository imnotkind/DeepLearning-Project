import pandas as pd
from annot2vec import annot_to_words_fun
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
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
import sys, os
import copy
import re

BOARD_SIZE = (8, 8, 8)
PIECE_TO_INDEX = {'P' : 0, 'R' : 1, 'N' : 2, 'B' : 3, 'Q' : 4, 'K' : 5}
MOVE_INDEX = 6
OTHER_INDEX = 7
CAPTURE_INDEX = (6, 6)
CASTLING_INDEX = (7, 7)
CHECK_INDEX = (7, 0)
################################## Parameters ##################################
epoches = 2# 1000
batch_size= 50
patience = 50
learning_rate = 0.000001
valid_size = 0.2
PATH_FILE = '../dohki/data/gameknot/'
SPLIT_INDEX = int(278 * 0.7)
MAX_INDEX = 308
################################################################################

def board_to_mat(fen, move):
    if re.search('[a-h][1-8]', move) is None:
        return np.nan
    fen, color = fen[2:].split()[0:2]
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
    # last move
    if not 'O' in move:
        alpha, num = re.findall('[a-h][1-8]', move)[0]
        board_mat[MOVE_INDEX, 8 - int(num), ord(alpha) - ord('a')] = 1

    # check
    if '+' in move:
        board_mat[OTHER_INDEX, CHECK_INDEX[0], CHECK_INDEX[1]] = 1

    # who moves
    if 'RNBQK' in move:
        for piece in PIECE_TO_INDEX.keys():
            if piece in move:
                board_mat[OTHER_INDEX, PIECE_TO_INDEX[piece], PIECE_TO_INDEX[piece]] = 1
    else:
        board_mat[OTHER_INDEX, PIECE_TO_INDEX['P'], PIECE_TO_INDEX['P']] = 1

    # capture
    if 'x' in move:
        board_mat[OTHER_INDEX, CAPTURE_INDEX[0], CAPTURE_INDEX[1]] = 1
    
    # castling
    if 'O-O' in move:
        board_mat[OTHER_INDEX, CASTLING_INDEX[0], CASTLING_INDEX[1]] = 1

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

def data_load(vectorize_sentence, is_train=True):
    df = pd.DataFrame(columns=['move', 'fen', 'annotation'])
    if is_train:
        iterator = range(SPLIT_INDEX)
    else:
        iterator = range(SPLIT_INDEX, MAX_INDEX)

    for i in iterator:
        file_path = PATH_FILE + 'gameknot_p%d.csv' % (i+1)
        try:
            _df_ = pd.read_csv(file_path, sep=',', usecols=['move','fen','annotation'], encoding='iso-8859-1')
            _df_ = _df_.dropna()
            df = df.append(_df_.iloc[:])
        except:
            continue
    
    X = df.apply(lambda x:board_to_mat(x['fen'], x['move']), axis=1)
    y = df.apply(lambda x:vectorize_sentence(x['annotation']),
                axis=1)
   
    # check nan
    where = [~np.isnan(np.sum(X.iloc[i])) and ~np.isnan(np.sum(y.iloc[i])) for i in range(len(X))]

    X = np.stack(X[where])
    y = np.stack(y[where])
    return X, y

if __name__ == "__main__":
    argv = sys.argv
    if len(argv) > 2:
        epoches, batch_size, patience = list(map(int, sys.argv[1:4]))
        learning_rate = float(sys.argv[4])
        layers = eval(sys.argv[5])
        model = sys.argv[6]
        file_path = sys.argv[7]

        cv = True
    else:
        layers = (256,128,50)
        model = 'fcn'
        file_path = None
        cv = False

    save_info = {'epoches' : epoches,
                  'batch_size' : batch_size,
                  'learning_rate': learning_rate,
                  'layers' : str(layers),
                  'model' : model}
    
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
    train_X, train_y = data_load(vectorize_sentence, is_train=True)
    test_X, test_y = data_load(vectorize_sentence, is_train=False)

    if model == 'fcn':
        net = FCNet(input_shape=BOARD_SIZE, output_shape=num_features,
                    num_layers=layers).cuda()
    else:
        net = Net(input_shape=BOARD_SIZE, output_shape=num_features,
                  num_filters=layers).cuda()
    initial_state = copy.deepcopy(net.state_dict())
    criterion = nn.MSELoss()

    # 5-fold-cross-validation
    if cv:
        kf = KFold(n_splits=5)
        mean_validRMSE = 0.0
        for train, valid in kf.split(train_X):
            net.load_state_dict(initial_state)
            optimizer = optim.Adam(net.parameters(), lr=learning_rate)

            min_valid_loss = np.infty
            best_state = None
            patience_cnt = 0

            for epoch in range(epoches):
                running_loss = 0.0
                random.shuffle(train)
                for i in range(int(len(train) / batch_size)):
                    xs, ys = train_X[train[i * batch_size: (i+1) * batch_size]], train_y[train[i * batch_size:(i+1) * batch_size]]
                    inputs = Variable(torch.tensor(xs), requires_grad=True).float().cuda()
                    labels = Variable(torch.tensor(ys), requires_grad=True).float().cuda()
                    optimizer.zero_grad()

                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()


                inputs = Variable(torch.tensor(train_X[valid])).float().cuda()
                labels = Variable(torch.tensor(train_y[valid])).float().cuda()

                outputs = net(inputs)
                valid_loss = criterion(outputs, labels).item()
                if valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    best_state = copy.deepcopy(net.state_dict())
                    patience_cnt = 0

                if epoch % 10 == 9:
                    print('[%04d/%d] train MSE: %.8f, valid MSE: %.8f' % (epoch+1, epoches, running_loss / (i + 1), valid_loss))

                patience_cnt += 1
                if patience_cnt > patience:
                    break
            mean_validRMSE += np.sqrt(min_valid_loss)
        mean_validRMSE /= 5
        save_info['validRMSE'] = mean_validRMSE

    # split data
    train, valid = train_test_split(range(len(train_X)), test_size=valid_size, shuffle=True)

    net.load_state_dict(initial_state)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    best_score = -np.infty
    best_state = None
    patience_cnt = 0
    for epoch in range(epoches):
        running_loss = 0.0
        random.shuffle(train)
        for i in range(int(len(train) / batch_size)):
            xs, ys = train_X[train[i * batch_size: (i+1) * batch_size]], train_y[train[i * batch_size:(i+1) * batch_size]]
            inputs = Variable(torch.tensor(xs), requires_grad=True).float().cuda()
            labels = Variable(torch.tensor(ys), requires_grad=True).float().cuda()
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


        inputs = Variable(torch.tensor(train_X[valid])).float().cuda()
        labels = Variable(torch.tensor(train_y[valid])).float().cuda()

        outputs = net(inputs)
        test_loss = criterion(outputs, labels)
        score = - test_loss.item()
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(net.state_dict())
            patience_cnt = 0

        if epoch % 10 == 9:
            print('[%04d/%d] loss: %.8f, test: %.8f' % (epoch+1, epoches, running_loss / (i + 1), test_loss.item()))

        patience_cnt += 1
        if patience_cnt > patience:
            break

    net.load_state_dict(best_state)

    # calculate test RMSE
    inputs = Variable(torch.tensor(test_X)).float().cuda()
    labels = Variable(torch.tensor(test_y)).float().cuda()

    outputs = net(inputs)
    loss = criterion(outputs, labels)
    testRMSE = np.sqrt(loss.item())
    save_info['testRMSE'] = testRMSE
    print('test RMSE: %.6f' % testRMSE)
    
    torch.save(net, 'FCNnet.torch.model')

    if file_path is not None:
        save_info = dict(map(lambda item: (item[0], [item[1]]), save_info.items()))
        _df = pd.DataFrame(save_info, index=None)
        if os.path.exists(file_path):
            _df.to_csv(file_path, mode='a', header=False)
        else:
            _df.to_csv(file_path)
    
