import json
import torch
from gensim.models import Word2Vec
from collections import defaultdict
from learner import board_to_mat
from torch.autograd import Variable

from pwn import *
import googletrans


class FenToWord(object):
    def __init__(self, net_path="bestFCNet.torch.model",
                 w2v_path="bestAnnotWords.word2vec.model",
                 w2w_path="bestAnnotWords.word2vec.model"):
        self.net = torch.load(net_path).cuda()
        self.w2v_model = Word2Vec.load(w2v_path)
        self.w2v = dict(zip(self.w2v_model.wv.index2word, self.w2v_model.wv.vectors))
        with open('word2weights.json.dict', 'r') as f:
            w2w = json.load(f)
        max_idf = max(w2w.values())
        self.w2w = defaultdict(lambda:max_idf, w2w)

        self.gt = googletrans.Translator()

    def fen_to_words(self, fen, lastmove, num_words=1):
        mat = board_to_mat(str(fen), str(lastmove))
        in_ = Variable(torch.tensor(board_to_mat(fen, lastmove))).float().cuda()
        output = self.net(in_.view(1, 8, 8, 8))
        words = self.w2v_model.wv.most_similar(output.cpu().data.numpy())
        eng_words = list(map(lambda x: x[0], words))[:num_words]
        kor_words = list(map(lambda e: self.translate(e), eng_words))
        return kor_words

    def translate(self, eng_word):
        sentence = 'That was ' + eng_word
        translated = self.gt.translate(sentence, src='en', dest='ko').text

        d = {'그것은': '그건', '었다': '었어', '습니다': '어', '였다': '였어'}
        for k, v in d.items():
            translated = translated.replace(k, v)

        return translated


def cb(r, f2w):
    with r:
        try:
            q = json.loads(r.recvline().strip().decode())
            assert type(q) == dict
            result = f2w.fen_to_words(q['fen'], q['move'], 1)
            r.sendline(result[0])

        except Exception as e:
            r.sendline(repr(e))

def main():
    f2w = FenToWord()
    print('[*] FenToWord loaded.')

    while True:
        with listen(51119) as server:
            with server.wait_for_connection() as conn:
                try:
                    # TODO: blocking to non-blocking
                    #threading.Thread(target=cb, args=(r,)).start()

                    cb(conn, f2w)
                except EOFError:
                    pass
    

if __name__ == '__main__':
    main()
