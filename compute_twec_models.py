from twec.twec import TWEC
from gensim.models.word2vec import Word2Vec


def compute_twec_models():
    aligner = TWEC(size=30, siter=10, diter=10, workers=4)

    # train the compass: the text should be the concatenation of the text from the slices
    parlament_compass = aligner.train_compass("examples/training/compass_parlament.txt", overwrite=True) # keep an eye on the overwrite behaviour

    # now you can train slices and they will be already aligned
    # these are gensim word2vec objects
    slice_one = aligner.train_slice("examples/training/parlament_06.txt", save=True)
    slice_two = aligner.train_slice("examples/training/parlament_16.txt", save=True)


if __name__ == '__main__':
    compute_twec_models()

'''
Credit (except for small changes) for the code inside compute_model: https://github.com/valedica/twec 
'''
