from twec.twec import TWEC
from gensim.models.word2vec import Word2Vec


def compute_models():
    aligner = TWEC(size=30, siter=10, diter=10, workers=4)

    # train the compass: the text should be the concatenation of the text from the slices
    compass = aligner.train_compass("examples/training/compass.txt", overwrite=False) # keep an eye on the overwrite behaviour

    # now you can train slices and they will be already aligned
    # these are gensim word2vec objects
    slice_one = aligner.train_slice("examples/training/arxiv_14.txt", save=True)
    slice_two = aligner.train_slice("examples/training/arxiv_9.txt", save=True)
    


def get_NN():
    '''
    gibt die 10 NN eines Begriffes mit Grundlage des Embeddings des Wortes im Kompass
    aus beiden Modellen zurueck
    '''
    model1 = Word2Vec.load("model/arxiv_9.model")
    model2 = Word2Vec.load("model/arxiv_14.model")
    model_compass = Word2Vec.load("model/compass.model")

    word_of_interest = "good"
    # der Vektor des wortes aus dem Kompass-Modell
    vec_compass = model_compass[word_of_interest]
    print(vec_compass)

    sims = model_compass.wv.most_similar(word_of_interest, topn=10) 
    print(f'Nearest Neighbors von "{word_of_interest}" in compass: ') 
    for nn_tuple in sims: 
        print(str(x)+'.\t{0:<33}{1:>18}'.format(nn_tuple[0], nn_tuple[1])) 
        x += 1 
        print('\n')
   
    # finde 10 NN zum Vektor aus dem Kompass-Modell in den anderen beiden Modellen
    sims_model1 = model1.wv.most_similar(positive=[vec_compass], topn=10)
    print(f'Nearest Neighbors von "{word_of_interest}"from model1 in model2: ') 
    for nn_tuple in sims_model1: 
        print(str(x)+'.\t{0:<33}{1:>18}'.format(nn_tuple[0], nn_tuple[1])) 
        x += 1 
        print('\n')
    
    sims_model2 = model2.wv.most_similar(positive=[vec_compass], topn=10)
    print(f'Nearest Neighbors von "{word_of_interest}"from model1 in model2: ') 
    for nn_tuple in model2: 
        print(str(x)+'.\t{0:<33}{1:>18}'.format(nn_tuple[0], nn_tuple[1])) 
        x += 1 
        print('\n')


def main(): #TODO: write me
    compute_models()
    get_NN()


if __name__ == '__main__':
    main()

'''
Credit (except for small changes) for the code inside compute_model: https://github.com/valedica/twec 
rest by Sonja Huber
'''