from twec.twec import TWEC
from gensim.models.word2vec import Word2Vec


def compare_position():
    '''
    Gibt die 10 NN eines Begriffes mit Grundlage des Embeddings des Wortes im Kompass
    aus beiden Modellen zurueck. Der Fokus liegt auf einer Position im Vektorraum statt auf
    einem Wort.
    by Sonja Huber
    '''
    model1 = Word2Vec.load("model/arxiv_9.model")
    model2 = Word2Vec.load("model/arxiv_14.model")
    model_compass = Word2Vec.load("model/compass.model")

    word_of_interest = "good"
    # der Vektor des wortes aus dem Kompass-Modell
    vec_compass = model_compass.wv[word_of_interest]
    

    sims = model_compass.wv.most_similar(word_of_interest, topn=10) 
    x=1
    print(f'Nearest Neighbors von "{word_of_interest}" in compass: ') 
    for nn_tuple in sims: 
        print(str(x)+'.\t{0:<33}{1:>18}'.format(nn_tuple[0], nn_tuple[1])) 
        x += 1 
    print('\n')    
   
    # finde 10 NN zum Vektor aus dem Kompass-Modell in den anderen beiden Modellen
    sims_model1 = model1.wv.most_similar(positive=[vec_compass], topn=10)
    x=1
    print(f'Nearest Neighbors of position "{word_of_interest}" from compass in model1: ') 
    for nn_tuple in sims_model1: 
        print(str(x)+'.\t{0:<33}{1:>18}'.format(nn_tuple[0], nn_tuple[1])) 
        x += 1 
    print('\n')    
    
    sims_model2 = model2.wv.most_similar(positive=[vec_compass], topn=10)
    x=1
    print(f'Nearest Neighbors of position "{word_of_interest}" from compass in model2: ') 
    for nn_tuple in sims_model2: 
        print(str(x)+'.\t{0:<33}{1:>18}'.format(nn_tuple[0], nn_tuple[1])) 
        x += 1 
    print('\n')    

if __name__ == '__main__':
    compare_position()
