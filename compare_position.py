from twec.twec import TWEC
from gensim.models.word2vec import Word2Vec


def compare_position():
    '''
    Gibt die 10 NN eines Begriffes mit Grundlage des Embeddings des Wortes im Kompass
    aus beiden Modellen zurueck. Der Fokus liegt auf einer Position im Vektorraum statt auf
    einem Wort.

    Die Pfade zu den Modellen müssen angepasst werden. Ausserdem ist es empfehlenswert, 
    die Strings, welche gedruckt werden, aussagekräftiger zu machen, etwa 'model1' mit dem 
    tatsächlichen Namen des Korpus zu ersetzen.    
    '''
    model1 = Word2Vec.load("model/parlament_06.model")
    model2 = Word2Vec.load("model/parlament_16.model")
    model_compass = Word2Vec.load("model/compass.model")

    word_of_interest = "Freiheit"
    
    # der Vektor des wortes aus dem ersten Model
    vec_modell1 = model1.wv[word_of_interest]

    # finde 10 NN zum Vektor im ersten Modell
    sims_model1 = model1.wv.most_similar(positive=[vec_modell1], topn=10)
    x=1
    print(f'Nearest Neighbors of position "{word_of_interest}" in model1: ') 
    for nn_tuple in sims_model1: 
        print(str(x)+'.\t{0:<33}{1:>18}'.format(nn_tuple[0], nn_tuple[1])) 
        x += 1 
    print('\n')    
    
    sims_model2 = model2.wv.most_similar(positive=[vec_modell1], topn=10)
    x=1
    print(f'Nearest Neighbors of position "{word_of_interest}" from model1 in model2: ') 
    for nn_tuple in sims_model2: 
        print(str(x)+'.\t{0:<33}{1:>18}'.format(nn_tuple[0], nn_tuple[1])) 
        x += 1 
    print('\n')    

if __name__ == '__main__':
    compare_position()
