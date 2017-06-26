from neural import NNmaker
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import theano
import random
import numpy as np
import names

def main():
    '''Checking the import worked'''
    maker = NNmaker()
    model, DNA = maker.makeRanNN(100,10)

def load_and_condition_MNIST_data():
    ''' loads and shapes MNIST image data '''
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    theano.config.floatX = 'float32'
    X_train = X_train.astype(theano.config.floatX) #before conversion were uint8
    X_test = X_test.astype(theano.config.floatX)
    X_train.resize(len(y_train), 784) # 28 pix x 28 pix = 784 pixels
    X_test.resize(len(y_test), 784)
    y_train_ohe = np_utils.to_categorical(y_train)
    return X_train, y_train, X_test, y_test, y_train_ohe

def mixGenDNA(DNA_pool, X_train, y_train, X_test, y_test, classes, gen):
    '''Takes in the  DNA pool and mixes them randomly amongst two partners '''
    #[name, test_acc, DNA]
    DNA_Pool = DNA_pool.copy()
    NewGenDNA = {}
    while len(DNA_Pool) > 1:
        mates = np.random.choice(list(DNA_Pool.keys()),2, replace=False)
        acc1 = DNA_Pool[mates[0]][1]
        acc2 = DNA_Pool[mates[1]][1]

        DNA1 = DNA_Pool[mates[0]][3]
        DNA2 = DNA_Pool[mates[1]][3]

        del DNA_Pool[mates[0]]
        del DNA_Pool[mates[1]]

        genes = list(DNA1.keys())
        mom = np.random.choice(genes,int(len(genes)/2), replace = False)
        momDNA = d = {g:DNA1[g] for g in mom}
        for key, value in momDNA.items():
            DNA2[key] = value

        name = names.get_first_name()
        maker = NNmaker()
        model = maker.reporduce_CNN(X_train, X_test, classes, DNA2)
        score = training_Time(model,DNA2, X_train, y_train, X_test, y_test, name, gen, mates[0], mates[1] )
        NewGenDNA[name]= [name, score[0],score[1], DNA2, name, gen]
        DNA_pool.update(NewGenDNA)
    return DNA_pool, len(NewGenDNA)
    # print('{} and {} had a baby'.format(mates[0],mates[1]))
    # print(DNA1)



def evolve():
    '''The main function to run the evolution theory'''
    rng_seed = 20 # set random number generator seed
    X_train,X_test, y_train, y_test, classes = get_data()

    np.random.seed(rng_seed)
    DNAPop = {}
    num_gen = 5
    num_pop = 5
    for pop in range(num_pop):
        maker = NNmaker()
        name = names.get_first_name()
        model,DNA = maker.makerandom_CNN(X_train,X_test, classes)
        score = training_Time(model, DNA, X_train, y_train, X_test, y_test, name, 1, 'none', 'none')

        DNAPop[name]= [name, score[1],score[0], DNA, 1]

    for gen in range(num_gen):
        print('Generation: ' + str((gen + 1)))
        # for pop in range(num_pop):

        DNAPop, childs = mixGenDNA(DNAPop, X_train, y_train, X_test, y_test,classes, (gen+2))
        DNAPop = killOff(DNAPop,childs)

        # print(NewGenDNA)
def killOff(DNAPop, childs):
    '''Removes the models that are not up to par'''
    pop = []
    for key in DNAPop.keys():
        pop.append([DNAPop[key][1], key])
    sorts = sorted(pop)
    print(sorts)
    for i in range(childs):
        for x, person in enumerate(sorts):
            if randint(0,3) != 3:
                print('kill')
                del DNAPop[person[1]]
                sorts.pop(x)
                break
        else:
            continue
        # add death to Mongo
    return DNAPop


def training_Time(model,DNA, X_train, y_train, X_test, y_test, name, gen, parent1, parent2):
    '''Train the models and return the accuracy'''
    #[name, score[0],score[1], DNA2, name, gen]

    model.fit(X_train, y_train, batch_size=DNA['BS'], epochs=DNA['epochs'],
      verbose=1, validation_data=(X_test, y_test), initial_epoch=0)
    score = model.evaluate(X_test, y_test, verbose=0)
    print('score 1: ' + str(score[0]))
    print('score 2: ' + str(score[1]))
    uploadModels(DNA, model, name, score[0],score[1], gen, parent1, parent2)
    # with open('model.pkl', 'wb') as f:
    #     pickle.dump(model, f)
    return score

def uploadModels(DNA, model, name, score1, score2, gen, parent1, parent2):
        '''Upload the models to a mongo database'''
        uri = str(os.environ.get("MONGODB_URI_MODELS"))
        client = pymongo.MongoClient(uri)
        db = client.get_default_database()
        m = db['models']
        m.insert_one({'name': name, 'DNA':DNA, 'score': score2, 'loss':score1, 'gen': gen, 'parent1': parent1, 'parent2': parent2 })



if __name__ == '__main__':
    # get_data()
    evolve()
    # popdna = killOff({'Robert': ['Robert', 0.1087, {'NumN': 315, 'NumL': 3, 'LM': [4, 0, 5], 'LA': ['tanh', 'relu', 'tanh'], 'LI': 'random_uniform', 'LR': 0.0038882690316379698}], 'Brian': ['Brian', 0.1115, {'NumN': 485, 'NumL': 4, 'LM': [2, 2, 5, 0], 'LA': ['tanh', 'softplus', 'softsign', 'softsign'], 'LI': 'random_uniform', 'LR': 0.004628537323412493}], 'Celine': ['Celine', 0.14979999999999999, {'NumN': 32, 'NumL': 0, 'LM': [], 'LA': [], 'LI': 'random_uniform', 'LR': 0.0833583941408591}]},2)
    #mixGenDNA(popdna)
