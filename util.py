"""
"""
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np 

# get logger
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('util')

def load_data(path=str, threshold=float):
    # load data
    data = np.loadtxt(path)
    logger.info(data.shape)

    # shuffle data
    np.random.shuffle(data)
    
    # extract X, y
    X = data[:, :-10]
    y = data[:, -10:]
    y = np.argmax(y, axis=1) # convert one-hot encodings to integers
    logger.info(X.shape)
    logger.info(y.shape)
    
    train_X, val_X = np.split(X, [int(threshold*len(X))])
    train_y, val_y = np.split(y, [int(threshold*len(y))])
    logger.info(train_X.shape)
    logger.info(train_y.shape)
    
    return train_X, train_y, val_X, val_y

def main():
    train_X, train_y, val_X, val_y = load_data('./data/semeion.data', 0.8)
    
    i = 900
    x = train_X[i].reshape(16,16)
    logger.info(x)
    logger.info(train_y[i])
   
    plt.imshow(x, cmap='gray')
    # plt.savefig('test.png')
    plt.show()

if __name__ == '__main__':
    main()

