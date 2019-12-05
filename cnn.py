"""
"""
# import from public libraries
import numpy as np

# import from own libraries
from util import load_data 

def reshape_data(data, size=int):
    '''
    Reshape 256 features into 16*16 matrix
    '''
    return np.reshape(data,(-1,size,size))

def main():
    train_X, train_y, val_X, val_y = load_data('./data/semeion.data', 0.8)
    
    # reshape X
    train_X = reshape_data(train_X, 16)
    print(train_X.shape)
    val_X = reshape_data(val_X, 16)
    print(val_X.shape)
    
if __name__ == '__main__':
    main()