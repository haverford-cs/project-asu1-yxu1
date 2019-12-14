"""
"""
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np 


def load_data(path=str, threshold=float):
    # load data
    data = np.loadtxt(path)

    # shuffle data
    np.random.seed(42)
    np.random.shuffle(data)
    
    # extract X, y
    X = data[:, :-10]
    y = data[:, -10:]
    y = np.argmax(y, axis=1) # convert one-hot encodings to integers

    train_X, val_X = np.split(X, [int(threshold*len(X))])
    train_y, val_y = np.split(y, [int(threshold*len(y))])
    
    return train_X, train_y, val_X, val_y

def draw_wrong_img(X, y, y_hat, filename):
    """
    Regenerate the images when their labels are predicted wrong.
    """
    i = 1
    for input_, prediction, label in zip(X, y_hat, y):
        if prediction != label:
            name = './img/{f}/{f}'.format(f=filename) + str(i)
            plt.imshow(input_.reshape(16,16), cmap="gray_r")
            plt.title('{}: True label: {}. Prediction: {}'.format(filename.upper(), label, prediction))
            plt.savefig('{}.png'.format(name))
            i+=1
            
def process_image(img):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (16,16))
    img[img<255]=1
    img[img==255]=0
    return img.reshape((256,))

def test_self_data():
    f = []
    
    for file_ in glob.glob('./data/self_data/*.png'): 
        f.append(file_)
        
    print(f)
    
    X = []
    for img in f:
        X.append(process_image(img))
        print(process_image(img).reshape(16,16))
    return np.array(X)
    
    
def main():
    train_X, train_y, val_X, val_y = load_data('./data/semeion.data', 0.8)
    
    # i = 900
    # x = train_X[i]
    # print(x.shape)
    print(val_X.shape)
   
    # plt.imshow(x, cmap="gray_r")
    # plt.savefig('test.png')
    # plt.show()
    
    test_self_data(train_X)    
 

if __name__ == '__main__':
    main()

