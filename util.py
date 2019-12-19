"""
Utils for KNN, SVM, and CNN learning.
Author: Yongxin (Fiona) Xu
Date: 12/18/2019
"""
import csv
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np 
import seaborn as sns


def load_data(path, threshold):
    '''
    Load data from data file.
    Args:
        path(str): path to data file
        threshod(float): ratio of training and validation dset,
            e.g. 0.8 means 80% of data is training data and 20% is validation
    Returns:
        train_X(np.ndarray): training data's parameters
        train_y(np.ndarray): training data's labels
        val_X(np.ndarray): validation data's parameters
        val_y(np.ndarray): validation data's labels
    '''
    # load data
    data = np.loadtxt(path)
    # shuffle data
    np.random.seed(42)
    np.random.shuffle(data)   
    # extract X, y
    X = data[:, :-10]
    y = data[:, -10:]
    # convert one-hot encodings to integers
    y = np.argmax(y, axis=1) 
    train_X, val_X = np.split(X, [int(threshold*len(X))])
    train_y, val_y = np.split(y, [int(threshold*len(y))])   
    return train_X, train_y, val_X, val_y


def draw_wrong_img(X, y, y_hat, filename):
    """
    Generate the images of digits that were predicted wrong.
    Args:
        X(np.ndarray): data's parameters
        y(np.ndarray): data's true labels
        y_hat(np.ndarray): data's predicted labels
        filename(str): name of generated images file, used for creating 
            the path to save images (e.g. './img/(filename)/(filename).png')
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
    """
    Use OpenCV to resize an image into 16x16 and turn it to numpy array with binary value.
    Args:
        img(str): path to an image
    Returns:
        (np.ndarray): numpy array representation of this image
    """
    
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (16,16))
    img[img<=230]=1
    img[img>230]=0
    return img.reshape((256,))


def transform_image(imgs):
    """
    Transform a list of imgs into a 2-d numpy matrix, that is input's features.
    Args:
        imgs(list): a list str type paths to images
    Returns:
        (np.dnarry): input's features
    """
    
    X = []
    for img in imgs:
        X.append(process_image(img))
    return np.array(X)


def read_csv(path, img_path):
    """
    Read CSV file
    Args:
        path(str): path to the CSV file
        img_path(str): path to the image files
    Returns:
        imgs(list): list of paths to iamge files
        label(np.dnarray): input's label
    """
    
    with open(path, encoding='utf-8-sig') as file:
        readCSV = csv.reader(file, delimiter=',')
        imgs = []
        label = []
        for row in readCSV:
            imgs.append(img_path+row[0]+'.png')
            label.append(float(row[1]))
        label = np.array(label)
        return imgs, label

    
def get_my_data(path, img_path):
    """
    Get self-generated data, i.e. image data
    Args:
        path(str): path to the CSV file
        img_path(str): path to the image files
    Returns:
        features(np.dnarray): input's features
        label(np.dnarray): input's label
    """
    
    imgs, label = read_csv(path, img_path)
    features = transform_image(imgs)
    return features, label


def draw_heatmap(array, filename):
    """
    Draw a heatmap based on the confusion matrix for visualization
    Args:
        array(np.dnarray): confusion matrix
        filename(str): path to save the plot
    """
    
    ax = sns.heatmap(array, linewidth=0.2, annot=True, cmap="YlGnBu")
    plt.savefig(filename)
    plt.show()
 
    
def main():
    train_X, train_y, val_X, val_y = load_data('./data/semeion.data', 0.8)
    
    
if __name__ == '__main__':
    main()

