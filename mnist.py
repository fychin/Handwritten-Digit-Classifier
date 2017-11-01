from struct import unpack
from collections import Counter
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pylab as plt


def loadmnist(imagefile, labelfile):

    # Open the images with gzip in read binary mode
    images = open(imagefile, 'rb')
    labels = open(labelfile, 'rb')

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)
    N = labels.read(4)
    N = unpack('>I', N)[0]

    # Get data
    x = np.zeros((N, rows*cols), dtype=np.uint8)  # Initialize numpy array
    y = np.zeros(N, dtype=np.uint8)  # Initialize numpy array
    for i in range(N):
        for j in range(rows*cols):
            tmp_pixel = images.read(1)  # Just a single byte
            tmp_pixel = unpack('>B', tmp_pixel)[0]
            x[i][j] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]

    images.close()
    labels.close()
    return (x, y)


def displaychar(image):
    plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()


def splitTrainingSet(training_data, training_labels, split_size):
    """
    Splits the training data into two separate sets: training and validation
    sets. The training set has the size of split_size passed as argument,
    and the validation set has the size of total size - split_size.
    Returns:
        4-tuple of training-set with size of split_size, training-set-labels 
        of size split_size, and the remaining data as validation-set and its
        labels.
    """
    training_set = training_data[:split_size]
    training_set_labels = training_labels[:split_size]
    validation_set_len = len(training_data) - split_size
    validation_set = training_data[split_size:]
    validation_set_labels = training_labels[split_size:]
    return (training_set, training_set_labels, validation_set, validation_set_labels)


def classifyTrainingData(training_set, training_labels):
    """
    Classify each training data to the respective class.
    Returns:
        A dictionary where the key is the class [0-9] and the value
        is the list of training data associated with the label.
    """
    classified_training_data = {}
    for idx, data in enumerate(training_set):
        current_label = training_labels[idx]
        # map current data to it's label
        if current_label in classified_training_data:
            list_of_data = classified_training_data[current_label]
            list_of_data.append(data)
        # add data to new key
        else:
            classified_training_data[current_label] = [data]
    return classified_training_data


def _findClassProbabilities(training_labels):
    """
    Given the training labels, find the distribution of each class [0-9]
    occurring within the list.
    Returns: 
        label_count_dict: dictionary with class [0-9] as key, and 
            the value as the fraction of class probabilities pi0-pi9.
    """
    label_count_dict = Counter(training_labels)
    total_label_size = len(training_labels)
    
    for label, count in label_count_dict.iteritems():
        label_count_dict[label] = count / float(total_label_size)

    return label_count_dict


def findGaussianParams(training_set, training_labels):
    """
    Given the training data and its labels, find the parameters for the 
    Gaussian distribution, namely class prob, mean and covariance for
    each class [0-9].
    Returns: 
        3-tuple dictionaries containing the class probabilities, mean, 
        and covariance.
    """
    classified_data = classifyTrainingData(training_set, training_labels)
    class_prob_distribution = _findClassProbabilities(training_labels)

    mean_j, cov_j = {}, {}
    for curr_class, list_of_data in classified_data.items():
        mean[curr_class] = np.mean(list_of_data, axis=0)
        cov[curr_class] = np.cov(list_of_data, rowvar=False)
    return (class_prob_distribution, mean_j, cov_j)


def fitGaussian(data, class_prob_dict, mean_dict, cov_dict, c_val):
    """
    Given data, a 784-dimensional vector, fit the Gaussian distribution
    to every class j: P_j * N(mean_j, cov_j), and return a list of 
    probabilities from P_0 to P_9.
    Params:
        c_val: used to smooth the covariance matrix by performing
                cov_j = cov_j + (c_val * I)
    Returns:
        gaussian_prob_list: a list of probabilities from fitting the gaussian,
            containing prob P_0 to P_9.
    """
    gaussian_prob_list = []

    # for each j, calculate: P(x|j) = pi_j * P_j
    # where P_j = N(mean_j, cov_j)
    for j in range(len(class_prob_dict)):
        # smooth the covariance matrix by c_val
        smoothed_cov = cov_dict[j] + (c_val * np.identity(784))
        # take the log of P_j since P_j is small
        logP_j = multivariate_normal.logpdf(data, mean_dict[j], smoothed_cov)
        logpi = np.log(class_prob_dict[j])
        # Since we are taking the log of P_j,
        # P(x|j) = log(pi_j) + log(P_j)
        this_class_prob = logpi + logP_j
        gaussian_prob_list.append(this_class_prob)
    return gaussian_prob_list


def getTestError(validation_set, validation_labels, c_val, 
                        class_prob_dict, mean_dict, cov_dict):
    """Prints out the test error for a given c_val.
    Returns:
        2-tuple of error rate (float) and list of prediction errors (pair)
        error rate: float, error rate on validation set.
        prediction errors: list of pairs, where the first element contains
                            the predicted label, and the second containing
                            the data.
    """
    error_count = 0
    prediction_errors = []
    for idx, data in enumerate(validation_set):
        prob_list = fitGaussian(data, class_prob_dict, mean_dict, cov_dict, c_val)
        predicted_label = prob_list.index(max(prob_list))   # key where the prob is max
        # check if prediction is wrong
        if predicted_label != validation_labels[idx]:
            error_count += 1
            prediction_errors.append((predicted_label, data))
    # get error rate
    error_rate = error_count / float(len(validation_set))
    return (error_rate, prediction_errors)


def main():
    # Load training data
    x, y = loadmnist('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')
    # len(x[0]) = 784

    # Split training set
    training_set, training_labels, validation_set, validation_labels = splitTrainingSet(x, y, 50000)
    # training_set is 50000, validation_set is 10000

    # Find mean and covariance for each class [0-9]
    class_prob, mean_dict, cov_dict = findGaussianParams(training_set, training_labels)

    # Choose appropriate c_value to smooth covariance matrices
    c_value_list = [0.1, 10, 50, 100, 1000, 5000, 10000]
    list_of_error_rates = []
    for cval in c_value_list:
        err_rate, p_list = getTestError(validation_set, validation_labels, cval, 
                                        class_prob, mean_dict, cov_dict)
        list_of_error_rates.append(err_rate)

    # Get best c_value, which is the c_val that yields the minimum error rate
    best_c_value = c_value_list[list_of_error_rates.index(min(list_of_error_rates))]

    # Load test data
    test_x, test_label = loadmnist('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')

    # Get test error 
    test_error_rate, pred_errors = getTestError(test_x, test_label, best_c_value, 
                                                class_prob, mean_dict, cov_dict)

    # Output test error
    print(test_error_rate)

    

    

if __name__ == '__main__':
    main()