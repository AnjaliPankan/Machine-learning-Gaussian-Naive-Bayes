import pandas as pd
import numpy as np
import math

# Load both training and test datasets.
data_train = pd.read_csv("./mnist_train.csv", header=None)
data_test = pd.read_csv("./mnist_test.csv", header=None)

# Determine number of classes and attributes from training dataset.
num_classes = len(np.unique(data_train[0]))
num_attributes = data_train.shape[1] - 1

# Determine number of test images.
test_data_count = data_test.shape[0]

# Calculate the mean and standard deviation of all classes for
# all attributes. Store these values in separate matrices with
# same dimentions: num_classes x num_attributes.
def naive_bayes():
    means = np.zeros((num_classes, num_attributes), dtype=np.float64)
    stdevs = np.zeros((num_classes, num_attributes), dtype=np.float64)
    
    digit_array = np.array(data_train[0])
    # Loop through each class in the training dataset.
    for class_num, class_name in enumerate(np.unique(data_train[0])):
        # For the current class, find the indices in the training dataset.
        indices = np.where(class_name == digit_array)
        class_total_instances = len(indices[0])

        # For the current class, determine mean for each attribute.
        for col in range(num_attributes):
            for row in indices[0]:
                means[class_num][col] = means[class_num][col] + data_train[col+1][row]

            means[class_num][col] = means[class_num][col] / class_total_instances

        # For the current class, determine standard deviation for each attribute.
        for col in range(num_attributes):
            mean_diff_sqr = 0.0
            for row in indices[0]:
                mean_diff_sqr = mean_diff_sqr + ((data_train[col+1][row] - means[class_num][col]) ** 2)

            stdevs[class_num][col] = math.sqrt(mean_diff_sqr / (class_total_instances - 1))
        
    return means, stdevs

means, stdevs = naive_bayes()

# Display mean and standard deviation of each attribute for each class.
for a in range(num_attributes):
    print("\t[", a+1, "]")
    print("\t", "mean\t\t\t", "standard deviation\n")
    for c in range(num_classes):
        print(c, "\t", means[c][a], "\t\t\t", stdevs[c][a]) 

# Calculate prior probabilities from the training dataset.
_, fractions = np.unique(data_train[0], return_counts=True)
pri_probs = fractions / len(data_train[0])
print("prior probabilities: ", pri_probs)

# Determine the likelihood estimation of a given test data.
# Used log to eliminate arithmatic undeflow.
def log_gaussian_likelihood(x, mean, stdev):
    # Removing those attributes where the standard deviation is zero.
    remove_indices = np.where(stdev == 0)[0]
    
    stdev_cleaned = np.delete(stdev, remove_indices)
    mean_cleaned = np.delete(mean, remove_indices)
    x_cleaned = np.delete(x, remove_indices)
    sqrt_2_pi =  math.sqrt(2 * math.pi)

    lgl = -(np.sum(np.log(sqrt_2_pi * stdev_cleaned)) + 0.5 * np.sum(((x_cleaned - mean_cleaned) / stdev_cleaned) ** 2)).reshape(-1, 1)

    return lgl

# Predict the digit using Gaussian naive Bayes classification.
def predict(x):
    c_predict = np.argmax([np.log(pri_probs[c]) + log_gaussian_likelihood(x, means[c], stdevs[c]) for c in range(num_classes)])

    return c_predict

# Determine the confusion matrix.
def confusion_matrix(predicted_digits, actual_digits):
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    
    for i in range(len(predicted_digits)):
        cm[actual_digits[i]][predicted_digits[i]] = cm[actual_digits[i]][predicted_digits[i]] + 1

    return cm

# Determine precision macroaverage.
def p_macroaverage():
    p_macro = np.zeros((num_classes), dtype=np.float64)
    
    for i in range(num_classes):
        sum_t = 0 
        frac = confusion_matrix[i][i]
        for j in range(num_classes):
            sum_t = sum_t + confusion_matrix[j][i]

        p_macro[i] = frac / sum_t

    return np.sum(p_macro) / num_classes

# Determine recall macroaverage.
def r_macroaverage():
    r_macro = np.zeros((num_classes), dtype=np.float64)

    for i in range(num_classes):
        sum_t = 0 
        frac = confusion_matrix[i][i]
        for j in range(num_classes):
            sum_t = sum_t + confusion_matrix[i][j]

        r_macro[i] = frac / sum_t

    return np.sum(r_macro) / num_classes

# Get a row from the test dataset for a given index.
def get_test_data_row(index):
    test_row = np.zeros((num_attributes), dtype=np.float64)

    for i in range(num_attributes):
        test_row[i] = data_test[i+1][index]

    return np.array(test_row)

# Predict digit correspond to each row in the test dataset.  
predicted_digits = np.zeros((test_data_count), dtype=np.int32)
for n in range(test_data_count):
    x = get_test_data_row(n)
    predicted_digits[n] = predict(x)

actual_digits = np.array(data_test[0], dtype=np.int32)

# Determine accuracy of prediction compared to the actual values. 
accuracy = (actual_digits == predicted_digits).sum()/float(actual_digits.size)
print("accuracy: ", accuracy)

confusion_matrix = confusion_matrix(predicted_digits, actual_digits)
print(confusion_matrix)

p_m = p_macroaverage()
r_m = r_macroaverage()
print("p_macroaverage: ", p_m)
print("r_macroaverage: ", r_m)

# Determine the F1-score.
f1_score = (2 * p_m * r_m) / (p_m + r_m) 
print("F1-score: ", f1_score)
