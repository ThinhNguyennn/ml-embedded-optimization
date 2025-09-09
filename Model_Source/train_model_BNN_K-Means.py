import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.cluster import KMeans
import datetime as dt
import resources
import matplotlib.pyplot as plt
from bitarray import bitarray
from tqdm import tqdm 
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape the input data
x_train = np.reshape(x_train, (60000, 784))
x_test = np.reshape(x_test, (10000, 784))

# convert the data to binary
x_train[x_train <= 127] = 0
x_train[x_train > 127] = 1
x_test[x_test <= 127] = 0
x_test[x_test > 127] = 1

# create the one_hot datatype for labels
y_train = np.matrix(np.eye(10)[y_train])
y_test = np.matrix(np.eye(10)[y_test])

# convert labels to binary
y_train[y_train == 0] = -1
y_test[y_test == 0] = -1

# define network parameters
LearningRate = 1
BatchSize = 200
Epochs = 10
Momentum = 0.99
NumOfTrainSample = 60000
NumOfTestSample = 10000
NumInput = 784
NumHidden = 512
NumOutput = 10

# hidden layer
Wh = np.matrix(np.random.uniform(-1, 1, (NumHidden, NumInput)))
bh = np.random.uniform(-1, 1, (1, NumHidden))
del_Wh = np.zeros((NumHidden, NumInput))
del_bh = np.zeros((1, NumHidden))

# output layer
Wo = np.random.uniform(-1, 1, (NumOutput, NumHidden))
bo = np.random.uniform(-1, 1, (1, NumOutput))
del_Wo = np.zeros((NumOutput, NumHidden))
del_bo = np.zeros((1, NumOutput))

# train the network with back propagation, SGD
SampleIdx = np.arange(NumOfTrainSample)
t_start = t1 = dt.datetime.now()
BiAcc = np.zeros(Epochs)
MSE = np.zeros(Epochs)
max_accuracy = 0
best_weights = defaultdict(lambda: defaultdict(list))
IdxCost = 0
Cost = np.zeros(np.int32(np.ceil(NumOfTrainSample / BatchSize)))
concatenated_images_dict = {}
best_weights = {}
def get_feature(img):
    intensity = img.sum() / img.shape[0]
    return np.array([intensity])

# Function to preprocess dataset by clustering
def Preprocess(dataset, cluster_num):
    preprocessed_dataset = defaultdict(lambda: defaultdict(list))
    for label, images in dataset.items():
        if len(images) >= cluster_num:
            features = [get_feature(img) for img in images]
            model = KMeans(cluster_num, n_init=10)
            model.fit(features)
            preds = model.predict(features)
            for i, img in enumerate(images):
                cluster = preds[i]
                preprocessed_dataset[label][cluster].append(img)
        else:
            single_cluster = 0
            for img in images:
                preprocessed_dataset[label][single_cluster].append(img)
    return preprocessed_dataset

cluster_num = 4

x_train_clusters = Preprocess({i: [img for j, img in enumerate(x_train) if y_train[j][0, i] == 1] for i in range(10)}, cluster_num)
x_test_clusters = Preprocess({i: [img for j, img in enumerate(x_test) if y_test[j][0, i] == 1] for i in range(10)}, cluster_num)

# =============================================================================
# for label, clusters in x_train_clusters.items():
#     print(f"Class {label}:")
#     for cluster, images in clusters.items():
#         print(f"  Cluster {cluster}: {len(images)} samples")
# =============================================================================

def convert_weights_to_binary(weights):
  """Converts weights from -1/1 to 0/1."""
  return np.where(weights > 0, 1, 0)

def convert_to_bitarray(binary_matrix):
    flattened_matrix = binary_matrix.flatten()
    binary_list = [bool(x) for x in flattened_matrix]  
    bit_array = bitarray(binary_list)
    return bit_array
      
for ep in range(Epochs):
    t1 = dt.datetime.now()
    for label, clusters in tqdm(x_train_clusters.items(), desc=f"Epoch {ep+1}/{Epochs}"):
        for cluster, images in clusters.items():                    
            # Shuffle the training samples
            np.random.shuffle(SampleIdx)
            for i in range(0, NumOfTrainSample - BatchSize, BatchSize):
                # Mini-batch Gradient descent algorithm
                Batch_sample = SampleIdx[i:i + BatchSize]
        
                # print(Batch_sample)
                x = np.matrix(x_train[Batch_sample, :])
                y = np.matrix(y_train[Batch_sample, :])
        
                # Feedforward propagation
                a = np.sign(np.dot(x, Wh.T) + bh)
                o = np.sign(np.dot(a, Wo.T) + bo)
        
                # calculate mean square error
                Cost[IdxCost] = np.mean(np.mean(np.power((y - o), 2), axis=1))
                IdxCost += 1
        
                # calculate loss function
                do = (y - o)
                dWo = np.matrix(np.dot(do.T, a) / BatchSize)
                dbo = np.mean(do, 0)
                WoUpdate = LearningRate * dWo + Momentum * del_Wo
                boUpdate = LearningRate * dbo + Momentum * del_bo
                del_Wo = WoUpdate
                del_bo = boUpdate
        
                # back propagate error through sign function
                dh = np.dot(do, Wo)
                dWh = np.dot(dh.T, x) / BatchSize
                dbh = np.mean(dh, 0)
        
                # Update Weight
                WhUpdate = LearningRate * dWh + Momentum * del_Wh
                bhUpdate = LearningRate * dbh + Momentum * del_bh
                del_Wh = WhUpdate
                del_bh = bhUpdate
        
                Wo = Wo + WoUpdate
                bo += boUpdate
                Wh = Wh + WhUpdate
                bh += bhUpdate
        
            MSE[ep] = np.mean(Cost)
            IdxCost = 0
    t2 = dt.datetime.now()
    training_time = t2 - t1
    print("Training epoch:", ep)
    print("MSE %f" % MSE[ep])
    print("Training time: %f seconds" % training_time.seconds)
    BiAccTest = np.zeros((10, cluster_num))
    
    for label, clusters in x_test_clusters.items():
        for cluster, images in clusters.items():
            # Calculate accuracy for each cluster on test dataset
            x_test_clusters[label][cluster] = np.matrix(np.eye(10)[label])
            x_test_clusters[label][cluster][x_test_clusters[label][cluster] == 0] = -1
            
            if (label, cluster) not in concatenated_images_dict:
                concatenated_images_dict[(label, cluster)] = np.vstack(images)
                   
            feed_forward = resources.FeedForward(concatenated_images_dict[(label, cluster)], Wh, bh, Wo, bo)
            BiOutN = feed_forward.OutN            
            acc_test = resources.AccTest(BiOutN, x_test_clusters[label][cluster])
            BiAccuracy = acc_test.accuracy
            BiAccTest[label][cluster] = BiAccuracy
            print("BiAccuracy",BiAccuracy)
    # Calculate the overall accuracy on the test dataset
    overall_accuracy = np.mean(BiAccTest)
    BiAcc[ep] = overall_accuracy
    
    print("Overall Test Accuracy:", overall_accuracy)
    # Nếu Overall Test Accuracy của epoch sau thấp hơn epoch trước
    if ep > 0 and BiAcc[ep] < BiAcc[ep - 1]:
        LearningRate *= 0.99
        print("Đã giảm LearningRate còn :", LearningRate)

    if overall_accuracy > max_accuracy:
        max_accuracy = overall_accuracy
        print("max_accuracy",max_accuracy)
        if label not in best_weights:
            best_weights = {}
        best_weights= {
        'Wh': convert_weights_to_binary(Wh.copy()),
        'bh': convert_weights_to_binary(bh.copy()),
        'Wo': convert_weights_to_binary(Wo.copy()),
        'bo': convert_weights_to_binary(bo.copy())
        }

for key, value in best_weights.items():
    best_weights[key] = convert_to_bitarray(value)

# Save the dictionary to a .npy file
np.save("BNN_model_cluster_29.npy", best_weights)
      
t_end = dt.datetime.now() - t_start
print("Total time: ", t_end)
print("Maximum Accuracy of Binary Weights: ")
print(max_accuracy)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))
axs[0].plot(BiAcc, "b-")
axs[0].set_ylabel('Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].set_title("Accuracy of Binary Weights")

axs[1].plot(MSE, "r-")
axs[1].set_ylabel('Loss')
axs[1].set_xlabel('Epoch')
axs[1].set_title("Loss of Binary Weights")

plt.savefig("figure.png")
plt.show()



