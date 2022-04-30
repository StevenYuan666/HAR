import matplotlib.pyplot as plt
import numpy as np

dropout_train_loss = []
dropout_train_accuracy = []
dropout_test_loss = []
dropout_test_accuracy = []

k_0_train_loss = []
k_0_train_accuracy = []
k_0_test_loss = []
k_0_test_accuracy = []

k_1_train_loss = []
k_1_train_accuracy = []
k_1_test_loss = []
k_1_test_accuracy = []

dropout_record = open("phase1_result/phone_record_dropout.txt", "r")
k_0_record = open("phase1_result/phone_record_k0.txt", "r")
k_1_record = open("phase1_result/phone_record_k1.txt", "r")

for line in dropout_record.readlines():
    line = line.replace("\n", "")
    line = line.replace("[", "")
    line = line.replace("]", "")
    l = line.split(" ")
    dropout_train_loss.append(float(l[3]))
    dropout_train_accuracy.append(float(l[5]))
    dropout_test_loss.append(float(l[7]))
    dropout_test_accuracy.append(float(l[9]))

for line in k_0_record.readlines():
    line = line.replace("\n", "")
    line = line.replace("[", "")
    line = line.replace("]", "")
    l = line.split(" ")
    k_0_train_loss.append(float(l[3]))
    k_0_train_accuracy.append(float(l[5]))
    k_0_test_loss.append(float(l[7]))
    k_0_test_accuracy.append(float(l[9]))

for line in k_1_record.readlines():
    line = line.replace("\n", "")
    line = line.replace("[", "")
    line = line.replace("]", "")
    l = line.split(" ")
    k_1_train_loss.append(float(l[3]))
    k_1_train_accuracy.append(float(l[5]))
    k_1_test_loss.append(float(l[7]))
    k_1_test_accuracy.append(float(l[9]))

dropout_record.close()
k_0_record.close()
k_1_record.close()

print(dropout_train_loss)
print(len(dropout_train_loss))

length = len(dropout_train_loss)
x = np.array(range(1, length * 250, 250))
print(x)
plt.plot(x, dropout_train_loss, label="Dropout")
plt.plot(x, k_0_train_loss, label="K=0")
plt.plot(x, k_1_train_loss, label="K=1")
plt.xlabel("Training Iterations")
plt.ylabel("SoftMax Cross Entropy")
plt.title("Training Loss")
plt.legend()
plt.savefig("./Training_Loss.png")
plt.show()

plt.plot(x, dropout_train_accuracy, label="Dropout")
plt.plot(x, k_0_train_accuracy, label="K=0")
plt.plot(x, k_1_train_accuracy, label="K=1")
plt.xlabel("Training Iterations")
plt.ylabel("Accuracy Score")
plt.title("Training Accuracy")
plt.legend()
plt.savefig("./Training_Accuracy.png")
plt.show()

plt.plot(x, dropout_test_loss, label="Dropout")
plt.plot(x, k_0_test_loss, label="K=0")
plt.plot(x, k_1_test_loss, label="K=1")
plt.xlabel("Training Iterations")
plt.ylabel("SoftMax Cross Entropy")
plt.title("Validation Loss")
plt.legend()
plt.savefig("./Validation_Loss.png")
plt.show()

plt.plot(x, dropout_test_accuracy, label="Dropout")
plt.plot(x, k_0_test_accuracy, label="K=0")
plt.plot(x, k_1_test_accuracy, label="K=1")
plt.xlabel("Training Iterations")
plt.ylabel("Accuracy Score")
plt.title("Validation Accuracy")
plt.legend()
plt.savefig("./Validation_Accuracy.png")
plt.show()
