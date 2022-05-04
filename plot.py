import matplotlib.pyplot as plt
import numpy as np

# phone
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

k_2_train_loss = []
k_2_train_accuracy = []
k_2_test_loss = []
k_2_test_accuracy = []

k_3_train_loss = []
k_3_train_accuracy = []
k_3_test_loss = []
k_3_test_accuracy = []

dropout_record = open("phase2_result/phone_record_dropout.txt", "r")
k_0_record = open("phase2_result/phone_record_k0.txt", "r")
k_1_record = open("phase2_result/phone_record_k1.txt", "r")
k_2_record = open("phase2_result/phone_record_k2.txt", "r")
k_3_record = open("phase2_result/phone_record_k3.txt", "r")

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

for line in k_2_record.readlines():
    line = line.replace("\n", "")
    line = line.replace("[", "")
    line = line.replace("]", "")
    l = line.split(" ")
    k_2_train_loss.append(float(l[3]))
    k_2_train_accuracy.append(float(l[5]))
    k_2_test_loss.append(float(l[7]))
    k_2_test_accuracy.append(float(l[9]))

for line in k_3_record.readlines():
    line = line.replace("\n", "")
    line = line.replace("[", "")
    line = line.replace("]", "")
    l = line.split(" ")
    k_3_train_loss.append(float(l[3]))
    k_3_train_accuracy.append(float(l[5]))
    k_3_test_loss.append(float(l[7]))
    k_3_test_accuracy.append(float(l[9]))

dropout_record.close()
k_0_record.close()
k_1_record.close()
k_2_record.close()
k_3_record.close()

length = len(dropout_train_loss)
x = np.array(range(1, length * 250, 250))
plt.plot(x, dropout_train_loss, label="Dropout")
plt.plot(x, k_0_train_loss, label="K=0")
plt.plot(x, k_1_train_loss, label="K=1")
plt.plot(x, k_2_train_loss, label="K=2")
plt.plot(x, k_3_train_loss, label="K=3")
plt.xlabel("Training Iterations")
plt.ylabel("SoftMax Cross Entropy")
plt.title("Phone Training Loss")
plt.legend()
plt.savefig("./phase2_result/Phone_Training_Loss.png")
plt.show()

plt.plot(x, dropout_train_accuracy, label="Dropout")
plt.plot(x, k_0_train_accuracy, label="K=0")
plt.plot(x, k_1_train_accuracy, label="K=1")
plt.plot(x, k_2_train_accuracy, label="K=2")
plt.plot(x, k_3_train_accuracy, label="K=3")
plt.xlabel("Training Iterations")
plt.ylabel("Accuracy Score")
plt.title("Phone Training Accuracy")
plt.legend()
plt.savefig("./phase2_result/Phone_Training_Accuracy.png")
plt.show()

plt.plot(x, dropout_test_loss, label="Dropout")
plt.plot(x, k_0_test_loss, label="K=0")
plt.plot(x, k_1_test_loss, label="K=1")
plt.plot(x, k_2_test_loss, label="K=2")
plt.plot(x, k_3_test_loss, label="K=3")
plt.xlabel("Training Iterations")
plt.ylabel("SoftMax Cross Entropy")
plt.title("Phone Validation Loss")
plt.legend()
plt.savefig("./phase2_result/Phone_Validation_Loss.png")
plt.show()

plt.plot(x, dropout_test_accuracy, label="Dropout")
plt.plot(x, k_0_test_accuracy, label="K=0")
plt.plot(x, k_1_test_accuracy, label="K=1")
plt.plot(x, k_2_test_accuracy, label="K=2")
plt.plot(x, k_3_test_accuracy, label="K=3")
plt.xlabel("Training Iterations")
plt.ylabel("Accuracy Score")
plt.title("Phone Validation Accuracy")
plt.legend()
plt.savefig("./phase2_result/Phone_Validation_Accuracy.png")
plt.show()

# watch
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

k_2_train_loss = []
k_2_train_accuracy = []
k_2_test_loss = []
k_2_test_accuracy = []

k_3_train_loss = []
k_3_train_accuracy = []
k_3_test_loss = []
k_3_test_accuracy = []

dropout_record = open("phase2_result/watch_record_dropout.txt", "r")
k_0_record = open("phase2_result/watch_record_k0.txt", "r")
k_1_record = open("phase2_result/watch_record_k1.txt", "r")
k_2_record = open("phase2_result/watch_record_k2.txt", "r")
k_3_record = open("phase2_result/watch_record_k3.txt", "r")

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

for line in k_2_record.readlines():
    line = line.replace("\n", "")
    line = line.replace("[", "")
    line = line.replace("]", "")
    l = line.split(" ")
    k_2_train_loss.append(float(l[3]))
    k_2_train_accuracy.append(float(l[5]))
    k_2_test_loss.append(float(l[7]))
    k_2_test_accuracy.append(float(l[9]))

for line in k_3_record.readlines():
    line = line.replace("\n", "")
    line = line.replace("[", "")
    line = line.replace("]", "")
    l = line.split(" ")
    k_3_train_loss.append(float(l[3]))
    k_3_train_accuracy.append(float(l[5]))
    k_3_test_loss.append(float(l[7]))
    k_3_test_accuracy.append(float(l[9]))

dropout_record.close()
k_0_record.close()
k_1_record.close()
k_2_record.close()
k_3_record.close()

length = len(dropout_train_loss)
x = np.array(range(1, length * 250, 250))
plt.plot(x, dropout_train_loss, label="Dropout")
plt.plot(x, k_0_train_loss, label="K=0")
plt.plot(x, k_1_train_loss, label="K=1")
plt.plot(x, k_2_train_loss, label="K=2")
plt.plot(x, k_3_train_loss, label="K=3")
plt.xlabel("Training Iterations")
plt.ylabel("SoftMax Cross Entropy")
plt.title("Watch Training Loss")
plt.legend()
plt.savefig("./phase2_result/Watch_Training_Loss.png")
plt.show()

plt.plot(x, dropout_train_accuracy, label="Dropout")
plt.plot(x, k_0_train_accuracy, label="K=0")
plt.plot(x, k_1_train_accuracy, label="K=1")
plt.plot(x, k_2_train_accuracy, label="K=2")
plt.plot(x, k_3_train_accuracy, label="K=3")
plt.xlabel("Training Iterations")
plt.ylabel("Accuracy Score")
plt.title("Watch Training Accuracy")
plt.legend()
plt.savefig("./phase2_result/Watch_Training_Accuracy.png")
plt.show()

plt.plot(x, dropout_test_loss, label="Dropout")
plt.plot(x, k_0_test_loss, label="K=0")
plt.plot(x, k_1_test_loss, label="K=1")
plt.plot(x, k_2_test_loss, label="K=2")
plt.plot(x, k_3_test_loss, label="K=3")
plt.xlabel("Training Iterations")
plt.ylabel("SoftMax Cross Entropy")
plt.title("Watch Validation Loss")
plt.legend()
plt.savefig("./phase2_result/Watch_Validation_Loss.png")
plt.show()

plt.plot(x, dropout_test_accuracy, label="Dropout")
plt.plot(x, k_0_test_accuracy, label="K=0")
plt.plot(x, k_1_test_accuracy, label="K=1")
plt.plot(x, k_2_test_accuracy, label="K=2")
plt.plot(x, k_3_test_accuracy, label="K=3")
plt.xlabel("Training Iterations")
plt.ylabel("Accuracy Score")
plt.title("Watch Validation Accuracy")
plt.legend()
plt.savefig("./phase2_result/Watch_Validation_Accuracy.png")
plt.show()
