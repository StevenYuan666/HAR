import matplotlib.pyplot as plt
import numpy as np


def get_record(path):
    train_loss = []
    train_accuracy = []
    validation_loss = []
    validation_accuracy = []

    record = open(path, 'r')

    test_loss = 0
    test_accuracy = 0

    for line in record.readlines():
        try:
            if line.split(" ")[0] == "Step:":
                line = line.replace("\n", "")
                line = line.replace("[", "")
                line = line.replace("]", "")
                l = line.split(" ")
                train_loss.append(float(l[3]))
                train_accuracy.append(float(l[5]))
                validation_loss.append(float(l[7]))
                validation_accuracy.append(float(l[9]))
            if line.split(" ")[0] == "Target":
                line = line.replace("\n", "")
                line = line.replace("[", "")
                line = line.replace("]", "")
                l = line.split(" ")
                test_loss = float(l[5])
                test_accuracy = float(l[2])

        except IndexError:
            break

    record.close()

    return train_loss, train_accuracy, validation_loss, validation_accuracy, test_loss, test_accuracy


def phone():
    phone_k0_run1_train_loss, phone_k0_run1_train_accuracy, phone_k0_run1_test_loss, phone_k0_run1_test_accuracy \
        = get_record(path="phase3_result/phone_k0_run1.txt")
    phone_k0_run2_train_loss, phone_k0_run2_train_accuracy, phone_k0_run2_test_loss, phone_k0_run2_test_accuracy \
        = get_record(path="phase3_result/phone_k0_run2.txt")
    phone_k0_run3_train_loss, phone_k0_run3_train_accuracy, phone_k0_run3_test_loss, phone_k0_run3_test_accuracy \
        = get_record(path="phase3_result/phone_k0_run3.txt")

    phone_k1_run1_train_loss, phone_k1_run1_train_accuracy, phone_k1_run1_test_loss, phone_k1_run1_test_accuracy \
        = get_record(path="phase3_result/phone_k1_run1.txt")
    phone_k1_run2_train_loss, phone_k1_run2_train_accuracy, phone_k1_run2_test_loss, phone_k1_run2_test_accuracy \
        = get_record(path="phase3_result/phone_k1_run2.txt")
    phone_k1_run3_train_loss, phone_k1_run3_train_accuracy, phone_k1_run3_test_loss, phone_k1_run3_test_accuracy \
        = get_record(path="phase3_result/phone_k1_run3.txt")

    phone_k2_run1_train_loss, phone_k2_run1_train_accuracy, phone_k2_run1_test_loss, phone_k2_run1_test_accuracy \
        = get_record(path="phase3_result/phone_k2_run1.txt")
    phone_k2_run2_train_loss, phone_k2_run2_train_accuracy, phone_k2_run2_test_loss, phone_k2_run2_test_accuracy \
        = get_record(path="phase3_result/phone_k2_run2.txt")
    phone_k2_run3_train_loss, phone_k2_run3_train_accuracy, phone_k2_run3_test_loss, phone_k2_run3_test_accuracy \
        = get_record(path="phase3_result/phone_k2_run3.txt")

    phone_k3_run1_train_loss, phone_k3_run1_train_accuracy, phone_k3_run1_test_loss, phone_k3_run1_test_accuracy \
        = get_record(path="phase3_result/phone_k3_run1.txt")
    phone_k3_run2_train_loss, phone_k3_run2_train_accuracy, phone_k3_run2_test_loss, phone_k3_run2_test_accuracy \
        = get_record(path="phase3_result/phone_k3_run2.txt")
    phone_k3_run3_train_loss, phone_k3_run3_train_accuracy, phone_k3_run3_test_loss, phone_k3_run3_test_accuracy \
        = get_record(path="phase3_result/phone_k3_run3.txt")

    phone_k0_train_loss = [phone_k0_run1_train_loss, phone_k0_run2_train_loss, phone_k0_run3_train_loss]
    phone_k0_train_accuracy = [phone_k0_run1_train_accuracy, phone_k0_run2_train_accuracy, phone_k0_run3_train_accuracy]
    phone_k0_test_loss = [phone_k0_run1_test_loss, phone_k0_run2_test_loss, phone_k0_run3_test_loss]
    phone_k0_test_accuracy = [phone_k0_run1_test_accuracy, phone_k0_run2_test_accuracy, phone_k0_run3_test_accuracy]

    phone_k1_train_loss = [phone_k1_run1_train_loss, phone_k1_run2_train_loss, phone_k1_run3_train_loss]
    phone_k1_train_accuracy = [phone_k1_run1_train_accuracy, phone_k1_run2_train_accuracy, phone_k1_run3_train_accuracy]
    phone_k1_test_loss = [phone_k1_run1_test_loss, phone_k1_run2_test_loss, phone_k1_run3_test_loss]
    phone_k1_test_accuracy = [phone_k1_run1_test_accuracy, phone_k1_run2_test_accuracy, phone_k1_run3_test_accuracy]

    phone_k2_train_loss = [phone_k2_run1_train_loss, phone_k2_run2_train_loss, phone_k2_run3_train_loss]
    phone_k2_train_accuracy = [phone_k2_run1_train_accuracy, phone_k2_run2_train_accuracy, phone_k2_run3_train_accuracy]
    phone_k2_test_loss = [phone_k2_run1_test_loss, phone_k2_run2_test_loss, phone_k2_run3_test_loss]
    phone_k2_test_accuracy = [phone_k2_run1_test_accuracy, phone_k2_run2_test_accuracy, phone_k2_run3_test_accuracy]

    phone_k3_train_loss = [phone_k3_run1_train_loss, phone_k3_run2_train_loss, phone_k3_run3_train_loss]
    phone_k3_train_accuracy = [phone_k3_run1_train_accuracy, phone_k3_run2_train_accuracy, phone_k3_run3_train_accuracy]
    phone_k3_test_loss = [phone_k3_run1_test_loss, phone_k3_run2_test_loss, phone_k3_run3_test_loss]
    phone_k3_test_accuracy = [phone_k3_run1_test_accuracy, phone_k3_run2_test_accuracy, phone_k3_run3_test_accuracy]

    length = len(phone_k0_run1_train_loss)
    x = np.array(range(1, length * 100, 100))

    # Training Loss
    _, _, bars1 = plt.errorbar(x, np.mean(phone_k0_train_loss, axis=0), xerr=None,
                               yerr=np.std(phone_k0_train_loss, axis=0), label='K=0')
    [bar.set_alpha(0.3) for bar in bars1]

    _, _, bars2 = plt.errorbar(x, np.mean(phone_k1_train_loss, axis=0), xerr=None,
                               yerr=np.std(phone_k1_train_loss, axis=0), label='K=1')
    [bar.set_alpha(0.3) for bar in bars2]

    _, _, bars3 = plt.errorbar(x, np.mean(phone_k2_train_loss, axis=0), xerr=None,
                               yerr=np.std(phone_k2_train_loss, axis=0), label='K=2')
    [bar.set_alpha(0.3) for bar in bars3]

    _, _, bars4 = plt.errorbar(x, np.mean(phone_k3_train_loss, axis=0), xerr=None,
                               yerr=np.std(phone_k3_train_loss, axis=0), label='K=3')
    [bar.set_alpha(0.3) for bar in bars4]

    plt.xlabel("Training Iterations")
    plt.ylabel("Training SoftMax Loss")
    plt.title("Phone Average Training Loss Curve")  # 三个run的average和std
    plt.legend()
    plt.savefig("./phase3_result/Phone_Training_Loss.png")
    plt.show()

    # Training Accuracy
    _, _, bars1 = plt.errorbar(x, np.mean(phone_k0_train_accuracy, axis=0), xerr=None,
                               yerr=np.std(phone_k0_train_accuracy, axis=0), label='K=0')
    [bar.set_alpha(0.3) for bar in bars1]

    _, _, bars2 = plt.errorbar(x, np.mean(phone_k1_train_accuracy, axis=0), xerr=None,
                               yerr=np.std(phone_k1_train_accuracy, axis=0), label='K=1')
    [bar.set_alpha(0.3) for bar in bars2]

    _, _, bars3 = plt.errorbar(x, np.mean(phone_k2_train_accuracy, axis=0), xerr=None,
                               yerr=np.std(phone_k2_train_accuracy, axis=0), label='K=2')
    [bar.set_alpha(0.3) for bar in bars3]

    _, _, bars4 = plt.errorbar(x, np.mean(phone_k3_train_accuracy, axis=0), xerr=None,
                               yerr=np.std(phone_k3_train_accuracy, axis=0), label='K=3')
    [bar.set_alpha(0.3) for bar in bars4]

    plt.xlabel("Training Iterations")
    plt.ylabel("Training Accuracy")
    plt.title("Phone Average Training Accuracy Curve")  # 三个run的average和std
    plt.legend()
    plt.savefig("./phase3_result/Phone_Training_Accuracy.png")
    plt.show()

    # Validation Error
    _, _, bars1 = plt.errorbar(x, np.mean(phone_k0_test_loss, axis=0), xerr=None,
                               yerr=np.std(phone_k0_test_loss, axis=0), label='K=0')
    [bar.set_alpha(0.3) for bar in bars1]

    _, _, bars2 = plt.errorbar(x, np.mean(phone_k1_test_loss, axis=0), xerr=None,
                               yerr=np.std(phone_k1_test_loss, axis=0), label='K=1')
    [bar.set_alpha(0.3) for bar in bars2]

    _, _, bars3 = plt.errorbar(x, np.mean(phone_k2_test_loss, axis=0), xerr=None,
                               yerr=np.std(phone_k2_test_loss, axis=0), label='K=2')
    [bar.set_alpha(0.3) for bar in bars3]

    _, _, bars4 = plt.errorbar(x, np.mean(phone_k3_test_loss, axis=0), xerr=None,
                               yerr=np.std(phone_k3_test_loss, axis=0), label='K=3')
    [bar.set_alpha(0.3) for bar in bars4]

    plt.xlabel("Training Iterations")
    plt.ylabel("Validation SoftMax Loss")
    plt.title("Phone Average Validation Loss Curve")  # 三个run的average和std
    plt.legend()
    plt.savefig("./phase3_result/Phone_Validation_Loss.png")
    plt.show()

    # Validation Accuracy
    _, _, bars1 = plt.errorbar(x, np.mean(phone_k0_test_accuracy, axis=0), xerr=None,
                               yerr=np.std(phone_k0_test_accuracy, axis=0), label='K=0')
    [bar.set_alpha(0.3) for bar in bars1]

    _, _, bars2 = plt.errorbar(x, np.mean(phone_k1_test_accuracy, axis=0), xerr=None,
                               yerr=np.std(phone_k1_test_accuracy, axis=0), label='K=1')
    [bar.set_alpha(0.3) for bar in bars2]

    _, _, bars3 = plt.errorbar(x, np.mean(phone_k2_test_accuracy, axis=0), xerr=None,
                               yerr=np.std(phone_k2_test_accuracy, axis=0), label='K=2')
    [bar.set_alpha(0.3) for bar in bars3]

    _, _, bars4 = plt.errorbar(x, np.mean(phone_k3_test_accuracy, axis=0), xerr=None,
                               yerr=np.std(phone_k3_test_accuracy, axis=0), label='K=3')
    [bar.set_alpha(0.3) for bar in bars4]

    plt.xlabel("Training Iterations")
    plt.ylabel("Validation Accuracy")
    plt.title("Phone Average Validation Accuracy Curve")  # 三个run的average和std
    plt.legend()
    plt.savefig("./phase3_result/Phone_Validation_Accuracy.png")
    plt.show()

    # Phone Test
    k0_accuracy = [0.1737, 0.1809, 0.1923]
    k0_loss = [15.3820, 14.7726, 15.3516]
    k1_accuracy = [0.1680, 0.1583, 0.1926]
    k1_loss = [14.3930, 14.8340, 14.0141]
    k2_accuracy = [0.1884, 0.1924, 0.2013]
    k2_loss = [15.2263, 13.8932, 14.1229]
    k3_accuracy = [0.1916, 0.1880, 0.1949]
    k3_loss = [11.7219, 12.0024, 12.9763]

    plt.bar(x=0, height=np.mean(k0_accuracy), xerr=None, yerr=np.std(k0_accuracy), label="K=0")
    plt.bar(x=1, height=np.mean(k1_accuracy), xerr=None, yerr=np.std(k1_accuracy), label="K=1")
    plt.bar(x=2, height=np.mean(k2_accuracy), xerr=None, yerr=np.std(k2_accuracy), label="K=2")
    plt.bar(x=3, height=np.mean(k3_accuracy), xerr=None, yerr=np.std(k3_accuracy), label="K=3")
    plt.xlabel("Value of K")
    plt.ylabel("Test Accuracy")
    plt.title("Phone Average Test Accuracy")  # 三个run的average和std
    plt.legend()
    plt.savefig("./phase3_result/Phone_Test_Accuracy.png")
    plt.show()

    plt.bar(x=0, height=np.mean(k0_loss), xerr=None, yerr=np.std(k0_loss), label="K=0")
    plt.bar(x=1, height=np.mean(k1_loss), xerr=None, yerr=np.std(k1_loss), label="K=1")
    plt.bar(x=2, height=np.mean(k2_loss), xerr=None, yerr=np.std(k2_loss), label="K=2")
    plt.bar(x=3, height=np.mean(k3_loss), xerr=None, yerr=np.std(k3_loss), label="K=3")
    plt.xlabel("Value of K")
    plt.ylabel("Test SoftMax Loss")
    plt.title("Phone Average Test SoftMax Loss")  # 三个run的average和std
    plt.legend()
    plt.savefig("./phase3_result/Phone_Test_Loss.png")
    plt.show()


def watch():
    watch_k0_run1_train_loss, watch_k0_run1_train_accuracy, watch_k0_run1_test_loss, watch_k0_run1_test_accuracy \
        = get_record(path="phase4_result/watch_k0_run1.txt")
    watch_k0_run2_train_loss, watch_k0_run2_train_accuracy, watch_k0_run2_test_loss, watch_k0_run2_test_accuracy \
        = get_record(path="phase4_result/watch_k0_run2.txt")
    watch_k0_run3_train_loss, watch_k0_run3_train_accuracy, watch_k0_run3_test_loss, watch_k0_run3_test_accuracy \
        = get_record(path="phase4_result/watch_k0_run3.txt")

    watch_k1_run1_train_loss, watch_k1_run1_train_accuracy, watch_k1_run1_test_loss, watch_k1_run1_test_accuracy \
        = get_record(path="phase4_result/watch_k1_run1.txt")
    watch_k1_run2_train_loss, watch_k1_run2_train_accuracy, watch_k1_run2_test_loss, watch_k1_run2_test_accuracy \
        = get_record(path="phase4_result/watch_k1_run2.txt")
    watch_k1_run3_train_loss, watch_k1_run3_train_accuracy, watch_k1_run3_test_loss, watch_k1_run3_test_accuracy \
        = get_record(path="phase4_result/watch_k1_run3.txt")

    watch_k2_run1_train_loss, watch_k2_run1_train_accuracy, watch_k2_run1_test_loss, watch_k2_run1_test_accuracy \
        = get_record(path="phase4_result/watch_k2_run1.txt")
    watch_k2_run2_train_loss, watch_k2_run2_train_accuracy, watch_k2_run2_test_loss, watch_k2_run2_test_accuracy \
        = get_record(path="phase4_result/watch_k2_run2.txt")
    watch_k2_run3_train_loss, watch_k2_run3_train_accuracy, watch_k2_run3_test_loss, watch_k2_run3_test_accuracy \
        = get_record(path="phase4_result/watch_k2_run3.txt")

    watch_k3_run1_train_loss, watch_k3_run1_train_accuracy, watch_k3_run1_test_loss, watch_k3_run1_test_accuracy \
        = get_record(path="phase4_result/watch_k3_run1.txt")
    watch_k3_run2_train_loss, watch_k3_run2_train_accuracy, watch_k3_run2_test_loss, watch_k3_run2_test_accuracy \
        = get_record(path="phase4_result/watch_k3_run2.txt")
    watch_k3_run3_train_loss, watch_k3_run3_train_accuracy, watch_k3_run3_test_loss, watch_k3_run3_test_accuracy \
        = get_record(path="phase4_result/watch_k3_run3.txt")

    watch_k0_train_loss = [watch_k0_run1_train_loss, watch_k0_run2_train_loss, watch_k0_run3_train_loss]
    watch_k0_train_accuracy = [watch_k0_run1_train_accuracy, watch_k0_run2_train_accuracy, watch_k0_run3_train_accuracy]
    watch_k0_test_loss = [watch_k0_run1_test_loss, watch_k0_run2_test_loss, watch_k0_run3_test_loss]
    watch_k0_test_accuracy = [watch_k0_run1_test_accuracy, watch_k0_run2_test_accuracy, watch_k0_run3_test_accuracy]

    watch_k1_train_loss = [watch_k1_run1_train_loss, watch_k1_run2_train_loss, watch_k1_run3_train_loss]
    watch_k1_train_accuracy = [watch_k1_run1_train_accuracy, watch_k1_run2_train_accuracy, watch_k1_run3_train_accuracy]
    watch_k1_test_loss = [watch_k1_run1_test_loss, watch_k1_run2_test_loss, watch_k1_run3_test_loss]
    watch_k1_test_accuracy = [watch_k1_run1_test_accuracy, watch_k1_run2_test_accuracy, watch_k1_run3_test_accuracy]

    watch_k2_train_loss = [watch_k2_run1_train_loss, watch_k2_run2_train_loss, watch_k2_run3_train_loss]
    watch_k2_train_accuracy = [watch_k2_run1_train_accuracy, watch_k2_run2_train_accuracy, watch_k2_run3_train_accuracy]
    watch_k2_test_loss = [watch_k2_run1_test_loss, watch_k2_run2_test_loss, watch_k2_run3_test_loss]
    watch_k2_test_accuracy = [watch_k2_run1_test_accuracy, watch_k2_run2_test_accuracy, watch_k2_run3_test_accuracy]

    watch_k3_train_loss = [watch_k3_run1_train_loss, watch_k3_run2_train_loss, watch_k3_run3_train_loss]
    watch_k3_train_accuracy = [watch_k3_run1_train_accuracy, watch_k3_run2_train_accuracy, watch_k3_run3_train_accuracy]
    watch_k3_test_loss = [watch_k3_run1_test_loss, watch_k3_run2_test_loss, watch_k3_run3_test_loss]
    watch_k3_test_accuracy = [watch_k3_run1_test_accuracy, watch_k3_run2_test_accuracy, watch_k3_run3_test_accuracy]

    length = len(watch_k0_run1_train_loss)
    x = np.array(range(1, length * 100, 100))

    # Training Loss
    _, _, bars1 = plt.errorbar(x, np.mean(watch_k0_train_loss, axis=0), xerr=None,
                               yerr=np.std(watch_k0_train_loss, axis=0), label='K=0')
    [bar.set_alpha(0.3) for bar in bars1]

    _, _, bars2 = plt.errorbar(x, np.mean(watch_k1_train_loss, axis=0), xerr=None,
                               yerr=np.std(watch_k1_train_loss, axis=0), label='K=1')
    [bar.set_alpha(0.3) for bar in bars2]

    _, _, bars3 = plt.errorbar(x, np.mean(watch_k2_train_loss, axis=0), xerr=None,
                               yerr=np.std(watch_k2_train_loss, axis=0), label='K=2')
    [bar.set_alpha(0.3) for bar in bars3]

    _, _, bars4 = plt.errorbar(x, np.mean(watch_k3_train_loss, axis=0), xerr=None,
                               yerr=np.std(watch_k3_train_loss, axis=0), label='K=3')
    [bar.set_alpha(0.3) for bar in bars4]

    plt.xlabel("Training Iterations")
    plt.ylabel("Training SoftMax Loss")
    plt.title("Watch Average Training Loss Curve")  # 三个run的average和std
    plt.legend()
    plt.savefig("./phase4_result/Watch_Training_Loss.png")
    plt.show()

    # Training Accuracy
    _, _, bars1 = plt.errorbar(x, np.mean(watch_k0_train_accuracy, axis=0), xerr=None,
                               yerr=np.std(watch_k0_train_accuracy, axis=0), label='K=0')
    [bar.set_alpha(0.3) for bar in bars1]

    _, _, bars2 = plt.errorbar(x, np.mean(watch_k1_train_accuracy, axis=0), xerr=None,
                               yerr=np.std(watch_k1_train_accuracy, axis=0), label='K=1')
    [bar.set_alpha(0.3) for bar in bars2]

    _, _, bars3 = plt.errorbar(x, np.mean(watch_k2_train_accuracy, axis=0), xerr=None,
                               yerr=np.std(watch_k2_train_accuracy, axis=0), label='K=2')
    [bar.set_alpha(0.3) for bar in bars3]

    _, _, bars4 = plt.errorbar(x, np.mean(watch_k3_train_accuracy, axis=0), xerr=None,
                               yerr=np.std(watch_k3_train_accuracy, axis=0), label='K=3')
    [bar.set_alpha(0.3) for bar in bars4]

    plt.xlabel("Training Iterations")
    plt.ylabel("Training Accuracy")
    plt.title("Watch Average Training Accuracy Curve")  # 三个run的average和std
    plt.legend()
    plt.savefig("./phase4_result/Watch_Training_Accuracy.png")
    plt.show()

    # Validation Error
    _, _, bars1 = plt.errorbar(x, np.mean(watch_k0_test_loss, axis=0), xerr=None,
                               yerr=np.std(watch_k0_test_loss, axis=0), label='K=0')
    [bar.set_alpha(0.3) for bar in bars1]

    _, _, bars2 = plt.errorbar(x, np.mean(watch_k1_test_loss, axis=0), xerr=None,
                               yerr=np.std(watch_k1_test_loss, axis=0), label='K=1')
    [bar.set_alpha(0.3) for bar in bars2]

    _, _, bars3 = plt.errorbar(x, np.mean(watch_k2_test_loss, axis=0), xerr=None,
                               yerr=np.std(watch_k2_test_loss, axis=0), label='K=2')
    [bar.set_alpha(0.3) for bar in bars3]

    _, _, bars4 = plt.errorbar(x, np.mean(watch_k3_test_loss, axis=0), xerr=None,
                               yerr=np.std(watch_k3_test_loss, axis=0), label='K=3')
    [bar.set_alpha(0.3) for bar in bars4]

    plt.xlabel("Training Iterations")
    plt.ylabel("Validation SoftMax Loss")
    plt.title("Watch Average Validation Loss Curve")  # 三个run的average和std
    plt.legend()
    plt.savefig("./phase4_result/Watch_Validation_Loss.png")
    plt.show()

    # Validation Accuracy
    _, _, bars1 = plt.errorbar(x, np.mean(watch_k0_test_accuracy, axis=0), xerr=None,
                               yerr=np.std(watch_k0_test_accuracy, axis=0), label='K=0')
    [bar.set_alpha(0.3) for bar in bars1]

    _, _, bars2 = plt.errorbar(x, np.mean(watch_k1_test_accuracy, axis=0), xerr=None,
                               yerr=np.std(watch_k1_test_accuracy, axis=0), label='K=1')
    [bar.set_alpha(0.3) for bar in bars2]

    _, _, bars3 = plt.errorbar(x, np.mean(watch_k2_test_accuracy, axis=0), xerr=None,
                               yerr=np.std(watch_k2_test_accuracy, axis=0), label='K=2')
    [bar.set_alpha(0.3) for bar in bars3]

    _, _, bars4 = plt.errorbar(x, np.mean(watch_k3_test_accuracy, axis=0), xerr=None,
                               yerr=np.std(watch_k3_test_accuracy, axis=0), label='K=3')
    [bar.set_alpha(0.3) for bar in bars4]

    plt.xlabel("Training Iterations")
    plt.ylabel("Validation Accuracy")
    plt.title("Watch Average Validation Accuracy Curve")  # 三个run的average和std
    plt.legend()
    plt.savefig("./phase4_result/Watch_Validation_Accuracy.png")
    plt.show()

    # Phone Test
    k0_accuracy = [0.6013, 0.6526, 0.6083]
    k0_loss = [1.2493, 1.1666, 1.2320]
    k1_accuracy = [0.6429, 0.6317, 0.6627]
    k1_loss = [1.1894, 1.2254, 1.1697]
    k2_accuracy = [0.6572, 0.6322, 0.6453]
    k2_loss = [1.1679, 1.1582, 1.1748]
    k3_accuracy = [0.6172, 0.6238, 0.6265]
    k3_loss = [1.2128, 1.1687, 1.2445]

    plt.bar(x=0, height=np.mean(k0_accuracy), xerr=None, yerr=np.std(k0_accuracy), label="K=0")
    plt.bar(x=1, height=np.mean(k1_accuracy), xerr=None, yerr=np.std(k1_accuracy), label="K=1")
    plt.bar(x=2, height=np.mean(k2_accuracy), xerr=None, yerr=np.std(k2_accuracy), label="K=2")
    plt.bar(x=3, height=np.mean(k3_accuracy), xerr=None, yerr=np.std(k3_accuracy), label="K=3")
    plt.xlabel("Value of K")
    plt.ylabel("Test Accuracy")
    plt.title("Watch Average Test Accuracy")  # 三个run的average和std
    plt.legend()
    plt.savefig("./phase4_result/Watch_Test_Accuracy.png")
    plt.show()

    plt.bar(x=0, height=np.mean(k0_loss), xerr=None, yerr=np.std(k0_loss), label="K=0")
    plt.bar(x=1, height=np.mean(k1_loss), xerr=None, yerr=np.std(k1_loss), label="K=1")
    plt.bar(x=2, height=np.mean(k2_loss), xerr=None, yerr=np.std(k2_loss), label="K=2")
    plt.bar(x=3, height=np.mean(k3_loss), xerr=None, yerr=np.std(k3_loss), label="K=3")
    plt.xlabel("Value of K")
    plt.ylabel("Test SoftMax Loss")
    plt.title("Watch Average Test SoftMax Loss")  # 三个run的average和std
    plt.legend()
    plt.savefig("./phase4_result/Watch_Test_Loss.png")
    plt.show()


def analyze(phase, k, num_runs, iterations=2000, step=100, flag="watch"):
    train_loss = []
    train_accuracy = []
    validation_loss = []
    validation_accuracy = []
    test_loss = []
    test_accuracy = []
    for i in range(k + 1):
        temp_k_train_loss = []
        temp_k_train_accuracy = []
        temp_k_validation_loss = []
        temp_k_validation_accuracy = []
        temp_k_test_loss = []
        temp_k_test_accuracy = []
        for j in range(1, num_runs + 1):
            temp_path = "./phase" + str(phase) + "_result/" + flag + "_k" + str(i) + "_run" + str(j) + ".txt"
            temp_train_loss, temp_train_accuracy, temp_validation_loss, temp_validation_accuracy, temp_test_loss, \
                temp_test_accuracy = get_record(path=temp_path)
            temp_k_train_loss.append(temp_train_loss)
            temp_k_train_accuracy.append(temp_train_accuracy)
            temp_k_validation_loss.append(temp_validation_loss)
            temp_k_validation_accuracy.append(temp_validation_accuracy)
            temp_k_test_loss.append(temp_test_loss)
            temp_k_test_accuracy.append(temp_test_accuracy)
        train_loss.append(temp_k_train_loss)
        train_accuracy.append(temp_k_train_accuracy)
        validation_loss.append(temp_k_validation_loss)
        validation_accuracy.append(temp_k_validation_accuracy)
        test_loss.append(temp_k_test_loss)
        test_accuracy.append(temp_k_test_accuracy)

    length = iterations // step
    x = np.array(range(1, length * step, 100))

    # Training Loss
    for i, e in enumerate(train_loss):
        _, _, bars = plt.errorbar(x, np.mean(e, axis=0), xerr=None,
                               yerr=np.std(e, axis=0), label='K=' + str(i))
        [bar.set_alpha(0.3) for bar in bars]
    plt.xlabel("Training Iterations")
    plt.ylabel("Training SoftMax Loss")
    plt.title("Watch Average Training Loss Curve")
    plt.legend()
    plt.savefig("./phase" + str(phase) + "_result/" + flag.upper() + "_Training_Loss.png")
    plt.show()

    # Training Accuracy
    for i, e in enumerate(train_accuracy):
        _, _, bars = plt.errorbar(x, np.mean(e, axis=0), xerr=None,
                               yerr=np.std(e, axis=0), label='K=' + str(i))
        [bar.set_alpha(0.3) for bar in bars]
    plt.xlabel("Training Iterations")
    plt.ylabel("Training Accuracy")
    plt.title("Watch Average Training Accuracy Curve")
    plt.legend()
    plt.savefig("./phase" + str(phase) + "_result/" + flag.upper() + "_Training_Accuracy.png")
    plt.show()

    # Validation Loss
    for i, e in enumerate(validation_loss):
        _, _, bars = plt.errorbar(x, np.mean(e, axis=0), xerr=None,
                               yerr=np.std(e, axis=0), label='K=' + str(i))
        [bar.set_alpha(0.3) for bar in bars]
    plt.xlabel("Training Iterations")
    plt.ylabel("Validation SoftMax Loss")
    plt.title("Watch Average Validation Loss Curve")  # 三个run的average和std
    plt.legend()
    plt.savefig("./phase" + str(phase) + "_result/" + flag.upper() + "_Validation_Loss.png")
    plt.show()

    # Validation Accuracy
    for i, e in enumerate(validation_accuracy):
        _, _, bars = plt.errorbar(x, np.mean(e, axis=0), xerr=None,
                               yerr=np.std(e, axis=0), label='K=' + str(i))
        [bar.set_alpha(0.3) for bar in bars]
    plt.xlabel("Training Iterations")
    plt.ylabel("Validation Accuracy")
    plt.title("Watch Average Validation Accuracy Curve")
    plt.legend()
    plt.savefig("./phase" + str(phase) + "_result/" + flag.upper() + "_Validation_Accuracy.png")
    plt.show()

    # Test Accuracy
    for i, e in enumerate(test_accuracy):
        plt.bar(x=i, height=np.mean(e), xerr=None, yerr=np.std(e), label="K=" + str(i))
    plt.xlabel("Value of K")
    plt.ylabel("Test Accuracy")
    plt.title("Watch Average Test Accuracy")  # 三个run的average和std
    plt.legend()
    plt.savefig("./phase" + str(phase) + "_result/" + flag.upper() + "_Test_Accuracy.png")
    plt.show()

    # Test Loss
    for i, e in enumerate(test_loss):
        plt.bar(x=i, height=np.mean(e), xerr=None, yerr=np.std(e), label="K=" + str(i))
    plt.xlabel("Value of K")
    plt.ylabel("Test SoftMax Loss")
    plt.title("Watch Average Test SoftMax Loss")
    plt.legend()
    plt.savefig("./phase" + str(phase) + "_result/" + flag.upper() + "_Test_Loss.png")
    plt.show()




if __name__ == '__main__':
    # phone()
    # watch()
    analyze(phase=5, num_runs=10, k=10)
