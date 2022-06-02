from sklearn.decomposition import PCA
import data_preprocessing as dp
import matplotlib.pyplot as plt

def pca_analyze():
    pca = PCA(n_components=2)
    x, y = dp.get_user(user_id=6)
    x = x.reshape(x.shape[0], -1)
    pca.fit(x)
    x = pca.transform(x)



    return x


if __name__ == '__main__':
    print(pca_analyze())

