import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import data_preprocessing as d


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        '''
        self.conv1 = nn.Conv2d(1, 64, (2, 2))
        self.dropout1 = nn.Dropout(p=0.1)
        self.conv2 = nn.Conv2d(64, 128, (2, 2))
        self.dropout2 = nn.Dropout(p=0.2)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(128, 256)
        self.dropout3 = nn.Dropout(p=0.5)
        self.dense2 = nn.Linear(256, 18)
        '''
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 2), # [32, 1, 80, 6] -> [32, 64, 79, 5]
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(64, 128, 2), # [32, 64, 79, 5] -> [32, 128, 78, 4]
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(39936, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 18),
            nn.Softmax()

        )

    def forward(self, x):
        '''
        x = x.view(-1, 1, 6, 80)
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        x = self.dropout3(x)
        output = F.softmax(self.dense2(x))
        return output
        '''
        # x.to(torch.float64)
        x = torch.tensor(x)
        x = self.features(x)
        x = x.view(-1, 39936)
        output = self.classifier(x)
        return output


def train(data):
    model = CNN()
    model = model.double()

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    if data == "phone":
        x_train, x_test, y_train, y_test = d.get_phone_data()
    else:
        x_train, x_test, y_train, y_test = d.get_watch_data()
    train_data = []
    for i in range(len(x_train)):
        train_data.append([torch.tensor(x_train[i]), torch.tensor(y_train[i])])
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=32)
    y_test = torch.tensor(y_test)

    # 开始训练
    for epoch in range(15):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            # print(type(batch_x))
            output = model(batch_x)  # batch_x=[32,1,80,6]
            loss = loss_function(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # accuracy
            if step % 100 == 0:
                test_output = model(x_test)
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                # accuracy = ((pred_y == y_test.data).sum()) / float(y_test.size(0))
                accuracy = ((pred_y == y_test.data.numpy()).astype(int).sum()) / float(y_test.size(0))
                print('Epoch: ', epoch, 'Step: ', step, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)


if __name__ == "__main__":
    train(data='phone')
