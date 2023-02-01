import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
import os


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 7)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    

# Define the CNN architecture
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define the transform to pre-process the images
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


#read data and prepare for the model
def get_data():
    images_dir = r"C:\BeCode\computervisionData\HAM10000_skin_mnist\HAM10000_images"
    df_original = pd.read_csv(r"C:\BeCode\computervisionData\HAM10000_skin_mnist\HAM10000_metadata.csv")
    df = df_original.sample(n=50)

    # Preprocess the image data
    images = []
    labels = []
    for i in range(len(df)):
        # Load the image
        img_path = os.path.join(images_dir, df.iloc[i]['image_id'] + '.jpg')
        img_path = os.path.join(images_dir, df.iloc[i]['image_id'] + '.jpg')
        img = Image.open(img_path).convert('RGB')

        img = transform(img)
        images.append(img)
        labels.append(df.iloc[i]['dx'])


    # Create the train and test datasets
    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

    train_set = tuple(zip(train_images, train_labels))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    test_set = tuple(zip(val_images, val_labels))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    return train_loader, test_loader



#Initialize and train model
def initNtrain_model(train_loader):
    # Initialize the model
    model = Net1()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))

    print('Finished Training')
    return model

# Evaluate the model on the test set
def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test set: %d %%' % (100 * correct / total))


def main():
    traind, testd = get_data()
    model = initNtrain_model(train_loader=traind)
    evaluate_model(model, testd)
    # Save the model
    torch.save(model.state_dict(), 'skin_mole_classifier.pth')


if __name__ == '__main__':
    main()