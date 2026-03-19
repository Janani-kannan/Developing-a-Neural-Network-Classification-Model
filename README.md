# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="941" height="946" alt="image" src="https://github.com/user-attachments/assets/2203389b-7745-407e-a979-f7978abcce38" />

## DESIGN STEPS
### STEP 1: 

Load the dataset, remove irrelevant columns (ID), handle missing values, encode categorical features using Label Encoding,
and encode the target class (Segmentation).

### STEP 2: 

Split the dataset into training and testing sets, then normalize the input features using 
StandardScaler for better neural network performance.

### STEP 3: 

Convert the scaled training and testing data into PyTorch tensors and create
DataLoader objects for batch-wise training and evaluation.

### STEP 4: 

Design a feedforward neural network with multiple fully connected layers and ReLU activation functions, 
ending with an output layer for multi-class classification.

### STEP 5: 

Train the model using CrossEntropyLoss and Adam optimizer by performing forward propagation, 
loss calculation, backpropagation, and weight updates over multiple epochs.


### STEP 6: 


Evaluate the trained model on test data using accuracy, confusion matrix,
and classification report, and perform prediction on a sample input.


## PROGRAM

### Name: JANANI K

### Register Number: 212224230102

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 =nn.Linear(input_size,32)
        self.fc2=nn.Linear(32,16)
        self.fc3=nn.Linear(16,8)
        self.fc4=nn.Linear(8,4)





    def forward(self, x):
      x=F.relu(self.fc1(x))
      x=F.relu(self.fc2(x))
      x=F.relu(self.fc3(x))
      x=self.fc4(x)
      return x

        
# Initialize the Model, Loss Function, and Optimizer
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()


    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')'

# Initialize model

model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model,train_loader,criterion,optimizer,epochs=100)

```

### Dataset Information

<img width="1244" height="258" alt="image" src="https://github.com/user-attachments/assets/6a5fe7b3-5858-4c80-973c-3f80e8e7d49e" />


### OUTPUT

<img width="1177" height="384" alt="image" src="https://github.com/user-attachments/assets/f56b7a5f-2320-459a-b448-aca95faaefeb" />


## Confusion Matrix

<img width="683" height="573" alt="image" src="https://github.com/user-attachments/assets/bddde1ec-11dd-41b8-8c1c-6dfa6cbde27e" />


## Classification Report

<img width="1038" height="425" alt="image" src="https://github.com/user-attachments/assets/a9078620-54bb-4970-840c-231257ad6701" />

### New Sample Data Prediction

<img width="888" height="108" alt="image" src="https://github.com/user-attachments/assets/951bbbf1-ff1d-49b5-8129-3974ba22fa00" />

## RESULT

This program has been executed successfully.
