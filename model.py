from torch.nn import Module
from torch.nn import LSTM, Linear, Embedding, Sequential, Flatten
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, Optimizer
from torch.nn import Softmax, Tanh, ReLU
from torch.utils.data import DataLoader
import torch



class CinderellaModel(Module):
    def __init__(self, vocabSize:int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.embedding = Embedding(vocabSize, 128)

        self.lstm1 = LSTM(128, 256, 2, batch_first=True, dropout=0.3)

        self.output = Linear(256, vocabSize)
        # specifying layers

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def forward(self, x):
        x  = self.embedding.forward(x)

        x, _ = self.lstm1.forward(x)

        x = x[:, -1, :]

        x = self.output.forward(x)

        return x
    
    def initialize(self, optimizer:Optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss

    def trainStep(self, x, y):
        self.optimizer.zero_grad()
        # clearing the gradiets 

        logits = self.forward(x)
        # getting predictions 

        loss = self.loss(logits, y)
        loss.backward()
        # gradient descent

        self.optimizer.step()
        # applying gradients

        return loss.item()
    
    def fit(self, dataloader:DataLoader, epochs:int):
        self.to(self.device)
        
        for epoch in range(epochs):
            epochLoss = 0
            for i, (x, y) in enumerate(dataloader):
                loss = self.trainStep(x.to(self.device), y.to(self.device))
                epochLoss += loss

                print(f"[{epoch+1}/{epochs} Batch:{i+1}/{len(dataloader)}] Loss: {epochLoss/(i+1)}", end="\r")

            print(f"[{epoch+1}/{epochs}] Loss: {epochLoss/len(dataloader)}")

        print("Finished Training!")

    def predit(self, x):
        with torch.no_grad():
            logits = self.forward(x.to(self.device))
            return Softmax(-1).forward(logits)
    

if __name__ == '__main__':
    from torch import randint
    from torch import int32
    import torch

    from loadData import getDataLoader

    import uuid

    loader = getDataLoader("data.txt", 128, "tokenizer.json", 64)
    model = CinderellaModel(1200)
    model.initialize(Adam(model.parameters()), CrossEntropyLoss())
    

    EPOCHS = 100
    MODEL_NAME = "model100.pth"
    try:
        model.fit(loader, EPOCHS)
        torch.save(model.state_dict(), MODEL_NAME)
    except KeyboardInterrupt:
        print("saving model...")
        torch.save(model.state_dict(), f"model{str(uuid.uuid1())}.pth")

    # for x, y in loader:
    #     print(model.forward(x).shape)
    #     break
    