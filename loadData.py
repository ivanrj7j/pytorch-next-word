from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from torch import tensor
from torch import int32

class Cinderella(Dataset):
    def __init__(self, filePath:str, sequenceLength:int, tokenizerPath:str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        with open(filePath, "r") as f:
            self.content = f.read()

        self.tokenizer = self.loadPretrainedTokenizer(tokenizerPath)
        self.tokens = self.tokenize(self.content.split('\n'))
        self.sequenceLength = sequenceLength

    @classmethod
    def initializeWithTokenizer(self, filePath:str, sequenceLength:int, *args, **kwargs):
        tokenizer = Tokenizer(BPE())
        trainer = BpeTrainer(vocab_size=1200, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        tokenizer.train_from_iterator([self.content], trainer)
        return self(filePath, sequenceLength, tokenizer, *args, **kwargs)

    def loadPretrainedTokenizer(self, tokenizerPath:str) -> Tokenizer:
        return Tokenizer.from_file(tokenizerPath)
        
    def tokenize(self, batch:list[str]) -> list:
        encodings =  self.tokenizer.encode_batch(batch)
        tokens = []

        for encoding in encodings:
            tokens.extend(encoding.ids)

        return tokens
    
    def decode(self, sequence) -> str:
        return self.tokenizer.decode(sequence).replace(" ##", "")

    def __len__(self):
        return len(self.tokens)-1
    
    def __getitem__(self, index):
        if index == len(self.tokens):
            raise IndexError("Index out of range")

        startIndex = max(0, index - self.sequenceLength)
        tokens = self.tokens[startIndex:index]

        if len(tokens) < self.sequenceLength:
            x = [0 for _ in range(self.sequenceLength - len(tokens))]
            tokens = x+tokens

        return tensor(tokens), tensor(self.tokens[index])
    

def getDataLoader(filePath:str, sequenceLength:int, tokenizerPath:str, batchSize:int, *args, **kwargs) -> DataLoader:
    dataset = Cinderella(filePath, sequenceLength, tokenizerPath, *args, **kwargs)
    return DataLoader(dataset, batch_size=batchSize, shuffle=True)


if __name__ == "__main__":
    # dataset = Cinderella("data.txt", 128, "tokenizer.json")

    # dataset.initializeTokenizer()


    # tokenized = dataset.tokenize("Hello world")
    # print(tokenized)
    # print(dataset.decode(tokenized))

    # print(dataset.content)
    # data = dataset.tokenizer.encode(dataset.content).ids
    # print(dataset.tokens)
    # print(len(dataset))

    # print(dataset[56])
    # print(dataset.tokens[-1])

    dataloader = getDataLoader("data.txt", 128, "tokenizer.json", 64)
    for x, y in dataloader:
        print(y)
        