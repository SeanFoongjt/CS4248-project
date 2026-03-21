import torch
import spacy
from transformers import AutoTokenizer, AutoModel

nlp = spacy.load("en_core_web_md")

class TextEmbedder:
    def __init__(self, method='bert'):
        self.method = method
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.method == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model = AutoModel.from_pretrained("bert-base-uncased").to(self.device)
            self.dim = 768
        elif self.method == 'spacy_glove':
            self.dim = 300
        else:
            self.dim = 128 

    def embed_nodes(self, concepts):
        if not concepts:
            return torch.empty((0, self.dim))

        if self.method == 'bert':
            inputs = self.tokenizer(concepts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, 0, :].cpu()
        elif self.method == 'spacy_glove':
            embeddings = [nlp(c).vector for c in concepts]
            return torch.tensor(embeddings, dtype=torch.float)
        else:
            return torch.randn((len(concepts), self.dim), dtype=torch.float)