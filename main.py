import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict, Counter



class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(WordEmbedding, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
        
        
    def forward(self, x):
        embeds = self.embedding(x)
        
        out = self.output_layer(embeds)
        
        return out

class EmbeddingTrainer:
    def __init__(self, text, embedding_dim=100, window_size=2):
        
        self.window_size = window_size
        
        words = text.split()
        word_counts = Counter(words)
        
        self.vocab = {}
        for idx, (word, count) in enumerate(word_counts.items()):
            self.vocab[word] = idx
            
        self.vocab_size = len(self.vocab)
        
        self.training_pairs = self._create_training_pairs(words)
        
        #initialize model
        self.model = WordEmbedding(self.vocab_size, embedding_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def _create_training_pairs(self, words):
        training_pairs = []
        
        for current_position, word in enumerate(words):
            
            start = max(0, current_position - self.window_size)
            end = min(len(words), current_position + self.window_size)
            
            center_word_id = self.vocab[word]
            
            for context_position, current_position in enumerate(range(start, end)):
                #context_position = from(start, end)
                
                if context_position == current_position:
                    continue
                
            
                context_word = words[context_position]
                #context_word from words
                
                
                context_word_id = self.vocab[context_word]
                
                
                training_pairs.append((center_word_id, context_word_id))
            
        return training_pairs
    
    
    def train(self, num_epochs=5):
        for epoch in range(num_epochs):
            total_loss = 0
            
            for center_word_id, context_word_id in self.training_pairs:
                center_word_id = torch.tensor([center_word_id])
                context_word_id = torch.tensor([context_word_id])
                
                self.optimizer.zero_grad()
                output = self.model(center_word_id)
                
                loss = self.criterion(output, context_word_id)
                total_loss += loss.item()
                
                loss.backward()
                self.optimizer.step()
            
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(self.training_pairs):.4f}")
            
    
    def get_embedding(self, word):
        if word in self.vocab:
            idx = torch.tensor([self.vocab[word]])
            return self.model.embedding(idx).detach().numpy()
        return None
    
    def find_similar_words(self, word, n=5):
        
        word_embedding = self.get_embedding(word)
        
        similarities = []
        
        for word in self.vocab:
            vocab_embedding = self.get_embedding(word)[0]
            similarity = np.dot(word_embedding, vocab_embedding) / (
                np.linalg.norm(word_embedding) * np.linalg.norm(vocab_embedding)
            )
            similarities.append((word, similarity))
            
        return similarities



sample_text = """
The quick brown fox jumps over the lazy dog.
The fox is quick and brown.
The dog is lazy and sleeps.
"""
#not gonna get an ideal result because small data lol

trainer = EmbeddingTrainer(sample_text, embedding_dim=50, window_size=2)
trainer.train(num_epochs=100)

print("\nWords similar to 'fox':")
similar_words = trainer.find_similar_words('fox')
for word, similarity in similar_words:
    print(f"{word}: {float(similarity):.4f}")
    

            
        
    
    
    
    
    
    
    
    
                
                            
             
        
        
        
        
            
            
        
            
        