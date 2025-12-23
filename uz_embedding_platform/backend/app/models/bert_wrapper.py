from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np


class BERTEmbedding:
    """BERT embedding model with multiple pooling strategies"""
    
    def __init__(self, model_name="bert-base-multilingual-cased", pooling_strategy="cls"):
        """
        Initialize BERT model
        
        Args:
            model_name: HuggingFace model name
            pooling_strategy: 'cls', 'mean', 'max', or 'weighted'
        """
        print(f"Loading transformer model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.pooling_strategy = pooling_strategy
        self.model.eval()
    
    def get_sentence_vector(self, text, pooling_strategy=None):
        """
        Get sentence embedding with specified pooling strategy
        
        Args:
            text: Input text
            pooling_strategy: Override default strategy if specified
        
        Returns:
            Embedding vector
        """
        if pooling_strategy is None:
            pooling_strategy = self.pooling_strategy
        
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.model(**tokens)
        
        # Get the appropriate vector based on pooling strategy
        if pooling_strategy == "cls":
            vector = self._cls_pooling(outputs)
        elif pooling_strategy == "mean":
            vector = self._mean_pooling(outputs, tokens)
        elif pooling_strategy == "max":
            vector = self._max_pooling(outputs)
        elif pooling_strategy == "weighted":
            vector = self._weighted_pooling(outputs)
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
        
        return vector.tolist() if isinstance(vector, np.ndarray) else vector.squeeze().tolist()
    
    def _cls_pooling(self, outputs):
        """CLS token pooling (standard BERT)"""
        return outputs.last_hidden_state[:, 0, :].squeeze()
    
    def _mean_pooling(self, outputs, tokens):
        """Mean pooling of all tokens"""
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = tokens['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return (sum_embeddings / sum_mask).squeeze()
    
    def _max_pooling(self, outputs):
        """Max pooling over tokens"""
        token_embeddings = outputs.last_hidden_state
        return torch.max(token_embeddings, dim=1)[0].squeeze()
    
    def _weighted_pooling(self, outputs):
        """Weighted pooling of last 4 layers"""
        # Get last 4 hidden states
        hidden_states = outputs.hidden_states[-4:]  # Last 4 layers
        
        # Simple weighted average (linear weights)
        weights = torch.tensor([0.25, 0.25, 0.25, 0.25])
        weighted_output = torch.zeros_like(hidden_states[0])
        
        for i, hidden_state in enumerate(hidden_states):
            weighted_output += weights[i] * hidden_state
        
        # Take mean over tokens
        return torch.mean(weighted_output, dim=1).squeeze()
    
    def get_all_pooling_vectors(self, text):
        """Get vectors from all pooling strategies"""
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.model(**tokens)
        
        vectors = {
            "cls": self._cls_pooling(outputs),
            "mean": self._mean_pooling(outputs, tokens),
            "max": self._max_pooling(outputs),
            "weighted": self._weighted_pooling(outputs)
        }
        
        return {k: v.squeeze().tolist() for k, v in vectors.items()}
