import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer

# Disable parallelism in tokenizers to prevent conflicts
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Embedding Model Wrapper using HuggingFace models
class EmbeddingModelWrapper:
    DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"

    def __init__(self, model_path=None):
        self.model_path = model_path or self.DEFAULT_MODEL
        self.model = AutoModel.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model.eval()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def emb_mean_pooling(self, model_output, attention_mask):
        # Perform mean pooling on the token embeddings
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def get_embeddings(self, sentences):
        # Get embeddings for a list of sentences
        encoded_input = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = self.emb_mean_pooling(model_output, encoded_input['attention_mask'])
        return embeddings

    def get_similarity(self, embedding1, embedding2):
        # Compute cosine similarity between two embeddings
        return self.cos(embedding1.unsqueeze(0), embedding2.unsqueeze(0)).item()

# BERT Syntax Similarity Calculator
class BERTSyntaxSimilarityCalculator:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def get_syntax_embeddings(self, sentence):
        # Get syntax embeddings using BERT
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

    def calculate_syntax_similarity(self, sentence1, sentence2):
        emb1 = self.get_syntax_embeddings(sentence1)
        emb2 = self.get_syntax_embeddings(sentence2)
        return self.cos(emb1, emb2).item()

# SBERT Syntax Similarity Calculator
class SBERTSyntaxSimilarityCalculator:
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def get_embeddings(self, sentence):
        # Get embeddings using SBERT
        embedding = self.model.encode(sentence, convert_to_tensor=True)
        return embedding

    def calculate_similarity(self, sentence1, sentence2):
        emb1 = self.get_embeddings(sentence1).unsqueeze(0)
        emb2 = self.get_embeddings(sentence2).unsqueeze(0)
        return self.cos(emb1, emb2).item()

def calculate_semantic_similarity(sentence1, sentence2, model_name='sentence-transformers/all-mpnet-base-v2'):
    model = SentenceTransformer(model_name)
    emb1 = model.encode(sentence1, convert_to_tensor=True)
    emb2 = model.encode(sentence2, convert_to_tensor=True)
    # Compute cosine similarity
    cosine_similarity = F.cosine_similarity(emb1, emb2, dim=0).item()
    # Scale between 0 and 1
    scaled_similarity = (cosine_similarity + 1) / 2
    return scaled_similarity


if __name__ == "__main__":
    # Define two sentences
    sentence1 = "I hate cats."
    sentence2 = "I love cats."

    print("Sentence 1:", sentence1)
    print("Sentence 2:", sentence2)
    print("")

    # Using EmbeddingModelWrapper
    embedding_model = EmbeddingModelWrapper()
    embeddings = embedding_model.get_embeddings([sentence1, sentence2])
    similarity_score = embedding_model.get_similarity(embeddings[0], embeddings[1])
    print("SemScore Similarity Score:", similarity_score)

    # Using BERT Syntax Similarity Calculator
    bert_syntax_calculator = BERTSyntaxSimilarityCalculator()
    bert_syntax_similarity = bert_syntax_calculator.calculate_syntax_similarity(sentence1, sentence2)
    print("BERT Syntax Similarity Score:", bert_syntax_similarity)

    # Using SBERT Syntax Similarity Calculator
    sbert_syntax_calculator = SBERTSyntaxSimilarityCalculator()
    sbert_syntax_similarity = sbert_syntax_calculator.calculate_similarity(sentence1, sentence2)
    print("SBERT Syntax Similarity Score:", sbert_syntax_similarity)

    # Using calculate_semantic_similarity function
    semantic_similarity = calculate_semantic_similarity(sentence1, sentence2)
    print("Cosine Similarity Score:", semantic_similarity)
