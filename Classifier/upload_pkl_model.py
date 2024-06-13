import pandas as pd
import spacy
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from textstat import textstat
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import nltk
nltk.download('punkt')

sampled_df = pd.read_csv('.\sampled_travel_with_f.csv')
import torch
import torch.nn as nn
import torch.optim as optim


class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=3, num_encoder_layers=3, dim_feedforward=128, dropout=0.5):
        super(TransformerRegressor, self).__init__()

        self.linear_in = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.linear_out = nn.Linear(d_model, 5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = self.linear_out(x[:, 0, :])
        return x


def training_and_saving():
    # Extract features and labels
    features = []
    for feature in tqdm(sampled_df['features'], desc="Processing features"):
        features.append(eval(feature))  # Convert string representation of lists to actual lists

    features_df = pd.DataFrame(features)

    # Prepare the labels
    labels = sampled_df['Rating'] - 1  # Shift labels to be 0-4 instead of 1-5 for zero-indexing

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_df, labels, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Instantiate the Transformer model
    input_dim = X_train_tensor.shape[1]
    model = TransformerRegressor(input_dim=input_dim)

    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training the model
    num_epochs = 20
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # Evaluate the model
    model.eval()
    test_loss = 0.0
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.unsqueeze(1))
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            predictions.extend(torch.argmax(outputs, dim=1).tolist())
            actuals.extend(labels.tolist())

    # Calculate accuracy
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    accuracy = np.mean(predictions == actuals)  # Calculate accuracy as the percentage of correct predictions

    print(f"Test Loss: {test_loss / len(test_loader):.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    # Example review code
    example_index = 0  # Change this index to test different reviews
    example_features = X_test_tensor[example_index].unsqueeze(0).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        example_output = model(example_features)
        example_prediction = torch.argmax(example_output).item() + 1  # Convert back to original scale 1-5
        example_actual = y_test_tensor[example_index].item() + 1  # Convert back to original scale 1-5
        print(f"\nExample review features: {X_test.iloc[example_index].tolist()}")
        print(f"Predicted rating: {example_prediction}, Actual rating: {example_actual}")


    with open('transformer_model7000.pkl', 'wb') as file:
        pickle.dump(model, file)


def extract_features(review):
    # Lexical features
    word_count = len(word_tokenize(review))
    avg_word_length = np.mean([len(word) for word in word_tokenize(review)])
    unique_word_count = len(set(word_tokenize(review)))
    char_count = len(review)

    # Syntactic features
    sentence_count = len(sent_tokenize(review))
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    punctuation_count = sum([1 for char in review if char in string.punctuation])

    doc = nlp(review)
    pos_counts = doc.count_by(spacy.attrs.POS)
    pos_distribution = [pos_counts.get(i, 0) / word_count for i in range(1, 18)]  # 1 to 17 are POS tag IDs

    # Semantic features
    tfidf_scores = tfidf_vectorizer.transform([review]).toarray().flatten()

    # Use Universal Sentence Encoder for sentence embeddings
    use_embedding = use_embed([review]).numpy().flatten()

    # Use Sentence-BERT for sentence embeddings
    sbert_embedding = sbert_model.encode([review]).flatten()

    ner_counts = len(doc.ents)

    # Sentiment features
    sentiment = TextBlob(review).sentiment
    sentiment_polarity = sentiment.polarity
    sentiment_subjectivity = sentiment.subjectivity

    # Readability features
    flesch_reading_ease = textstat.flesch_reading_ease(review)
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(review)

    # Combine all features into a single feature vector
    features = [
                   word_count, avg_word_length, unique_word_count, char_count,
                   sentence_count, avg_sentence_length, punctuation_count, ner_counts,
                   sentiment_polarity, sentiment_subjectivity, flesch_reading_ease, flesch_kincaid_grade
               ] + pos_distribution + tfidf_scores.tolist() + use_embedding.tolist() + sbert_embedding.tolist()

    return features

def load_model():
    with open('transformer_model7000.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def predict_rating(review):
    model = load_model()
    nlp = spacy.load('en_core_web_sm')

    # Load Universal Sentence Encoder and Sentence-BERT
    use_embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Initialize TF-IDF vectorizer (fitting on all reviews for consistency)
    tfidf_vectorizer = TfidfVectorizer(max_features=13)
    data = pd.read_csv('sampled_travel_with_f.csv')
    tfidf_vectorizer.fit(data['Review'])
    features = extract_features(review)
    features_df = pd.DataFrame([features])
    features_tensor = torch.tensor(features_df.values, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        output = model(features_tensor.unsqueeze(1))
        prediction = torch.argmax(output).item() + 1  # Convert back to original scale 1-5
    return prediction