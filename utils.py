# utils.py
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english') + stopwords.words('swedish'))

def load_data(csv_file):
    df = pd.read_csv(csv_file)
    # Drop rows with missing essential fields
    df = df.dropna(subset=['Title', 'Link', 'Organization'])
    
    # Handle 'Publish Date' if present
    if 'Publish Date (ISO)' in df.columns:
        df['Publish Date'] = pd.to_datetime(df['Publish Date (ISO)'], errors='coerce')
    else:
        df['Publish Date'] = pd.NaT  # Not a Time (NaT) for missing dates
    
    # Handle 'Deadline for Application'
    if 'Deadline for Application (ISO)' in df.columns:
        df['Deadline for Application'] = pd.to_datetime(df['Deadline for Application (ISO)'], errors='coerce')
    elif 'Application Deadline (ISO)' in df.columns:
        df['Deadline for Application'] = pd.to_datetime(df['Application Deadline (ISO)'], errors='coerce')
    else:
        df['Deadline for Application'] = pd.NaT  # Handle missing deadlines
    
    # Drop rows with invalid or missing deadlines
    df = df.dropna(subset=['Deadline for Application'])
    
    # Extract year and month from deadlines
    df['Deadline Year'] = df['Deadline for Application'].dt.year
    df['Deadline Month'] = df['Deadline for Application'].dt.month
    
    # Define a function to clean the Title
    def clean_title(title):
        # Remove specific substrings
        patterns_to_remove = [
            r"master's thesis", 
            r"30hp", 
            r"30 hp", 
            r"15 hp",
            r"project",
            r"master",
            r"thesis",
            r"examensarbete",
            r"credit"
            
        ]
        for pattern in patterns_to_remove:
            title = re.sub(pattern, '', title, flags=re.IGNORECASE)
        
        # Remove colon signs
        title = title.replace(':', '')
        
        # Remove parentheses that contain only numbers, e.g., (123)
        title = re.sub(r'\(\d+\)', '', title)
        
        # Remove any extra whitespace
        title = re.sub(r'\s+', ' ', title).strip()
        
        return title
    
    # Apply the cleaning function to the Title column
    df['Title'] = df['Title'].apply(clean_title)
    
    return df

def generate_embeddings_and_clusters(data, model_name='all-MiniLM-L6-v2', num_clusters=5):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(data['Title'].tolist(), show_progress_bar=True)
    
    # PCA for dimensionality reduction
    pca = PCA(n_components=50)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(reduced_embeddings)
    
    data = data.copy()
    data['Cluster'] = clusters
    return data, embeddings, clusters, kmeans
