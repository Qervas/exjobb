# utils_cth.py
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from fuzzywuzzy import process

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english') + stopwords.words('swedish'))

def generate_abbreviation(term, existing_abbrv):
    """
    Generates a unique abbreviation for a given term.
    
    Parameters:
    - term: str, the term to abbreviate.
    - existing_abbrv: set, existing abbreviations to avoid duplicates.
    
    Returns:
    - abbrv: str, the unique abbreviation.
    """
    words = term.split()
    abbrv = ''.join(word[0].upper() for word in words)
    original_abbrv = abbrv
    count = 1
    while abbrv in existing_abbrv:
        abbrv = f"{original_abbrv}{count}"
        count += 1
    existing_abbrv.add(abbrv)
    return abbrv

def create_dynamic_mapping(df, column_name):
    """
    Creates a dynamic mapping dictionary for a given column.
    
    Parameters:
    - df: pandas DataFrame.
    - column_name: str, the column to create mappings for.
    
    Returns:
    - mapping_dict: dict, mapping from full term to abbreviation.
    """
    unique_terms = set()
    for entry in df[column_name].dropna():
        # Remove trailing commas and then split
        cleaned_entry = entry.rstrip(',')
        terms = [term.strip() for term in cleaned_entry.split(',')]
        unique_terms.update(terms)
    
    mapping_dict = {}
    existing_abbrv = set()
    for term in sorted(unique_terms):
        abbrv = generate_abbreviation(term, existing_abbrv)
        mapping_dict[term] = abbrv
    return mapping_dict

def map_terms_dynamic(terms, mapping_dict, threshold=80):
    """
    Maps a list of terms to their corresponding abbreviations using fuzzy matching.
    
    Parameters:
    - terms: list of str, the terms to map.
    - mapping_dict: dict, mapping from full term to abbreviation.
    - threshold: int, minimum score for a match to be considered valid.
    
    Returns:
    - list of str, mapped terms.
    """
    mapped_terms = []
    for term in terms:
        term_clean = term.strip().lower()
        if not term_clean:
            continue
        # Perform fuzzy matching
        match, score = process.extractOne(term_clean, [k.lower() for k in mapping_dict.keys()])
        if score >= threshold:
            # Retrieve the original key to get the correct mapping
            original_key = next(k for k in mapping_dict.keys() if k.lower() == match)
            mapped_terms.append(mapping_dict[original_key])
        else:
            # If no good match found, keep the original term (title-cased)
            mapped_terms.append(term.strip())
    return mapped_terms

def load_data_cth(csv_file):
    """
    Loads and preprocesses Chalmers Exjobb project data from a CSV file.
    
    Parameters:
    - csv_file: str, path to the CSV file.
    
    Returns:
    - df: pandas DataFrame with cleaned and preprocessed data.
    - edu_mapping: dict, mapping for Educational Area.
    - subj_mapping: dict, mapping for Subject Area.
    """
    try:
        # Read CSV with appropriate handling for bad lines
        df = pd.read_csv(csv_file, on_bad_lines='skip')  # Skips malformed rows

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
        elif 'Apply By' in df.columns:
            df['Deadline for Application'] = pd.to_datetime(df['Apply By'], errors='coerce')
        else:
            df['Deadline for Application'] = pd.NaT  # Handle missing deadlines

        # Drop rows with invalid or missing deadlines
        df = df.dropna(subset=['Deadline for Application'])

        # Extract year and month from deadlines
        df['Deadline Year'] = df['Deadline for Application'].dt.year
        df['Deadline Month'] = df['Deadline for Application'].dt.month

        # Define a function to clean the Title
        def clean_title(title):
            """
            Cleans the project title by removing specific substrings and unwanted characters.

            Parameters:
            - title: str, original project title.

            Returns:
            - cleaned_title: str, cleaned project title.
            """
            # Remove specific phrases only if they appear at the start or end
            phrases_to_remove = [
                r"^(Master's thesis\s*[:\-]?)",  # At the beginning
                r"([:\-]\s*Master's thesis)$",  # At the end
                r"\b30hp\b",
                r"\b30 hp\b",
                r"\b15 hp\b",
                r"^(Project\s*[:\-]?)",  # At the beginning
                r"([:\-]\s*Project)$",  # At the end
                r"^(Master\s*[:\-]?)",  # At the beginning
                r"([:\-]\s*Master)$",  # At the end
                r"^(Thesis\s*[:\-]?)",  # At the beginning
                r"([:\-]\s*Thesis)$",  # At the end
                r"^(Examensarbete\s*[:\-]?)",  # At the beginning
                r"([:\-]\s*Examensarbete)$",  # At the end
                r"^\bCredit\b"  # Exact match
            ]

            for pattern in phrases_to_remove:
                title = re.sub(pattern, '', title, flags=re.IGNORECASE).strip()

            # Remove colon signs
            title = title.replace(':', '')

            # Remove parentheses that contain only numbers, e.g., (123)
            title = re.sub(r'\(\d+\)', '', title)

            # Remove any extra whitespace
            title = re.sub(r'\s+', ' ', title).strip()

            return title

        # Apply the cleaning function to the Title column
        df['Title'] = df['Title'].apply(clean_title)

        # Clean trailing commas from specific columns
        for column in ['Subject Area', 'Educational Area', 'Location', 'Country', 'Organization']:
            if column in df.columns:
                df[column] = df[column].str.rstrip(',')

        # Create dynamic mapping dictionaries
        edu_mapping = create_dynamic_mapping(df, 'Educational Area') if 'Educational Area' in df.columns else {}
        subj_mapping = create_dynamic_mapping(df, 'Subject Area') if 'Subject Area' in df.columns else {}

        # Process 'Educational Area' and 'Subject Area' to map terms dynamically
        if 'Educational Area' in df.columns:
            df['Educational Area'] = df['Educational Area'].apply(
                lambda x: ', '.join(map_terms_dynamic(x.split(','), edu_mapping)) if pd.notna(x) else pd.NA
            )

        if 'Subject Area' in df.columns:
            df['Subject Area'] = df['Subject Area'].apply(
                lambda x: ', '.join(map_terms_dynamic(x.split(','), subj_mapping)) if pd.notna(x) else pd.NA
            )

        return df, edu_mapping, subj_mapping
    except Exception as e:
        raise e

def generate_embeddings_and_clusters_cth(data, model_name='all-MiniLM-L6-v2', num_clusters=5):
    """
    Generates embeddings from the 'Title' column and performs clustering.
    
    Parameters:
    - data: pandas DataFrame containing the project data.
    - model_name: str, name of the SentenceTransformer model.
    - num_clusters: int, number of clusters for KMeans.
    
    Returns:
    - data: pandas DataFrame with an added 'Cluster' column.
    - embeddings: numpy array of shape (n_samples, embedding_dim).
    - clusters: numpy array of cluster labels.
    - kmeans_model: trained KMeans model.
    """
    try:
        # Ensure 'Title' column exists and is not empty
        if 'Title' not in data.columns:
            raise ValueError("Missing 'Title' column in data.")

        # Remove any entries with empty 'Title' after cleaning
        initial_count = data.shape[0]
        data = data[data['Title'].str.strip().astype(bool)]
        cleaned_count = data.shape[0]

        if data.empty:
            raise ValueError("No data available after removing empty Titles.")

        # Initialize SentenceTransformer model
        model = SentenceTransformer(model_name)

        # Generate embeddings
        embeddings = model.encode(data['Title'].tolist(), show_progress_bar=True)

        # Ensure embeddings are not empty
        if embeddings.size == 0:
            raise ValueError("Embeddings generation resulted in an empty array.")

        # Ensure embeddings are 2D
        if len(embeddings.shape) != 2:
            raise ValueError(f"Embeddings have incorrect shape: {embeddings.shape}")

        # Perform PCA for dimensionality reduction
        pca = PCA(n_components=50, random_state=42)
        reduced_embeddings = pca.fit_transform(embeddings)

        # Ensure reduced_embeddings are 2D
        if len(reduced_embeddings.shape) != 2:
            raise ValueError(f"Reduced embeddings have incorrect shape: {reduced_embeddings.shape}")

        # Perform KMeans clustering
        if num_clusters > reduced_embeddings.shape[0]:
            raise ValueError(f"Number of clusters ({num_clusters}) cannot exceed number of samples ({reduced_embeddings.shape[0]}).")

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(reduced_embeddings)

        # Assign clusters to data
        data = data.copy()
        data['Cluster'] = clusters

        return data, embeddings, clusters, kmeans
    except Exception as e:
        raise e
