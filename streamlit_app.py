import streamlit as st
import pandas as pd
import plotly.express as px
import os
import base64
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import re  # Regular expression library
from math import ceil
import io  # For in-memory buffer

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english') + stopwords.words('swedish'))

# Set page configuration
st.set_page_config(page_title="Exjobb Project Analysis", layout="wide")

# Load and preprocess data
@st.cache_data
def load_data():
    # Load CSV file
    csv_file = 'data/exjobb_projects.csv'
    df = pd.read_csv(csv_file)
    # Drop missing values
    df = df.dropna(subset=['Title', 'Main Research Field', 'Application Deadline (Display)', 'Link'])
    # Convert date format
    df['Application Deadline'] = pd.to_datetime(df['Application Deadline (ISO)'], errors='coerce')
    # Drop rows with invalid dates
    df = df.dropna(subset=['Application Deadline'])
    # Extract year and month
    df['Deadline Year'] = df['Application Deadline'].dt.year
    df['Deadline Month'] = df['Application Deadline'].dt.month
    # Clean "master thesis" phrase from titles
    df['Title'] = df['Title'].apply(lambda x: re.sub(r'master thesis', '', x, flags=re.IGNORECASE).strip())
    return df

df = load_data()

# Generate embeddings for project titles and perform clustering
@st.cache_data
def generate_embeddings_and_clusters(data, model_name='all-MiniLM-L6-v2', num_clusters=5):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(data['Title'].tolist(), show_progress_bar=True)
    
    # Use PCA for dimensionality reduction before clustering
    pca = PCA(n_components=50)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Use KMeans for clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(reduced_embeddings)
    
    data = data.copy()
    data['Cluster'] = clusters
    return data, embeddings, clusters, kmeans

# Set default number of clusters
default_num_clusters = 5

# User-adjustable number of clusters
num_clusters = st.sidebar.slider("Select number of clusters", min_value=2, max_value=20, value=default_num_clusters, step=1)

with st.spinner("Generating embeddings and performing clustering..."):
    df, embeddings, clusters, kmeans_model = generate_embeddings_and_clusters(df, num_clusters=num_clusters)

# Page title
st.title("Linköping University Exjobb Project Data Analysis and Visualization")

# Sidebar filter options
st.sidebar.header("Filter Options")

# Search keyword
search_query = st.sidebar.text_input("Search project title keywords")

# Filter by main research field
fields = sorted(df['Main Research Field'].unique())
selected_fields = st.sidebar.multiselect("Main Research Field", options=fields, default=fields)

# Filter by organization
organizations = sorted(df['Organization'].unique())
selected_orgs = st.sidebar.multiselect("Organization", options=organizations, default=organizations)

# Filter by deadline year (multi-select)
years = sorted(df['Deadline Year'].unique())
selected_years = st.sidebar.multiselect("Deadline Year", options=years, default=years)

# Filter by deadline month (multi-select)
months = sorted(df['Deadline Month'].unique())
selected_months = st.sidebar.multiselect("Deadline Month", options=months, default=months)

# Filter by cluster
clusters_options = sorted(df['Cluster'].unique())
selected_clusters = st.sidebar.multiselect("Cluster", options=clusters_options, default=clusters_options)

# Sorting options
st.sidebar.header("Sorting Options")
sort_by = st.sidebar.selectbox("Sort by", options=['Organization', 'Main Research Field', 'Application Deadline'], index=0)
sort_order = st.sidebar.radio("Sort order", options=['Ascending', 'Descending'], index=1)

# Dynamic visualization options
st.sidebar.header("Dynamic Visualization Options")
visualization_options = st.sidebar.multiselect(
    "Select visualization charts",
    options=['Project Title Word Cloud', 'Cluster Scatter Plot', 'Project Count Trend', 'Project Title and Organization Relationship'],
    default=['Project Title Word Cloud', 'Cluster Scatter Plot', 'Project Count Trend', 'Project Title and Organization Relationship']
)

# Ensure selected years and months are not empty (if empty, interpret as selecting all)
if not selected_years:
    selected_years = years

if not selected_months:
    selected_months = months

if not selected_clusters:
    selected_clusters = clusters_options

# Filter data based on selected criteria
filtered_df = df[
    (df['Main Research Field'].isin(selected_fields)) &
    (df['Organization'].isin(selected_orgs)) &
    (df['Deadline Year'].isin(selected_years)) &
    (df['Deadline Month'].isin(selected_months)) &
    (df['Cluster'].isin(selected_clusters))
]

# Further filter based on search keyword
if search_query:
    filtered_df = filtered_df[filtered_df['Title'].str.contains(search_query, case=False, na=False)]

# Sort data based on sorting options
ascending = True if sort_order == 'Ascending' else False
filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)

# Pagination settings
items_per_page = 20
total_pages = ceil(filtered_df.shape[0] / items_per_page)

# User selects page number (move to the top of the main page)
st.markdown("### Pagination")
page = st.number_input("Select page number", min_value=1, max_value=total_pages, value=1, step=1)

# Calculate start and end indices
start_idx = (page - 1) * items_per_page
end_idx = start_idx + items_per_page
paginated_df = filtered_df.iloc[start_idx:end_idx]

# Display the number of filtered projects
st.markdown(f"### Filtered Results: {filtered_df.shape[0]} projects (showing {start_idx + 1} to {min(end_idx, filtered_df.shape[0])})")

# Display project data table
# To make links clickable, we need to use HTML format
def make_clickable(link):
    return f'<a href="{link}" target="_blank">View Details</a>'

paginated_df_display = paginated_df.copy()
paginated_df_display['Link'] = paginated_df_display['Link'].apply(make_clickable)

# Select columns to display
columns_to_display = ['Title', 'Organization', 'Main Research Field', 'Application Deadline (Display)', 'Link', 'Cluster']

# Display table with clickable links
st.markdown(paginated_df_display[columns_to_display].to_html(escape=False, index=False), unsafe_allow_html=True)

# Visualization section
st.markdown("## Data Visualization")

# Chart 1: Distribution of Main Research Fields (Bar Chart)
fig1 = px.bar(
    filtered_df,
    x='Main Research Field',
    title='Distribution of Main Research Fields',
    labels={'Main Research Field': 'Main Research Field', 'Main Research Field': 'Project Count'},
    height=400,
    color='Main Research Field',
    template='plotly_white'
)
st.plotly_chart(fig1, use_container_width=True, key='fig1')

# Chart 2: Distribution of Organizations (Pie Chart)
fig2 = px.pie(
    filtered_df,
    names='Organization',
    title='Distribution of Organizations',
    height=400,
    hole=0.3,
    template='plotly_white'
)
st.plotly_chart(fig2, use_container_width=True, key='fig2')

# Chart 3: Application Deadline Trend (Line Chart)
# Aggregate data
deadline_trend = filtered_df.groupby('Application Deadline').size().reset_index(name='Project Count')

# Ensure application deadlines are sorted
deadline_trend = deadline_trend.sort_values('Application Deadline')

# Create line chart
fig3 = px.line(
    deadline_trend,
    x='Application Deadline',
    y='Project Count',
    title='Application Deadline Trend',
    labels={'Application Deadline': 'Application Deadline', 'Project Count': 'Project Count'},
    markers=True,
    height=400,
    template='plotly_white'
)
st.plotly_chart(fig3, use_container_width=True, key='fig3')

# Chart 4: Distribution of Project Count by Year and Month (Heatmap)
heatmap_data = filtered_df.groupby(['Deadline Year', 'Deadline Month']).size().reset_index(name='Project Count')
heatmap_pivot = heatmap_data.pivot(index='Deadline Year', columns='Deadline Month', values='Project Count').fillna(0)

fig4 = px.imshow(
    heatmap_pivot,
    labels=dict(x="Deadline Month", y="Deadline Year", color="Project Count"),
    x=heatmap_pivot.columns,
    y=heatmap_pivot.index,
    title="Distribution of Project Count by Year and Month",
    aspect="auto",
    color_continuous_scale='Blues',
    template='plotly_white'
)
st.plotly_chart(fig4, use_container_width=True, key='fig4')

# Chart 5: Relationship between Main Research Field and Organization (Bubble Chart)
fig5 = px.scatter(
    filtered_df,
    x='Main Research Field',
    y='Organization',
    color='Main Research Field',
    hover_data=['Title', 'Application Deadline (Display)'],
    title='Relationship between Main Research Field and Organization',
    template='plotly_white',
    height=600
    # size parameter removed to avoid type issues
)
st.plotly_chart(fig5, use_container_width=True, key='fig5')

# Dynamic visualization charts

# Chart 6: Project Title Word Cloud
if 'Project Title Word Cloud' in visualization_options:
    st.markdown("### Project Title Word Cloud")
    text = ' '.join(filtered_df['Title'].dropna().tolist())
    wordcloud = WordCloud(stopwords=stop_words, background_color='white', width=800, height=400).generate(text)
    
    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc, use_container_width=True)  # Removed key parameter

# Chart 7: Cluster Scatter Plot (PCA reduced to 2D)
if 'Cluster Scatter Plot' in visualization_options:
    st.markdown("### Cluster Scatter Plot (PCA reduced to 2D)")
    @st.cache_data
    def get_pca_projection(embeddings):
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(embeddings)
        return pca_result
    
    pca_result = get_pca_projection(embeddings)
    
    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]
    
    fig6 = px.scatter(
        df,
        x='PCA1',
        y='PCA2',
        color='Cluster',
        hover_data=['Title', 'Organization', 'Main Research Field', 'Application Deadline (Display)'],
        title='Cluster Scatter Plot (PCA reduced to 2D)',
        template='plotly_white',
        height=600
    )
    st.plotly_chart(fig6, use_container_width=True, key='fig6_dynamic')

# Add data download button
st.markdown("## Data Download")

def download_link(object_to_download, download_filename, download_link_text):
    """
    Generate CSV download link
    """
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
    b64 = base64.b64encode(object_to_download.encode()).decode()  # Convert to base64
    return f'<a href="data:file/csv;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

# Add download link
st.markdown(
    download_link(filtered_df, 'filtered_projects.csv', '📥 Download filtered data (CSV)'),
    unsafe_allow_html=True
)

# Add Excel download link (corrected)
def download_link_excel(object_to_download, download_filename, download_link_text):
    """
    Generate Excel download link
    """
    if isinstance(object_to_download, pd.DataFrame):
        buffer = io.BytesIO()
        object_to_download.to_excel(buffer, index=False, engine='openpyxl')
        buffer.seek(0)
        object_to_download = buffer.read()
    b64 = base64.b64encode(object_to_download).decode()  # Convert to base64
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

st.markdown(
    download_link_excel(filtered_df, 'filtered_projects.xlsx', '📥 Download filtered data (Excel)'),
    unsafe_allow_html=True
)