# pages/liu.py
import streamlit as st
import pandas as pd
import plotly.express as px
import base64
import io
from math import ceil
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import sys
import os
from sklearn.decomposition import PCA
# Add the root directory to sys.path to import utils.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import load_data, generate_embeddings_and_clusters, stop_words

# Set page configuration
st.set_page_config(page_title="LiU Exjobb Analysis", layout="wide")

# Load and preprocess data
@st.cache_data
def load_university_data():
    csv_file = 'data/liu_exjobb_projects.csv'  # Path to LiU data
    df = load_data(csv_file)
    return df

df = load_university_data()

# Generate embeddings and clusters
@st.cache_data
def process_data(df, num_clusters):
    return generate_embeddings_and_clusters(df, num_clusters=num_clusters)

# Sidebar for number of clusters
default_num_clusters = 5
num_clusters = st.sidebar.slider("Select number of clusters", min_value=2, max_value=20, value=default_num_clusters, step=1)

with st.spinner("Generating embeddings and performing clustering..."):
    df, embeddings, clusters, kmeans_model = process_data(df, num_clusters=num_clusters)

# Page title
st.title("LiU Exjobb Project Data Analysis and Visualization")

# Sidebar filter options
st.sidebar.header("Filter Options")

# Search keyword
search_query = st.sidebar.text_input("Search project title keywords")

# Filter by main research field
if 'Main Research Field' in df.columns:
    # Convert to string and drop NaN
    df['Main Research Field'] = df['Main Research Field'].astype(str)
    fields = sorted(df['Main Research Field'].dropna().unique())
    selected_fields = st.sidebar.multiselect("Main Research Field", options=fields, default=fields)
else:
    selected_fields = []

# Filter by organization
organizations = sorted(df['Organization'].astype(str).dropna().unique())
selected_orgs = st.sidebar.multiselect("Organization", options=organizations, default=organizations)

# Filter by deadline year
years = sorted(df['Deadline Year'].dropna().unique())
selected_years = st.sidebar.multiselect("Deadline Year", options=years, default=years)

# Filter by deadline month
months = sorted(df['Deadline Month'].dropna().unique())
selected_months = st.sidebar.multiselect("Deadline Month", options=months, default=months)

# Filter by cluster
clusters_options = sorted(df['Cluster'].dropna().unique())
selected_clusters = st.sidebar.multiselect("Cluster", options=clusters_options, default=clusters_options)

# Sorting options
st.sidebar.header("Sorting Options")
sort_by = st.sidebar.selectbox("Sort by", options=['Organization', 'Deadline for Application'], index=0)
sort_order = st.sidebar.radio("Sort order", options=['Ascending', 'Descending'], index=1)

# Dynamic visualization options
st.sidebar.header("Dynamic Visualization Options")
visualization_options = st.sidebar.multiselect(
    "Select visualization charts",
    options=['Project Title Word Cloud', 'Cluster Scatter Plot', 'Project Count Trend', 'Project Title and Organization Relationship'],
    default=['Project Title Word Cloud', 'Cluster Scatter Plot', 'Project Count Trend', 'Project Title and Organization Relationship']
)

# Ensure selected years and months are not empty
if not selected_years:
    selected_years = years

if not selected_months:
    selected_months = months

if not selected_clusters:
    selected_clusters = clusters_options

# Filter data based on selected criteria
filtered_df = df[
    (df['Organization'].isin(selected_orgs)) &
    (df['Deadline Year'].isin(selected_years)) &
    (df['Deadline Month'].isin(selected_months)) &
    (df['Cluster'].isin(selected_clusters))
]

# Apply additional filters if applicable
if selected_fields:
    filtered_df = filtered_df[filtered_df['Main Research Field'].isin(selected_fields)]

# Further filter based on search keyword
if search_query:
    filtered_df = filtered_df[filtered_df['Title'].str.contains(search_query, case=False, na=False)]

# Sort data based on sorting options
ascending = True if sort_order == 'Ascending' else False
filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)

# Pagination settings
items_per_page = 20
total_pages = ceil(filtered_df.shape[0] / items_per_page) if filtered_df.shape[0] > 0 else 1

# User selects page number
st.markdown("### Pagination")
page = st.number_input("Select page number", min_value=1, max_value=total_pages, value=1, step=1)

# Calculate start and end indices
start_idx = (page - 1) * items_per_page
end_idx = start_idx + items_per_page
paginated_df = filtered_df.iloc[start_idx:end_idx]

# Display the number of filtered projects
st.markdown(f"### Filtered Results: {filtered_df.shape[0]} projects (showing {start_idx + 1} to {min(end_idx, filtered_df.shape[0])})")

# Display project data table
def make_clickable(link):
    return f'<a href="{link}" target="_blank">View Details</a>'

paginated_df_display = paginated_df.copy()
paginated_df_display['Link'] = paginated_df_display['Link'].apply(make_clickable)

# Select columns to display based on available columns
columns_to_display = ['Title', 'Organization', 'Main Research Field', 'Application Deadline (Display)', 'Link', 'Cluster']

# Ensure all columns exist
columns_to_display = [col for col in columns_to_display if col in paginated_df_display.columns]

st.markdown(paginated_df_display[columns_to_display].to_html(escape=False, index=False), unsafe_allow_html=True)

# Visualization section
st.markdown("## Data Visualization")

# Chart 1: Distribution of Main Research Fields (Bar Chart)
if 'Main Research Field' in filtered_df.columns and not filtered_df['Main Research Field'].empty:
    fig1 = px.bar(
        filtered_df,
        x='Main Research Field',
        title='Distribution of Main Research Fields',
        labels={'Main Research Field': 'Main Research Field', 'count': 'Project Count'},
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

# Chart 3: Project Title Word Cloud
if 'Project Title Word Cloud' in visualization_options and not filtered_df['Title'].dropna().empty:
    st.markdown("### Project Title Word Cloud")
    text = ' '.join(filtered_df['Title'].dropna().astype(str).tolist())
    wordcloud = WordCloud(stopwords=stop_words, background_color='white', width=800, height=400).generate(text)
    
    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc, use_container_width=True)

# Chart 4: Application Deadline Trend (Line Chart)
deadline_trend = filtered_df.groupby('Deadline for Application').size().reset_index(name='Project Count')
deadline_trend = deadline_trend.sort_values('Deadline for Application')

fig3 = px.line(
    deadline_trend,
    x='Deadline for Application',
    y='Project Count',
    title='Application Deadline Trend',
    labels={'Deadline for Application': 'Application Deadline', 'Project Count': 'Project Count'},
    markers=True,
    height=400,
    template='plotly_white'
)
st.plotly_chart(fig3, use_container_width=True, key='fig3')

# Chart 5: Distribution of Project Count by Year and Month (Heatmap)
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

# Chart 6: Cluster Scatter Plot (PCA reduced to 2D)
if 'Cluster Scatter Plot' in visualization_options and not filtered_df.empty:
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
        hover_data=['Title', 'Organization', 'Application Deadline (Display)'],
        title='Cluster Scatter Plot (PCA reduced to 2D)',
        template='plotly_white',
        height=600
    )
    st.plotly_chart(fig6, use_container_width=True, key='fig6_dynamic')

# Add data download buttons
st.markdown("## Data Download")

def download_link(object_to_download, download_filename, download_link_text, filetype='csv'):
    """
    Generate a download link for CSV or Excel files.
    """
    if isinstance(object_to_download, pd.DataFrame):
        if filetype == 'csv':
            object_to_download = object_to_download.to_csv(index=False).encode()
            mime_type = 'text/csv'
        elif filetype == 'excel':
            buffer = io.BytesIO()
            object_to_download.to_excel(buffer, index=False, engine='openpyxl')
            object_to_download = buffer.getvalue()
            mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    else:
        raise ValueError("object_to_download must be a pandas DataFrame")
    
    b64 = base64.b64encode(object_to_download).decode()
    return f'<a href="data:{mime_type};base64,{b64}" download="{download_filename}">{download_link_text}</a>'

# Add CSV download link
st.markdown(
    download_link(filtered_df, 'filtered_projects_liu.csv', '📥 Download filtered data (CSV)', filetype='csv'),
    unsafe_allow_html=True
)

# Add Excel download link
st.markdown(
    download_link(filtered_df, 'filtered_projects_liu.xlsx', '📥 Download filtered data (Excel)', filetype='excel'),
    unsafe_allow_html=True
)
