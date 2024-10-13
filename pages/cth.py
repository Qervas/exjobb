# pages/cth.py
import streamlit as st
import pandas as pd
import plotly.express as px
import base64
import io
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import sys
import os
from sklearn.decomposition import PCA
import re
import html

# Import Streamlit components
import streamlit.components.v1 as components

# Add the root directory to sys.path to import utils_cth.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils_cth import load_data_cth, generate_embeddings_and_clusters_cth, stop_words

# Set page configuration
st.set_page_config(page_title="CTH Exjobb Analysis", layout="wide")

# Title of the app
st.title("CTH Exjobb Project Data Analysis and Visualization")

# Load and preprocess data
@st.cache_data
def load_university_data():
    csv_file = 'data/cth_exjobb_projects.csv'  # Path to CTH data
    df, edu_mapping, subj_mapping = load_data_cth(csv_file)
    return df, edu_mapping, subj_mapping

df, edu_mapping, subj_mapping = load_university_data()

# Display the number of projects loaded
st.write(f"**Total Projects Loaded:** {df.shape[0]}")

# Generate embeddings and clusters
@st.cache_data
def process_data(df, num_clusters):
    return generate_embeddings_and_clusters_cth(df, num_clusters=num_clusters)

# Function to create multiselect filters
def create_multiselect_filter(df, column_name, display_name):
    if column_name in df.columns:
        if df[column_name].dtype == 'object':
            df[column_name] = df[column_name].astype(str)

        # If the column is "Subject Area", "Educational Area", "Location", or "Country", split the tags into separate options
        if column_name in ['Subject Area', 'Educational Area', 'Location', 'Country', 'Organization']:
            unique_tags = df[column_name].dropna().str.split(', ').explode().unique()
            options = sorted(unique_tags)
        else:
            options = sorted(df[column_name].dropna().unique())

        # Let users select multiple options, and by default, select all options
        selected_options = st.sidebar.multiselect(display_name, options=options, default=options)

        return selected_options
    else:
        return []
    
# Sidebar for number of clusters
st.sidebar.header("Clustering Options")
default_num_clusters = 5
num_clusters = st.sidebar.slider("Select number of clusters", min_value=2, max_value=20, value=default_num_clusters, step=1)

with st.spinner("Generating embeddings and performing clustering..."):
    try:
        df, embeddings, clusters, kmeans_model = process_data(df, num_clusters=num_clusters)
        st.success("**Embeddings and clustering generated successfully.**")
    except ValueError as ve:
        st.error(f"Error in embeddings/clustering: {ve}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.stop()


# Assign clusters to dataframe
df['Cluster'] = clusters

# Sidebar filter options
st.sidebar.header("Filter Options")

# Search keyword
search_query = st.sidebar.text_input("Search project title keywords")

# Sorting options 
st.sidebar.header("Sorting Options")
sort_by_options = []
if 'Organization' in df.columns:
    sort_by_options.append('Organization')
if 'Deadline for Application' in df.columns:
    sort_by_options.append('Deadline for Application')
if 'Publish Date' in df.columns:
    sort_by_options.append('Publish Date')  # Added date
sort_by = st.sidebar.selectbox("Sort by", options=sort_by_options, index=0 if sort_by_options else 0)
sort_order = st.sidebar.radio("Sort order", options=['Ascending', 'Descending'], index=1)

# Add a legend to explain the abbreviations
st.sidebar.header("Legend: Abbreviation Meanings")
with st.sidebar.expander("Click to see abbreviation meanings", expanded=False):
    st.write("Below are the meanings of the abbreviations used in the Subject Area and Educational Area columns:")

    # Educational Area Mappings
    st.markdown("### Educational Area Abbreviations")
    for full_term, abbr in edu_mapping.items():
        st.markdown(f"**{abbr}**: {full_term}")

    # Subject Area Mappings
    st.markdown("### Subject Area Abbreviations")
    for full_term, abbr in subj_mapping.items():
        st.markdown(f"**{abbr}**: {full_term}")

# Create filters - General filters first
selected_educational_areas = create_multiselect_filter(df, 'Educational Area', "Educational Area")
selected_countries = create_multiselect_filter(df, 'Country', "Country")
selected_years = create_multiselect_filter(df, 'Deadline Year', "Deadline Year")
selected_months = create_multiselect_filter(df, 'Deadline Month', "Deadline Month")
selected_clusters = create_multiselect_filter(df, 'Cluster', "Cluster")

# Create filters - Specific filters at the end
selected_subject_areas = create_multiselect_filter(df, 'Subject Area', "Subject Area")
selected_locations = create_multiselect_filter(df, 'Location', "Location")
selected_orgs = create_multiselect_filter(df, 'Organization', "Organization")



# Dynamic visualization options
st.sidebar.header("Dynamic Visualization Options")
visualization_options = st.sidebar.multiselect(
    "Select visualization charts",
    options=[
        'Project Title Word Cloud', 
        'Cluster Scatter Plot', 
        'Project Count Trend', 
        'Project Subject Area Distribution', 
        'Project Educational Area Distribution'
    ],
    default=['Project Title Word Cloud', 'Cluster Scatter Plot', 'Project Count Trend', 'Project Subject Area Distribution', 'Project Educational Area Distribution']
)

# Filter data based on selected criteria
filtered_df = df.copy()

# Apply filters only if selections are made in the sidebar
if selected_orgs and selected_orgs != list(df['Organization'].unique()):
    filtered_df = filtered_df[
        filtered_df['Organization'].apply(lambda x: any(org in x.split(', ') for org in selected_orgs) if pd.notna(x) else False)
    ]

if selected_years and selected_years != list(df['Deadline Year'].unique()):
    filtered_df = filtered_df[filtered_df['Deadline Year'].isin(selected_years) | filtered_df['Deadline Year'].isna()]

if selected_months and selected_months != list(df['Deadline Month'].unique()):
    filtered_df = filtered_df[filtered_df['Deadline Month'].isin(selected_months) | filtered_df['Deadline Month'].isna()]

if selected_clusters and selected_clusters != list(df['Cluster'].unique()):
    filtered_df = filtered_df[filtered_df['Cluster'].isin(selected_clusters) | filtered_df['Cluster'].isna()]

if selected_subject_areas and selected_subject_areas != list(df['Subject Area'].unique()):
    filtered_df = filtered_df[
        filtered_df['Subject Area'].apply(lambda x: any(area in x.split(', ') for area in selected_subject_areas) if pd.notna(x) else False)
    ]

if selected_educational_areas and selected_educational_areas != list(df['Educational Area'].unique()):
    filtered_df = filtered_df[
        filtered_df['Educational Area'].apply(lambda x: any(area in x.split(', ') for area in selected_educational_areas) if pd.notna(x) else False)
    ]

if selected_locations and selected_locations != list(df['Location'].unique()):
    filtered_df = filtered_df[
        filtered_df['Location'].apply(lambda x: any(location in x.split(', ') for location in selected_locations) if pd.notna(x) else False)
    ]

if selected_countries and selected_countries != list(df['Country'].unique()):
    filtered_df = filtered_df[
        filtered_df['Country'].apply(lambda x: any(country in x.split(', ') for country in selected_countries) if pd.notna(x) else False)
    ]

# Further filter based on search keyword
if search_query:
    filtered_df = filtered_df[filtered_df['Title'].str.contains(search_query, case=False, na=False)]

# Sort data based on sorting options
if sort_by:
    ascending = True if sort_order == 'Ascending' else False
    filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)

# Display the number of filtered projects
st.markdown(f"### **Filtered Results:** {filtered_df.shape[0]} projects")

# Ensure 'Link' and 'Contact Info' columns are of string dtype
if 'Link' in filtered_df.columns:
    filtered_df['Link'] = filtered_df['Link'].astype(str)
if 'Contact Info' in filtered_df.columns:
    filtered_df['Contact Info'] = filtered_df['Contact Info'].astype(str)

# Function to convert links and emails into clickable HTML links with shortened display text
def make_clickable(row):
    link = row.get('Link', '')
    apply_by_detail = row.get('Apply By Detail', '')  # Ensure this column exists
    contact_info = row.get('Contact Info', '')

    # Convert 'Link' to a clickable URL
    if pd.notna(link) and link.strip() and link.lower() != 'nan':
        link_html = f'<a href="{link}" target="_blank">View Details</a>'
    else:
        link_html = 'N/A'

    # Convert 'Apply By Detail' to clickable links or emails with shortened text
    if pd.notna(apply_by_detail) and apply_by_detail.strip() and apply_by_detail.lower() != 'nan':
        # Initialize the processed text
        processed_text = apply_by_detail

        # Find all URLs
        url_pattern = re.compile(r'(https?://\S+)')
        urls = url_pattern.findall(apply_by_detail)
        for url in urls:
            processed_text = processed_text.replace(url, f'<a href="{url}" target="_blank">Link</a>')

        # Find all emails
        email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+')
        emails = email_pattern.findall(apply_by_detail)
        for email in emails:
            processed_text = processed_text.replace(email, f'<a href="mailto:{email}">Email</a>')

        apply_by_html = processed_text
    else:
        apply_by_html = 'N/A'

    # Replace the 'Link' and 'Apply By Detail' with clickable HTML
    row['Link'] = link_html
    row['Apply By Detail'] = apply_by_html

    return row

# Apply the function to the DataFrame
paginated_df_display = filtered_df.copy().apply(make_clickable, axis=1)

# Function to render tags for Educational Area and Subject Area
def render_tags(cell_value):
    if pd.isna(cell_value):
        return 'N/A'
    tags = [tag.strip() for tag in cell_value.split(',')]
    tag_html = ""
    color_map = {}
    # Assign colors based on unique tags
    for tag in tags:
        if tag not in color_map and tag != 'N/A':
            # Generate a consistent color based on the tag name
            color = "#{:06x}".format(abs(hash(tag)) % 0xFFFFFF)
            color_map[tag] = color
    for tag in tags:
        color = color_map.get(tag, "#e0e0e0")  # Default color for 'N/A' or undefined tags
        tag_html += f'<span style="background-color: {color}; color: white; border-radius: 5px; padding: 3px 7px; margin: 2px; display: inline-block; font-size: 12px;">{html.escape(tag)}</span> '
    return tag_html

# Apply the render_tags function to the relevant columns
if 'Educational Area' in paginated_df_display.columns:
    paginated_df_display['Educational Area'] = paginated_df_display['Educational Area'].apply(render_tags)
if 'Subject Area' in paginated_df_display.columns:
    paginated_df_display['Subject Area'] = paginated_df_display['Subject Area'].apply(render_tags)

# Select columns to display based on available columns
columns_to_display = [
    'Title', 
    'Organization', 
    'Subject Area', 
    'Educational Area', 
    'Location', 
    'Country', 
    'Deadline for Application', 
    'Apply By Detail',    # Now rendered as hyperlinks
    'Link', 
    'Cluster'
]

# Ensure all columns exist
columns_to_display = [col for col in columns_to_display if col in paginated_df_display.columns]

# Function to generate HTML table with dark mode support
def generate_html_table(df, columns):
    """
    Generates an HTML table from a DataFrame with dark mode support.
    
    Parameters:
    - df: pandas DataFrame.
    - columns: list of str, columns to include in the table.
    
    Returns:
    - html_table: str, HTML representation of the table.
    """
    # Start table
    html_table = """
    <style>
        /* Common table styles */
        table {
            border-collapse: collapse;
            width: 100%;
            font-family: Arial, sans-serif;
            font-size: 14px;
        }
        th, td {
            text-align: left;
            padding: 8px;
            vertical-align: top;
            border: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-child(even){background-color: #f9f9f9}
        a {
            color: #1E90FF;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }

        /* Dark mode styles */
        @media (prefers-color-scheme: dark) {
            table {
                border-color: #444;
            }
            th {
                background-color: #333;
                color: #fff;
            }
            tr:nth-child(even){
                background-color: #2a2a2a;
            }
            tr:nth-child(odd){
                background-color: #1e1e1e;
            }
            td {
                color: #ddd;
                border-color: #555;
            }
            a {
                color: #1E90FF;
            }
        }
    </style>
    <table>
        <thead>
            <tr>
    """
    # Add headers
    for col in columns:
        html_table += f"<th>{html.escape(col)}</th>"
    html_table += "</tr></thead><tbody>"

    # Add rows
    for _, row in df.iterrows():
        html_table += "<tr>"
        for col in columns:
            cell = row[col]
            if pd.isna(cell):
                cell = 'N/A'
            html_table += f"<td>{cell}</td>"
        html_table += "</tr>"
    html_table += "</tbody></table>"
    return html_table

# Generate HTML table
html_table = generate_html_table(paginated_df_display, columns_to_display)

# Render the HTML table using Streamlit components
components.html(html_table, height=600, scrolling=True)

# Visualization section
st.markdown("## **Data Visualization**")

# Chart 1: Distribution of Subject Areas (Bar Chart)
if 'Subject Area' in filtered_df.columns and not filtered_df['Subject Area'].dropna().empty:
    if 'Project Subject Area Distribution' in visualization_options:
        # Extract raw subject areas without HTML tags for accurate counting
        subject_series = filtered_df['Subject Area'].dropna().str.replace(r'<[^>]+>', '', regex=True)
        subject_counts = subject_series.str.split(', ').explode().value_counts().reset_index()
        subject_counts.columns = ['Subject Area', 'Project Count']

        fig1 = px.bar(
            subject_counts,
            x='Subject Area',
            y='Project Count',
            title='Distribution of Subject Areas',
            labels={'Subject Area': 'Subject Area', 'Project Count': 'Project Count'},
            height=400,
            color='Subject Area',
            template='plotly_white'
        )
        st.plotly_chart(fig1, use_container_width=True, key='fig1_cth')

# Chart 2: Distribution of Educational Areas (Bar Chart)
if 'Educational Area' in filtered_df.columns and not filtered_df['Educational Area'].dropna().empty:
    if 'Project Educational Area Distribution' in visualization_options:
        # Extract raw educational areas without HTML tags for accurate counting
        edu_series = filtered_df['Educational Area'].dropna().str.replace(r'<[^>]+>', '', regex=True)
        edu_counts = edu_series.str.split(', ').explode().value_counts().reset_index()
        edu_counts.columns = ['Educational Area', 'Project Count']

        fig2 = px.bar(
            edu_counts,
            x='Educational Area',
            y='Project Count',
            title='Distribution of Educational Areas',
            labels={'Educational Area': 'Educational Area', 'Project Count': 'Project Count'},
            height=400,
            color='Educational Area',
            template='plotly_white'
        )
        st.plotly_chart(fig2, use_container_width=True, key='fig2_cth')

# Chart 3: Project Title Word Cloud
if 'Project Title Word Cloud' in visualization_options and not filtered_df['Title'].dropna().empty:
    st.markdown("### **Project Title Word Cloud**")
    text = ' '.join(filtered_df['Title'].dropna().astype(str).tolist())
    wordcloud = WordCloud(stopwords=stop_words, background_color='white', width=800, height=400).generate(text)

    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc, use_container_width=True)

# Chart 4: Application Deadline Trend (Line Chart)
if 'Deadline for Application' in filtered_df.columns and not filtered_df['Deadline for Application'].dropna().empty:
    if 'Project Count Trend' in visualization_options:
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
        st.plotly_chart(fig3, use_container_width=True, key='fig3_cth')

# Chart 5: Distribution of Project Count by Year and Month (Heatmap)
if {'Deadline Year', 'Deadline Month'}.issubset(filtered_df.columns):
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
    st.plotly_chart(fig4, use_container_width=True, key='fig4_cth')

# Chart 6: Cluster Scatter Plot (PCA reduced to 2D)
if 'Cluster Scatter Plot' in visualization_options and not filtered_df.empty:
    st.markdown("### **Cluster Scatter Plot (PCA reduced to 2D)**")

    @st.cache_data
    def get_pca_projection(embeddings):
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(embeddings)
        return pca_result

    try:
        pca_result = get_pca_projection(embeddings)
        df['PCA1'] = pca_result[:, 0]
        df['PCA2'] = pca_result[:, 1]

        fig6 = px.scatter(
            df,
            x='PCA1',
            y='PCA2',
            color='Cluster',
            hover_data=['Title', 'Organization', 'Deadline for Application'],
            title='Cluster Scatter Plot (PCA reduced to 2D)',
            template='plotly_white',
            height=600
        )
        st.plotly_chart(fig6, use_container_width=True, key='fig6_cth')
    except Exception as e:
        st.error(f"Error generating PCA scatter plot: {e}")

# Add data download buttons
st.markdown("## **Data Download**")

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
    download_link(filtered_df, 'filtered_projects_cth.csv', '📥 Download filtered data (CSV)', filetype='csv'),
    unsafe_allow_html=True
)

# Add Excel download link
st.markdown(
    download_link(filtered_df, 'filtered_projects_cth.xlsx', '📥 Download filtered data (Excel)', filetype='excel'),
    unsafe_allow_html=True
)
