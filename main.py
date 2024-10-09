import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

df = pd.read_csv('sample_dataset.csv')

def format_value(value):
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.1f}K"
    else:
        return str(value)

total_repositories = len(df)
total_stars = abs(df['stars_count'].sum())
total_forks = abs(df['forks_count'].sum())
total_watchers = abs(df['watchers'].sum())
avg_pull_requests = df['pull_requests'].mean()
avg_stars_per_repo = total_stars / total_repositories if total_repositories > 0 else 0
avg_forks_per_repo = total_forks / total_repositories if total_repositories > 0 else 0
most_common_language = df['primary_language'].mode()[0] if not df['primary_language'].isnull().all() else "N/A"
total_pull_requests = df['pull_requests'].sum()

st.title('GitHub Projects Dashboard')
st.header("Summary Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Total Repositories", value=total_repositories)
with col2:
    st.metric(label="Total Stars", value=format_value(total_stars))
with col3:
    st.metric(label="Total Forks", value=format_value(total_forks))
with col4:
    st.metric(label="Total Watchers", value=format_value(total_watchers))

col5, col6, col7, col8 = st.columns(4)

with col5:
    st.metric(label="Total Pull Requests", value=format_value(total_pull_requests))
with col6:
    st.metric(label="Avg. Forks/Repo", value=format_value(avg_forks_per_repo))
with col7:
    st.metric(label="Avg. Stars/Repo", value=format_value(avg_stars_per_repo))
with col8:
    st.metric(label="Most Common Language", value=most_common_language)

st.markdown("---")

left_col, right_col = st.columns(2)

with left_col:
    st.header("Commits Count by Year")
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df['year'] = df['created_at'].dt.year
    commits_per_year = df.groupby('year')['commit_count'].sum().reset_index()
    commits_per_year.columns = ['Year', 'Total Commits']
    commits_per_year['Total Commits'] = commits_per_year['Total Commits'].apply(format_value)
    st.table(commits_per_year.set_index('Year'))

with right_col:
    st.header("Top 5 Repositories by Stars")
    top_repos = df.nlargest(5, 'stars_count')[['name', 'stars_count']]
    top_repos['stars_count'] = top_repos['stars_count'].apply(format_value)
    st.table(top_repos.set_index('name'))

df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
df['year'] = df['created_at'].dt.year
lang_year_df = df.groupby(['year', 'primary_language']).size().reset_index(name='count')
lang_year_df['count'] = lang_year_df['count'].abs()
top_languages = lang_year_df.groupby('primary_language')['count'].sum().nlargest(5).index
top_lang_year_df = lang_year_df[lang_year_df['primary_language'].isin(top_languages)]

all_years = pd.DataFrame({'year': range(df['year'].min(), df['year'].max() + 1)})
complete_data = (top_lang_year_df.merge(all_years, on='year', how='right')
                 .fillna({'count': 0}))

st.header("Primary Languages Usage Over the Years")
with st.container():
    st.subheader("Top Primary Language Over Time")
    all_primary_languages = df['primary_language'].dropna().unique().tolist()
    selected_language = st.selectbox("Select a Primary Language:", all_primary_languages)
    selected_lang_data = complete_data[complete_data['primary_language'] == selected_language]

    if selected_lang_data.shape[0] > 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        selected_lang_data = selected_lang_data.sort_values('year')
        x = selected_lang_data['year']
        y = selected_lang_data['count']
        X_Y_Spline = make_interp_spline(x, y)
        X_ = np.linspace(x.min(), x.max(), 500)
        Y_ = X_Y_Spline(X_)
        ax.plot(X_, Y_, label=selected_language, color='blue', linewidth=2)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel('Number of Repositories', fontsize=14)
        ax.set_title(f'{selected_language} Usage Over the Years', fontsize=16)
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(title='Primary Language')
        st.pyplot(fig)
    else:
        st.warning(f"Not enough data to plot usage for '{selected_language}'. Only available in one year.")

with st.container():
    st.subheader("Top 5 Primary Languages Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, language in enumerate(top_languages):
        lang_data = complete_data[complete_data['primary_language'] == language]
        lang_data = lang_data.sort_values('year')
        x = lang_data['year']
        y = lang_data['count']
        X_Y_Spline = make_interp_spline(x, y)
        X_ = np.linspace(x.min(), x.max(), 500)
        Y_ = X_Y_Spline(X_)
        ax.plot(X_, Y_, label=language, color=colors[i], linewidth=2)

    ax.set_ylim(bottom=0)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Number of Repositories', fontsize=14)
    ax.set_title('Top 5 Primary Languages Over Time (Smoothed)', fontsize=16)
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(title='Primary Languages')
    st.pyplot(fig)

with st.container():
    st.header("Tech Stack Density")
    languages_split = df['languages_used'].dropna().str.split(',').explode().str.strip()
    tech_stack_counts = languages_split.value_counts().reset_index()
    tech_stack_counts.columns = ['Technology', 'Count']
    top_n = 10
    tech_stack_counts = tech_stack_counts.nlargest(top_n, 'Count')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(tech_stack_counts['Technology'], tech_stack_counts['Count'], color='teal')
    ax.set_xlabel('Technology', fontsize=14)
    ax.set_ylabel('Number of Repositories', fontsize=14)
    ax.set_title('Tech Stack Density: Number of Repositories Using Each Technology', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    st.pyplot(fig)

with st.container():
    st.header("Pull Requests vs Forks Over Time")
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df['year'] = df['created_at'].dt.year
    time_series_activity = df.groupby('year').agg({'pull_requests': 'sum', 'forks_count': 'sum'}).reset_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_series_activity['year'], time_series_activity['pull_requests'], color='blue', marker='o', label='Total Pull Requests', linewidth=2)
    ax.plot(time_series_activity['year'], time_series_activity['forks_count'], color='orange', marker='o', label='Total Forks', linewidth=2)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('Pull Requests vs Forks Over Time', fontsize=16)
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(title='Metrics')
    st.pyplot(fig)

st.markdown("---")
st.markdown("Developed by Deepa ([GitHub](https://github.com/Deep95y))")
