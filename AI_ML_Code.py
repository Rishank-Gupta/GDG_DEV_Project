# Netflix Content Analysis & Preprocessing 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Task 1: Load and Inspect Data
# -------------------------------

df = pd.read_csv("netflix_titles.csv")

df.head(10)
df.shape
df.info()
df.describe(include='all')


df.drop_duplicates(subset='show_id', inplace=True)

df.drop(columns=['description'], inplace=True)

# -------------------------------
# Task 2: Data Cleaning & Feature Engineering
# -------------------------------

df.isnull().sum()

df['country'].fillna('Unknown', inplace=True)
df['director'].fillna('No Director Listed', inplace=True)

df['duration_minutes'] = np.where(
    df['type'] == 'Movie',
    df['duration'].str.replace(' min', '').astype(float),
    np.nan
)

df['seasons'] = np.where(
    df['type'] == 'TV Show',
    df['duration'].str.replace(' Seasons', '').str.replace(' Season', '').astype(float),
    np.nan
)

df['Is_Recent'] = np.where(df['release_year'] >= 2015, 1, 0)

# -------------------------------
# Task 3: Exploratory Data Analysis
# -------------------------------

sns.countplot(x='type', data=df)
plt.title('Movies vs TV Shows')
plt.show()

# Histogram of release year
plt.hist(df['release_year'], bins=20)
plt.xlabel('Release Year')
plt.ylabel('Count')
plt.title('Distribution of Release Year')
plt.show()

# Top 10 countries by number of titles
df['country'].value_counts().head(10).plot(kind='bar')
plt.xlabel('Country')
plt.ylabel('Number of Titles')
plt.title('Top 10 Countries on Netflix')
plt.show()

# Boxplot: Movie duration vs Is_Recent
movie_df = df[df['type'] == 'Movie']
sns.boxplot(x='Is_Recent', y='duration_minutes', data=movie_df)
plt.title('Movie Duration: Recent vs Older')
plt.show()

# -------------------------------
# Optional: Word Cloud for Genres
# -------------------------------

from wordcloud import WordCloud

genres_text = " ".join(df['listed_in'].dropna())
wordcloud = WordCloud(background_color='white').generate(genres_text)

plt.imshow(wordcloud)
plt.axis('off')
plt.show()

# -------------------------------
# Correlation Heatmap
# -------------------------------

numeric_df = df[['release_year', 'duration_minutes', 'seasons', 'Is_Recent']]
sns.heatmap(numeric_df.corr(), annot=True)
plt.title('Correlation Heatmap')
plt.show()
