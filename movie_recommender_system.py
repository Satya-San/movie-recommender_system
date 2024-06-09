import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load custom ratings
my_ratings_df = pd.read_csv("My_Ratings.csv")

# Load movie titles
movie_titles_df = pd.read_csv("Movie_Id_Titles.csv")

# Load movie ratings
movies_rating_df = pd.read_csv('u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

# Merge the two dataframes on the item_id
movies_merged_df = pd.merge(movies_rating_df, movie_titles_df, on='item_id')

# Visualization
sns.set_style('white')
plt.figure(figsize=(10,4))
movies_merged_df['rating'].hist(bins=50)
plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()

# Calculate the mean rating for each movie
mean_ratings = movies_merged_df.groupby('title')['rating'].mean()

# Calculate the number of ratings for each movie
ratings_count = movies_merged_df.groupby('title')['rating'].count()

# Create a dataframe with mean ratings and the number of ratings
ratings_df = pd.DataFrame({'mean_rating': mean_ratings, 'ratings_count': ratings_count})

# Visualization of ratings count
plt.figure(figsize=(10,4))
ratings_df['ratings_count'].hist(bins=50)
plt.title("Distribution of Ratings Count")
plt.xlabel("Number of Ratings")
plt.ylabel("Frequency")
plt.show()

# Jointplot of mean rating and number of ratings
sns.jointplot(x='mean_rating', y='ratings_count', data=ratings_df, alpha=0.5)
plt.show()

# Pivot table to create a matrix of user ratings for each movie
movie_matrix = movies_merged_df.pivot_table(index='user_id', columns='title', values='rating')

# Function to get movie recommendations
def get_recommendations(movie_title, min_ratings=100):
    movie_ratings = movie_matrix[movie_title]
    similar_movies = movie_matrix.corrwith(movie_ratings)
    corr_movie = pd.DataFrame(similar_movies, columns=['correlation'])
    corr_movie.dropna(inplace=True)
    corr_movie = corr_movie.join(ratings_df['ratings_count'])
    recommendations = corr_movie[corr_movie['ratings_count'] > min_ratings].sort_values('correlation', ascending=False)
    return recommendations

# Example: Get recommendations for a specific movie
movie_title = "Star Wars (1977)"
recommendations = get_recommendations(movie_title)
print(f"Recommendations for {movie_title}:")
print(recommendations.head())
