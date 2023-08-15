from tqdm.notebook import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class IMDBData:
    def __init__(self):
        self.merged_data = None

    def load_data_with_progress(self, filename, chunk_size=100000):
        chunks = []
        with tqdm(total=100, desc=f"Reading {filename}") as pbar:
            for chunk in pd.read_csv(filename, sep='\t', compression='gzip', chunksize=chunk_size, low_memory=False):
                chunks.append(chunk)
                pbar.update(1)
        return pd.concat(chunks, ignore_index=True)

    def merge_data_with_progress(self, dataframes, on_column):
        with tqdm(total=len(dataframes) - 1, desc="Merging DataFrames") as pbar:
            self.merged_data = dataframes[0]
            for df in dataframes[1:]:
                self.merged_data = pd.merge(self.merged_data, df, on=on_column)
                pbar.update(1)


    def preprocess_data(self):
        # Preprocessing
        self.merged_data.replace('\\N', pd.NA, inplace=True)
        self.merged_data['startYear'] = pd.to_numeric(self.merged_data['startYear'], errors='coerce')
        self.merged_data['endYear'] = pd.to_numeric(self.merged_data['endYear'], errors='coerce')
        self.merged_data['runtimeMinutes'] = pd.to_numeric(self.merged_data['runtimeMinutes'], errors='coerce')
        self.merged_data = self.merged_data[self.merged_data['isAdult'] == '0']
        self.merged_data.dropna(subset=['startYear'], inplace=True)
        self.merged_data['endYear'].fillna('Unknown', inplace=True)
        self.merged_data['runtimeMinutes'].fillna(self.merged_data['runtimeMinutes'].median(), inplace=True)
        self.merged_data['genres'].fillna('Unknown', inplace=True)
        self.merged_data['directors'].fillna('Unknown', inplace=True)
        self.merged_data['writers'].fillna('Unknown', inplace=True)

    def scale_and_cluster_features(self):
        # Convert 'runtimeMinutes' to numeric
        self.merged_data['runtimeMinutes'] = pd.to_numeric(self.merged_data['runtimeMinutes'], errors='coerce')

        # Select the numerical features
        numerical_features = self.merged_data[['averageRating', 'numVotes', 'runtimeMinutes']].fillna(0)
        genres_split = self.merged_data['genres'].str.get_dummies(sep=',')
        encoded_genres = pd.get_dummies(genres_split)
        X = pd.concat([numerical_features, encoded_genres], axis=1)

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(X_scaled)
        self.merged_data['cluster'] = kmeans.labels_

    def recommend_by_title(self, movie_title, top_n=5):  # Add self as the first parameter
        filtered_movies = self.merged_data[self.merged_data['primaryTitle'] == movie_title]  # Add self.

        if filtered_movies.empty:
            print(f"No movie found with title {movie_title}")
            return []

        cluster = filtered_movies['cluster'].iloc[0]
        recommendations = self.merged_data[self.merged_data['cluster'] == cluster]  # Add self.

        # Sort recommendations by rating and number of votes
        recommendations = recommendations.sort_values(by=['averageRating', 'numVotes'], ascending=False)

        # Select top N recommendations
        top_recommendations = recommendations.head(top_n)

        # Prepare information for display
        recommended_movies = []
        for i, row in top_recommendations.iterrows():
            movie_info = {
                'title': row['primaryTitle'],
                'genre': row['genres'],
                'rating': row['averageRating'],
                'votes': row['numVotes'],
                # Add more details as needed
            }
            recommended_movies.append(movie_info)

        return recommended_movies



    # def recommend_by_director(self, director_name, top_n=5):
    #     # Recommendation function based on director
    #     # ...

    def recommend_by_genre(self, genre, top_n=5):
        filtered_movies = self.merged_data[self.merged_data['genres'].str.contains(genre, case=False)]

        if filtered_movies.empty:
            print(f"No movies found with genre {genre}")
            return []

        # Sort recommendations by rating and number of votes
        recommendations = filtered_movies.sort_values(by=['averageRating', 'numVotes'], ascending=False)

        # Select top N recommendations
        top_recommendations = recommendations.head(top_n)

        # Prepare information for display
        recommended_movies = []
        for i, row in top_recommendations.iterrows():
            movie_info = {
                'title': row['primaryTitle'],
                'genre': row['genres'],
                'rating': row['averageRating'],
                'votes': row['numVotes'],
                # Add more details as needed
            }
            recommended_movies.append(movie_info)

        return recommended_movies


# # Example usage
# imdb_data = IMDBData()
# title_basics = imdb_data.load_data_with_progress('imdb/title.basics.tsv.gz')
# title_crew = imdb_data.load_data_with_progress('imdb/title.crew.tsv.gz')
# name_basics = imdb_data.load_data_with_progress('imdb/name.basics.tsv.gz')
# title_ratings = pd.read_csv('imdb/title.ratings.tsv.gz', sep='\t', compression='gzip', low_memory=False)
# dataframes_to_merge_titles = [title_basics, title_ratings, title_crew]
# merged_titles = imdb_data.merge_data_with_progress(dataframes_to_merge_titles, 'tconst')
# imdb_data.preprocess_data()
# imdb_data.scale_and_cluster_features()
# # Call other methods as needed
