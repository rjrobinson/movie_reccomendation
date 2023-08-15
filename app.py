from flask import Flask, render_template, request, Response
from models.netflix_data import NetflixData  # Import other necessary modules and functions from your Notebook
import seaborn as sns
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Add this line
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)

# Example usage:
netflix_data = NetflixData()


# # Call other methods as needed
@app.route('/')
def index():
    return render_template("index.html")


@app.route('/recommend', methods=['POST'])
def recommend_by():
    query = request.form['query'].upper()
    recommendations = netflix_data.recommend_by(query)  # Call your combined recommendation function
    return render_template('index.html', recommendations=recommendations)


@app.route('/top_10_genres_plot.png')
def top_10_genres_plot():
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(figsize=(10, 6))

    top_10_genres = netflix_data.data_netflix['genre_type'].value_counts().head(10)
    sns.barplot(x=top_10_genres.index, y=top_10_genres.values, palette="colorblind", ax=ax)

    ax.set_title('Top 10 Genres on Netflix')
    ax.set_xlabel('Genre')
    ax.set_ylabel('Number of Titles')
    ax.tick_params(axis='x', rotation=45, labelsize=8)

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)

    return Response(img_buffer.getvalue(), content_type='image/png')

@app.route('/average_rating_by_genre_plot.png')
def average_rating_by_genre_plot():
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(figsize=(10, 6))

    avg_rating_by_genre = netflix_data.data_netflix.groupby('genre_type')['rating'].mean().sort_values(ascending=False)
    sns.barplot(x=avg_rating_by_genre.index, y=avg_rating_by_genre.values, palette="colorblind", ax=ax)

    ax.set_title('Average Rating by Genre on Netflix')
    ax.set_xlabel('Genre')
    ax.set_ylabel('Average Rating')
    ax.tick_params(axis='x', rotation=45, labelsize=8)

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)

    return Response(img_buffer.getvalue(), content_type='image/png')

@app.route('/release_year_histogram_plot.png')
def release_year_histogram_plot():
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(data=netflix_data.data_netflix, x='release_year', bins=20, kde=False, color='purple', ax=ax)

    ax.set_title('Distribution of Release Years on Netflix')
    ax.set_xlabel('Release Year')
    ax.set_ylabel('Number of Titles')

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)

    return Response(img_buffer.getvalue(), content_type='image/png')

if __name__ == '__main__':
    app.run(debug=True)
