import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the movie data
df = pd.read_csv('./movie_metadata.csv')

# Clean the movie titles by stripping whitespace and removing non-breaking spaces
df['movie_title'] = df['movie_title'].str.strip().str.replace('\xa0', '')

# Print the first few rows and column names to verify the data
print(df.head())
print(df.columns)

# Function to combine relevant features into a single string
def combine_features(row):
    keywords = row['plot_keywords'].split('|') if isinstance(row['plot_keywords'], str) else []
    keywords_str = ' '.join(keywords) + ' ' + ' '.join(keywords)  # Repeat keywords for emphasis
    
    genres = row['genres'].split('|') if isinstance(row['genres'], str) else []
    genres_str = ' '.join(genres) * 10  # Repeat genres 10 times for even more emphasis
    
    return (f"{genres_str} "
            f"{row['director_name']} {row['director_name']} "
            f"{row['imdb_score']} {row['imdb_score']} "
            f"{row['actor_1_name']} {row['actor_2_name']} {row['actor_3_name']} "
            f"{keywords_str}")

# Create a new column with combined features
df['combined_features'] = df.apply(combine_features, axis=1)

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Create feature matrix
feature_matrix = tfidf.fit_transform(df['combined_features'])

def get_recommendations(movie_titles, num_recommendations=10, include_sequels=True):
    matching_indices = []
    input_genres = set()
    input_actors = set()
    input_directors = set()
    input_years = set()
    input_languages = set()
    input_countries = set()
    input_keywords = set()
    for title in movie_titles:
        exact_match = df[df['movie_title'].str.lower() == title.lower()]
        if not exact_match.empty:
            matching_indices.append(exact_match.index[0])
            input_genres.update(exact_match['genres'].iloc[0].split('|'))
            input_actors.update([
                exact_match['actor_1_name'].iloc[0],
                exact_match['actor_2_name'].iloc[0],
                exact_match['actor_3_name'].iloc[0]
            ])
            input_directors.add(exact_match['director_name'].iloc[0])
            input_years.add(int(exact_match['title_year'].iloc[0]) if pd.notna(exact_match['title_year'].iloc[0]) else None)
            input_languages.add(exact_match['language'].iloc[0])
            input_countries.add(exact_match['country'].iloc[0])
            input_keywords.update(exact_match['plot_keywords'].iloc[0].split('|') if isinstance(exact_match['plot_keywords'].iloc[0], str) else [])

    input_actors = {actor for actor in input_actors if isinstance(actor, str) and actor.strip()}
    input_directors = {director for director in input_directors if isinstance(director, str) and director.strip()}
    input_languages = {lang for lang in input_languages if isinstance(lang, str) and lang.strip()}
    input_countries = {country for country in input_countries if isinstance(country, str) and country.strip()}
    input_years = {year for year in input_years if year is not None}
    
    if len(matching_indices) == 0:
        print(f"No matching movies found for: {movie_titles}")
        print("Available movie titles (first 10):")
        print(df['movie_title'].head(10).tolist())
        return []
    
    print(f"Found {len(matching_indices)} matching movies:")
    print(df.loc[matching_indices, 'movie_title'].tolist())
    
    avg_vector = feature_matrix[matching_indices].mean(axis=0).A1
    similarity_scores = cosine_similarity(avg_vector.reshape(1, -1), feature_matrix)[0]
    
    # Calculate affinity scores
    max_similarity = np.max(similarity_scores)
    affinity_scores = similarity_scores / max_similarity
    
    # Apply genre, actor, director, year, language, country, keyword, rating, budget, gross, and popularity boost
    for idx, score in enumerate(affinity_scores):
        movie_genres = set(df.iloc[idx]['genres'].split('|'))
        genre_overlap = len(input_genres.intersection(movie_genres))
        genre_boost = 1 + (genre_overlap * 0.05)  # 5% boost for each matching genre
        
        # Extra boost for War genre
        if 'War' in input_genres and 'War' in movie_genres:
            genre_boost *= 1.2  # 20% additional boost for War movies
        
        # Actor boost
        movie_actors = {
            df.iloc[idx]['actor_1_name'],
            df.iloc[idx]['actor_2_name'],
            df.iloc[idx]['actor_3_name']
        }
        actor_overlap = len(input_actors.intersection(movie_actors))
        actor_boost = 1 + (actor_overlap * 0.1)  # 10% boost for each matching actor
        
        # Director boost
        director_boost = 1.15 if df.iloc[idx]['director_name'] in input_directors else 1.0
        
        # Year boost (for movies within 10 years before and after the average year of input movies)
        year_boost = 1.0
        movie_year = df.iloc[idx]['title_year']
        if movie_year and input_years:
            avg_input_year = sum(input_years) / len(input_years)
            if abs(movie_year - avg_input_year) <= 10:
                year_boost = 1.2  # 20% boost for movies within 10 years
            elif abs(movie_year - avg_input_year) <= 20:
                year_boost = 1.1  # 10% boost for movies within 20 years
        
        # Language boost
        language_boost = 1.1 if df.iloc[idx]['language'] in input_languages else 1.0
        
        # Country boost
        country_boost = 1.1 if df.iloc[idx]['country'] in input_countries else 1.0
        
        # Keyword boost
        movie_keywords = set(df.iloc[idx]['plot_keywords'].split('|') if isinstance(df.iloc[idx]['plot_keywords'], str) else [])
        keyword_overlap = len(input_keywords.intersection(movie_keywords))
        keyword_boost = 1 + (keyword_overlap * 0.05)  # 5% boost for each matching keyword
        
        # Rating boost
        imdb_score = df.iloc[idx]['imdb_score']
        if imdb_score > 7.5:
            rating_boost = 1.17  # 17% boost for highly-rated movies
        else:
            rating_boost = 1.0
        
        # Budget boost
        budget = df.iloc[idx]['budget']
        budget_boost = 1.0
        if budget > 100000000:  # Adjust this threshold as needed
            budget_boost = 1.1  # 10% boost for high-budget movies
        
        # Gross boost (for popular movies)
        gross = df.iloc[idx]['gross']
        gross_boost = 1.0
        if gross > 200000000:  # Adjust this threshold as needed
            gross_boost = 1.15  # 15% boost for high-grossing movies
        
        # Popularity boost
        num_voted_users = df.iloc[idx]['num_voted_users']
        popularity_boost = 1.0
        if num_voted_users > 500000:  # Adjust this threshold as needed
            popularity_boost = 1.1  # 10% boost for highly voted movies
        
        affinity_scores[idx] *= (genre_boost * actor_boost * director_boost * year_boost * 
                                 language_boost * country_boost * keyword_boost * rating_boost * 
                                 budget_boost * gross_boost * popularity_boost)

    if include_sequels:
        # Boost sequels and prequels
        for idx, score in enumerate(affinity_scores):
            movie_title = df.iloc[idx]['movie_title'].lower()
            for input_title in movie_titles:
                input_title_lower = input_title.lower()
                if movie_title.startswith(input_title_lower) or input_title_lower.startswith(movie_title):
                    affinity_scores[idx] *= 1.2  # 20% boost for sequels and prequels
                    break
    
    # Normalize affinity scores to be between 0 and 1
    affinity_scores = (affinity_scores - np.min(affinity_scores)) / (np.max(affinity_scores) - np.min(affinity_scores))
    
    # Calculate user affinity scores
    max_affinity = np.max(affinity_scores)
    user_affinity_scores = np.zeros_like(affinity_scores)
    for idx, score in enumerate(affinity_scores):
        relative_score = score / max_affinity
        if relative_score > 0.8:
            user_affinity_scores[idx] = 97 + (relative_score - 0.8) * 30  # 97-100
        elif relative_score > 0.6:
            user_affinity_scores[idx] = 90 + (relative_score - 0.6) * 70  # 90-97
        elif relative_score > 0.4:
            user_affinity_scores[idx] = 80 + (relative_score - 0.4) * 100  # 80-90
        elif relative_score > 0.2:
            user_affinity_scores[idx] = 60 + (relative_score - 0.2) * 150  # 60-80
        else:
            user_affinity_scores[idx] = relative_score * 300  # 0-60

    # Sort movies by user affinity score
    movie_affinity = list(enumerate(user_affinity_scores))
    movie_affinity.sort(key=lambda x: x[1], reverse=True)
    
    # Remove input movies from recommendations
    movie_affinity = [item for item in movie_affinity if item[0] not in matching_indices]
    
    if not include_sequels:
        # Remove sequels and prequels from recommendations
        movie_affinity = [item for item in movie_affinity if not any(df.iloc[item[0]]['movie_title'].lower().startswith(title.lower()) or title.lower().startswith(df.iloc[item[0]]['movie_title'].lower()) for title in movie_titles)]
    
    # Get the top N unique recommendations
    seen_titles = set()
    unique_recommendations = []
    for idx, score in movie_affinity:
        title = df.iloc[idx]['movie_title']
        if title not in seen_titles:
            unique_recommendations.append((idx, score))
            seen_titles.add(title)
        if len(unique_recommendations) == num_recommendations:
            break
    
    # Create a list of dictionaries with movie information and affinity score
    columns_to_include = ['movie_title', 'director_name', 'duration', 'actor_1_name', 
                          'actor_2_name', 'actor_3_name', 'genres', 'language', 
                          'country', 'content_rating', 'title_year', 'imdb_score',
                          'budget', 'gross', 'num_voted_users','synopsis']
    
    recommendations = []
    for idx, score in unique_recommendations:
        movie_info = df.iloc[idx][columns_to_include].to_dict()
        movie_info['affinity_score'] = float(affinity_scores[idx]) if not np.isnan(affinity_scores[idx]) else None
        movie_info['user_affinity_score'] = int(score)
        # Replace NaN values with None
        for key, value in movie_info.items():
            if pd.isna(value):
                movie_info[key] = None
        recommendations.append(movie_info)
    
    return recommendations