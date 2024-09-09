from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
from fastapi.responses import JSONResponse
from script import get_recommendations
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import unquote

app = FastAPI()


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class MovieRequest(BaseModel):
    titles: List[str]

@app.get("/")
async def root():
    return {"message": "Welcome to the Movie Recommendation API"}

@app.get("/index")
async def index():
    # Load the movie data
    df = pd.read_csv('./movie_metadata.csv')

    # Sort by imdb_score in descending order and get top 100
    top_100 = df.sort_values('imdb_score', ascending=False).head(100)

    # Select 20 random movies from the top 100
    random_20 = top_100.sample(n=20)

    # Function to convert problematic values to JSON-compatible format
    def convert_to_json_compatible(val):
        if pd.isna(val):
            return None
        if isinstance(val, (np.integer, np.floating)):
            return val.item()
        if isinstance(val, str):
            return val
        return str(val)

    # Apply the conversion function to all elements
    json_compatible_data = random_20.applymap(convert_to_json_compatible).to_dict(orient='records')

    # Use JSONResponse to ensure proper JSON serialization
    return JSONResponse(content=json_compatible_data)

@app.get("/search")
async def fetch_by_name(name: str):
    # Load the movie data
    df = pd.read_csv('./movie_metadata.csv')

    # Convert search term and movie titles to lowercase
    name_lower = name.lower()
    df['movie_title_lower'] = df['movie_title'].str.lower()

    # Create masks for different priority levels
    exact_match = df['movie_title_lower'] == name_lower
    starts_with = df['movie_title_lower'].str.startswith(name_lower)
    contains = df['movie_title_lower'].str.contains(name_lower, na=False)

    # Combine results with priority
    matching_movies = pd.concat([
        df[exact_match],
        df[starts_with & ~exact_match],
        df[contains & ~starts_with & ~exact_match]
    ])

    # Sort the matching movies
    def sort_key(title):
        title_lower = title.lower()
        if title_lower == name_lower:
            return (0, 0)  # Exact match gets highest priority
        if title_lower.startswith(name_lower):
            # Extract sequel number if present
            remaining = title_lower[len(name_lower):].strip()
            if remaining.isdigit():
                return (1, int(remaining))
            elif not remaining:  # This is the first movie without a number
                return (1, 0)
            else:
                return (1, float('inf'))
        return (2, 0)  # Contains match gets lowest priority

    matching_movies = matching_movies.sort_values('movie_title', key=lambda x: x.map(sort_key))

    # Function to convert problematic values to JSON-compatible format
    def convert_to_json_compatible(val):
        if pd.isna(val):
            return None
        if isinstance(val, (np.integer, np.floating)):
            return val.item()
        if isinstance(val, str):
            return val
        return str(val)

    # Apply the conversion function to all elements
    json_compatible_data = matching_movies.applymap(convert_to_json_compatible).to_dict(orient='records')

    # Use JSONResponse to ensure proper JSON serialization
    return JSONResponse(content=json_compatible_data)


@app.get("/recommend")
async def recommend_movies(titles: str, include_sequels: bool = True):
    # Decode the URL-encoded titles and remove trailing non-breaking spaces
    decoded_titles = unquote(titles).replace('\xa0', '').strip()
    movie_titles = [title.strip() for title in decoded_titles.split(',')]
    recommendations = get_recommendations(movie_titles, include_sequels=include_sequels)
    return {"recommendations": recommendations}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


