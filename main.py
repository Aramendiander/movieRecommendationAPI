from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from script import get_recommendations

app = FastAPI()

class MovieRequest(BaseModel):
    titles: List[str]

@app.get("/")
async def root():
    return {"message": "Welcome to the Movie Recommendation API"}

@app.get("/recommend")
async def recommend_movies(titles: str, include_sequels: bool = True):
    movie_titles = titles.split(',')
    recommendations = get_recommendations(movie_titles, include_sequels=include_sequels)
    return {"recommendations": recommendations}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

""" recommendations = get_recommendations(['Toy Story', 'Shrek', 'Spirited Away'])

print("Recommendations:")
for movie, affinity in recommendations:
    print(f"{movie} (Affinity: {affinity:.4f})") """

