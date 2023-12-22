import os
import openai
from neo4j import GraphDatabase, Result
import pandas as pd
from openai import OpenAI, APIError
from time import sleep
from dotenv import load_dotenv

# load env variables using dotenv
load_dotenv()

class MovieVectorSearchIndexConfiguration:
    """
    Class used to create the embeddings from movie taglines
    Used to create an index in Neo4j
    Done before utilizing the chatbot
    """

    def __init__(self):
        """
        Connect to the database
        """
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )
        self.driver.verify_connectivity()

    def collect_movies(self, limit = None):
        """
        Collect all movies that have a tagline
        """
        query = """MATCH (m:Movie) WHERE m.tagline IS NOT NULL
                RETURN ID(m) AS movieId, m.title AS title, m.tagline AS tagline"""

        if limit is not None:
            query += f" LIMIT {limit}"

        movies = self.driver.execute_query(
            query,
            result_transformer_=Result.to_df
        )

        print(len(movies))

        return movies
    
    def generate_embeddings(self, file_name, movies):
        """
        Create embeddings for the movie taglines using OpenAI text-embedding-ada-002
        """
        openai.api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI()
        
        embeddings = []
        for _, n in movies.iterrows():
        
            successful_call = False
            while not successful_call:
                try:
                    res = client.embeddings.create(
                        model="text-embedding-ada-002",
                        input=f"{n['title']}: {n['tagline']}",
                        encoding_format="float")
                    successful_call = True
                except APIError as e:
                    print(e)
                    print("Retrying in 5 seconds...")
                    sleep(5)

            print(n['title'])

            embeddings.append({"movieId": n['movieId'], "embedding": res.data[0].embedding})

        embedding_df = pd.DataFrame(embeddings)
        embedding_df.head()
        embedding_df.to_csv(file_name, index=False)

#Run
config = MovieVectorSearchIndexConfiguration()
config.generate_embeddings('openai-embeddings-full.csv', config.collect_movies())
print('Embeddings Created!')
