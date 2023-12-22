from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.chains import RetrievalQA

import os
from dotenv import load_dotenv

# project modules
from llm import llm, embeddings

# load env variables using dotenv
load_dotenv()

# Neo4j Vector Storage for the embeddings
neo4jvector = Neo4jVector.from_existing_index(
    embeddings,                              
    url=os.getenv("NEO4J_URI"),             
    username=os.getenv("NEO4J_USERNAME"),   
    password=os.getenv("NEO4J_PASSWORD"),  
    index_name="movieTaglines",                
    node_label="Movie",                      
    text_node_property="tagline",              
    embedding_node_property="embedding",
    retrieval_query="""
                    RETURN
                        node.tagline AS text,
                        score,
                        {
                            title: node.title,
                            directors: [ (person)-[:DIRECTED]->(node) | person.name ],
                            actors: [ (person)-[r:ACTED_IN]->(node) | [person.name, r.roles] ],
                            writers: [ (person)-[:WROTE]->(node) | person.name ],
                            reviewers: [ (person)-[:REVIEWED]->(node) | person.name ],
                            rating: [ (person)-[r:REVIEWED]->(node) | r.rating ],
                            reviewer_summary: [ (person)-[r:REVIEWED]->(node) | r.summary ],
                            producers: [ (person)-[:PRODUCED]->(node) | person.name ]
                        } AS metadata
                    """
)

# Returns the vector storage as a retriever
retriever = neo4jvector.as_retriever()

# Chain to user the retriever to pass documents to the LLM
# "stuff" inserts documents into the prompt
kg_qa = RetrievalQA.from_chain_type(
    llm,                  
    chain_type="stuff",   
    retriever=retriever,
    # verbose=True,
    return_source_documents=True  
)
