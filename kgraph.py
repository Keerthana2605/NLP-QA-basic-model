import os
import subprocess
import time
from pathlib import Path

from haystack.nodes import Text2SparqlRetriever
from haystack.document_stores import GraphDBKnowledgeGraph, InMemoryKnowledgeGraph
from haystack.utils import fetch_archive_from_http

def tutorial10_knowledge_graph():
    # Let's first fetch some triples that we want to store in our knowledge graph
    # Here: exemplary triples from the wizarding world
    graph_dir = "data/tutorial10/"
    s3_url = "https://fandom-qa.s3-eu-west-1.amazonaws.com/triples_and_config.zip"
    fetch_archive_from_http(url=s3_url, output_dir=graph_dir)

    # Fetch a pre-trained BART model that translates text queries to SPARQL queries
    model_dir = "../saved_models/tutorial10_knowledge_graph/"
    s3_url = "https://fandom-qa.s3-eu-west-1.amazonaws.com/saved_models/hp_v3.4.zip"
    fetch_archive_from_http(url=s3_url, output_dir=model_dir)

    # Initialize a in memory knowledge graph and use "tutorial_10_index" as the name of the index
    kg = InMemoryKnowledgeGraph(index="tutorial_10_index")
    # Delete the index as it might have been already created in previous runs
    kg.delete_index()
    # Create the index
    kg.create_index()
    # Import triples of subject, predicate, and object statements from a ttl file
    kg.import_from_ttl_file(index="tutorial_10_index", path=Path(graph_dir) / "triples.ttl")
    print(f"The last triple stored in the knowledge graph is: {kg.get_all_triples()[-1]}")
    print(f"There are {len(kg.get_all_triples())} triples stored in the knowledge graph.")

    # Load a pre-trained model that translates text queries to SPARQL queries
    kgqa_retriever = Text2SparqlRetriever(knowledge_graph=kg, model_name_or_path=model_dir + "hp_v3.4") 

    query = "In which house is Harry Potter?"
    print(f'Translating the text query "{query}" to a SPARQL query and executing it on the knowledge graph...')
    result = kgqa_retriever.retrieve(query=query)
    print(result)
    # Correct SPARQL query: select ?a { hp:Harry_potter hp:house ?a . }
    # Correct answer: Gryffindor

    print("Executing a SPARQL query with prefixed names of resources...")
    result = kgqa_retriever._query_kg(
        sparql_query="select distinct ?sbj where { ?sbj hp:job hp:Keeper_of_keys_and_grounds . }"
    )
    print(result)
    # Paraphrased question: Who is the keeper of keys and grounds?
    # Correct answer: Rubeus Hagrid

    print("Executing a SPARQL query with full names of resources...")
    result = kgqa_retriever._query_kg(
        sparql_query="select distinct ?obj where { <https://deepset.ai/harry_potter/Hermione_granger> <https://deepset.ai/harry_potter/patronus> ?obj . }"
    )
    print(result)
    # Paraphrased question: What is the patronus of Hermione?
    # Correct answer: Otter

if __name__ == "__main__":
    tutorial10_knowledge_graph()