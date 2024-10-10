import os
import sys

from langchain_community.graphs import Neo4jGraph
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

from graphdatascience import GraphDataScience

from gbx_lm.langchain.chat_gbx import ChatGBX
from gbx_lm.langchain import GBXPipeline
from gbx_lm.langchain import SimpleGraphTransformer
from gbx_lm.langchain.examples.common import get_bert_mlx_embeddings

from typing import List
from langchain_community.graphs.graph_document import GraphDocument
from langchain_core.documents import Document

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import numpy as np

from langchain_community.vectorstores import Neo4jVector

MODEL_ID = "GreenBitAI/Llama-3-8B-instruct-layer-mix-bpw-4.0-mlx"
MAX_TOKENS = 100
TEMPERATURE = 0
MAX_WORKERS = 10
NUM_ARTICLES = 1

# download and install neo4j desktop version from https://neo4j.com/download/
# install APOC and Graph Data Science Plugins


# Setting Up the Neo4j Environment
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "bonanza-fax-1342"

graph = Neo4jGraph(refresh_schema=True)

# download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    print("Punkt tokenizer data is already downloaded.")
except LookupError:
    print("Punkt tokenizer data not found. Downloading it is necessary.")
    nltk.download('punkt_tab')


def num_tokens_from_string(string: str) -> int:
    tokens = word_tokenize(string)
    return len(tokens)

# dataset
news = pd.read_csv(
    "https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv"
)

news["tokens"] = [
    num_tokens_from_string(f"{row['title']} {row['text']}")
    for i, row in news.iterrows()
]

print(news.head())

# # Text Chunking
#
import seaborn as sns
import matplotlib.pyplot as plt

# Debug Info
# sns.histplot(news["tokens"], kde=False)
# plt.title('Distribution of chunk sizes')
# plt.xlabel('Token count')
# plt.ylabel('Frequency')
# plt.show()

# Initialize GBX model
def init_gbx_model():
    llm = GBXPipeline.from_model_id(
        model_id=MODEL_ID,
        pipeline_kwargs={"max_tokens": MAX_TOKENS, "temp": TEMPERATURE}
    )
    return ChatGBX(llm=llm)

chat_gbx = init_gbx_model()

# Expanded list of allowed nodes
allowed_nodes = [
    "Person", "Organization", "Location", "Event", "Date", "Number",
    "Product", "Concept", "Technology", "Document", "Topic",
    "Project", "Process", "Law", "Award", "Industry",
    "Animal", "Plant", "Disease", "Drug", "CelestialBody"
]

# Expanded list of allowed relationships
allowed_relationships = [
    "ASSOCIATED_WITH", "LOCATED_IN", "WORKS_FOR", "OCCURRED_ON", "HAS_VALUE",
    "FOUNDED", "OWNS", "PRODUCES", "PARTICIPATES_IN", "LEADS",
    "BELONGS_TO", "COLLABORATES_WITH", "INFLUENCES", "SUPPORTS",
    "OPPOSES", "STUDIED_BY", "CREATED_BY", "CAUSED_BY", "RESULTED_IN",
    "PART_OF", "PRECEDES", "SUCCEEDS", "REGULATES", "TREATS",
    "SYMBOLIZES", "CLASSIFIES", "RELATED_TO"
]

llm_transformer = SimpleGraphTransformer(
    llm=chat_gbx,
    # allowed_nodes=allowed_nodes,
    # allowed_relationships=allowed_relationships,
    node_properties=["description"],
    relationship_properties=["description"],
    strict_mode=False
)


def process_text(text: str) -> List[GraphDocument]:
    doc = Document(page_content=text)
    try:
        result = llm_transformer.convert_to_graph_documents([doc])
        if not result or not result[0].nodes:
            print(f"Warning: No entities extracted from text: {text[:100]}...")
        else:
            print(f"Extracted {len(result[0].nodes)} entities and {len(result[0].relationships)} relationships from text: {text[:100]}...")
        return result
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return []

graph_documents = []

# with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#     futures = [
#         executor.submit(process_text, f"{row['title']} {row['text']}")
#         for i, row in news.head(NUM_ARTICLES).iterrows()
#     ]
#
#     for future in tqdm(
#         as_completed(futures), total=len(futures), desc="Processing documents"
#     ):
#         try:
#             graph_document = future.result()
#             if graph_document:
#                 graph_documents.extend(graph_document)
#         except Exception as e:
#             print(f"Error in future: {str(e)}")


for i, row in tqdm(news.head(NUM_ARTICLES).iterrows(), total=NUM_ARTICLES, desc="Processing documents"):
    try:
        text = f"{row['title']} {row['text']}"
        graph_document = process_text(text)
        if graph_document:
            graph_documents.extend(graph_document)
    except Exception as e:
        print(f"Error processing document {i}: {str(e)}")

print(f"Total graph documents processed: {len(graph_documents)}")
print(f"Total entities extracted: {sum(len(doc.nodes) for doc in graph_documents)}")
print(f"Total relationships extracted: {sum(len(doc.relationships) for doc in graph_documents)}")


graph.add_graph_documents(
    graph_documents,
    baseEntityLabel=True,
    include_source=True
)

# In this example, we extract graph information from 2,000 articles and store results to Neo4j.
# We have extracted around 13,000 entities and 16,000 relationships.
# Here is an example of an extracted document in the graph.
#
entity_dist = graph.query(
    """
MATCH (d:Document)
RETURN d.text AS text,
       count {(d)-[:MENTIONS]->()} AS entity_count
"""
)
entity_dist_df = pd.DataFrame.from_records(entity_dist)
entity_dist_df["token_count"] = [
    num_tokens_from_string(str(el)) for el in entity_dist_df["text"]
]
# Scatter plot with regression line
sns.lmplot(
    x="token_count", y="entity_count", data=entity_dist_df, line_kws={"color": "red"}
)
plt.title("Entity Count vs Token Count Distribution")
plt.xlabel("Token Count")
plt.ylabel("Entity Count")
plt.show()

degree_dist = graph.query(
    """
MATCH (e:__Entity__)
RETURN count {(e)-[:!MENTIONS]-()} AS node_degree
"""
)
degree_dist_df = pd.DataFrame.from_records(degree_dist)

# Calculate mean and median
mean_degree = np.mean(degree_dist_df['node_degree'])
percentiles = np.percentile(degree_dist_df['node_degree'], [25, 50, 75, 90])
# Create a histogram with a logarithmic scale
plt.figure(figsize=(12, 6))
sns.histplot(degree_dist_df['node_degree'], bins=50, kde=False, color='blue')
# Use a logarithmic scale for the x-axis
plt.yscale('log')
# Adding labels and title
plt.xlabel('Node Degree')
plt.ylabel('Count (log scale)')
plt.title('Node Degree Distribution')
# Add mean, median, and percentile lines
plt.axvline(mean_degree, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_degree:.2f}')
plt.axvline(percentiles[0], color='purple', linestyle='dashed', linewidth=1, label=f'25th Percentile: {percentiles[0]:.2f}')
plt.axvline(percentiles[1], color='orange', linestyle='dashed', linewidth=1, label=f'50th Percentile: {percentiles[1]:.2f}')
plt.axvline(percentiles[2], color='yellow', linestyle='dashed', linewidth=1, label=f'75th Percentile: {percentiles[2]:.2f}')
plt.axvline(percentiles[3], color='brown', linestyle='dashed', linewidth=1, label=f'90th Percentile: {percentiles[3]:.2f}')
# Add legend
plt.legend()
# Show the plot
plt.show()



graph.query("""
MATCH (n:`__Entity__`)
RETURN "node" AS type,
       count(*) AS total_count,
       count(n.description) AS non_null_descriptions
UNION ALL
MATCH (n)-[r:!MENTIONS]->()
RETURN "relationship" AS type,
       count(*) AS total_count,
       count(r.description) AS non_null_descriptions
""")


# Entity Resolution
bert_mlx_embeddings = get_bert_mlx_embeddings()
vector = Neo4jVector.from_existing_graph(
    bert_mlx_embeddings,
    node_label='__Entity__',
    text_node_properties=['id', 'description'],
    embedding_node_property='embedding'
)

# project graph

gds = GraphDataScience(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
)

# remove existing graph
gds.graph.drop("entities")

G, result = gds.graph.project(
    "entities",  # Graph name
    "__Entity__",  # Node projection
    "*",  # Relationship projection
    nodeProperties=["embedding"]  # Configuration parameters
)

similarity_threshold = 0.95

gds.knn.mutate(
  G,
  nodeProperties=['embedding'],
  mutateRelationshipType= 'SIMILAR',
  mutateProperty= 'score',
  similarityCutoff=similarity_threshold
)


gds.wcc.write(
    G,
    writeProperty="wcc",
    relationshipTypes=["SIMILAR"]
)

word_edit_distance = 3
potential_duplicate_candidates = graph.query(
    """MATCH (e:`__Entity__`)
    WHERE size(e.id) > 4 // longer than 4 characters
    WITH e.wcc AS community, collect(e) AS nodes, count(*) AS count
    WHERE count > 1
    UNWIND nodes AS node
    // Add text distance
    WITH distinct
      [n IN nodes WHERE apoc.text.distance(toLower(node.id), toLower(n.id)) < $distance | n.id] AS intermediate_results
    WHERE size(intermediate_results) > 1
    WITH collect(intermediate_results) AS results
    // combine groups together if they share elements
    UNWIND range(0, size(results)-1, 1) as index
    WITH results, index, results[index] as result
    WITH apoc.coll.sort(reduce(acc = result, index2 IN range(0, size(results)-1, 1) |
            CASE WHEN index <> index2 AND
                size(apoc.coll.intersection(acc, results[index2])) > 0
                THEN apoc.coll.union(acc, results[index2])
                ELSE acc
            END
    )) as combinedResult
    WITH distinct(combinedResult) as combinedResult
    // extra filtering
    WITH collect(combinedResult) as allCombinedResults
    UNWIND range(0, size(allCombinedResults)-1, 1) as combinedResultIndex
    WITH allCombinedResults[combinedResultIndex] as combinedResult, combinedResultIndex, allCombinedResults
    WHERE NOT any(x IN range(0,size(allCombinedResults)-1,1)
        WHERE x <> combinedResultIndex
        AND apoc.coll.containsAll(allCombinedResults[x], combinedResult)
    )
    RETURN combinedResult
    """, params={'distance': word_edit_distance})
potential_duplicate_candidates[:5]


system_prompt = """You are a data processing assistant. Your task is to identify duplicate entities in a list and decide which of them should be merged.
The entities might be slightly different in format or content, but essentially refer to the same thing. Use your analytical skills to determine duplicates.

Here are the rules for identifying duplicates:
1. Entities with minor typographical differences should be considered duplicates.
2. Entities with different formats but the same content should be considered duplicates.
3. Entities that refer to the same real-world object or concept, even if described differently, should be considered duplicates.
4. If it refers to different numbers, dates, or products, do not merge results
"""
user_template = """
Here is the list of entities to process:
{entities}

Please identify duplicates, merge them, and provide the merged list.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from retry import retry

class DuplicateEntities(BaseModel):
    entities: List[str] = Field(
        description="Entities that represent the same object or real-world entity and should be merged"
    )


class Disambiguate(BaseModel):
    merge_entities: Optional[List[DuplicateEntities]] = Field(
        description="Lists of entities that represent the same object or real-world entity and should be merged"
    )


extraction_llm = chat_gbx.with_structured_output(
    Disambiguate
)

extraction_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        (
            "human",
            user_template,
        ),
    ]
)

extraction_chain = extraction_prompt | extraction_llm

@retry(tries=3, delay=2)
def entity_resolution(entities):
    try:
        result = extraction_chain.invoke({"entities": entities})
        if result is None:
            print("Extraction chain returned None")
            return []

        if not hasattr(result, 'merge_entities'):
            print(f"Unexpected result type: {type(result)}")
            print(f"Result content: {result}")
            return []

        return list(result.merge_entities)
    except Exception as e:
        print(f"Error in entity_resolution: {str(e)}")
        return []


# Usage
resolved_entities = entity_resolution(['Star Ocean The Second Story R', 'Star Ocean: The Second Story R'])
print(f"Resolved entities: {resolved_entities}")

entity_resolution({"entities": ['December 16, 2023',
   'December 2, 2023',
   'December 23, 2023',
   'December 26, 2023',
   'December 30, 2023',
   'December 5, 2023',
   'December 9, 2023']})

merged_entities = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # Submitting all tasks and creating a list of future objects
    futures = [
        executor.submit(entity_resolution, el['combinedResult'])
        for el in potential_duplicate_candidates
    ]

    for future in tqdm(
        as_completed(futures), total=len(futures), desc="Processing documents"
    ):
        to_merge = future.result()
        if to_merge:
            merged_entities.extend(to_merge)

print(merged_entities[:10])


graph.query("""
UNWIND $data AS candidates
CALL {
  WITH candidates
  MATCH (e:__Entity__) WHERE e.id IN candidates
  RETURN collect(e) AS nodes
}
CALL apoc.refactor.mergeNodes(nodes, {properties: {
    `.*`: 'discard'
}})
YIELD node
RETURN count(*)
""", params={"data": merged_entities})


G.drop()

# Element Summarization
# remove if existing
gds.graph.drop("communities")

G, result = gds.graph.project(
    "communities",  # Graph name
    "__Entity__",  # Node projection
    {
        "_ALL_": {
            "type": "*",
            "orientation": "UNDIRECTED",
            "properties": {"weight": {"property": "*", "aggregation": "COUNT"}},
        }
    },
)

wcc = gds.wcc.stats(G)
print(f"Component count: {wcc['componentCount']}")
print(f"Component distribution: {wcc['componentDistribution']}")

gds.leiden.write(
    G,
    writeProperty="communities",
    includeIntermediateCommunities=True,
    relationshipWeightProperty="weight",
)

graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:__Community__) REQUIRE c.id IS UNIQUE;")


graph.query("""
MATCH (e:`__Entity__`)
UNWIND range(0, size(e.communities) - 1 , 1) AS index
CALL {
  WITH e, index
  WITH e, index
  WHERE index = 0
  MERGE (c:`__Community__` {id: toString(index) + '-' + toString(e.communities[index])})
  ON CREATE SET c.level = index
  MERGE (e)-[:IN_COMMUNITY]->(c)
  RETURN count(*) AS count_0
}
CALL {
  WITH e, index
  WITH e, index
  WHERE index > 0
  MERGE (current:`__Community__` {id: toString(index) + '-' + toString(e.communities[index])})
  ON CREATE SET current.level = index
  MERGE (previous:`__Community__` {id: toString(index - 1) + '-' + toString(e.communities[index - 1])})
  ON CREATE SET previous.level = index - 1
  MERGE (previous)-[:IN_COMMUNITY]->(current)
  RETURN count(*) AS count_1
}
RETURN count(*)
""")

graph.query("""
MATCH (c:__Community__)<-[:IN_COMMUNITY*]-(:__Entity__)<-[:MENTIONS]-(d:Document)
WITH c, count(distinct d) AS rank
SET c.community_rank = rank;
""")

community_size = graph.query(
    """
MATCH (c:__Community__)<-[:IN_COMMUNITY*]-(e:__Entity__)
WITH c, count(distinct e) AS entities
RETURN split(c.id, '-')[0] AS level, entities
"""
)
community_size_df = pd.DataFrame.from_records(community_size)
percentiles_data = []
for level in community_size_df["level"].unique():
    subset = community_size_df[community_size_df["level"] == level]["entities"]
    num_communities = len(subset)
    percentiles = np.percentile(subset, [25, 50, 75, 90, 99])
    percentiles_data.append(
        [
            level,
            num_communities,
            percentiles[0],
            percentiles[1],
            percentiles[2],
            percentiles[3],
            percentiles[4],
            max(subset)
        ]
    )

# Create a DataFrame with the percentiles
percentiles_df = pd.DataFrame(
    percentiles_data,
    columns=[
        "Level",
        "Number of communities",
        "25th Percentile",
        "50th Percentile",
        "75th Percentile",
        "90th Percentile",
        "99th Percentile",
        "Max"
    ],
)
print(percentiles_df)

community_info = graph.query("""
MATCH (c:`__Community__`)<-[:IN_COMMUNITY*]-(e:__Entity__)
WHERE c.level IN [0,1,4]
WITH c, collect(e ) AS nodes
WHERE size(nodes) > 1
CALL apoc.path.subgraphAll(nodes[0], {
	whitelistNodes:nodes
})
YIELD relationships
RETURN c.id AS communityId,
       [n in nodes | {id: n.id, description: n.description, type: [el in labels(n) WHERE el <> '__Entity__'][0]}] AS nodes,
       [r in relationships | {start: startNode(r).id, type: type(r), end: endNode(r).id, description: r.description}] AS rels
""")

print(community_info[0])


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

community_template = """Based on the provided nodes and relationships that belong to the same graph community,
generate a natural language summary of the provided information:
{community_info}

Summary:"""  # noqa: E501

community_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given an input triples, generate the information summary. No pre-amble.",
        ),
        ("human", community_template),
    ]
)

community_chain = community_prompt | chat_gbx | StrOutputParser()

def prepare_string(data):
    nodes_str = "Nodes are:\n"
    for node in data['nodes']:
        node_id = node['id']
        node_type = node['type']
        if 'description' in node and node['description']:
            node_description = f", description: {node['description']}"
        else:
            node_description = ""
        nodes_str += f"id: {node_id}, type: {node_type}{node_description}\n"

    rels_str = "Relationships are:\n"
    for rel in data['rels']:
        start = rel['start']
        end = rel['end']
        rel_type = rel['type']
        if 'description' in rel and rel['description']:
            description = f", description: {rel['description']}"
        else:
            description = ""
        rels_str += f"({start})-[:{rel_type}]->({end}){description}\n"

    return nodes_str + "\n" + rels_str

def process_community(community):
    stringify_info = prepare_string(community)
    summary = community_chain.invoke({'community_info': stringify_info})
    return {"community": community['communityId'], "summary": summary}


print(prepare_string(community_info[0]))


summaries = []
with ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_community, community): community for community in community_info}

    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing communities"):
        summaries.append(future.result())


# Store summaries
graph.query("""
UNWIND $data AS row
MERGE (c:__Community__ {id:row.community})
SET c.summary = row.summary
""", params={"data": summaries})

print(summaries)


def print_graph_details(graph):
    # Query to get all nodes
    node_query = """
    MATCH (n)
    RETURN DISTINCT labels(n) AS labels, 
           COUNT(n) AS count, 
           COLLECT(DISTINCT KEYS(n))[0] AS properties
    """

    # Query to get all relationships
    rel_query = """
    MATCH ()-[r]->()
    RETURN DISTINCT type(r) AS type, 
           COUNT(r) AS count, 
           COLLECT(DISTINCT KEYS(r))[0] AS properties
    """

    # Execute queries
    nodes = graph.query(node_query)
    relationships = graph.query(rel_query)

    # Print node information
    print("Nodes in the graph:")
    for node in nodes:
        labels = ', '.join(node['labels'])
        print(f"  Label(s): {labels}")
        print(f"  Count: {node['count']}")
        print(f"  Properties: {', '.join(node['properties'])}")
        print()

    # Print relationship information
    print("Relationships in the graph:")
    for rel in relationships:
        print(f"  Type: {rel['type']}")
        print(f"  Count: {rel['count']}")
        print(f"  Properties: {', '.join(rel['properties'])}")
        print()

# print_graph_details(graph)
gds._driver.close()
graph._driver.close()