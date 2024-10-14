from .config import *

import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from transformers import AutoTokenizer

from examples.graph_rag.data_processing.news_processor import download_nltk_data, load_news_data
from examples.graph_rag.data_processing.text_processor import process_text
from examples.graph_rag.graph_operations.neo4j_graph import Neo4jGraphOperations
from examples.graph_rag.graph_operations.graph_data_science import GraphDataScienceOperations
from examples.graph_rag.llm.gbx_model import init_gbx_model, create_llm_transformer
from examples.graph_rag.llm.entity_resolution import create_extraction_chain, entity_resolution
from examples.graph_rag.utils.visualization import (
    plot_token_distribution,
    plot_entity_count_vs_token_count,
    plot_node_degree_distribution
)
from examples.graph_rag.utils.debug import print_community_info, print_summaries, print_percentiles

from ..common import get_bert_mlx_embeddings


def get_embeddings():
    return get_bert_mlx_embeddings()

def initialize_data(debug=False):
    download_nltk_data()
    news = load_news_data()
    if debug:
        plot_token_distribution(news)
    return news

def process_documents(news, llm_transformer, num_articles, debug=False):
    graph_documents = []
    for i, row in tqdm(news.head(num_articles).iterrows(), total=num_articles, desc="Processing documents"):
        try:
            text = f"{row['title']} {row['text']}"
            graph_document = process_text(text, llm_transformer, debug)
            if graph_document:
                graph_documents.extend(graph_document)
        except Exception as e:
            print(f"Error processing document {i}: {str(e)}")
    # if debug:
    print(f"Total graph documents processed: {len(graph_documents)}")
    print(f"Total entities extracted: {sum(len(doc.nodes) for doc in graph_documents)}")
    print(f"Total relationships extracted: {sum(len(doc.relationships) for doc in graph_documents)}")
    return graph_documents

def analyze_entity_distribution(neo4j_ops, debug=False):
    if not debug:
        return
    entity_dist = neo4j_ops.query(
        """
        MATCH (d:Document)
        RETURN d.text AS text,
               count {(d)-[:MENTIONS]->()} AS entity_count
        """
    )
    entity_dist_df = pd.DataFrame.from_records(entity_dist)
    entity_dist_df["token_count"] = [
        len(str(el).split()) for el in entity_dist_df["text"]
    ]
    plot_entity_count_vs_token_count(entity_dist_df)

def analyze_node_degree_distribution(neo4j_ops, debug=False):
    if not debug:
        return
    degree_dist = neo4j_ops.query(
        """
        MATCH (e:__Entity__)
        RETURN count {(e)-[:!MENTIONS]-()} AS node_degree
        """
    )
    degree_dist_df = pd.DataFrame.from_records(degree_dist)
    plot_node_degree_distribution(degree_dist_df)

def perform_entity_resolution(neo4j_ops, chat_gbx, debug=False):
    bert_mlx_embeddings = get_embeddings()
    vector = neo4j_ops.create_vector_store(bert_mlx_embeddings)

    potential_duplicate_candidates = neo4j_ops.query(
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
        """, params={'distance': 3})

    extraction_chain = create_extraction_chain(chat_gbx)

    merged_entities = []

    for el in tqdm(potential_duplicate_candidates, desc="Processing documents"):
        to_merge = entity_resolution(el['combinedResult'], extraction_chain)
        if to_merge:
            merged_entities.extend(to_merge)

    if debug:
        print_summaries(merged_entities[:10])

    # Merge entities in the graph
    neo4j_ops.query("""
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

def perform_community_detection(gds_ops, neo4j_ops, debug=False):
    gds_ops.project_graph(
        "communities",
        "__Entity__",
        {
            "_ALL_": {
                "type": "*",
                "orientation": "UNDIRECTED",
                "properties": {"weight": {"property": "*", "aggregation": "COUNT"}},
            }
        },
    )

    wcc = gds_ops.gds.wcc.stats(gds_ops.G)
    if debug:
        print(f"Component count: {wcc['componentCount']}")
        print(f"Component distribution: {wcc['componentDistribution']}")

    gds_ops.run_leiden(
        write_property="communities",
        include_intermediate_communities=True,
        relationship_weight_property="weight"
    )

    neo4j_ops.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:__Community__) REQUIRE c.id IS UNIQUE;")

    neo4j_ops.query("""
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

    neo4j_ops.query("""
    MATCH (c:__Community__)<-[:IN_COMMUNITY*]-(:__Entity__)<-[:MENTIONS]-(d:Document)
    WITH c, count(distinct d) AS rank
    SET c.community_rank = rank;
    """)

def analyze_community_size(neo4j_ops, debug=False):
    community_size = neo4j_ops.query(
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
    if debug:
        print_percentiles(percentiles_df)

def summarize_communities(neo4j_ops, chat_gbx, debug=False):
    community_info = neo4j_ops.query("""
    MATCH (c:`__Community__`)<-[:IN_COMMUNITY*]-(e:__Entity__)
    WHERE c.level IN [0,1,4]
    WITH c, collect(e) AS nodes
    WHERE size(nodes) > 1
    CALL apoc.path.subgraphAll(nodes[0], {
        whitelistNodes:nodes
    })
    YIELD relationships
    RETURN c.id AS communityId,
           [n in nodes | {id: n.id, description: n.description, type: [el in labels(n) WHERE el <> '__Entity__'][0]}] AS nodes,
           [r in relationships | {start: startNode(r).id, type: type(r), end: endNode(r).id, description: r.description}] AS rels
    """)

    if debug:
        print_community_info(community_info)

    community_template = """Based on the provided nodes and relationships that belong to the same graph community,
    generate a natural language summary of the provided information:
    {community_info}

    Summary:"""

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

    if debug:
        print(prepare_string(community_info[0]))

    summaries = []
    for community in tqdm(community_info, desc="Processing communities"):
        result = process_community(community)
        summaries.append(result)

    # Store summaries
    neo4j_ops.query("""
    UNWIND $data AS row
    MERGE (c:__Community__ {id:row.community})
    SET c.summary = row.summary
    """, params={"data": summaries})

    if debug:
        print_summaries(summaries)
    return summaries

def get_community_summary(neo4j_ops, community_id):
    query = """
    MATCH (c:__Community__ {id: $id})
    RETURN c.id AS id, c.summary AS summary
    """
    result = neo4j_ops.query(query, params={"id": community_id})
    return result[0] if result else None

def get_all_community_summaries(neo4j_ops):
    query = """
    MATCH (c:__Community__)
    RETURN c.id AS id, c.summary AS summary
    """
    return neo4j_ops.query(query)

def search_communities_by_summary(neo4j_ops, keyword):
    query = """
    MATCH (c:__Community__)
    WHERE c.summary CONTAINS $keyword
    RETURN c.id AS id, c.summary AS summary
    """
    return neo4j_ops.query(query, params={"keyword": keyword})

def clear_existing_embeddings(neo4j_ops):
    try:
        # Remove embeddings from entities
        neo4j_ops.query("MATCH (e:__Entity__) REMOVE e.embedding")
        print("Removed embeddings from entities.")

        # Remove embeddings from communities
        neo4j_ops.query("MATCH (c:__Community__) REMOVE c.embedding")
        print("Removed embeddings from communities.")
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")
        print("Continuing with embedding generation...")

def truncate_text(text, max_tokens=512):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens)

def Generate_embeddings_index(neo4j_ops, debug=False, clear_embeddings=False):
    # Step 1: Remove existing embeddings and indexes
    if clear_embeddings:
        neo4j_ops.safe_drop_index_and_embeddings('entity_embeddings', '__Entity__')
        neo4j_ops.safe_drop_index_and_embeddings('community_embeddings', '__Community__')

    # Step 2: Generate and store new embeddings
    embeddings = get_embeddings()
    embedding_dimension = len(embeddings.embed_query("test"))
    if debug:
        print(f"Embedding dimension: {embedding_dimension}")

    # Check and create vector index for entities
    neo4j_ops.create_vector_index('vector', '__Entity__', 'embedding', embedding_dimension)

    # Generate embeddings for entities
    entities = neo4j_ops.query("MATCH (e:__Entity__) WHERE e.embedding IS NULL RETURN e.id AS id, e.name AS name, e.type AS type")
    for entity in tqdm(entities, desc="Generating entity embeddings"):
        try:
            text_to_embed = entity.get('name') or entity['id']
            entity_type = entity.get('type', 'Unknown')
            combined_text = f"Entity ID: {entity['id']}, Type: {entity_type}, Name: {text_to_embed}"
            truncated_text = truncate_text(combined_text)
            embedding = embeddings.embed_query(truncated_text)
            neo4j_ops.query(
                "MATCH (e:__Entity__ {id: $id}) SET e.embedding = $embedding",
                params={"id": entity['id'], "embedding": embedding}
            )
        except Exception as e:
            print(f"Error generating embedding for entity {entity['id']}: {str(e)}")

    # Check and create vector index for communities
    neo4j_ops.create_vector_index('community_embeddings', '__Community__', 'embedding', embedding_dimension)

    # Generate embeddings for communities
    communities = neo4j_ops.query("MATCH (c:__Community__) WHERE c.embedding IS NULL RETURN c.id AS id, c.summary AS summary")
    for community in tqdm(communities, desc="Generating community embeddings"):
        try:
            summary = community.get('summary') or f"Community {community['id']}"
            truncated_summary = truncate_text(summary)
            embedding = embeddings.embed_query(truncated_summary)
            neo4j_ops.query(
                "MATCH (c:__Community__ {id: $id}) SET c.embedding = $embedding",
                params={"id": community['id'], "embedding": embedding}
            )
        except Exception as e:
            print(f"Error generating embedding for community {community['id']}: {str(e)}")

    if debug:
        print("Entity and community embeddings generated and stored.")
        print("Vector indexes created for entities and communities.")

    # Verify index creation
    if debug:
        indexes = neo4j_ops.query("SHOW INDEXES")
        for index in indexes:
            if index['name'] in ['entity_embeddings', 'community_embeddings']:
                print(f"Index: {index['name']}, Type: {index['type']}, Properties: {index['properties']}")


def create_graph_database(neo4j_ops, chat_gbx, num_articles=2000, debug=False):

    # Initialize data and models
    news = initialize_data(debug)
    llm_transformer = create_llm_transformer(chat_gbx, debug)

    # Process documents
    graph_documents = process_documents(news, llm_transformer, num_articles, debug)
    neo4j_ops.add_graph_documents(graph_documents)

    # Analyze entity distribution and node degree
    analyze_entity_distribution(neo4j_ops, debug)
    analyze_node_degree_distribution(neo4j_ops, debug)

    # Perform entity resolution
    perform_entity_resolution(neo4j_ops, chat_gbx, debug)

    # Perform community detection
    gds_ops = GraphDataScienceOperations()
    perform_community_detection(gds_ops, neo4j_ops, debug)

    # Analyze community size
    analyze_community_size(neo4j_ops, debug)

    # Summarize communities
    summarize_communities(neo4j_ops, chat_gbx, debug)

    # Generate embeddings for entities
    Generate_embeddings_index(neo4j_ops, debug, clear_embeddings=False)

    gds_ops.close()


def perform_graph_rag(neo4j_ops, chat_gbx, query, mode='community', num_result=5, debug=False):
    try:
        embeddings = get_embeddings()
        truncated_query = truncate_text(query)
        query_embedding = embeddings.embed_query(truncated_query)
        if debug:
            print(f"Query embedding dimension: {len(query_embedding)}")

        if mode == 'community':
            search_query = """
            CALL db.index.vector.queryNodes('community_embeddings', $k, $query_embedding)
            YIELD node, score
            RETURN node.id AS id, node.summary AS summary, score
            ORDER BY score DESC
            LIMIT $k
            """
        else:  # entity mode
            search_query = """
            CALL db.index.vector.queryNodes('vector', $k, $query_embedding)
            YIELD node, score
            RETURN node.id AS id, node.name AS summary, score
            ORDER BY score DESC
            LIMIT $k
            """

        parameters = {
            "k": num_result,
            "query_embedding": query_embedding
        }

        results = neo4j_ops.query(search_query, params=parameters)

        if debug:
            print(f"Similar {mode}s:", results)

        if not results:
            return f"No relevant {mode}s found for the given query."

        # Prepare context for LLM
        context = f"Query: {query}\n\n"
        context += f"Related {'Communities' if mode == 'community' else 'Entities'}:\n"
        for result in results:
            summary = result.get('summary') or f"No summary available for {mode} {result['id']}"
            truncated_summary = truncate_text(summary)
            context += f"- {result['id']}: {truncated_summary} (Score: {result['score']:.4f})\n"

        # Generate response using LLM
        rag_template = """Based on the following context and query, provide a comprehensive answer:

        Context:
        {context}

        Query: {query}

        Answer:"""

        rag_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an AI assistant with access to a knowledge graph. Provide informative answers based on the given context. If the information is not in the context, say so.",
                ),
                ("human", rag_template),
            ]
        )

        rag_chain = rag_prompt | chat_gbx | StrOutputParser()

        response = rag_chain.invoke({"context": context, "query": query})

        if debug:
            print("Context:", context)

        return response

    except Exception as e:
        error_message = f"An error occurred during RAG processing: {str(e)}"
        print(error_message)
        return error_message


def main():
    parser = argparse.ArgumentParser(description="Run graph operations with multiple modes")
    parser.add_argument("--mode", choices=["create", "query", "rag"], required=True, help="Operation mode")
    parser.add_argument("--community_id", help="Community ID for summary query")
    parser.add_argument("--keyword", help="Keyword for community search")
    parser.add_argument("--rag_query", help="Query for GraphRAG retrieval")
    parser.add_argument("--rag_mode", choices=["community", "entity"], default="community", help="RAG retrieval mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum number of tokens for LLM output")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for LLM")
    parser.add_argument("--num_articles", type=int, default=2000, help="Number of articles to process")
    parser.add_argument("--model_id", type=str, default= "GreenBitAI/Llama-3-8B-instruct-layer-mix-bpw-4.0-mlx", help="huggingFace model id")
    args = parser.parse_args()

    neo4j_ops = Neo4jGraphOperations()
    chat_gbx = init_gbx_model(args.model_id, max_tokens=args.max_tokens, temperature=args.temperature)

    try:
        if args.mode == "create":
            create_graph_database(neo4j_ops, chat_gbx, args.num_articles, args.debug)
            print("Graph database created successfully.")
        elif args.mode == "query":
            if args.community_id:
                result = get_community_summary(neo4j_ops, args.community_id)
                print(f"Community summary for ID {args.community_id}:", result)
            elif args.keyword:
                results = search_communities_by_summary(neo4j_ops, args.keyword)
                print(f"Communities matching keyword '{args.keyword}':", results)
            else:
                results = get_all_community_summaries(neo4j_ops)
                print("All community summaries:", results)
        elif args.mode == "rag":
            if args.rag_query:
                response = perform_graph_rag(neo4j_ops, chat_gbx, args.rag_query, mode=args.rag_mode,
                                             num_result=10, debug=args.debug)
                print("GraphRAG Response:", response)
            else:
                print("Please provide a query for GraphRAG retrieval using --rag_query")

        # if args.debug:
        #     neo4j_ops.print_graph_details()

    finally:
        neo4j_ops.close()

if __name__ == "__main__":
    main()