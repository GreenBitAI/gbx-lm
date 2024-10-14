from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector


class Neo4jGraphOperations:
    def __init__(self):
        self.graph = Neo4jGraph(refresh_schema=True)

    def add_graph_documents(self, graph_documents):
        self.graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True
        )

    def query(self, query_string, params=None):
        return self.graph.query(query_string, params=params)

    def create_vector_store(self, embeddings):
        return Neo4jVector.from_existing_graph(
            embeddings,
            node_label='__Entity__',
            text_node_properties=['id', 'description'],
            embedding_node_property='embedding'
        )

    def print_graph_details(self):
        node_query = """
        MATCH (n)
        RETURN DISTINCT labels(n) AS labels, 
               COUNT(n) AS count, 
               COLLECT(DISTINCT KEYS(n))[0] AS properties
        """
        rel_query = """
        MATCH ()-[r]->()
        RETURN DISTINCT type(r) AS type, 
               COUNT(r) AS count, 
               COLLECT(DISTINCT KEYS(r))[0] AS properties
        """
        nodes = self.query(node_query)
        relationships = self.query(rel_query)

        print("Nodes in the graph:")
        for node in nodes:
            labels = ', '.join(node['labels'])
            print(f"  Label(s): {labels}")
            print(f"  Count: {node['count']}")
            print(f"  Properties: {', '.join(node['properties'])}")
            print()

        print("Relationships in the graph:")
        for rel in relationships:
            print(f"  Type: {rel['type']}")
            print(f"  Count: {rel['count']}")
            print(f"  Properties: {', '.join(rel['properties'])}")
            print()

    def safe_drop_index_and_embeddings(self, index_name, node_label):
        try:
            self.query(f"DROP INDEX {index_name} IF EXISTS")
            print(f"Index {index_name} dropped successfully (if it existed).")
        except Exception as e:
            print(f"Note: Could not drop index {index_name}. It might not exist or the command is not supported: {str(e)}")

        try:
            self.query(f"MATCH (n:{node_label}) REMOVE n.embedding")
            print(f"Embeddings removed from {node_label} nodes.")
        except Exception as e:
            print(f"Error removing embeddings from {node_label} nodes: {str(e)}")

    def create_vector_index(self, index_name, node_label, property_name, dimension):

        self.safe_drop_index_and_embeddings(index_name, node_label)

        create_query = f"""
        CALL db.index.vector.createNodeIndex(
          '{index_name}',
          '{node_label}',
          '{property_name}',
          {dimension},
          'cosine'
        )
        """
        try:
            self.query(create_query)
            print(f"Vector index {index_name} created successfully.")
        except Exception as e:
            print(f"Error creating vector index: {str(e)}")
            print("Proceeding without vector index. This may affect performance.")

    def close(self):
        self.graph._driver.close()

