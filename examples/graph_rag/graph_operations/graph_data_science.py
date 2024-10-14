from graphdatascience import GraphDataScience
import os

class GraphDataScienceOperations:
    def __init__(self):
        self.gds = GraphDataScience(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
        )
        self.G = None

    def project_graph(self, name, node_projection, relationship_projection, node_properties=None):
        self.gds.graph.drop(name)  # Remove existing graph if any
        self.G, result = self.gds.graph.project(
            name,
            node_projection,
            relationship_projection,
            nodeProperties=node_properties
        )
        return result

    def run_knn(self, node_properties, relationship_type, property_name, similarity_cutoff):
        return self.gds.knn.mutate(
            self.G,
            nodeProperties=node_properties,
            mutateRelationshipType=relationship_type,
            mutateProperty=property_name,
            similarityCutoff=similarity_cutoff
        )

    def run_wcc(self, write_property, relationship_types):
        return self.gds.wcc.write(
            self.G,
            writeProperty=write_property,
            relationshipTypes=relationship_types
        )

    def run_leiden(self, write_property, include_intermediate_communities, relationship_weight_property):
        return self.gds.leiden.write(
            self.G,
            writeProperty=write_property,
            includeIntermediateCommunities=include_intermediate_communities,
            relationshipWeightProperty=relationship_weight_property
        )

    def close(self):
        if self.G:
            self.G.drop()
