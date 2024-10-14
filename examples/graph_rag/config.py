import os

# Neo4j Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "bonanza-fax-1342"

# Set environment variables
os.environ["NEO4J_URI"] = NEO4J_URI
os.environ["NEO4J_USERNAME"] = NEO4J_USERNAME
os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD

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