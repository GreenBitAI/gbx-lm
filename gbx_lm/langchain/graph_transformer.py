import json
import re
from typing import List, Dict, Any, Sequence, Optional, Union, cast, Tuple, Type
from gbx_lm.langchain import ChatGBX
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from pydantic import BaseModel, Field, create_model
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate
)
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser


system_prompt = (
    "# Knowledge Graph Instructions for GPT-4\n"
    "## 1. Overview\n"
    "You are a top-tier algorithm designed for extracting information in structured "
    "formats to build a knowledge graph.\n"
    "Try to capture as much information from the text as possible without "
    "sacrificing accuracy. Do not add any information that is not explicitly "
    "mentioned in the text.\n"
    "- **Nodes** represent entities and concepts.\n"
    "- The aim is to achieve simplicity and clarity in the knowledge graph, making it\n"
    "accessible for a vast audience.\n"
    "## 2. Labeling Nodes\n"
    "- **Consistency**: Ensure you use available types for node labels.\n"
    "Ensure you use basic or elementary types for node labels.\n"
    "- For example, when you identify an entity representing a person, "
    "always label it as **'person'**. Avoid using more specific terms "
    "like 'mathematician' or 'scientist'."
    "- **Node IDs**: Never utilize integers as node IDs. Node IDs should be "
    "names or human-readable identifiers found in the text.\n"
    "- **Relationships** represent connections between entities or concepts.\n"
    "Ensure consistency and generality in relationship types when constructing "
    "knowledge graphs. Instead of using specific and momentary types "
    "such as 'BECAME_PROFESSOR', use more general and timeless relationship types "
    "like 'PROFESSOR'. Make sure to use general and timeless relationship types!\n"
    "## 3. Coreference Resolution\n"
    "- **Maintain Entity Consistency**: When extracting entities, it's vital to "
    "ensure consistency.\n"
    'If an entity, such as "John Doe", is mentioned multiple times in the text '
    'but is referred to by different names or pronouns (e.g., "Joe", "he"),'
    "always use the most complete identifier for that entity throughout the "
    'knowledge graph. In this example, use "John Doe" as the entity ID.\n'
    "Remember, the knowledge graph should be coherent and easily understandable, "
    "so maintaining consistency in entity references is crucial.\n"
    "## 4. Strict Compliance\n"
    "Adhere to the rules strictly. Non-compliance will result in termination."
)

default_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        (
            "human",
            (
                "Tip: Make sure to answer in the correct format and do "
                "not include any explanations. "
                "Use the given format to extract information from the "
                "following input: {input}"
            ),
        ),
    ]
)


examples = [
    {
        "text": (
            "Adam is a software engineer in Microsoft since 2009, "
            "and last year he got an award as the Best Talent"
        ),
        "head": "Adam",
        "head_type": "Person",
        "relation": "WORKS_FOR",
        "tail": "Microsoft",
        "tail_type": "Company",
    },
    {
        "text": (
            "Adam is a software engineer in Microsoft since 2009, "
            "and last year he got an award as the Best Talent"
        ),
        "head": "Adam",
        "head_type": "Person",
        "relation": "HAS_AWARD",
        "tail": "Best Talent",
        "tail_type": "Award",
    },
    {
        "text": (
            "Microsoft is a tech company that provide "
            "several products such as Microsoft Word"
        ),
        "head": "Microsoft Word",
        "head_type": "Product",
        "relation": "PRODUCED_BY",
        "tail": "Microsoft",
        "tail_type": "Company",
    },
    {
        "text": "Microsoft Word is a lightweight app that accessible offline",
        "head": "Microsoft Word",
        "head_type": "Product",
        "relation": "HAS_CHARACTERISTIC",
        "tail": "lightweight app",
        "tail_type": "Characteristic",
    },
    {
        "text": "Microsoft Word is a lightweight app that accessible offline",
        "head": "Microsoft Word",
        "head_type": "Product",
        "relation": "HAS_CHARACTERISTIC",
        "tail": "accessible offline",
        "tail_type": "Characteristic",
    },
]


class UnstructuredRelation(BaseModel):
    head: str = Field(
        description=(
            "extracted head entity like Microsoft, Apple, John. "
            "Must use human-readable unique identifier."
        )
    )
    head_type: str = Field(
        description="type of the extracted head entity like Person, Company, etc"
    )
    relation: str = Field(description="relation between the head and the tail entities")
    tail: str = Field(
        description=(
            "extracted tail entity like Microsoft, Apple, John. "
            "Must use human-readable unique identifier."
        )
    )
    tail_type: str = Field(
        description="type of the extracted tail entity like Person, Company, etc"
    )

def create_unstructured_prompt(
    node_labels: Optional[List[str]] = None, rel_types: Optional[List[str]] = None
) -> ChatPromptTemplate:
    node_labels_str = str(node_labels) if node_labels else ""
    rel_types_str = str(rel_types) if rel_types else ""
    base_string_parts = [
        "You are a top-tier algorithm designed for extracting information in "
        "structured formats to build a knowledge graph. Your task is to identify "
        "the entities and relations requested with the user prompt from a given "
        "text. You must generate the output in a JSON format containing a list "
        'with JSON objects. Each object should have the keys: "head", '
        '"head_type", "relation", "tail", and "tail_type". The "head" '
        "key must contain the text of the extracted entity with one of the types "
        "from the provided list in the user prompt.",
        f'The "head_type" key must contain the type of the extracted head entity, '
        f"which must be one of the types from {node_labels_str}."
        if node_labels
        else "",
        f'The "relation" key must contain the type of relation between the "head" '
        f'and the "tail", which must be one of the relations from {rel_types_str}.'
        if rel_types
        else "",
        f'The "tail" key must represent the text of an extracted entity which is '
        f'the tail of the relation, and the "tail_type" key must contain the type '
        f"of the tail entity from {node_labels_str}."
        if node_labels
        else "",
        "Attempt to extract as many entities and relations as you can. Maintain "
        "Entity Consistency: When extracting entities, it's vital to ensure "
        'consistency. If an entity, such as "John Doe", is mentioned multiple '
        "times in the text but is referred to by different names or pronouns "
        '(e.g., "Joe", "he"), always use the most complete identifier for '
        "that entity. The knowledge graph should be coherent and easily "
        "understandable, so maintaining consistency in entity references is "
        "crucial.",
        "IMPORTANT NOTES:\n- Don't add any explanation and text.",
    ]
    system_prompt = "\n".join(filter(None, base_string_parts))

    system_message = SystemMessage(content=system_prompt)
    parser = JsonOutputParser(pydantic_object=UnstructuredRelation)

    human_string_parts = [
        "Based on the following example, extract entities and "
        "relations from the provided text.\n\n",
        "Use the following entity types, don't use other entity "
        "that is not defined below:"
        "# ENTITY TYPES:"
        "{node_labels}"
        if node_labels
        else "",
        "Use the following relation types, don't use other relation "
        "that is not defined below:"
        "# RELATION TYPES:"
        "{rel_types}"
        if rel_types
        else "",
        "Below are a number of examples of text and their extracted "
        "entities and relationships."
        "{examples}\n"
        "For the following text, extract entities and relations as "
        "in the provided example."
        "{format_instructions}\nText: {input}",
    ]
    human_prompt_string = "\n".join(filter(None, human_string_parts))
    human_prompt = PromptTemplate(
        template=human_prompt_string,
        input_variables=["input"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "node_labels": node_labels,
            "rel_types": rel_types,
            "examples": examples,
        },
    )

    human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message, human_message_prompt]
    )
    return chat_prompt


def format_property_key(s: str) -> str:
    words = s.split()
    if not words:
        return s
    first_word = words[0].lower()
    capitalized_words = [word.capitalize() for word in words[1:]]
    return "".join([first_word] + capitalized_words)

def _parse_and_clean_json(
    argument_json: Dict[str, Any],
) -> Tuple[List[Node], List[Relationship]]:
    nodes = []
    for node in argument_json["nodes"]:
        if not node.get("id"):  # Id is mandatory, skip this node
            continue
        node_properties = {}
        if "properties" in node and node["properties"]:
            for p in node["properties"]:
                node_properties[format_property_key(p["key"])] = p["value"]
        nodes.append(
            Node(
                id=node["id"],
                type=node.get("type", "Node"),
                properties=node_properties,
            )
        )
    relationships = []
    for rel in argument_json["relationships"]:
        # Mandatory props
        if (
            not rel.get("source_node_id")
            or not rel.get("target_node_id")
            or not rel.get("type")
        ):
            continue

        # Node type copying if needed from node list
        if not rel.get("source_node_type"):
            try:
                rel["source_node_type"] = [
                    el.get("type")
                    for el in argument_json["nodes"]
                    if el["id"] == rel["source_node_id"]
                ][0]
            except IndexError:
                rel["source_node_type"] = None
        if not rel.get("target_node_type"):
            try:
                rel["target_node_type"] = [
                    el.get("type")
                    for el in argument_json["nodes"]
                    if el["id"] == rel["target_node_id"]
                ][0]
            except IndexError:
                rel["target_node_type"] = None

        rel_properties = {}
        if "properties" in rel and rel["properties"]:
            for p in rel["properties"]:
                rel_properties[format_property_key(p["key"])] = p["value"]

        source_node = Node(
            id=rel["source_node_id"],
            type=rel["source_node_type"],
        )
        target_node = Node(
            id=rel["target_node_id"],
            type=rel["target_node_type"],
        )
        relationships.append(
            Relationship(
                source=source_node,
                target=target_node,
                type=rel["type"],
                properties=rel_properties,
            )
        )
    return nodes, relationships

class _Graph(BaseModel):
    nodes: Optional[List]
    relationships: Optional[List]


def _get_additional_info(input_type: str) -> str:
    # Check if the input_type is one of the allowed values
    if input_type not in ["node", "relationship", "property"]:
        raise ValueError("input_type must be 'node', 'relationship', or 'property'")

    # Perform actions based on the input_type
    if input_type == "node":
        return (
            "Ensure you use basic or elementary types for node labels.\n"
            "For example, when you identify an entity representing a person, "
            "always label it as **'Person'**. Avoid using more specific terms "
            "like 'Mathematician' or 'Scientist'"
        )
    elif input_type == "relationship":
        return (
            "Instead of using specific and momentary types such as "
            "'BECAME_PROFESSOR', use more general and timeless relationship types "
            "like 'PROFESSOR'. However, do not sacrifice any accuracy for generality"
        )
    elif input_type == "property":
        return ""
    return ""


def optional_enum_field(
    enum_values: Optional[List[str]] = None,
    description: str = "",
    input_type: str = "node",
    llm_type: Optional[str] = None,
    **field_kwargs: Any,
) -> Any:
    """Utility function to conditionally create a field with an enum constraint."""
    # Only openai supports enum param
    if enum_values and llm_type == "openai-chat":
        return Field(
            ...,
            enum=enum_values,
            description=f"{description}. Available options are {enum_values}",
            **field_kwargs,
        )
    elif enum_values:
        return Field(
            ...,
            description=f"{description}. Available options are {enum_values}",
            **field_kwargs,
        )
    else:
        additional_info = _get_additional_info(input_type)
        return Field(..., description=description + additional_info, **field_kwargs)


def create_simple_model(
    node_labels: Optional[List[str]] = None,
    rel_types: Optional[List[str]] = None,
    node_properties: Union[bool, List[str]] = False,
    llm_type: Optional[str] = None,
    relationship_properties: Union[bool, List[str]] = False,
) -> Type[_Graph]:
    """
    Create a simple graph model with optional constraints on node
    and relationship types.

    Args:
        node_labels (Optional[List[str]]): Specifies the allowed node types.
            Defaults to None, allowing all node types.
        rel_types (Optional[List[str]]): Specifies the allowed relationship types.
            Defaults to None, allowing all relationship types.
        node_properties (Union[bool, List[str]]): Specifies if node properties should
            be included. If a list is provided, only properties with keys in the list
            will be included. If True, all properties are included. Defaults to False.
        relationship_properties (Union[bool, List[str]]): Specifies if relationship
            properties should be included. If a list is provided, only properties with
            keys in the list will be included. If True, all properties are included.
            Defaults to False.
        llm_type (Optional[str]): The type of the language model. Defaults to None.
            Only openai supports enum param: openai-chat.

    Returns:
        Type[_Graph]: A graph model with the specified constraints.

    Raises:
        ValueError: If 'id' is included in the node or relationship properties list.
    """

    node_fields: Dict[str, Tuple[Any, Any]] = {
        "id": (
            str,
            Field(..., description="Name or human-readable unique identifier."),
        ),
        "type": (
            str,
            optional_enum_field(
                node_labels,
                description="The type or label of the node.",
                input_type="node",
                llm_type=llm_type,
            ),
        ),
    }

    if node_properties:
        if isinstance(node_properties, list) and "id" in node_properties:
            raise ValueError("The node property 'id' is reserved and cannot be used.")
        # Map True to empty array
        node_properties_mapped: List[str] = (
            [] if node_properties is True else node_properties
        )

        class Property(BaseModel):
            """A single property consisting of key and value"""

            key: str = optional_enum_field(
                node_properties_mapped,
                description="Property key.",
                input_type="property",
                llm_type=llm_type,
            )
            value: str = Field(..., description="value")

        node_fields["properties"] = (
            Optional[List[Property]],
            Field(None, description="List of node properties"),
        )
    SimpleNode = create_model("SimpleNode", **node_fields)  # type: ignore

    relationship_fields: Dict[str, Tuple[Any, Any]] = {
        "source_node_id": (
            str,
            Field(
                ...,
                description="Name or human-readable unique identifier of source node",
            ),
        ),
        "source_node_type": (
            str,
            optional_enum_field(
                node_labels,
                description="The type or label of the source node.",
                input_type="node",
                llm_type=llm_type,
            ),
        ),
        "target_node_id": (
            str,
            Field(
                ...,
                description="Name or human-readable unique identifier of target node",
            ),
        ),
        "target_node_type": (
            str,
            optional_enum_field(
                node_labels,
                description="The type or label of the target node.",
                input_type="node",
                llm_type=llm_type,
            ),
        ),
        "type": (
            str,
            optional_enum_field(
                rel_types,
                description="The type of the relationship.",
                input_type="relationship",
                llm_type=llm_type,
            ),
        ),
    }
    if relationship_properties:
        if (
            isinstance(relationship_properties, list)
            and "id" in relationship_properties
        ):
            raise ValueError(
                "The relationship property 'id' is reserved and cannot be used."
            )
        # Map True to empty array
        relationship_properties_mapped: List[str] = (
            [] if relationship_properties is True else relationship_properties
        )

        class RelationshipProperty(BaseModel):
            """A single property consisting of key and value"""

            key: str = optional_enum_field(
                relationship_properties_mapped,
                description="Property key.",
                input_type="property",
                llm_type=llm_type,
            )
            value: str = Field(..., description="value")

        relationship_fields["properties"] = (
            Optional[List[RelationshipProperty]],
            Field(None, description="List of relationship properties"),
        )
    SimpleRelationship = create_model("SimpleRelationship", **relationship_fields)  # type: ignore

    class DynamicGraph(_Graph):
        """Represents a graph document consisting of nodes and relationships."""

        nodes: Optional[List[SimpleNode]] = Field(description="List of nodes")  # type: ignore
        relationships: Optional[List[SimpleRelationship]] = Field(  # type: ignore
            description="List of relationships"
        )

    return DynamicGraph

def map_to_base_node(node: Any) -> Node:
    """Map the SimpleNode to the base Node."""
    properties = {}
    if hasattr(node, "properties") and node.properties:
        for p in node.properties:
            properties[format_property_key(p.key)] = p.value
    return Node(id=node.id, type=node.type, properties=properties)

def map_to_base_relationship(rel: Any) -> Relationship:
    """Map the SimpleRelationship to the base Relationship."""
    source = Node(id=rel.source_node_id, type=rel.source_node_type)
    target = Node(id=rel.target_node_id, type=rel.target_node_type)
    properties = {}
    if hasattr(rel, "properties") and rel.properties:
        for p in rel.properties:
            properties[format_property_key(p.key)] = p.value
    return Relationship(
        source=source, target=target, type=rel.type, properties=properties
    )

def _format_nodes(nodes: List[Node]) -> List[Node]:
    return [
        Node(
            id=el.id.title() if isinstance(el.id, str) else el.id,
            type=el.type.capitalize()  # type: ignore[arg-type]
            if el.type
            else None,  # handle empty strings  # type: ignore[arg-type]
            properties=el.properties,
        )
        for el in nodes
    ]

def _format_relationships(rels: List[Relationship]) -> List[Relationship]:
    return [
        Relationship(
            source=_format_nodes([el.source])[0],
            target=_format_nodes([el.target])[0],
            type=el.type.replace(" ", "_").upper(),
            properties=el.properties,
        )
        for el in rels
    ]

def _parse_chatmlx_output(content: str) -> Tuple[List[Node], List[Relationship]]:

    nodes = []
    relationships = []

    # 分离节点和关系部分
    sections = re.split(r'\*\*(Nodes|Relationships)\*\*', content, flags=re.IGNORECASE)
    nodes_section = ''
    relationships_section = ''
    for i, section in enumerate(sections):
        if section.strip().lower() == 'nodes':
            nodes_section = sections[i + 1]
        elif section.strip().lower() == 'relationships':
            relationships_section = sections[i + 1]

    # 提取节点
    node_pattern = r'\*\s*(.*?)(?:\s*(?:-|:|：)\s*(.*?))?(?:\n|$)'
    for match in re.finditer(node_pattern, nodes_section):
        node_id, node_type = match.groups()
        node_id = re.sub(r'\s*\([^)]*\)', '', node_id).strip()
        node_type = node_type.strip() if node_type else 'Unknown'
        if '(' in node_id and ')' in node_id:
            parts = node_id.split('(')
            node_id = parts[0].strip()
            if node_type == 'Unknown':
                node_type = parts[1].split(')')[0].strip()
        if len(node_id) <= 50:
            nodes.append(Node(id=node_id, type=node_type))

    # 提取关系
    rel_patterns = [
        r'\*\s*(.*?)\s+(?:is|are)\s+(?:a\s+)?(.*?)\s+(?:in|of|for)\s+(.*?)(?:\n|$)',
        r"\*\s*(.*?)'s\s+(.*?)\s+(?:is|are)\s+(.*?)(?:\n|$)",
        r'\*\s*(.*?)\s+-\s+(.*?)\s+-\s+(.*?)(?:\n|$)',
        r'\*\s*(.*?)\s+(.*?)\s+(.*?)(?:\n|$)'
    ]

    for pattern in rel_patterns:
        for match in re.finditer(pattern, relationships_section, re.IGNORECASE):
            source, rel_type, target = match.groups()
            source = source.strip()
            rel_type = rel_type.strip().upper()
            target = target.strip()

            source_node = next(
                (node for node in nodes if node.id.lower() in source.lower() or source.lower() in node.id.lower()),
                None)
            target_node = next(
                (node for node in nodes if node.id.lower() in target.lower() or target.lower() in node.id.lower()),
                None)

            if source_node and target_node and source_node != target_node:
                if rel_type == 'COMPANY':
                    rel_type = 'OPERATES_IN'
                elif rel_type == 'STOCK':
                    rel_type = 'LISTED_ON'
                elif 'ASSOCIATED' in rel_type:
                    rel_type = 'ASSOCIATED_WITH'
                elif rel_type in ['IS', 'ARE', 'WAS', 'WERE']:
                    rel_type = 'IS_A'

                if not any(r.source == source_node and r.target == target_node and r.type == rel_type for r in
                           relationships):
                    relationships.append(Relationship(source=source_node, target=target_node, type=rel_type))

    return _format_nodes(nodes), _format_relationships(relationships)

def _convert_to_graph_document(
    raw_schema: Dict[Any, Any],
) -> Tuple[List[Node], List[Relationship]]:
    # If there are validation errors
    if not raw_schema["parsed"]:
        # for MLX type reponse
        if isinstance(raw_schema.get("raw"), AIMessage):
            return _parse_chatmlx_output(raw_schema["raw"].content)
        elif isinstance(raw_schema.get("raw"), str):
            return _parse_chatmlx_output(raw_schema["raw"])
        try:
            try:  # OpenAI type response
                argument_json = json.loads(
                    raw_schema["raw"].additional_kwargs["tool_calls"][0]["function"][
                        "arguments"
                    ]
                )
            except Exception:  # Google type response
                try:
                    argument_json = json.loads(
                        raw_schema["raw"].additional_kwargs["function_call"][
                            "arguments"
                        ]
                    )
                except Exception:  # Ollama type response
                    argument_json = raw_schema["raw"].tool_calls[0]["args"]
                    if isinstance(argument_json["nodes"], str):
                        argument_json["nodes"] = json.loads(argument_json["nodes"])
                    if isinstance(argument_json["relationships"], str):
                        argument_json["relationships"] = json.loads(
                            argument_json["relationships"]
                        )

            nodes, relationships = _parse_and_clean_json(argument_json)
        except Exception:  # If we can't parse JSON
            return ([], [])
    else:  # If there are no validation errors use parsed pydantic object
        parsed_schema: _Graph = raw_schema["parsed"]
        nodes = (
            [map_to_base_node(node) for node in parsed_schema.nodes if node.id]
            if parsed_schema.nodes
            else []
        )

        relationships = (
            [
                map_to_base_relationship(rel)
                for rel in parsed_schema.relationships
                if rel.type and rel.source_node_id and rel.target_node_id
            ]
            if parsed_schema.relationships
            else []
        )
    # Title / Capitalize
    return _format_nodes(nodes), _format_relationships(relationships)


class SimpleGraphTransformer:
    def __init__(
            self,
            llm: ChatGBX,
            allowed_nodes: List[str] = [],
            allowed_relationships: List[str] = [],
            prompt: Optional[ChatPromptTemplate] = None,
            strict_mode: bool = True,
            node_properties: Union[bool, List[str]] = False,
            relationship_properties: Union[bool, List[str]] = False,
            is_debug=False
    ) -> None:
        self.llm = llm
        self.allowed_nodes = allowed_nodes
        self.allowed_relationships = allowed_relationships
        self.strict_mode = strict_mode
        self.node_properties = node_properties
        self.relationship_properties = relationship_properties
        self._function_call = True
        self.is_debug = is_debug
        try:
            llm.with_structured_output(_Graph)
        except NotImplementedError:
            self._function_call = False
        if not self._function_call:
            if node_properties or relationship_properties:
                print("Warning: 'node_properties' and 'relationship_properties' are ignored in the current implementation.")
            try:
                import json_repair  # type: ignore
                self.json_repair = json_repair
            except ImportError:
                raise ImportError(
                    "Could not import json_repair python package. "
                    "Please install it with `pip install json-repair`."
                )
            prompt = prompt or create_unstructured_prompt(
                allowed_nodes, allowed_relationships
            )
            self.chain = prompt | llm
        else:
            # Define chain
            try:
                llm_type = llm._llm_type  # type: ignore
            except AttributeError:
                llm_type = None
            schema = create_simple_model(
                allowed_nodes,
                allowed_relationships,
                node_properties,
                llm_type,
                relationship_properties,
            )
            structured_llm = llm.with_structured_output(schema, include_raw=True)
            prompt = prompt or default_prompt
            self.chain = prompt | structured_llm

    def process_response(
        self, document: Document, config: Optional[RunnableConfig] = None
    ) -> GraphDocument:
        """
        Processes a single document, transforming it into a graph document using
        an LLM based on the model's schema and constraints.
        """
        text = document.page_content
        try:
            if self.is_debug:
                print(f"Processing text: {text[:100]}...")
            raw_output = self.chain.invoke({"input": text}, config=config)
            if self.is_debug:
                print(f"Debug - Raw LLM output: {raw_output}")

            if self._function_call:
                raw_schema = cast(Dict[Any, Any], raw_output)
                nodes, relationships = _convert_to_graph_document(raw_schema)
            else:
                nodes, relationships = self.parse_output(raw_output)
            if self.is_debug:
                print(f"Extracted {len(nodes)} nodes and {len(relationships)} relationships")
            for node in nodes:
                if self.is_debug:
                    print(f"Node: {node.id} (Type: {node.type})")
            for rel in relationships:
                if self.is_debug:
                    print(f"Relationship: ({rel.source.id}) -[{rel.type}]-> ({rel.target.id})")

            # Strict mode filtering
            if self.strict_mode and (self.allowed_nodes or self.allowed_relationships):
                if self.allowed_nodes:
                    lower_allowed_nodes = [el.lower() for el in self.allowed_nodes]
                    nodes = [
                        node for node in nodes if node.type.lower() in lower_allowed_nodes
                    ]
                    relationships = [
                        rel
                        for rel in relationships
                        if rel.source.type.lower() in lower_allowed_nodes
                           and rel.target.type.lower() in lower_allowed_nodes
                    ]
                if self.allowed_relationships:
                    relationships = [
                        rel
                        for rel in relationships
                        if rel.type.lower()
                           in [el.lower() for el in self.allowed_relationships]
                    ]

            return GraphDocument(nodes=nodes, relationships=relationships, source=document)
        except Exception as e:
            print(f"Error processing document: {str(e)}")
            if self.is_debug:
                print(f"Problematic text: {text[:200]}...")
            return GraphDocument(nodes=[], relationships=[], source=document)

    def parse_output(self, output: Union[str, Any]) -> Tuple[List[Node], List[Relationship]]:
        if not isinstance(output, str):
            output = output.content

        try:
            parsed_data = json.loads(output)
            return self.process_parsed_data(parsed_data)
        except json.JSONDecodeError:
            pass

        json_match = re.search(r'\{.*\}|\[.*\]', output, re.DOTALL)
        if json_match:
            try:
                fixed_json = self.fix_json_structure(json_match.group(0))
                parsed_data = json.loads(fixed_json)
                return self.process_parsed_data(parsed_data)
            except json.JSONDecodeError:
                print(f"Failed to parse fixed JSON. Attempting partial parsing.")

        partial_data = self.parse_partial_json(output)
        if partial_data['nodes'] or partial_data['relationships']:
            return self.process_parsed_data(partial_data)

        print(f"Failed to extract any structured data. Raw output: {output}")
        return [], []

    def fix_json_structure(self, json_str: str) -> str:
        json_str = re.sub(r'[^{\[\]},:"]+', lambda m: m.group().strip(), json_str)
        json_str = re.sub(r'}\s*{', '},{', json_str)
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

        if not (json_str.startswith('{') or json_str.startswith('[')):
            json_str = '[' + json_str
        if not (json_str.endswith('}') or json_str.endswith(']')):
            json_str = json_str + ']'
        return json_str

    def parse_partial_json(self, json_str: str) -> Dict[str, Any]:
        result = {"nodes": [], "relationships": []}

        # extract nodes
        nodes_match = re.search(r'"nodes"\s*:\s*\[(.*?)\]', json_str, re.DOTALL)
        if nodes_match:
            nodes_str = nodes_match.group(1)
            node_matches = re.finditer(r'\{[^{}]*\}', nodes_str)
            for match in node_matches:
                try:
                    node = json.loads(match.group(0))
                    result['nodes'].append(node)
                except json.JSONDecodeError:
                    continue
        # extract relations
        rels_match = re.search(r'"relationships"\s*:\s*\[(.*?)\]', json_str, re.DOTALL)
        if rels_match:
            rels_str = rels_match.group(1)
            rel_matches = re.finditer(r'\{[^{}]*\}', rels_str)
            for match in rel_matches:
                try:
                    rel = json.loads(match.group(0))
                    result['relationships'].append(rel)
                except json.JSONDecodeError:
                    continue

        return result

    def process_parsed_data(self, data: Union[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]) -> Tuple[
        List[Node], List[Relationship]]:
        nodes = []
        relationships = []

        if isinstance(data, list):
            nodes_set = set()
            for item in data:
                if all(key in item for key in ["head", "head_type", "relation", "tail", "tail_type"]):
                    nodes_set.add((item["head"], item["head_type"]))
                    nodes_set.add((item["tail"], item["tail_type"]))
                    source_node = Node(id=item["head"], type=item["head_type"])
                    target_node = Node(id=item["tail"], type=item["tail_type"])
                    relationships.append(Relationship(source=source_node, target=target_node, type=item["relation"]))
            nodes = [Node(id=el[0], type=el[1]) for el in list(nodes_set)]
        elif isinstance(data, dict):
            nodes = [Node(id=node['id'], type=node['type']) for node in data.get('nodes', [])]
            relationships = [
                Relationship(
                    source=Node(id=rel['source'], type="Unknown"),
                    target=Node(id=rel['target'], type="Unknown"),
                    type=rel['type']
                ) for rel in data.get('relationships', [])
            ]

        return nodes, relationships

    def convert_to_graph_documents(
            self, documents: Sequence[Document], config: Optional[RunnableConfig] = None
    ) -> List[GraphDocument]:
        """Convert a sequence of documents into graph documents."""
        return [self.process_response(document, config) for document in documents]