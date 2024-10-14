from typing import List
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument

def process_text(text: str, llm_transformer, debug=False) -> List[GraphDocument]:
    doc = Document(page_content=text)
    try:
        result = llm_transformer.convert_to_graph_documents([doc])
        if not result or not result[0].nodes:
            print(f"Warning: No entities extracted from text: {text[:100]}...")
        else:
            if debug:
                print(f"Extracted {len(result[0].nodes)} entities and {len(result[0].relationships)} relationships from text: {text[:100]}...")
        return result
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return []

