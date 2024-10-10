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

def create_extraction_chain(chat_gbx):
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

    extraction_llm = chat_gbx.with_structured_output(Disambiguate)

    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_template),
    ])

    return extraction_prompt | extraction_llm

@retry(tries=3, delay=2)
def entity_resolution(entities, extraction_chain):
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

