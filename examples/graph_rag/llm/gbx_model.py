from gbx_lm.langchain.chat_gbx import ChatGBX
from gbx_lm.langchain import GBXPipeline
from gbx_lm.langchain import SimpleGraphTransformer


def init_gbx_model(model_id:str, max_tokens: int=100, temperature:float=0.0):
    llm = GBXPipeline.from_model_id(
        model_id=model_id,
        pipeline_kwargs={"max_tokens": max_tokens, "temp": temperature}
    )
    return ChatGBX(llm=llm)

def create_llm_transformer(chat_gbx, debug=False):
    return SimpleGraphTransformer(
        llm=chat_gbx,
        # allowed_nodes=allowed_nodes,
        # allowed_relationships=allowed_relationships,
        node_properties=["description"],
        relationship_properties=["description"],
        strict_mode=False,
        is_debug=debug
    )