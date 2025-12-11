import os
import logging
import ollama
import numpy as np
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

# !!! qwen2-7B maybe produce unparsable results and cause the extraction of graph to fail.
WORKING_DIR = "./nano_graphrag_cache_ollama_TEST"
MODEL = "gpt-oss:120b"

EMBED_MODEL = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", cache_folder=WORKING_DIR, device="cuda"
)


# We're using Sentence Transformers to generate embeddings for the BGE model
@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBED_MODEL.max_seq_length,
)
async def local_embedding(texts: list[str]) -> np.ndarray:
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)


async def ollama_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # remove kwargs that are not supported by ollama
    kwargs.pop("max_tokens", None)
    kwargs.pop("response_format", None)

    ollama_client = ollama.AsyncClient()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------
    response = await ollama_client.chat(model=MODEL, messages=messages, **kwargs)

    result = response["message"]["content"]
    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result, "model": MODEL}})
    # -----------------------------------------------------
    return result


def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)


def query():
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=ollama_model_if_cache,
        cheap_model_func=ollama_model_if_cache,
        embedding_func=local_embedding,
    )
    query_file = "/home/m202/M202/Richard/nano-graphrag/tests/query.txt"
    with open(query_file, 'r') as f:
        query_text = f.read().strip()
    
    print(f"Query from {query_file}:")
    print(query_text)
    print("\n" + "="*80 + "\n")
    print("global")
    # Execute query
    print(rag.query(query_text, param=QueryParam(mode="global")))
    # response = rag.query("What are the activities?", param=QueryParam(mode="global"))
    print("\n" + "local" + "\n")
    response = rag.query(query_text, param=QueryParam(mode="local"))
    print(response)


def insert():
    from time import time

    with open("./tests/test_4.txt", encoding="utf-8-sig") as f:
        FAKE_TEXT = f.read()

    remove_if_exist(f"{WORKING_DIR}/vdb_entities.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_full_docs.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_text_chunks.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_community_reports.json")
    remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")

    rag = GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=True,
        best_model_func=ollama_model_if_cache,
        cheap_model_func=ollama_model_if_cache,
        embedding_func=local_embedding,
    )
    start = time()
    rag.insert(FAKE_TEXT)
    print("indexing time:", time() - start)
    # rag = GraphRAG(working_dir=WORKING_DIR, enable_llm_cache=True)
    # rag.insert(FAKE_TEXT[half_len:])


if __name__ == "__main__":
    # insert()
    query()
