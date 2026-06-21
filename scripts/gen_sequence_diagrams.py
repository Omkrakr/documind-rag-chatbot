"""
Generates two sequence/interaction-flow diagrams:
  1. Document ingestion path
  2. Query / answer-generation path (incl. cache hit shortcut)

Rendered as swim-lane style flows (actor noted in each step) rather than
classic UML lifelines, for maximum rendering reliability.
"""
from graphviz import Digraph


def build_sequence(name, title, steps, actor_colors):
    g = Digraph(name, format="png")
    g.attr(rankdir="TB", bgcolor="white", fontname="Helvetica", nodesep="0.25", ranksep="0.18", pad="0.4")
    g.attr(label=title, labelloc="t", fontsize="15", fontcolor="#1F2A38", fontname="Helvetica-Bold")
    g.attr("node", fontname="Helvetica", fontsize="10", shape="box", style="rounded,filled", penwidth="1")
    g.attr("edge", fontname="Helvetica", fontsize="9", color="#5A6472", arrowsize="0.6")

    prev = None
    for i, (actor_from, actor_to, label) in enumerate(steps, start=1):
        node_id = f"s{i}"
        color = actor_colors.get(actor_from, "#EFF4FA")
        g.node(node_id, f"{i}. {actor_from} \u2192 {actor_to}\\n{label}", fillcolor=color, color="#5C7AA8")
        if prev:
            g.edge(prev, node_id)
        prev = node_id

    g.render(f"/home/claude/documind/docs/diagrams/{name}", cleanup=True)


actor_colors = {
    "Client": "#E8EEF7",
    "API": "#D6E4F0",
    "RAGPipeline": "#FCEBD5",
    "Chunker": "#E3F0E1",
    "Embedder": "#E3F0E1",
    "VectorStore": "#F3DCE3",
    "DB": "#F3DCE3",
    "Cache": "#E3F0E1",
    "Retriever": "#E3F0E1",
    "LLMProvider": "#FFF4E0",
}

# ------------------------------------------------------------------
# Ingestion sequence
# ------------------------------------------------------------------
ingestion_steps = [
    ("Client", "API", "POST /documents/upload (file)"),
    ("API", "DB", "create Document row, status=PROCESSING"),
    ("API", "RAGPipeline", "ingest_document(path, filename, doc_id)"),
    ("RAGPipeline", "Chunker", "loader.load() + chunk(text, doc_id)"),
    ("Chunker", "RAGPipeline", "return List[Chunk]"),
    ("RAGPipeline", "Embedder", "fit(all_texts) + embed(all_texts)"),
    ("Embedder", "RAGPipeline", "return vectors"),
    ("RAGPipeline", "VectorStore", "clear() + upsert(records)"),
    ("RAGPipeline", "Cache", "invalidate_all()"),
    ("API", "DB", "set status=READY, chunk_count=N"),
    ("API", "Client", "200 OK { document_id, status, chunk_count }"),
]
build_sequence("sequence_ingestion", "Sequence: Document Ingestion", ingestion_steps, actor_colors)

# ------------------------------------------------------------------
# Query sequence
# ------------------------------------------------------------------
query_steps = [
    ("Client", "API", "POST /chat/query { query, conversation_id? }"),
    ("API", "DB", "get-or-create Conversation"),
    ("API", "RAGPipeline", "answer_query(query)"),
    ("RAGPipeline", "Cache", "get(hash(query))"),
    ("Cache", "RAGPipeline", "[cache miss] \u2192 None"),
    ("RAGPipeline", "Retriever", "retrieve(query)"),
    ("Retriever", "Embedder", "embed([query])"),
    ("Retriever", "VectorStore", "search(query_vector, top_k)"),
    ("VectorStore", "Retriever", "ranked (chunk, score) list"),
    ("Retriever", "RAGPipeline", "List[RetrievedChunk] (above threshold)"),
    ("RAGPipeline", "LLMProvider", "generate(query, chunks)"),
    ("LLMProvider", "RAGPipeline", "grounded answer text"),
    ("RAGPipeline", "Cache", "set(hash(query), answer)"),
    ("API", "DB", "save user + assistant messages"),
    ("API", "Client", "200 OK { answer, sources[], cache_hit }"),
]
build_sequence("sequence_query", "Sequence: Query \u2192 Retrieval \u2192 Generation", query_steps, actor_colors)

print("done")
