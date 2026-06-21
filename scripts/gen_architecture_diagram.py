"""
Generates the component/architecture diagram for DocuMind using Graphviz.
"""
from graphviz import Digraph

g = Digraph("architecture", format="png")
g.attr(rankdir="TB", bgcolor="white", fontname="Helvetica", splines="ortho",
       nodesep="0.45", ranksep="0.55", pad="0.4")
g.attr("node", fontname="Helvetica", fontsize="11", shape="box", style="rounded,filled",
       penwidth="1.2")
g.attr("edge", fontname="Helvetica", fontsize="9", color="#5A6472", arrowsize="0.7")

# ---- Title ----
g.attr(label="DocuMind \u2014 Enterprise Document Q&A Assistant: Component Architecture",
       labelloc="t", fontsize="16", fontcolor="#1F2A38", fontname="Helvetica-Bold")

CLIENT = "#E8EEF7"
API = "#D6E4F0"
ORCH = "#FCEBD5"
CORE = "#E3F0E1"
STORE = "#F3DCE3"
EXT = "#EDEDED"

# ---- Client layer ----
with g.subgraph(name="cluster_client") as c:
    c.attr(label="Client", style="rounded,filled", fillcolor="#F7F9FC", color="#A9B7CC")
    c.node("client", "Web / Mobile Client\n(employee asking questions,\nuploading documents)",
           fillcolor=CLIENT, color="#5C7AA8")

# ---- API layer ----
with g.subgraph(name="cluster_api") as c:
    c.attr(label="API Layer  (FastAPI)", style="rounded,filled", fillcolor="#F2F6FB", color="#5C7AA8")
    c.node("api", "REST Endpoints\nPOST /documents/upload\nGET /documents/{id}/status\nPOST /chat/query\nGET /chat/{id}/history",
           fillcolor=API, color="#3F6694")

# ---- Orchestration ----
with g.subgraph(name="cluster_orch") as c:
    c.attr(label="Orchestration Layer", style="rounded,filled", fillcolor="#FFF8EF", color="#C99A4D")
    c.node("pipeline", "RAGPipeline (Facade)\ningest_document()\nanswer_query()",
           fillcolor=ORCH, color="#C99A4D")

# ---- Core services ----
with g.subgraph(name="cluster_core") as c:
    c.attr(label="Core Services", style="rounded,filled", fillcolor="#F2F8F1", color="#5E9C57")
    c.node("ingestion", "Ingestion Service\nDocumentLoaderFactory\nChunker (Strategy)", fillcolor=CORE, color="#5E9C57")
    c.node("retrieval", "Retrieval Service\nEmbedder + Retriever\n(top-k cosine search)", fillcolor=CORE, color="#5E9C57")
    c.node("generation", "Generation Service\nLLMProvider (Strategy)\nprompt builder", fillcolor=CORE, color="#5E9C57")
    c.node("cache", "Cache Manager\n(LRU + TTL)", fillcolor=CORE, color="#5E9C57")

# ---- Storage ----
with g.subgraph(name="cluster_store") as c:
    c.attr(label="Storage Layer", style="rounded,filled", fillcolor="#FBF1F4", color="#B05C7A")
    c.node("vecstore", "Vector Store\n(in-memory cosine /\nFAISS in production)", fillcolor=STORE, color="#B05C7A")
    c.node("rdb", "Relational DB\n(SQLite / PostgreSQL)\nusers, documents,\nconversations, messages", fillcolor=STORE, color="#B05C7A")

# ---- External ----
with g.subgraph(name="cluster_ext") as c:
    c.attr(label="External", style="rounded,filled", fillcolor="#F5F5F5", color="#8A8A8A")
    c.node("llm", "LLM Provider API\n(Claude / extractive\nfallback)", fillcolor=EXT, color="#8A8A8A")

# ---- Edges ----
g.edge("client", "api", label="HTTPS / JSON")
g.edge("api", "pipeline", label="invoke")

g.edge("pipeline", "ingestion", label="ingest_document()")
g.edge("pipeline", "retrieval", label="answer_query()")
g.edge("pipeline", "cache", label="get/set")
g.edge("pipeline", "generation", label="generate()")

g.edge("ingestion", "vecstore", label="upsert vectors")
g.edge("retrieval", "vecstore", label="search(top_k)")
g.edge("generation", "llm", label="API call")

g.edge("api", "rdb", label="repository\nread/write", style="dashed")
g.edge("pipeline", "rdb", style="invis")  # layout aid

g.render("/home/claude/documind/docs/diagrams/architecture_diagram", cleanup=True)
print("done")
