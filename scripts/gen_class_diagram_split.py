"""
Generates the class diagram for DocuMind, split into two readable halves:
  1. Ingestion side (DocumentLoader + Chunker families)
  2. Core/Orchestration side (Embedder + VectorStore + LLMProvider families,
     Retriever, LRUCache, RAGPipeline)
Splitting avoids one extremely wide, hard-to-read banner image.
"""
from graphviz import Digraph


def uml(g, name, attrs=None, methods=None, abstract=False, fillcolor="#FAFBFD"):
    title = f"\u00ab abstract \u00bb\\n{name}" if abstract else name
    attr_lines = "\\l".join(attrs) + "\\l" if attrs else ""
    method_lines = "\\l".join(methods) + "\\l" if methods else ""
    label = f"{{{title}|{attr_lines}|{method_lines}}}"
    g.node(name, label=label, fillcolor=fillcolor)


def base_graph(name, title):
    g = Digraph(name, format="png")
    g.attr(rankdir="BT", bgcolor="white", fontname="Helvetica", nodesep="0.45", ranksep="0.6", pad="0.4")
    g.attr(label=title, labelloc="t", fontsize="15", fontcolor="#1F2A38", fontname="Helvetica-Bold")
    g.attr("node", fontname="Helvetica", fontsize="10", shape="record", style="filled",
           fillcolor="#FAFBFD", color="#5C7AA8")
    g.attr("edge", fontname="Helvetica", fontsize="9", color="#5A6472")
    return g


# ============================================================
# Diagram 1: Ingestion side
# ============================================================
g1 = base_graph("class_diagram_ingestion", "DocuMind \u2014 Class Diagram: Ingestion Layer")

uml(g1, "DocumentLoader", methods=["+load(path): str"], abstract=True, fillcolor="#EFF4FA")
uml(g1, "TxtLoader", fillcolor="#F2F8F1")
uml(g1, "MarkdownLoader", fillcolor="#F2F8F1")
uml(g1, "PdfLoader", fillcolor="#F2F8F1")
uml(g1, "DocumentLoaderFactory", methods=["+get_loader(filename): DocumentLoader"], fillcolor="#FDF3D9")

uml(g1, "Chunker", methods=["+chunk(text, doc_id): List~Chunk~"], abstract=True, fillcolor="#EFF4FA")
uml(g1, "FixedSizeChunker", attrs=["chunk_size", "overlap"], fillcolor="#F2F8F1")
uml(g1, "SentenceAwareChunker", attrs=["chunk_size", "overlap_sentences"], fillcolor="#F2F8F1")
uml(g1, "ChunkerFactory", methods=["+get_chunker(strategy): Chunker"], fillcolor="#FDF3D9")

for impl in ["TxtLoader", "MarkdownLoader", "PdfLoader"]:
    g1.edge(impl, "DocumentLoader", arrowhead="empty", style="dashed")
for impl in ["FixedSizeChunker", "SentenceAwareChunker"]:
    g1.edge(impl, "Chunker", arrowhead="empty", style="dashed")
g1.edge("DocumentLoaderFactory", "DocumentLoader", label="creates", style="dotted", arrowhead="vee")
g1.edge("ChunkerFactory", "Chunker", label="creates", style="dotted", arrowhead="vee")

g1.render("/home/claude/documind/docs/diagrams/class_diagram_ingestion", cleanup=True)

# ============================================================
# Diagram 2: Core / Orchestration side
# ============================================================
g2 = base_graph("class_diagram_core", "DocuMind \u2014 Class Diagram: Retrieval, Generation & Orchestration")

uml(g2, "Embedder", methods=["+fit(corpus)", "+embed(texts): ndarray"], abstract=True, fillcolor="#EFF4FA")
uml(g2, "TfidfEmbedder", attrs=["max_features"], fillcolor="#F2F8F1")
uml(g2, "AnthropicEmbedder", attrs=["model"], fillcolor="#F2F8F1")
uml(g2, "EmbedderFactory", methods=["+create(provider): Embedder"], fillcolor="#FDF3D9")

uml(g2, "VectorStore", methods=["+upsert(records)", "+search(vector, k)", "+delete_by_document(id)"],
    abstract=True, fillcolor="#EFF4FA")
uml(g2, "InMemoryVectorStore", fillcolor="#FBF1F4")
uml(g2, "FaissVectorStore", attrs=["dim"], fillcolor="#FBF1F4")
uml(g2, "VectorStoreFactory", methods=["+create(backend): VectorStore"], fillcolor="#FDF3D9")

uml(g2, "LLMProvider", methods=["+generate(query, chunks): str"], abstract=True, fillcolor="#EFF4FA")
uml(g2, "ExtractiveLLMProvider", fillcolor="#FFF8EF")
uml(g2, "AnthropicLLMProvider", attrs=["model", "api_key"], fillcolor="#FFF8EF")
uml(g2, "LLMProviderFactory", methods=["+create(provider): LLMProvider"], fillcolor="#FDF3D9")

uml(g2, "Retriever", attrs=["top_k", "score_threshold"],
    methods=["+retrieve(query): List~RetrievedChunk~"], fillcolor="#E3F0E1")
uml(g2, "LRUCache", attrs=["max_size", "ttl_seconds"],
    methods=["+get(key)", "+set(key, value)"], fillcolor="#E3F0E1")
uml(g2, "RAGPipeline", methods=["+ingest_document(path, filename, doc_id)", "+answer_query(query): QueryResult"],
    fillcolor="#FFE9C7")

for impl in ["TfidfEmbedder", "AnthropicEmbedder"]:
    g2.edge(impl, "Embedder", arrowhead="empty", style="dashed")
for impl in ["InMemoryVectorStore", "FaissVectorStore"]:
    g2.edge(impl, "VectorStore", arrowhead="empty", style="dashed")
for impl in ["ExtractiveLLMProvider", "AnthropicLLMProvider"]:
    g2.edge(impl, "LLMProvider", arrowhead="empty", style="dashed")

g2.edge("EmbedderFactory", "Embedder", label="creates", style="dotted", arrowhead="vee")
g2.edge("VectorStoreFactory", "VectorStore", label="creates", style="dotted", arrowhead="vee")
g2.edge("LLMProviderFactory", "LLMProvider", label="creates", style="dotted", arrowhead="vee")

g2.edge("RAGPipeline", "Embedder", label="1", arrowhead="diamond", color="#8A6D1A")
g2.edge("RAGPipeline", "VectorStore", label="1", arrowhead="diamond", color="#8A6D1A")
g2.edge("RAGPipeline", "LLMProvider", label="1", arrowhead="diamond", color="#8A6D1A")
g2.edge("RAGPipeline", "LRUCache", label="1", arrowhead="diamond", color="#8A6D1A")
g2.edge("RAGPipeline", "Retriever", label="1", arrowhead="diamond", color="#8A6D1A")
g2.edge("Retriever", "Embedder", label="uses", style="dotted")
g2.edge("Retriever", "VectorStore", label="uses", style="dotted")

g2.render("/home/claude/documind/docs/diagrams/class_diagram_core", cleanup=True)

print("done")
