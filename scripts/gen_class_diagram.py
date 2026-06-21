"""
Generates a UML-style class diagram for DocuMind's core abstractions
(Strategy/Factory interfaces + implementations + the orchestrator).
"""
from graphviz import Digraph

g = Digraph("class_diagram", format="png")
g.attr(rankdir="BT", bgcolor="white", fontname="Helvetica", nodesep="0.5", ranksep="0.65", pad="0.4")
g.attr(label="DocuMind \u2014 Class Diagram (core abstractions)", labelloc="t",
       fontsize="16", fontcolor="#1F2A38", fontname="Helvetica-Bold")
g.attr("node", fontname="Helvetica", fontsize="10", shape="record", style="filled", fillcolor="#FAFBFD", color="#5C7AA8")
g.attr("edge", fontname="Helvetica", fontsize="9", color="#5A6472")


def uml(name, attrs=None, methods=None, abstract=False, fillcolor="#FAFBFD"):
    title = f"\u00ab abstract \u00bb\\n{name}" if abstract else name
    attr_lines = "\\l".join(attrs) + "\\l" if attrs else ""
    method_lines = "\\l".join(methods) + "\\l" if methods else ""
    label = f"{{{title}|{attr_lines}|{method_lines}}}"
    g.node(name, label=label, fillcolor=fillcolor)


# ---- Interfaces (abstract) ----
uml("DocumentLoader", methods=["+load(path): str"], abstract=True, fillcolor="#EFF4FA")
uml("Chunker", methods=["+chunk(text, doc_id): List~Chunk~"], abstract=True, fillcolor="#EFF4FA")
uml("Embedder", methods=["+fit(corpus)", "+embed(texts): ndarray"], abstract=True, fillcolor="#EFF4FA")
uml("VectorStore", methods=["+upsert(records)", "+search(vector, k)", "+delete_by_document(id)"], abstract=True, fillcolor="#EFF4FA")
uml("LLMProvider", methods=["+generate(query, chunks): str"], abstract=True, fillcolor="#EFF4FA")

# ---- Concrete implementations ----
uml("TxtLoader", fillcolor="#F2F8F1")
uml("MarkdownLoader", fillcolor="#F2F8F1")
uml("PdfLoader", fillcolor="#F2F8F1")

uml("FixedSizeChunker", attrs=["chunk_size", "overlap"], fillcolor="#F2F8F1")
uml("SentenceAwareChunker", attrs=["chunk_size", "overlap_sentences"], fillcolor="#F2F8F1")

uml("TfidfEmbedder", attrs=["max_features"], fillcolor="#F2F8F1")
uml("AnthropicEmbedder", attrs=["model"], fillcolor="#F2F8F1")

uml("InMemoryVectorStore", fillcolor="#FBF1F4")
uml("FaissVectorStore", attrs=["dim"], fillcolor="#FBF1F4")

uml("ExtractiveLLMProvider", fillcolor="#FFF8EF")
uml("AnthropicLLMProvider", attrs=["model", "api_key"], fillcolor="#FFF8EF")

# ---- Factories ----
uml("DocumentLoaderFactory", methods=["+get_loader(filename): DocumentLoader"], fillcolor="#FDF3D9")
uml("ChunkerFactory", methods=["+get_chunker(strategy): Chunker"], fillcolor="#FDF3D9")
uml("EmbedderFactory", methods=["+create(provider): Embedder"], fillcolor="#FDF3D9")
uml("VectorStoreFactory", methods=["+create(backend): VectorStore"], fillcolor="#FDF3D9")
uml("LLMProviderFactory", methods=["+create(provider): LLMProvider"], fillcolor="#FDF3D9")

# ---- Orchestrator + supporting classes ----
uml("Retriever", attrs=["top_k", "score_threshold"], methods=["+retrieve(query): List~RetrievedChunk~"], fillcolor="#E3F0E1")
uml("LRUCache", attrs=["max_size", "ttl_seconds"], methods=["+get(key)", "+set(key, value)"], fillcolor="#E3F0E1")
uml("RAGPipeline", methods=["+ingest_document(path, filename, doc_id)", "+answer_query(query): QueryResult"], fillcolor="#FFE9C7")

# ---- Inheritance (implements) ----
for impl in ["TxtLoader", "MarkdownLoader", "PdfLoader"]:
    g.edge(impl, "DocumentLoader", arrowhead="empty", style="dashed")
for impl in ["FixedSizeChunker", "SentenceAwareChunker"]:
    g.edge(impl, "Chunker", arrowhead="empty", style="dashed")
for impl in ["TfidfEmbedder", "AnthropicEmbedder"]:
    g.edge(impl, "Embedder", arrowhead="empty", style="dashed")
for impl in ["InMemoryVectorStore", "FaissVectorStore"]:
    g.edge(impl, "VectorStore", arrowhead="empty", style="dashed")
for impl in ["ExtractiveLLMProvider", "AnthropicLLMProvider"]:
    g.edge(impl, "LLMProvider", arrowhead="empty", style="dashed")

# ---- Factories create concrete classes ----
g.edge("DocumentLoaderFactory", "DocumentLoader", label="creates", style="dotted", arrowhead="vee")
g.edge("ChunkerFactory", "Chunker", label="creates", style="dotted", arrowhead="vee")
g.edge("EmbedderFactory", "Embedder", label="creates", style="dotted", arrowhead="vee")
g.edge("VectorStoreFactory", "VectorStore", label="creates", style="dotted", arrowhead="vee")
g.edge("LLMProviderFactory", "LLMProvider", label="creates", style="dotted", arrowhead="vee")

# ---- Composition: RAGPipeline owns one of each ----
g.edge("RAGPipeline", "Embedder", label="1", arrowhead="diamond", color="#8A6D1A")
g.edge("RAGPipeline", "VectorStore", label="1", arrowhead="diamond", color="#8A6D1A")
g.edge("RAGPipeline", "LLMProvider", label="1", arrowhead="diamond", color="#8A6D1A")
g.edge("RAGPipeline", "LRUCache", label="1", arrowhead="diamond", color="#8A6D1A")
g.edge("RAGPipeline", "Chunker", label="1", arrowhead="diamond", color="#8A6D1A")
g.edge("RAGPipeline", "Retriever", label="1", arrowhead="diamond", color="#8A6D1A")
g.edge("Retriever", "Embedder", label="uses", style="dotted")
g.edge("Retriever", "VectorStore", label="uses", style="dotted")

g.render("/home/claude/documind/docs/diagrams/class_diagram", cleanup=True)
print("done")
