"""
Data ingestion pipeline for Nigerian Student Rights Knowledge Graph.
Run once with: python load_data.py
"""
import sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector
from llm import llm, embeddings
from graph import graph

DATA_DIR = Path(__file__).parent / "data"
INDEX_NAME = "studentRightsIndex"
NODE_LABEL = "Document"
TEXT_PROPERTY = "text"
EMBEDDING_PROPERTY = "embedding"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def load_pdfs():
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"ERROR: No PDF files found in {DATA_DIR}")
        sys.exit(1)

    all_docs = []
    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.metadata["source_file"] = pdf_path.name
        all_docs.extend(pages)
        print(f"    -> {len(pages)} pages")
    return all_docs


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def create_vector_index(chunks):
    Neo4jVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        graph=graph,
        index_name=INDEX_NAME,
        node_label=NODE_LABEL,
        text_node_property=TEXT_PROPERTY,
        embedding_node_property=EMBEDDING_PROPERTY,
    )


def extract_knowledge_graph(chunks):
    """
    Extract named entities and relationships from document chunks
    and store them as a structured knowledge graph in Neo4j.
    Requires: pip install langchain-experimental
    """
    try:
        from langchain_experimental.graph_transformers import LLMGraphTransformer
    except ImportError:
        print("  Skipping KG extraction (langchain-experimental not installed).")
        print("  Install with: pip install langchain-experimental")
        return 0, 0

    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=[
            "Right", "Law", "Section", "Policy",
            "RegulatoryBody", "Institution", "Obligation", "Violation",
        ],
        allowed_relationships=[
            "GRANTS", "GOVERNS", "ENFORCES", "APPLIES_TO",
            "VIOLATES", "PROTECTS", "REFERENCES", "PART_OF",
        ],
        node_properties=["description"],
    )

    # Process a representative sample to limit API cost
    sample = chunks[:30]
    graph_docs = transformer.convert_to_graph_documents(sample)
    graph.add_graph_documents(
        graph_docs,
        baseEntityLabel=True,
        include_source=True,
    )

    total_nodes = sum(len(gd.nodes) for gd in graph_docs)
    total_rels = sum(len(gd.relationships) for gd in graph_docs)
    return total_nodes, total_rels


if __name__ == "__main__":
    print("=" * 60)
    print("Nigerian Student Rights — Knowledge Graph Ingestion")
    print("=" * 60)

    print(f"\n[1/4] Scanning {DATA_DIR} for PDFs...")
    documents = load_pdfs()
    print(f"       Total pages loaded: {len(documents)}")

    print("\n[2/4] Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"       Total chunks: {len(chunks)}")

    print("\n[3/4] Creating vector index in Neo4j...")
    create_vector_index(chunks)
    print("       Vector index created.")

    print("\n[4/4] Extracting knowledge graph entities...")
    nodes, rels = extract_knowledge_graph(chunks)
    if nodes or rels:
        print(f"       Nodes: {nodes}  |  Relationships: {rels}")

    print("\n✓ Ingestion complete. Run the chatbot with:")
    print("  streamlit run bot.py")
