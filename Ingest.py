# --> retriever engine
import os
from collections import defaultdict
import torch
import chromadb

os.environ["NLTK_DATA"] = os.path.expanduser("~/nltk_data")

from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.chroma import ChromaVectorStore

#  --> creating Storage

DIR_PATH = "./storage/data"
DB_PATH = "./storage/vectorDB"
DOCSTORE_PATH = "./storage/docstore.json" #--> to keep track of files

os.makedirs(DIR_PATH,exist_ok=True)
os.makedirs(DB_PATH,exist_ok=True)


# --> Defining Global default settings
device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device=device
)
Settings.embed_model = embed_model

Settings.node_parser = SemanticSplitterNodeParser(
    embed_model = embed_model,
    buffer_size = 3, #-->BUFFER SIZE (The Context Window for Splitting):Instead of looking at just one sentence to decide on a split, the parser groups this many consecutive sentences together to calculate a "smoothed" similarity score
    breakpoint_percentile_threshold=80 #-->BREAKPOINT PERCENTILE THRESHOLD (The Strictness of the Split): The parser calculates the cosine similarity between all consecutive groups. It then finds the similarity score that sits at this percentile.
)

# --> initializing Chroma Client
chroma_client = chromadb.PersistentClient(path=DB_PATH)
chroma_collection = chroma_client.get_or_create_collection(name="CORE")

# --> wrapping chroma client into llama-index
vectorstore = ChromaVectorStore(chroma_collection)


# --> loading docstore for file tracking,or creating if not found

if os.path.exists(DOCSTORE_PATH):
    docstore = SimpleDocumentStore.from_persist_path(DOCSTORE_PATH)
    print(f"Loaded existing docstore cache")
else:
    docstore = SimpleDocumentStore()
    print(f"Created a new docstore")


def reconcile_and_purge(data_dir: str, docstore: SimpleDocumentStore, chroma_collection):
    """Remove docstore entries and Chroma vectors for PDFs deleted from disk."""

    # 1. Files currently on disk
    current_files = {f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")}

    # 2. Files tracked in the docstore, mapped to their doc_ids
    tracked = defaultdict(list)
    for doc_id, doc in docstore.docs.items():
        if name := doc.metadata.get("file_name"):
            tracked[name].append(doc_id)

    # 3. Files that were deleted from disk
    missing_files = tracked.keys() - current_files
    if not missing_files:
        print("Reconciliation complete: No deleted files found.")
        return

    print(f"Reconciliation: Found {len(missing_files)} deleted file(s). Purging...")

    for name in missing_files:
        # Remove vectors from ChromaDB
        try:
            chroma_collection.delete(where={"file_name": name})
        except Exception as e:
            print(f"Chroma delete error for {name}: {e}")
        # Remove tracking records from the docstore
        for doc_id in tracked[name]:
            docstore.delete_document(doc_id, raise_error=False)

    print("Reconciliation and purge complete.")

reconcile_and_purge(DIR_PATH, docstore, chroma_collection) #--> checks for ghost entry and purge them
docstore.persist(DOCSTORE_PATH) #--> persist after purge

# --> creating ingestion PipeLine
pipeline = IngestionPipeline(
    transformations=[
        Settings.node_parser,
        Settings.embed_model,
    ],
    vector_store=vectorstore,
    docstore=docstore,
    docstore_strategy=DocstoreStrategy.UPSERTS,
) #--> ingestion pipeline does not require storagecontext which is used for inserting data into vectorDB
# 1. Pipeline automatically checks hashes:
#    - Does nothing if files already exist in docstore
#    - Embeds only new / modified files into ChromaDB

# pdf reader using pdf parser from disk
pdf_reader = SimpleDirectoryReader(
    input_dir=DIR_PATH,file_extractor={".pdf":PyMuPDFReader()},required_exts=[".pdf"]
)
documents = pdf_reader.load_data() #--> contains all the extracted pdf chuck

if not documents:
    print(f"[Ingest] No PDF documents found in '{DIR_PATH}'. Place .pdf files there to ingest.")
else:
    nodes = pipeline.run(documents=documents, show_progress=True) 
    print(f"Ingestion complete. Total semantic chunks: {len(nodes)}")

# Persist docstore with all newly ingested nodes and hashes
docstore.persist(DOCSTORE_PATH)
print("[Ingest] Docstore cache persisted to disk.")
