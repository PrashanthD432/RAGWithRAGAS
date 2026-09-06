from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter


HEADERS_TO_SPLIT_ON = [
    ("#", "Header"),
    ("##", "Section"),
]


def split_documents(documents):
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    return splitter.split_text(documents)


def print_chunks(chunks):
    print("\n================ CHUNKS =================\n")
    for i, chunk in enumerate(chunks, start=1):
        print(f"Chunk {i}")
        print("Metadata:")
        print(chunk.metadata)
        print("Content:")
        print(chunk.page_content)
        print("-" * 60)


def create_embedding_model(model_name):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def demonstrate_embeddings(chunks, embedding_model):
    print("\n================ EMBEDDINGS =================\n")
    for i, chunk in enumerate(chunks, start=1):
        header = chunk.metadata.get("Header", "")
        section = chunk.metadata.get("Section", "")
        embedding_text = (
            f"Header: {header}\nSection: {section}\n\n"
            f"Content:\n{chunk.page_content}"
        )
        vector = embedding_model.embed_query(embedding_text)

        print(f"Chunk {i}")
        print(f"Header  : {header}")
        print(f"Section : {section}")
        print(f"Embedding dimension: {len(vector)}")
        print(f"First 10 values: {vector[:10]}")
        print()


def prepare_section_documents(chunks):
    section_documents = []
    for chunk in chunks:
        header = chunk.metadata.get("Header", "")
        section = chunk.metadata.get("Section", "")
        content = (
            f"Document Header: {header}\n"
            f"Document Section: {section}\n\n{chunk.page_content}"
        )
        section_documents.append(
            Document(
                page_content=content,
                metadata={"header": header, "section": section},
            )
        )
    return section_documents


def create_vector_db(section_documents, embedding_model, index_path):
    vector_db = FAISS.from_documents(
        documents=section_documents,
        embedding=embedding_model,
    )
    print("Embeddings stored successfully in FAISS")
    vector_db.save_local(index_path)
    print("FAISS index saved")
    return vector_db


def retrieve_documents(vector_db, question, top_k):
    retrieved_docs = vector_db.similarity_search(question, k=top_k)
    print("\n================ RETRIEVED CHUNKS =================\n")
    for i, doc in enumerate(retrieved_docs, start=1):
        print(f"Result {i}")
        print("Header:")
        print(doc.metadata.get("header"))
        print("Section:")
        print(doc.metadata.get("section"))
        print("Content:")
        print(doc.page_content)
        print("-" * 60)
    return retrieved_docs
