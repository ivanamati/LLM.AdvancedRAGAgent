import os 
from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.readers.file import PDFReader

load_dotenv()

#### Creating index for pdf
def get_index(data, index_name):
    index = None
    if not os.path.exists(index_name):
        print("building index", index_name)
        index = VectorStoreIndex.from_documents(data)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )
    return index

#### Reading the pdf with PDFReader
pdf_path = os.path.join("data", "Hrvatska.pdf")
croatia_pdf = PDFReader().load_data(file=pdf_path)

#### creating the engine for quering
croatia_index = get_index(croatia_pdf, "croatia")
croatia_engine = croatia_index.as_query_engine()