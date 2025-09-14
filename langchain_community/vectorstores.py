# This is a shim to support the deprecated import of Chroma using langchain-chroma package.
try:
    from langchain_chroma import Chroma as BaseChroma
except ImportError:
    import warnings
    warnings.warn("langchain_chroma not installed. Using a dummy Chroma implementation.")
    class BaseChroma:
        def __init__(self, persist_directory, embedding_function, **kwargs):
            self.persist_directory = persist_directory
            self.embedding_function = embedding_function
        def as_retriever(self, **kwargs):
            # Dummy retriever that returns None or can be expanded as needed
            return None

# Re-export Chroma so that 'from langchain_community.vectorstores import Chroma' works
Chroma = BaseChroma
