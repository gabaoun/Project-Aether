from typing import List, Sequence, Generator
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import BaseNode, Document
from llama_index.core.embeddings import BaseEmbedding
from src.utils.logger import logger

class SemanticDoubleMergingSplitter(SemanticSplitterNodeParser):
    """
    A semantic splitter that performs a second pass to merge small, 
    semantically similar chunks to ensure optimal context window usage.
    """
    min_chunk_size: int = 200
    similarity_threshold: float = 0.85

    def __init__(
        self, 
        embed_model: BaseEmbedding, 
        min_chunk_size: int = 200, 
        similarity_threshold: float = 0.85,
        **kwargs
    ):
        super().__init__(embed_model=embed_model, **kwargs)
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold

    def get_nodes_generator(self, documents: Sequence[Document], **kwargs) -> Generator[BaseNode, None, None]:
        """Yields nodes one by one to save memory, processing doc by doc."""
        for doc in documents:
            nodes = super().get_nodes_from_documents([doc], **kwargs)
            if not nodes:
                continue

            current_node = nodes[0]
            for i in range(1, len(nodes)):
                next_node = nodes[i]
                if len(current_node.get_content()) < self.min_chunk_size:
                    # Merge if too small
                    current_node.set_content(current_node.get_content() + "\n" + next_node.get_content())
                else:
                    yield current_node
                    current_node = next_node
            
            yield current_node

    def get_nodes_from_documents(self, documents: Sequence[Document], **kwargs) -> List[BaseNode]:
        return list(self.get_nodes_generator(documents, **kwargs))
