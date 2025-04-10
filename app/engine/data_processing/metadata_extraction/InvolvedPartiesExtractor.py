from typing import List, Any, Dict

from llama_index.core.extractors import BaseExtractor
from pydantic import Field


class InvolvedPartiesExtractor(BaseExtractor):
    """
    Extractor for identifying involved parties in a document.
    Searches for specific entity names within the document text.
    """

    entities: List[str] = Field(
        default=None,
        description="A list of the names of the parties contained within the first page of the document.",
    )

    def __init__(self, entities, **kwargs: Any):
        """
        Initialize the InvolvedPartiesExtractor.

        Args:
            entities (List[str]): List of entity names to search for in the document.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.entities = entities

    async def aextract(self, nodes) -> List[Dict]:
        """
        Asynchronously extract involved parties from document nodes.

        Args:
            nodes: Collection of document nodes to process.

        Returns:
            List[Dict]: List of dictionaries containing involved parties for each node.
        """
        import re
        metadata_list = []

        # Compile regex patterns for each entity once
        entity_patterns = {
            entity: re.compile(r'\b' + re.escape(entity) + r'\b')
            for entity in self.entities
        }

        # Process each node to find matching entities
        for node in nodes:
            _inv_parties = []

            # Check for each entity pattern in the node text
            for entity, pattern in entity_patterns.items():
                if pattern.search(node.text):
                    _inv_parties.append(entity)

            metadata_list.append({"involved_parties": _inv_parties})

        return metadata_list
