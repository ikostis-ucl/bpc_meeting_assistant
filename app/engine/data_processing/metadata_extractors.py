from time import sleep
from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.async_utils import run_jobs
from llama_index.core.bridge.pydantic import (
    Field,
    SerializeAsAny,
)
from llama_index.core.extractors.interface import BaseExtractor
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.settings import Settings

# Template for keyword extraction in French
DEFAULT_KEYWORD_EXTRACT_TEMPLATE = """\
{context_str}. En français uniquement, donnez à {keywords} des mots-clés uniques pour ce document. \
Formulez-les en les séparant par des virgules. Mots-clés : """


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
        metadata_list = []
        # Process each node to find matching entities
        for node in nodes:
            _inv_parties = []
            _text = node.text.split()
            # Check each entity against the node text
            for ent in self.entities:
                if ent in _text:
                    _inv_parties.append(ent)
            metadata_list.append({"involved_parties": _inv_parties})

        return metadata_list


class KeywordExtractor(BaseExtractor):
    """
    Extracts keywords from document nodes using an LLM.
    Node-level extractor that populates the 'excerpt_keywords' metadata field.

    Args:
    llm (Optional[LLM]): LLM
    keywords (int): number of keywords to extract
    prompt_template (str): template for keyword extraction
    """

    llm: SerializeAsAny[LLM] = Field(description="The LLM to use for generation.")
    keywords: int = Field(
        default=5, description="The number of keywords to extract.", gt=0
    )
    prompt_template: str = Field(
        default=DEFAULT_KEYWORD_EXTRACT_TEMPLATE,
        description="Prompt template to use when generating keywords.",
    )

    def __init__(
            self,
            llm: Optional[LLM] = None,
            llm_predictor: Optional[LLM] = None,
            keywords: int = 5,
            prompt_template: str = DEFAULT_KEYWORD_EXTRACT_TEMPLATE,
            num_workers: int = 1,
            **kwargs: Any,
    ) -> None:
        """
        Initialize the KeywordExtractor.

        Args:
            llm: Language model for keyword generation.
            llm_predictor: Alternative LLM predictor (deprecated).
            keywords: Number of keywords to extract per node.
            prompt_template: Template string for keyword extraction prompt.
            num_workers: Number of parallel workers for extraction.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If keywords is less than 1.
        """
        if keywords < 1:
            raise ValueError("num_keywords must be >= 1")

        super().__init__(
            llm=llm or llm_predictor or Settings.llm,
            keywords=keywords,
            prompt_template=prompt_template,
            num_workers=num_workers,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        """
        Get the name of the extractor class.

        Returns:
            str: Class name string.
        """
        return "KeywordExtractor"

    async def _aextract_keywords_from_node(self, node: BaseNode) -> Dict[str, str]:
        """
        Extract keywords from a single node.

        Args:
            node: Document node to process.

        Returns:
            Dict[str, str]: Dictionary containing extracted keywords or empty if node type mismatch.
        """
        # Skip non-text nodes if text-only mode is enabled
        if self.is_text_node_only and not isinstance(node, TextNode):
            return {}

        # Extract content and generate keywords using LLM
        context_str = node.get_content(metadata_mode=self.metadata_mode)
        keywords = await self.llm.apredict(
            PromptTemplate(template=self.prompt_template),
            keywords=self.keywords,
            context_str=context_str,
        )

        return {"excerpt_keywords": keywords.strip()}

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        """
        Asynchronously extract keywords from multiple nodes.

        Args:
            nodes: Sequence of document nodes to process.

        Returns:
            List[Dict]: List of dictionaries containing extracted keywords for each node.
        """
        # Create extraction jobs for each node
        keyword_jobs = []
        for node in nodes:
            keyword_jobs.append(self._aextract_keywords_from_node(node))

        # Run extraction jobs in parallel
        metadata_list: List[Dict] = await run_jobs(
            keyword_jobs, show_progress=self.show_progress, workers=self.num_workers
        )

        # Add delay after processing
        sleep(10)
        return metadata_list
