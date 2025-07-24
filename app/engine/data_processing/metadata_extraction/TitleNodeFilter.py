import re
from typing import List

from llama_index.core.schema import TransformComponent, BaseNode

from app.utils.app_utils import pprint_debug

CHAR_LIMITER = 40


class TitleNodeFilter(TransformComponent):
    """Filter out nodes that contain only titles or headers."""

    def __call__(self, nodes: List[BaseNode], **kwargs) -> List[BaseNode]:
        filtered_nodes = []
        for node in nodes:
            if not self.is_title_node(node.text):
                filtered_nodes.append(node)
            # else:
            #     pprint_debug(f"Excluding title node: {repr(node.text[:100])}")
        return filtered_nodes

    @staticmethod
    def is_title_node(content: str) -> bool:
        """Check if a node contains only title/header content."""
        content = content.strip()

        # Filter very short content (likely titles)
        if len(content) < CHAR_LIMITER:
            return True

        # Filter markdown headers only
        if re.match(r'^#+\s+.*$', content) and '\n' not in content:
            return True

        # Filter content with limited sentence structure
        sentence_count = len(re.findall(r'[.!?]+', content))
        if sentence_count < 2 and len(content) < CHAR_LIMITER:
            return True

        # Filter all caps content (common for titles)
        if content.isupper() and len(content) < CHAR_LIMITER:
            return True

        return False
