import datetime
from typing import List

from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import VectorStoreQuery

from app.utils.temporal_retriever_utils import parse_datetime


class TemporalRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(self, embed_model, vector_store,
                 alpha=0.5,
                 query_mode='default',
                 similarity_top_k=20):
        # TODO: Test k-comptime relation, adjust default k

        self._vector_store = vector_store
        self._embed_model = embed_model
        self._alpha = alpha
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        self.datetime_span = None
        super().__init__()

    def set_datetime_span(self, end_date=None, start_date=None):
        if start_date is None:
            s_date = datetime.datetime(1970, 1, 1)
        else:
            s_date = parse_datetime(start_date)

        if end_date is None:
            e_date = datetime.datetime.now()
        else:
            e_date = parse_datetime(end_date)

        self.datetime_span = {"start_date": s_date, "end_date": e_date}

    def _calculate_temporal_score(self, node):
        """
        Given a date span, calculate the temporal score for each node.
        """
        # TODO: Finish this

        if (self.datetime_span[0].timestamp()
                <= node.metadata["meeting_datetime"].timestamp()
                <= self.datetime_span[1].timestamp()):

            tau = self._alpha / (self.datetime_span[1].timestamp() - node.metadata["meeting_datetime"].timestamp())


        pass

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        if query_bundle.embedding is None:
            query_embedding = self._embed_model.get_query_embedding(
                query_bundle.query_str
            )
        else:
            query_embedding = query_bundle.embedding

        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = self._vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores
