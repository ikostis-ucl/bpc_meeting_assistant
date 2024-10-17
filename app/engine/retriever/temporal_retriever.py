import datetime
from typing import List

import numpy as np
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.postprocessor.cohere_rerank import CohereRerank

from app.utils.temporal_retriever_utils import parse_datetime


class TemporalRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and temporal search."""

    def __init__(self, embed_model, index, key,
                 alpha=1,
                 query_mode='default',
                 similarity_top_n=5,
                 similarity_top_k=200,
                 cutoff_percentage=0.05):

        self._index = index
        self._embed_model = embed_model
        self._alpha = alpha
        self._query_mode = query_mode
        self._similarity_top_n = similarity_top_n
        self._similarity_top_k = similarity_top_k
        self._cutoff_percentage = cutoff_percentage if 0 <= cutoff_percentage <= 1 else 0.05
        self.datetime_span = None
        self.reranker = CohereRerank(api_key=key, top_n=self._similarity_top_k)
        super().__init__()

    def set_datetime_span(self, start_date=None, end_date=None):
        """
        Run this before querying to set the date.
        """
        if start_date is None or start_date == "":
            s_date = datetime.datetime(1970, 1, 1)
        elif isinstance(start_date, datetime.datetime):
            s_date = start_date
        else:
            s_date = parse_datetime(start_date)

        if end_date is None or end_date == "":
            e_date = datetime.datetime.now()
        elif isinstance(end_date, datetime.datetime):
            e_date = end_date
        else:
            e_date = parse_datetime(end_date)

        self.datetime_span = {"start_date": s_date, "end_date": e_date}

    def _calculate_temporal_score(self, node):
        """
        Given a date span, calculate the temporal score for each node.
        """
        meeting_datetime = datetime.datetime(day=int(node.metadata["meeting_datetime"][0]),
                                             month=int(node.metadata["meeting_datetime"][1]),
                                             year=int(node.metadata["meeting_datetime"][2]))
        if (self.datetime_span['start_date'].timestamp()
                <= meeting_datetime.timestamp()
                <= self.datetime_span['end_date'].timestamp()):

            try:
                return self._alpha / (self.datetime_span['end_date'].timestamp() - meeting_datetime.timestamp())
            except ZeroDivisionError:
                return self._alpha / 0.99  # timestamp = how many seconds have passed from 01/01/1970 00:00:00
        else:
            return None

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        if query_bundle.embedding is None:
            query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        else:
            query_embedding = query_bundle.embedding

        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = self._index.vector_store.query(vector_store_query)

        unranked_nodes = []
        nodes_with_scores = {}
        for result_index, node_id in enumerate(query_result.ids):
            if query_result.similarities is not None:
                t_score = self._calculate_temporal_score(self._index.docstore.docs[node_id])
                if t_score is None:
                    continue
                s_score = query_result.similarities[result_index]
                unranked_nodes.append(
                    NodeWithScore(node=self._index.docstore.docs[node_id], score=s_score))
                nodes_with_scores[node_id] = {"s_score": s_score, "t_score": t_score}

        self.reranker.top_n = len(unranked_nodes)
        ranked_nodes = self.reranker._postprocess_nodes(nodes=unranked_nodes, query_bundle=query_bundle)
        for node in ranked_nodes:
            nodes_with_scores[node.node_id]['s_score'] = node.score

        __s_scores = [s["s_score"] for s in nodes_with_scores.values()]
        __t_scores = [t["t_score"] for t in nodes_with_scores.values()]
        __mu_s = np.mean(__s_scores)
        __std_s = np.std(__s_scores)
        __mu_t = np.mean(__t_scores)
        __std_t = np.std(__t_scores)

        nodes = []
        for node_id, node_score in nodes_with_scores.items():
            t_score = ((node_score["t_score"] - __mu_t) / __std_t) * __std_s + __mu_s
            s_score = node_score["s_score"]
            n_score = t_score + s_score
            nodes.append(NodeWithScore(node=self._index.docstore.docs[node_id], score=n_score))

        sorted_nodes = sorted(nodes, key=lambda obj: obj.score, reverse=True)
        cutoff_index = int(len(sorted_nodes) * self._cutoff_percentage)
        if cutoff_index < self._similarity_top_n:
            cutoff_index = self._similarity_top_n
        result_nodes = sorted_nodes[:cutoff_index]

        return result_nodes
