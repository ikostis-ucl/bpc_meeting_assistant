from typing import List

import numpy as np
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.postprocessor.cohere_rerank import CohereRerank


class TemporalRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and temporal search."""

    def __init__(self, embed_model, index, key,
                 alpha=0.1,
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
        self.reranker = CohereRerank(api_key=key, top_n=self._similarity_top_n)

        self.negative_answer = ("Je suis désolé, mais je n'ai pu trouver aucune information relative à votre demande. "
                                "Veuillez essayer de reformuler votre requête ou d'en modifier la période.")

        super().__init__()

    def _calculate_temporal_score(self, node):
        """
        Given a date span, calculate the temporal score for each node.
        """
        meeting_datetime = node.metadata.get('meeting_datetime')

        if self.datetime_span['start_date'] <= meeting_datetime <= self.datetime_span['end_date']:
            try:
                return self._alpha / (self.datetime_span['end_date'] - meeting_datetime)
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
        if __s_scores and __t_scores:
            __mu_s = np.mean(__s_scores)
            __std_s = np.std(__s_scores)
            __mu_t = np.mean(__t_scores)
            __std_t = np.std(__t_scores)
        else:
            __mu_s = 0
            __std_s = 1
            __mu_t = 0
            __std_t = 1

        nodes = []
        for node_id, node_score in nodes_with_scores.items():
            t_score = ((node_score["t_score"] - __mu_t) / __std_t) * __std_s + __mu_s
            s_score = node_score["s_score"]
            z_score = t_score + s_score
            nodes.append(NodeWithScore(node=self._index.docstore.docs[node_id], score=z_score))

        if nodes:
            sorted_nodes = sorted(nodes, key=lambda obj: obj.score, reverse=True)
            cutoff_index = int(len(sorted_nodes) * self._cutoff_percentage)
            if cutoff_index < self._similarity_top_n:
                cutoff_index = self._similarity_top_n
            result_nodes = sorted_nodes[:cutoff_index]
        else:
            result_nodes = []

        return result_nodes
