import json
import os
import re
import unicodedata
from collections import defaultdict
from typing import Dict, Tuple, List

from llama_index.core import Settings
from llama_index.core.callbacks import TokenCountingHandler, CallbackManager
from llama_index.llms.groq import Groq

from app.utils.app_utils import pprint_error, pprint_debug
from app.utils.inference_utils import throttle_requests


class InputGuardrails:
    """
    Input guardrails using exclusively extracted domain thematics.
    """

    def __init__(self, args):
        """Initialize guardrails with multilingual Groq model and extracted thematics only."""
        self.args = args

        self.guard_llm = Groq(
            model=args.groq_model_gr,
            api_key=args.groq_api_key,
            temperature=0.0
        )

        self.model_tpm = args.groq_model_gr_tpm
        self.token_counter = TokenCountingHandler()
        Settings.callback_manager = CallbackManager([self.token_counter])

        self.min_query_length = 3
        self.max_query_length = 500

        self.domain_thematics = self._load_extracted_thematics()
        self.thematic_context = self._create_thematic_context()

    def _normalize_and_merge_thematics(self, thematics_data: Dict) -> Dict[str, str]:
        """
        Normalize thematic titles and merge duplicates with LLM-assisted description merging.

        Args:
            thematics_data: Raw thematics data from JSON file

        Returns:
            Dict[str, str]: Normalized and merged thematics with descriptions
        """

        # Helper function to normalize titles
        def normalize_title(title: str) -> str:
            normalized = re.sub(r'[*_\-]+', '', title)
            normalized = re.sub(r'[\s_]+', '_', normalized.lower())
            normalized = normalized.strip('_')
            return normalized

        grouped_thematics = defaultdict(list)

        for title, data in thematics_data['thematics'].items():
            normalized_title = normalize_title(title)
            grouped_thematics[normalized_title].append({
                'original_title': title,
                'description': data['description'],
                'frequency': data['frequency']
            })

        # Merge duplicates
        merged_thematics = {}
        merged_frequencies = {}

        for normalized_title, duplicates in grouped_thematics.items():
            if len(duplicates) == 1:
                duplicate = duplicates[0]
                merged_thematics[duplicate['original_title']] = duplicate['description']
                merged_frequencies[duplicate['original_title']] = duplicate['frequency']
            else:
                total_frequency = sum(d['frequency'] for d in duplicates)
                descriptions = [d['description'] for d in duplicates]
                original_titles = [d['original_title'] for d in duplicates]

                canonical_title = max(duplicates, key=lambda x: x['frequency'])['original_title']

                merged_description = self._merge_descriptions_with_llm(descriptions, canonical_title)
                merged_thematics[canonical_title] = merged_description
                merged_frequencies[canonical_title] = total_frequency

                pprint_debug(
                    f"Merged {len(duplicates)} duplicates into '{canonical_title}' (total frequency: {total_frequency})")
                pprint_debug(f"Original titles: {original_titles}")

        self._merged_frequencies = merged_frequencies
        return merged_thematics

    def _merge_descriptions_with_llm(self, descriptions: List[str], canonical_title: str) -> str:
        """
        Use Groq LLM to merge multiple descriptions into a single coherent one.

        Args:
            descriptions: List of descriptions to merge
            canonical_title: The canonical title for context

        Returns:
            str: Merged description
        """
        merge_prompt = f"""Tu es un expert en fusion de descriptions thématiques pour des projets de construction.
    
        TITRE DE LA THÉMATIQUE: {canonical_title}
    
        DESCRIPTIONS À FUSIONNER:
        {chr(10).join(f"Description {i + 1}: {desc}" for i, desc in enumerate(descriptions))}
    
        INSTRUCTIONS:
        1. Fusionne ces descriptions en une seule description cohérente et complète
        2. Conserve tous les aspects importants mentionnés dans chaque description
        3. Élimine les redondances tout en préservant les nuances
        4. Maintiens le même format et style que les descriptions originales
        5. Commence par "Description : "
        6. Garde la même langue que les descriptions originales
    
        DESCRIPTION FUSIONNÉE:"""

        try:
            response = self.guard_llm.complete(merge_prompt)
            merged_description = response.text.strip()

            if not merged_description.startswith("Description :"):
                merged_description = f"Description : {merged_description}"

            return merged_description

        except Exception as e:
            pprint_error(f"LLM merge failed for '{canonical_title}': {e}")
            # Fallback: concatenate descriptions with separator
            return f"Description : {' | '.join(desc.replace('Description : ', '').strip() for desc in descriptions)}"

    def _apply_pareto_principle(self, merged_thematics: Dict[str, str], thematics_data: Dict) -> Dict[str, str]:
        """
        Apply Pareto principle to select the most important thematics (80/20 rule).

        Args:
            merged_thematics: Normalized and merged thematics
            thematics_data: Original data for frequency information

        Returns:
            Dict[str, str]: Top thematics following Pareto principle
        """
        thematic_frequencies = [(title, self._merged_frequencies[title])
                                for title in merged_thematics.keys()]

        thematic_frequencies.sort(key=lambda x: x[1], reverse=True)

        total_thematics = len(thematic_frequencies)
        pareto_count = max(1, int(total_thematics * 0.2))

        selected_thematics = {}
        total_frequency = sum(freq for _, freq in thematic_frequencies)
        cumulative_frequency = 0

        pprint_debug(f"Applying Pareto principle: selecting top {pareto_count} out of {total_thematics} thematics")

        for i, (title, frequency) in enumerate(thematic_frequencies[:pareto_count]):
            selected_thematics[title] = merged_thematics[title]
            cumulative_frequency += frequency

            pprint_debug(f"Selected: {title} (frequency: {frequency})")

        coverage_percentage = (cumulative_frequency / total_frequency) * 100
        pprint_debug(f"Pareto selection covers {coverage_percentage:.1f}% of total frequency")

        return selected_thematics

    @staticmethod
    def _titles_match(title1: str, title2: str) -> bool:
        """Check if two titles match after normalization."""
        import re

        def normalize(title):
            normalized = re.sub(r'[*_\-]+', '', title)
            normalized = re.sub(r'[\s_]+', '_', normalized.lower())
            return normalized.strip('_')

        return normalize(title1) == normalize(title2)

    def _load_extracted_thematics(self) -> Dict[str, str]:
        """Load and process extracted thematics from JSON file."""
        try:
            with open(self.args.thematics_storage_path, 'r', encoding='utf-8') as f:
                thematics_data = json.load(f)

            pprint_debug(f"Loaded {len(thematics_data['thematics'])} raw thematics")

            if os.path.exists(self.args.merged_thematics_storage_path):
                pprint_debug("Loading existing merged thematics")
                with open(self.args.merged_thematics_storage_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)

                merged_thematics = cached_data.get('merged_thematics', {})
                self._merged_frequencies = cached_data.get('merged_frequencies', {})

                pprint_debug(f"Loaded {len(merged_thematics)} merged thematics from file")
            else:
                pprint_debug("No merged thematics found, running merge procedure")
                merged_thematics = self._normalize_and_merge_thematics(thematics_data)
                pprint_debug(f"After normalization and merging: {len(merged_thematics)} unique thematics")

                cache_data = {
                    'merged_thematics': merged_thematics,
                    'merged_frequencies': self._merged_frequencies
                }
                with open(self.args.merged_thematics_storage_path, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, indent=2, ensure_ascii=False)
                pprint_debug(f"Saved merged thematics to {self.args.merged_thematics_storage_path}")

            final_thematics = self._apply_pareto_principle(merged_thematics, thematics_data)
            pprint_debug(f"After Pareto principle: {len(final_thematics)} selected thematics")

            return final_thematics

        except FileNotFoundError:
            pprint_error(f"Thematics file not found: {self.args.thematics_storage_path}")
            return {}
        except Exception as e:
            pprint_error(f"Error loading thematics: {e}")
            return {}

    def _create_thematic_context(self) -> str:
        """Create formatted context using only extracted thematics."""
        context_parts = []

        for i, (thematic_name, description) in enumerate(self.domain_thematics.items(), 1):
            context_parts.append(f"{i}. {thematic_name}:\n{description}")

        return "\n\n".join(context_parts)

    def validate_query(self, query: str) -> Tuple[bool, str]:
        """Validate query using extracted thematics only."""
        basic_valid, basic_reason = self._basic_validation(query)
        if not basic_valid:
            return False, basic_reason

        domain_valid, domain_reason = self._domain_classification(query)
        if not domain_valid:
            return False, domain_reason

        return True, ""

    def _basic_validation(self, query: str) -> Tuple[bool, str]:
        """Basic validation checks."""
        query = re.sub(r'\s+', ' ', query.strip())

        if len(query) < self.min_query_length:
            return False, "too_short"

        if len(query) > self.max_query_length:
            return False, "too_long"

        normalized = unicodedata.normalize('NFKD', query)
        special_char_ratio = len(re.findall(r'[^\w\s\-\.\,\?\!\:\;\(\)]', normalized)) / len(query)
        if special_char_ratio > 0.3:
            return False, "invalid_format"

        if re.search(r'(.)\1{10,}', query):
            return False, "invalid_format"

        return True, ""

    @throttle_requests()
    def _domain_classification(self, query: str) -> Tuple[bool, str]:
        """Classify query using only extracted thematics."""

        classification_prompt = f"""Analysez cette requête par rapport aux thématiques extraites du projet.

            THÉMATIQUES DU PROJET (extraites automatiquement des documents):
            
            {self.thematic_context}
            
            INSTRUCTIONS:
            - Ces thématiques ont été extraites des documents réels du projet
            - Vérifiez si la requête correspond à AU MOINS UNE de ces thématiques
            - Questions autorisées: tout ce qui correspond aux thématiques ci-dessus
            - Questions interdites: tout ce qui ne correspond à AUCUNE thématique
            
            REQUÊTE: "{query}"
            
            Cette requête correspond-elle à au moins une des thématiques extraites du projet?
            
            Répondez uniquement: OUI ou NON
            
            Réponse:"""

        try:
            response = self.guard_llm.complete(classification_prompt)
            response_text = response.text.strip().upper()

            if "NON" in response_text:
                return False, "domain_mismatch"
            elif "OUI" in response_text:
                return True, ""
            else:
                return True, ""

        except Exception as e:
            pprint_error(f"Classification failed: {e}")
            return True, ""

    def get_rejection_message(self, reason: str) -> str:
        """Generate rejection message using extracted thematic names."""

        thematic_list = "\n".join([f"• {name.replace('_', ' ')}" for name in self.domain_thematics.keys()])

        rejection_messages = {
            "too_short": "Question trop courte. Veuillez formuler une question plus détaillée.",

            "too_long": "Question trop longue. Veuillez la raccourcir.",

            "invalid_format": "Format invalide. Utilisez du texte lisible.",

            "domain_mismatch": (
                f"Cette question ne correspond pas aux thématiques du projet. "
                f"Je peux vous aider avec:\n\n{thematic_list}\n\n"
                f"Veuillez reformuler votre question selon ces thématiques."
            )
        }

        return rejection_messages.get(reason, "Question non autorisée.")

    def get_domain_summary(self) -> str:
        """Summary of extracted thematics."""
        summary = f"Thématiques extraites ({len(self.domain_thematics)}):\n\n"
        for i, (name, desc) in enumerate(self.domain_thematics.items(), 1):
            summary += f"{i}. {name.replace('_', ' ')}:\n   {desc[:100]}...\n\n"
        return summary
