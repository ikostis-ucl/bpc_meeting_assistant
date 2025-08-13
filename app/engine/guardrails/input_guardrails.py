import json
import re
import unicodedata
from typing import Dict, Tuple

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

        # Initialize multilingual Groq model for domain classification
        self.guard_llm = Groq(
            model=args.groq_model_gr,
            api_key=args.groq_api_key,
            temperature=0.0
        )

        self.model_tpm = args.groq_model_gr_tpm
        self.token_counter = TokenCountingHandler()
        Settings.callback_manager = CallbackManager([self.token_counter])

        # Validation thresholds
        self.min_query_length = 3
        self.max_query_length = 500

        self.domain_thematics = self._load_extracted_thematics()
        self.thematic_context = self._create_thematic_context()

    def _load_extracted_thematics(self) -> Dict[str, str]:
        """Load domain thematics using Pareto Principle (70/30 rule) for selection."""
        try:
            with open(self.args.thematics_storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            all_thematics = data.get('thematics', {})
            frequency_data = data.get('thematic_frequency', {})

            if not all_thematics or not frequency_data:
                pprint_error("No thematics or frequency data found in storage file.")
                return {}

            # Calculate total occurrences
            total_occurrences = sum(frequency_data.values())
            target_coverage = total_occurrences * 0.7

            # Sort thematics by frequency (descending)
            sorted_thematics = sorted(
                frequency_data.items(),
                key=lambda x: x[1],
                reverse=True
            )

            # Select thematics until we reach 80% coverage
            selected_thematics = {}
            cumulative_count = 0

            for thematic_name, frequency in sorted_thematics:
                cumulative_count += frequency

                # Extract description from thematics data
                if thematic_name in all_thematics:
                    if isinstance(all_thematics[thematic_name], dict):
                        description = all_thematics[thematic_name].get('description', '')
                    else:
                        description = str(all_thematics[thematic_name])

                    selected_thematics[thematic_name] = description

                # Stop when we reach 70% coverage
                if cumulative_count >= target_coverage:
                    break

            coverage_percentage = (cumulative_count / total_occurrences) * 100
            pprint_debug(
                f"Pareto selection: {len(selected_thematics)} thematics covering {coverage_percentage:.1f}% of occurrences")

            return selected_thematics

        except FileNotFoundError:
            pprint_error(f"Thematics file not found: {self.args.thematics_storage_path}")
            return {}
        except json.JSONDecodeError:
            pprint_error(f"Invalid JSON in thematics file: {self.args.thematics_storage_path}")
            return {}
        except Exception as e:
            pprint_error(f"Error loading extracted thematics: {e}")
            return {}

    def _create_thematic_context(self) -> str:
        """Create formatted context using only extracted thematics."""
        context_parts = []

        for i, (thematic_name, description) in enumerate(self.domain_thematics.items(), 1):
            context_parts.append(f"{i}. {thematic_name}:\n{description}")

        return "\n\n".join(context_parts)

    def validate_query(self, query: str) -> Tuple[bool, str]:
        """Validate query using extracted thematics only."""
        # Basic validation
        basic_valid, basic_reason = self._basic_validation(query)
        if not basic_valid:
            return False, basic_reason

        # Domain classification using extracted thematics
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
                return True, ""  # Fallback to permissive

        except Exception as e:
            pprint_error(f"Classification failed: {e}")
            return True, ""  # Fallback to permissive

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
