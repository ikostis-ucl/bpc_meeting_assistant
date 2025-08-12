# app/engine/guardrails/input_guardrails.py
import json
import re
import unicodedata
from typing import Dict, Tuple

from llama_index.core import Settings
from llama_index.core.callbacks import TokenCountingHandler, CallbackManager
from llama_index.llms.groq import Groq

from app.utils.app_utils import pprint_error, pprint_console
from app.utils.inference_utils import throttle_requests


class InputGuardrails:
    """
    Enhanced input guardrails using Llama Guard 4 and domain-specific thematics.
    """

    def __init__(self, args):
        """Initialize guardrails with Llama Guard 4 and extracted thematics."""
        self.args = args

        # Initialize Llama Guard 4 for content screening
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

        # Load extracted domain thematics
        self.domain_thematics = self._load_domain_thematics()

        # Create thematic categories for Llama Guard
        self.thematic_categories = self._create_thematic_categories()

    def _load_domain_thematics(self) -> Dict[str, str]:
        """Load domain thematics from JSON file."""
        try:
            with open(self.args.thematics_storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                thematics = data.get('thematics', {})
                pprint_console(f"Loaded {len(thematics)} domain thematics for guardrails")
                return thematics
        except FileNotFoundError:
            pprint_error(f"Thematics file {self.args.thematics_storage_path} not found. Using fallback.")
            return self._get_fallback_thematics()
        except Exception as e:
            pprint_error(f"Error loading thematics file: {e}. Using fallback.")
            return self._get_fallback_thematics()

    def _get_fallback_thematics(self) -> Dict[str, str]:
        """Fallback thematics if extraction file is not available."""
        return {
            "Systemes_techniques_batiment": "Tous les aspects techniques du bâtiment incluant HVAC, électricité, plomberie, sprinklage, ventilation, chauffage, climatisation, systèmes de sécurité, éclairage et équipements techniques.",
            "Coordination_projet_communication": "Organisation d'équipes, réunions, échanges entre intervenants, communication client, coordination des métiers, planification collaborative et gestion des parties prenantes.",
            "Suivi_execution_travaux": "Avancement des travaux, supervision chantier, contrôle qualité, gestion des phases, suivi des délais, coordination des interventions et monitoring de l'exécution.",
            "Gestion_administrative_financiere": "Budgets, coûts, facturation, contrats, devis, aspects réglementaires, autorisations, conformité et gestion financière des projets.",
            "Conception_modification_plans": "Évolution des plans, adaptations techniques, changements de conception, validation des solutions, études techniques et modifications architecturales."
        }

    def _create_thematic_categories(self) -> str:
        """Create formatted thematic categories for Llama Guard prompt."""
        categories = []
        for i, (thematic, description) in enumerate(self.domain_thematics.items(), 1):
            categories.append(f"O{i}: {thematic}")
            categories.append(f"Définition: {description}")
            categories.append("")

        return "\n".join(categories)

    def validate_query(self, query: str) -> Tuple[bool, str]:
        """Comprehensive query validation using Llama Guard 4 and thematics."""
        # 1. Basic validation guardrails
        basic_valid, basic_reason = self._basic_validation(query)
        if not basic_valid:
            return False, basic_reason

        # 2. Llama Guard 4 thematic screening
        guard_valid, guard_reason = self._llama_guard_screening(query)
        if not guard_valid:
            return False, guard_reason

        return True, guard_reason

    @throttle_requests()
    def _llama_guard_screening(self, query: str) -> Tuple[bool, str]:
        """Use Llama Guard 4 to determine if query matches allowed thematics."""

        guard_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Tâche: Évaluer si la requête utilisateur correspond aux thématiques autorisées pour un assistant de gestion de comptes-rendus de réunions de projets de construction.

<|start_header_id|>assistant<|end_header_id|>

Je vais analyser la requête selon les thématiques autorisées définies ci-dessous.

# Politique de Sécurité

## Thématiques Autorisées

{self.thematic_categories}

## Instructions d'Évaluation

Évaluez si la requête utilisateur correspond à au moins une des thématiques autorisées (O1-O{len(self.domain_thematics)}). 

Une requête est considérée comme SÛRE si elle:
- Se rapporte directement à l'une des thématiques autorisées
- Concerne la gestion de projets de construction ou de réunions d'équipe
- Demande des informations sur les aspects techniques, organisationnels ou administratifs des projets

Une requête est considérée comme NON SÛRE si elle:
- Ne se rapporte à aucune des thématiques autorisées
- Concerne des sujets sans rapport avec la construction ou la gestion de projet
- Aborde des domaines extérieurs au périmètre professionnel défini

## Requête Utilisateur

{query}

## Évaluation

Fournissez votre évaluation au format suivant:

**safe** ou **unsafe**

Si unsafe, indiquez la raison: La requête ne correspond à aucune des thématiques autorisées pour cet assistant spécialisé en gestion de projets de construction.

<|eot_id|>"""

        try:
            response = self.guard_llm.complete(guard_prompt)
            response_text = response.text.strip().lower()

            if "unsafe" in response_text:
                return False, "Cette question ne correspond pas aux thématiques de gestion de projets de construction couvertes par cet assistant."
            elif "safe" in response_text:
                return True, ""
            else:
                # If response format is unclear, be permissive
                pprint_error(f"Unclear Llama Guard response: {response_text}")
                return True, ""

        except Exception as e:
            pprint_error(f"Llama Guard screening failed: {e}")
            # Fallback to permissive in case of error
            return True, ""

    def _basic_validation(self, query: str) -> Tuple[bool, str]:
        """Basic input validation rules."""
        # Length validation
        if len(query.strip()) < self.min_query_length:
            return False, f"Requête trop courte (minimum {self.min_query_length} caractères)"

        if len(query) > self.max_query_length:
            return False, f"Requête trop longue (maximum {self.max_query_length} caractères)"

        # Language detection (basic French check)
        if not self._is_likely_french(query):
            return False, "Veuillez formuler votre question en français"

        # Suspicious patterns
        if self._contains_suspicious_patterns(query):
            return False, "Format de requête non autorisé"

        return True, ""

    def _is_likely_french(self, text: str) -> bool:
        """Basic French language detection."""
        text = unicodedata.normalize('NFD', text.lower())
        french_chars = set('àâäæéèêëïîôöùûüÿçñ')
        french_words = {'le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'et', 'ou', 'que', 'qui', 'est', 'dans',
                        'sur', 'avec', 'pour', 'comment', 'pourquoi', 'quand', 'où', 'quel', 'quelle'}

        has_french_chars = any(char in french_chars for char in text)
        words = re.findall(r'\b\w+\b', text)
        has_french_words = any(word in french_words for word in words)

        return has_french_chars or has_french_words or len(words) <= 2

    def _contains_suspicious_patterns(self, query: str) -> bool:
        """Check for suspicious patterns."""
        suspicious_patterns = [
            r'<script.*?>', r'javascript:', r'sql.*?injection',
            r'union.*?select', r'drop.*?table', r'exec\s*\(',
            r'<.*?>', r'eval\s*\(', r'system\s*\('
        ]
        query_lower = query.lower()
        return any(re.search(pattern, query_lower, re.IGNORECASE) for pattern in suspicious_patterns)

    def get_rejection_message(self, reason: str) -> str:
        """Generate user-friendly rejection message."""
        base_message = "Désolé, votre question n'a pas pu être traitée.\n\n"

        thematic_list = "\n".join([f"• {name}: {desc[:100]}..."
                                   for name, desc in self.domain_thematics.items()])

        suggestion = f"""\n\nCet assistant est spécialisé dans les thématiques suivantes:

{thematic_list}

Veuillez reformuler votre question en rapport avec ces domaines."""

        return base_message + reason + suggestion

    def get_thematics_info(self) -> str:
        """Get information about loaded thematics for debugging."""
        if not self.domain_thematics:
            return "Aucune thématique chargée."

        info = f"Thématiques chargées ({len(self.domain_thematics)}):\n"
        for i, (name, desc) in enumerate(self.domain_thematics.items(), 1):
            info += f"  {i}. {name}: {desc[:80]}...\n"

        return info
