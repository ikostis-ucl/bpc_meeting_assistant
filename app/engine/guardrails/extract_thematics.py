import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List
import re

from llama_index.core import Settings
from llama_index.core.callbacks import TokenCountingHandler, CallbackManager
from llama_index.llms.groq import Groq
from tqdm import tqdm

from app.engine.data_processing.data_loaders import load_index
from app.utils.app_utils import pprint_console, pprint_error
from app.utils.inference_utils import throttle_requests


class DomainThematicsExtractor:
    # TODO: This adds any new thematics to the existing. Make a version that takes the union of thematics per document.
    """
    Extract domain-specific thematics from meeting documents to build productive guardrails vocabulary.
    """

    def __init__(self, args):
        """Initialize the thematics' extractor."""
        self.args = args

        # Initialize LLM for thematic extraction
        self.extraction_llm = Groq(
            model=args.groq_model_indexing_extraction,
            api_key=args.groq_api_key,
            temperature=0.0
        )

        self.model_tpm = self.args.groq_model_indexing_extraction_tpm
        self.token_counter = TokenCountingHandler()
        Settings.callback_manager = CallbackManager([self.token_counter])

        self.index, _ = load_index(self.args)

        os.makedirs(os.path.dirname(self.args.thematics_storage_path), exist_ok=True)
        self.thematics = {}

    def extract_thematics_from_index(self) -> Dict[str, str]:
        """
        Extract and build thematics from all documents in the index.

        Returns:
            Dict[str, str]: Dictionary of thematics with their descriptions
        """

        pprint_console(f"Processing {len(self.index.docstore.docs)} documents...")

        # Group documents by file name
        documents_by_file = self._group_documents_by_file()

        pprint_console(f"Found {len(documents_by_file)} unique documents")

        # Load existing thematics if file exists
        self._load_existing_thematics()

        # Process each document
        processed_documents = 0
        for file_name, nodes in tqdm(documents_by_file.items(), desc="[THEMATIC_EXTR] Processing documents"):
            try:
                # Combine all text from nodes of this document
                document_text = self._combine_document_nodes(nodes)

                # Extract thematics from this document
                new_thematics = self._extract_document_thematics(document_text, file_name)

                # Update existing thematics
                if new_thematics:
                    self._update_thematics(new_thematics)
                    self._save_thematics()

                processed_documents += 1

                if processed_documents % 5 == 0:  # Save periodically
                    self._save_thematics()

            except Exception as e:
                pprint_error(f"Error processing document {file_name}: {e}")
                continue

        # Final save
        self._save_thematics()

        pprint_console(
            f"Extraction completed! Found {len(self.thematics)} thematics across {processed_documents} documents")
        return self.thematics

    def _group_documents_by_file(self) -> Dict[str, List]:
        """Group index nodes by document file name."""
        documents = defaultdict(list)

        for node in self.index.docstore.docs.values():
            if hasattr(node, 'metadata') and node.metadata:
                file_name = node.metadata.get("file_name")
                if file_name:
                    documents[file_name].append(node)

        return dict(documents)

    @staticmethod
    def _combine_document_nodes(nodes: List) -> str:
        """Combine text from all nodes of a document."""
        document_text = []

        for node in nodes:
            if hasattr(node, 'text') and node.text:
                document_text.append(node.text.strip())

        return '\n\n'.join(document_text)

    def _analyze_document_structure(self, document_text: str) -> Dict[str, List[str]]:
        """Analyze the Markdown structure to extract section information."""

        structure = {
            'main_sections': [],
            'subsections': [],
            'dates_found': [],
            'key_topics': []
        }

        lines = document_text.split('\n')

        for line in lines:
            line = line.strip()

            # Extract main sections (# titles)
            if line.startswith('# '):
                section = line[2:].strip()
                if section and section not in structure['main_sections']:
                    structure['main_sections'].append(section)

            # Extract subsections (## ### titles)
            elif line.startswith('##'):
                subsection = re.sub(r'^#+\s*', '', line).strip()
                if subsection and subsection not in structure['subsections']:
                    structure['subsections'].append(subsection)

            # Extract dates (DD/MM/YY format)
            date_matches = re.findall(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', line)
            structure['dates_found'].extend(date_matches)

            # Extract potential key topics (capitalized words that might be technical terms)
            if len(line) > 10 and not line.startswith('#'):
                key_words = re.findall(r'\b[A-Z][a-zàâäæéèêëïîôöùûüÿç]{3,}\b', line)
                structure['key_topics'].extend(key_words)

        # Remove duplicates and limit size
        for key in structure:
            if isinstance(structure[key], list):
                structure[key] = list(set(structure[key]))[:10]  # Limit to 10 items

        return structure

    def _load_existing_thematics(self):
        """Load existing thematics from file if it exists."""
        try:
            if os.path.exists(self.args.thematics_storage_path):
                with open(self.args.thematics_storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.thematics = data.get('thematics', {})
                    pprint_console(f"Loaded {len(self.thematics)} existing thematics")
            else:
                self.thematics = {}
                pprint_console("No existing thematics file found, starting fresh")
        except Exception as e:
            pprint_error(f"Error loading existing thematics: {e}")
            self.thematics = {}

    @throttle_requests()
    def _extract_document_thematics(self, document_text: str, file_name: str) -> Dict[str, str]:
        """Extract thematics from a single document using LLM with enhanced Markdown awareness."""

        # Analyze document structure
        doc_structure = self._analyze_document_structure(document_text)

        # Prepare existing thematics context
        existing_context = ""
        if self.thematics:
            existing_list = []
            for i, (thematic, description) in enumerate(self.thematics.items(), 1):
                existing_list.append(f"S{i}: {thematic}")
                existing_list.append(f"{description}")

            existing_context = f"""
    THÉMATIQUES EXISTANTES:
    {chr(10).join(existing_list)}

    """

        # Prepare structure context
        structure_context = f"""
    STRUCTURE DU DOCUMENT DÉTECTÉE:
    - Sections principales: {', '.join(doc_structure['main_sections'][:5]) if doc_structure['main_sections'] else 'Aucune'}
    - Sous-sections: {', '.join(doc_structure['subsections'][:5]) if doc_structure['subsections'] else 'Aucune'}
    - Dates trouvées: {len(doc_structure['dates_found'])} entrées temporelles
    - Termes techniques identifiés: {', '.join(doc_structure['key_topics'][:8]) if doc_structure['key_topics'] else 'Aucun'}

    """

        extraction_prompt = f"""Tu es un expert en analyse de projets de construction et de réunions d'équipe.

    {existing_context}{structure_context}Analyse le document suivant (fichier: {file_name}) et identifie les THÉMATIQUES principales abordées.

    Si le document est au format Markdown, utilise la structure détectée ci-dessus pour mieux comprendre l'organisation du contenu et identifier les thématiques pertinentes. Sinon, analyse le contenu tel qu'il est présenté.

    OBJECTIF: Créer des thématiques LARGES et INCLUSIVES qui permettront aux utilisateurs de poser des questions variées sans être trop restrictifs.

    Une thématique représente un DOMAINE ÉTENDU qui peut englober de nombreux sujets et questions dans la gestion de projets de construction. Exemples de thématiques larges:

    - "Systemes_techniques_batiment": Tous les aspects techniques (HVAC, électricité, plomberie, sprinklage, ventilation, chauffage, climatisation, systèmes de sécurité, éclairage, etc.)
    - "Coordination_projet_communication": Organisation d'équipes, réunions, échanges entre intervenants, communication client, coordination des métiers, planification collaborative
    - "Suivi_execution_travaux": Avancement des travaux, supervision chantier, contrôle qualité, gestion des phases, suivi des délais, coordination des interventions
    - "Gestion_administrative_financiere": Budgets, coûts, facturation, contrats, devis, aspects réglementaires, autorisations, conformité
    - "Conception_modification_plans": Évolution des plans, adaptations techniques, changements de conception, validation des solutions, études techniques

    INSTRUCTIONS IMPORTANTES:
    1. Utilise la structure du document (si disponible) pour identifier les domaines abordés
    2. Crée des thématiques LARGES qui englobent plusieurs sous-sujets
    3. Si le document aborde des domaines DÉJÀ COUVERTS par les thématiques existantes, écris "AUCUNE_NOUVELLE_THEMATIQUE"
    4. Sinon, liste les NOUVELLES thématiques au format exact suivant:
       S<numéro>: <Nom_Thematique>
       <Description étendue qui liste plusieurs aspects couverts par cette thématique>
    5. Assure-toi que chaque thématique peut couvrir de nombreuses questions différentes
    6. Évite les thématiques trop spécifiques ou étroites

    FORMAT DE RÉPONSE REQUIS:
    S1: Nom_De_La_Thematique
    Description complète qui explique tous les aspects couverts par cette thématique, incluant les sous-domaines, les types de questions possibles, et les sujets connexes.

    DOCUMENT À ANALYSER:
    {document_text}

    NOUVELLES THÉMATIQUES:"""

        try:
            response = self.extraction_llm.complete(extraction_prompt)
            response_text = response.text.strip()

            if "AUCUNE_NOUVELLE_THEMATIQUE" in response_text.upper():
                return {}

            # Parse new thematics with the new format
            new_thematics = {}
            lines = response_text.split('\n')
            current_thematic = None
            current_description_parts = []

            for line in lines:
                line = line.strip()

                # Check if line starts with S<number>:
                if re.match(r'^S\d+:', line):
                    # Save previous thematic if exists
                    if current_thematic and current_description_parts:
                        description = ' '.join(current_description_parts).strip()
                        if description:
                            new_thematics[current_thematic] = description

                    # Start new thematic
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        current_thematic = parts[1].strip()
                        current_description_parts = []
                elif line and current_thematic:
                    # Add to description
                    current_description_parts.append(line)

            # Save last thematic
            if current_thematic and current_description_parts:
                description = ' '.join(current_description_parts).strip()
                if description:
                    new_thematics[current_thematic] = description

            if new_thematics:
                pprint_console(f"Found {len(new_thematics)} new thematics in {file_name}")
                for name, desc in new_thematics.items():
                    pprint_console(f"  - {name}: {desc[:80]}...")

            return new_thematics

        except Exception as e:
            pprint_error(f"LLM extraction failed for {file_name}: {e}")
            return {}

    def _update_thematics(self, new_thematics: Dict[str, str]):
        """Update existing thematics with new ones."""
        for thematic_name, description in new_thematics.items():
            if thematic_name not in self.thematics:
                self.thematics[thematic_name] = description

    def _save_thematics(self):
        """Save current thematics to file with proper formatting."""
        try:
            # Create formatted thematics for storage
            formatted_thematics = {}
            for i, (name, description) in enumerate(self.thematics.items(), 1):
                formatted_key = f"S{i}: {name}"
                formatted_thematics[formatted_key] = description

            data = {
                'extraction_date': datetime.now().isoformat(),
                'total_thematics': len(self.thematics),
                'thematics': self.thematics,  # Keep original for processing
                'formatted_thematics': formatted_thematics  # For display
            }

            with open(self.args.thematics_storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            pprint_error(f"Error saving thematics: {e}")

    def get_thematics_summary(self) -> str:
        """Get a formatted summary of all thematics."""
        if not self.thematics:
            return "Aucune thématique extraite."

        summary = f"THÉMATIQUES EXTRAITES ({len(self.thematics)}):\n"
        summary += "=" * 60 + "\n\n"

        for i, (name, description) in enumerate(self.thematics.items(), 1):
            summary += f"S{i}: {name}\n"
            summary += f"{description}\n\n"

        return summary

    def run_extraction(self):
        """Run the complete thematic extraction process."""
        pprint_console("Starting domain thematics extraction...")
        pprint_console(f"Extraction model: {self.args.groq_model_indexing_extraction}")
        pprint_console(f"Output file: {self.args.thematics_storage_path}")

        start_time = datetime.now()

        # Extract thematics
        thematics = self.extract_thematics_from_index()

        end_time = datetime.now()
        duration = end_time - start_time

        # Print summary
        pprint_console(f"Extraction completed in {duration}")
        pprint_console(self.get_thematics_summary())

        return thematics
