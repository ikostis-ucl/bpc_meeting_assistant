import json
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

from llama_index.core import Settings
from llama_index.core.callbacks import TokenCountingHandler, CallbackManager
from llama_index.llms.groq import Groq
from tqdm import tqdm

from app.engine.data_processing.data_loaders import load_index
from app.utils.app_utils import pprint_console, pprint_error, pprint_debug
from app.utils.inference_utils import throttle_requests


class ThematicsExtractor:
    """
    Extract domain-specific thematics from meeting documents and append all findings,
    tracking frequency of thematics across documents.
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

        self.processed_documents = set()
        self.thematic_frequency = defaultdict(int)
        self.thematics = {}

    def extract_thematics_from_index(self) -> Dict[str, any]:
        """
        Extract and append thematics from all documents in the index.

        Returns:
            Dict containing thematics with frequency
        """

        pprint_console(f"Processing {len(self.index.docstore.docs)} documents...")

        # Group documents by file name
        documents_by_file = self._group_documents_by_file()

        # Load existing data if file exists
        self._load_existing_data()

        # Process each document
        processed_count = 0
        for file_name, nodes in tqdm(documents_by_file.items(), desc="[APPENDING_EXTR] Processing documents"):
            try:
                # Skip if document already processed
                if file_name in self.processed_documents:
                    pprint_console(f"Document {file_name} already processed, skipping...")
                    continue

                # Combine all text from nodes of this document
                document_text = self._combine_document_nodes(nodes)

                # Extract thematics from this document
                document_thematics = self._extract_document_thematics(document_text, file_name)

                # Update frequency counters and store thematics
                if document_thematics:
                    for thematic_name, description in document_thematics.items():
                        self.thematic_frequency[thematic_name] += 1
                        # Keep the description (first one found or update if better)
                        if thematic_name not in self.thematics:
                            self.thematics[thematic_name] = description

                    # Mark document as processed
                    self.processed_documents.add(file_name)
                    self._save_data()

                processed_count += 1

                if processed_count % 5 == 0:
                    self._save_data()

            except Exception as e:
                pprint_error(f"Error processing document {file_name}: {e}")
                continue

        # Final save
        self._save_data()

        pprint_console(
            f"Extraction completed! Found {len(self.thematics)} unique thematics across {processed_count} documents")

        return {
            'thematics': self.thematics,
            'thematic_frequency': dict(self.thematic_frequency)
        }

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

    @staticmethod
    def _analyze_document_structure(document_text: str) -> Dict[str, List[str]]:
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

    def _load_existing_data(self):
        """Load existing data from file if it exists."""
        try:
            if os.path.exists(self.args.thematics_storage_path):
                with open(self.args.thematics_storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Load simplified structure
                self.thematics = data.get('thematics', {})
                frequency_data = data.get('thematic_frequency', {})
                self.thematic_frequency = defaultdict(int, frequency_data)
                self.processed_documents = set(data.get('processed_documents', []))

                pprint_console(
                    f"Loaded existing data: {len(self.processed_documents)} documents, {len(self.thematics)} unique thematics")
            else:
                pprint_console("No existing data file found, starting fresh.")
        except Exception as e:
            pprint_error(f"Error loading existing data: {e}")
            self.thematics = {}
            self.thematic_frequency = defaultdict(int)
            self.processed_documents = set()

    @throttle_requests()
    def _extract_document_thematics(self, document_text: str, file_name: str) -> Dict[str, str]:
        """Extract thematics from a single document using LLM."""

        # Analyze document structure
        doc_structure = self._analyze_document_structure(document_text)

        # Prepare structure context
        structure_context = f"""
            STRUCTURE DU DOCUMENT DÉTECTÉE:
            - Sections principales: {', '.join(doc_structure['main_sections'][:5]) if doc_structure['main_sections'] else 'Aucune'}
            - Sous-sections: {', '.join(doc_structure['subsections'][:5]) if doc_structure['subsections'] else 'Aucune'}
            - Dates trouvées: {len(doc_structure['dates_found'])} entrées temporelles
            - Termes techniques identifiés: {', '.join(doc_structure['key_topics'][:8]) if doc_structure['key_topics'] else 'Aucun'}
            
            """

        extraction_prompt = f"""Tu es un expert en analyse de projets de construction et de réunions d'équipe.

            {structure_context}Analyse le document suivant (fichier: {file_name}) et identifie TOUTES les THÉMATIQUES principales abordées.
            
            Si le document est au format Markdown, utilise la structure détectée ci-dessus pour mieux comprendre l'organisation du contenu et identifier les thématiques pertinentes.
            
            OBJECTIF: Créer des thématiques LARGES et INCLUSIVES qui représentent tous les domaines abordés dans ce document spécifique.
            
            Une thématique représente un DOMAINE ÉTENDU qui peut englober de nombreux sujets et questions dans la gestion de projets de construction. Exemples de thématiques larges:
            
            - "Systemes_techniques_batiment": Tous les aspects techniques (HVAC, électricité, plomberie, sprinklage, ventilation, chauffage, climatisation, systèmes de sécurité, éclairage, etc.)
            - "Coordination_projet_communication": Organisation d'équipes, réunions, échanges entre intervenants, communication client, coordination des métiers, planification collaborative
            - "Suivi_execution_travaux": Avancement des travaux, supervision chantier, contrôle qualité, gestion des phases, suivi des délais, coordination des interventions
            - "Gestion_administrative_financiere": Budgets, coûts, facturation, contrats, devis, aspects réglementaires, autorisations, conformité
            - "Conception_modification_plans": Évolution des plans, adaptations techniques, changements de conception, validation des solutions, études techniques
            
            INSTRUCTIONS IMPORTANTES:
            1. Utilise la structure du document (si disponible) pour identifier TOUS les domaines abordés
            2. Crée des thématiques LARGES qui englobent plusieurs sous-sujets
            3. Liste TOUTES les thématiques présentes dans ce document, même si elles peuvent exister ailleurs
            4. Format exact requis:
               S<numéro>: <Nom_Thematique>
               <Description étendue qui liste plusieurs aspects couverts par cette thématique>
            5. Assure-toi que chaque thématique peut couvrir de nombreuses questions différentes
            6. N'omets aucune thématique, même si elle semble mineure
            
            FORMAT DE RÉPONSE REQUIS:
            S1: Nom_De_La_Thematique
            Description complète qui explique tous les aspects couverts par cette thématique, incluant les sous-domaines, les types de questions possibles, et les sujets connexes.
            
            DOCUMENT À ANALYSER:
            {document_text}
            
            THÉMATIQUES TROUVÉES:"""

        try:
            response = self.extraction_llm.complete(extraction_prompt)
            response_text = response.text.strip()

            # Parse thematics
            document_thematics = {}
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
                            document_thematics[current_thematic] = description

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
                    document_thematics[current_thematic] = description

            return document_thematics

        except Exception as e:
            pprint_error(f"LLM extraction failed for {file_name}: {e}")
            return {}

    def _save_data(self):
        """Save simplified data structure to file."""
        try:
            data = {
                'extraction_date': datetime.now().isoformat(),
                'total_documents': len(self.processed_documents),
                'total_unique_thematics': len(self.thematics),
                'processed_documents': list(self.processed_documents),
                'thematics': {},
                'thematic_frequency': dict(self.thematic_frequency)
            }

            for thematic_name, description in self.thematics.items():
                frequency = self.thematic_frequency[thematic_name]
                data['thematics'][thematic_name] = {
                    'description': description,
                    'frequency': frequency
                }

            with open(self.args.thematics_storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            pprint_error(f"Error saving data: {e}")

    def get_comprehensive_summary(self) -> str:
        """Get a comprehensive summary of extracted thematics with frequency."""
        if not self.thematics:
            return "Aucune thématique extraite."

        summary = f"EXTRACTION COMPLÈTE - APPROCHE AJOUT\n"
        summary += "=" * 60 + "\n\n"

        # Overall statistics
        summary += f"STATISTIQUES GÉNÉRALES:\n"
        summary += f"- Documents traités: {len(self.processed_documents)}\n"
        summary += f"- Thématiques uniques: {len(self.thematics)}\n\n"

        # Sort thematics by frequency (most common first)
        sorted_thematics = sorted(
            self.thematics.items(),
            key=lambda x: self.thematic_frequency[x[0]],
            reverse=True
        )

        summary += "THÉMATIQUES PAR FRÉQUENCE:\n"
        summary += "-" * 40 + "\n"
        for i, (name, description) in enumerate(sorted_thematics, 1):
            frequency = self.thematic_frequency[name]
            summary += f"S{i}: {name} [{frequency} documents]\n"
            summary += f"{description}\n\n"

        return summary

    def run_extraction(self):
        """Run the complete appending thematic extraction process."""
        pprint_console("Starting appending domain thematics extraction...")

        start_time = datetime.now()

        # Extract thematics
        results = self.extract_thematics_from_index()

        end_time = datetime.now()
        duration = end_time - start_time

        # Print summary
        pprint_console(f"Extraction completed in {duration}")
        pprint_debug(self.get_comprehensive_summary())

        return results
