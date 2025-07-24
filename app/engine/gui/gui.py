import datetime
import gc
import os

import fitz
import gradio as gr
from PIL import Image

from app.utils.gui_utils import ANON_PATH, ANON_INDEX
from app.utils.inference_utils import pprint_qa

HOME_PATH = os.path.expanduser("~")

# Screen height constants for different resolutions
BIG_WIN, SMALL_WIN = 740, 665  # -> 1920x1080
DPI = 150


class GUI:
    """
    Graphical User Interface for the Meeting Minutes Assistant.

    This class handles the frontend interface and backend logic for the chatbot application.
    It manages user interactions, document display, and conversation flow.
    """

    def __init__(self, args, conv_agent):
        """
        Initialize the GUI with configuration and conversation agent.

        Args:
            args: Configuration arguments for the application.
            conv_agent: Conversation agent for handling queries.
        """
        self.args = args
        self.conv_agent = conv_agent

        if self.args.anon:
            self.idle_screen = "./app/assets/idle_screen_anon.pdf"
        else:
            self.idle_screen = "./app/assets/idle_screen.pdf"

        # Set the timespans using the configured time_freq value
        self.conv_agent.generate_timespans(starting_month_timestamp=self.conv_agent.start_date,
                                           ending_month_timestamp=self.conv_agent.end_date,
                                           time_freq=args.time_freq)

        # Example questions for the user interface
        self.examples = [
            "Quelle est la couleur choisie (RAL) pour les châssis ?",
            "Liste des décisions prises concernant le carrelage des salles des bains (SDBs) et les dates (jour/mois/année) auxquelles elles ont été prises.",
            "Quelle est la date de remise des espaces communs ?",
            "Quelles sont les décisions qui ont étés prises pour les acrotères des terrasses ?",
            "Pourrais-je avoir toutes les informations concernant l'ascenseur vélo ?",
            "Pourrais-je avoir un historique concernant les décisions prises pour les couvre-murs ?",
            "Informations concernant les faux-plafonds ?",
            "Quelles finitions pour les halls d'entrée ?",
            "Quelle isolation a été choisi pour les plafonds du sous-sols -1 ?",
            "Pourrais-je avoir une liste des remarques faite par le SECO ?"
        ]

    @staticmethod
    def exec_user_query(question, chat_history):
        """
        Execute the user's query and update chat history.

        Args:
            question: User's input question.
            chat_history: The current chat history.
        Returns:
            tuple: Empty string and updated chat history.

        Raises:
            SystemExit: If user enters 'exit' or 'quit'.

        """
        chat_history.append([question, None])
        return "", chat_history

    def query_conv_agent(self, chat_history, render_state):
        """
        Process the query through the conversation agent and update the interface.

        Returns:
            list: Updated chat history with agent's responses.
        """
        question = chat_history[-1][0]
        results = self.conv_agent.query_llm(query_string=question)
        pprint_qa(question=question, results=results)

        response_metadata = {}

        # Process each result and update metadata and chat history
        for answer, metadata, (_s_date, _e_date) in results:
            # Format dates for display
            start_date_str = datetime.datetime.fromtimestamp(_s_date).strftime('%d/%m/%Y')
            end_date_str = datetime.datetime.fromtimestamp(_e_date).strftime('%d/%m/%Y')
            timespan_key = f"{start_date_str} to {end_date_str}"

            # Organize metadata by timespan
            if timespan_key not in response_metadata:
                response_metadata[timespan_key] = {}

            # Process metadata for each node
            for node_id, node_values in metadata.items():
                file_name = node_values["file_name"]
                file_path = node_values["file_path"]
                page_number = node_values["page_number"]
                if file_name in response_metadata[timespan_key]:
                    response_metadata[timespan_key][file_name]["page_numbers"].append(page_number)
                else:
                    response_metadata[timespan_key][file_name] = {
                        "file_path": file_path,
                        "page_numbers": [page_number]
                    }

            # Update chat history with response
            chat_history.append([None, f"Du {start_date_str} au {end_date_str}:"])
            chat_history.append([None, f"{answer}"])
            chat_history.append([None, f"-------"])

        if render_state:
            render_state = not render_state

        # Force garbage collection after processing query
        gc.collect()

        return chat_history, response_metadata, render_state

    def run(self):
        """
        Launch the GUI application.

        Creates and configures the Gradio interface, sets up the layout,
        and handles the application lifecycle.
        """
        # CSS styling for container width
        head_style = """
            <style>
            @media (min-width: 1900px) {
                .gradio-container {
                    min-width: var(--size-full) !important;
                    min-height: 100vh !important;
                }
            }
            </style>
            """

        # Create Gradio interface
        with gr.Blocks(title="Meeting Minutes Assistant",
                       fill_height=True,
                       head=head_style,
                       theme=gr.themes.Default(primary_hue=gr.themes.colors.blue)) as app:

            chat_history = gr.State([])
            metadata = gr.State(
                {"default_1": {"file_name": "Home",
                               "file_path": self.idle_screen,
                               "page_number": 1}}
            )
            render_state = gr.State(True)

            # Layout definition with chat and document display
            with gr.Row():
                with gr.Column():
                    chatbot = gr.Chatbot(
                        value=chat_history.value,
                        elem_id="chatbot",
                        label='Chat History',
                        placeholder="👤 🤖️",
                        height=BIG_WIN,
                    )
                with gr.Column():
                    @gr.render(inputs=[render_state, metadata],
                               triggers=[app.load, chatbot.change])
                    def render_context(render_state, metadata):
                        """
                        Render PDF context based on conversation state.
                        Displays relevant PDF pages in the interface.
                        """
                        if render_state:
                            # Initial render state handling
                            pdf_info = {}
                            for node_id, node_values in metadata.items():
                                file_name = node_values["file_name"]
                                file_path = node_values["file_path"]
                                page_number = node_values.get("page_number") - 1

                                # Handle anonymization if in anon mode
                                if self.args.anon and file_name in ANON_INDEX:
                                    anon_file_name = ANON_INDEX[file_name]
                                    file_path = os.path.join(ANON_PATH, anon_file_name)
                                    display_name = anon_file_name
                                else:
                                    display_name = file_name

                                if display_name not in pdf_info:
                                    pdf_info[display_name] = {"file_path": file_path, "page_numbers": [page_number]}
                                else:
                                    pdf_info[display_name]["page_numbers"].append(page_number)

                            # Display PDFs in tabs
                            for pdf_name, info in pdf_info.items():
                                pdf_path = info["file_path"]
                                page_numbers = list(set(info["page_numbers"]))
                                pdf_pages = []

                                doc = fitz.open(pdf_path)
                                for pg_num in page_numbers:
                                    page = doc[pg_num]
                                    pix = page.get_pixmap(matrix=fitz.Matrix(DPI / 72, DPI / 72))
                                    img = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
                                    pdf_pages.append(img)
                                    pix = None

                                # Always close the document to prevent memory leaks
                                if 'doc' in locals():
                                    doc.close()

                                with gr.Tab(label=pdf_name):
                                    gr.Gallery(
                                        value=pdf_pages,
                                        allow_preview=True,
                                        preview=True,
                                        selected_index=0,
                                        type='pil',
                                        interactive=False,
                                        height=SMALL_WIN,
                                    )
                        else:
                            # Conversation state handling
                            for timespan, files in metadata.items():
                                with gr.Tab(label=timespan):
                                    for file_name, file_info in files.items():
                                        if self.args.anon and file_name in ANON_INDEX:
                                            anon_file_name = ANON_INDEX[file_name]
                                            pdf_path = os.path.join(ANON_PATH, anon_file_name)
                                            display_name = anon_file_name
                                        else:
                                            pdf_path = file_info["file_path"]
                                            display_name = file_name

                                        with gr.Tab(label=display_name):
                                            page_numbers = list(set(file_info["page_numbers"]))
                                            pdf_pages = []
                                            try:
                                                doc = fitz.open(pdf_path)
                                                for pg_num in page_numbers:
                                                    page = doc[pg_num - 1]
                                                    pix = page.get_pixmap(matrix=fitz.Matrix(DPI / 72, DPI / 72))
                                                    img = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
                                                    pdf_pages.append(img)
                                                    pix = None
                                            finally:
                                                if 'doc' in locals():
                                                    doc.close()

                                            gr.Gallery(
                                                value=pdf_pages,
                                                allow_preview=True,
                                                preview=True,
                                                selected_index=0,
                                                type='pil',
                                                interactive=False,
                                                height=int(SMALL_WIN * 0.9),
                                            )

                        # Force garbage collection after rendering
                        gc.collect()

            # Input components
            with gr.Row():
                input_textbox = gr.Textbox(
                    placeholder="Pose-moi une question sur le projet!",
                    scale=1,
                    show_label=False,
                    interactive=True,
                    container=False
                )

            with gr.Row():
                _examples = gr.Examples(examples=self.examples, inputs=[input_textbox])

            # Event handlers
            input_textbox.submit(
                fn=self.exec_user_query,
                inputs=[input_textbox, chatbot],
                outputs=[input_textbox, chatbot]
            ).success(
                fn=self.query_conv_agent,
                inputs=[chatbot, render_state],
                outputs=[chatbot, metadata, render_state]
            )

        # Launch application with appropriate configuration
        try:
            if self.args.prod:
                auth_pairs = os.getenv("GRADIO_AUTH_PAIRS").split(',')
                auth_users = [tuple(pair.split(':')) for pair in auth_pairs]

                app.queue(max_size=2)
                app.launch(
                    server_port=9000,
                    auth=auth_users,
                    max_threads=32,
                    favicon_path="./app/assets/bpc_logo.png"
                )
            else:
                app.queue(max_size=2)
                app.launch(
                    share=True,
                    inbrowser=False,
                    max_threads=8,
                    favicon_path="./app/assets/bpc_logo.png",
                    server_name="0.0.0.0",
                    server_port=7863
                )

        except (KeyboardInterrupt, SystemExit):
            app.close()
