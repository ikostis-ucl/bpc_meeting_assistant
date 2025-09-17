import datetime
import gc
import os

import fitz
import gradio as gr
from PIL import Image

from app.utils.benchmark_utils import BENCHMARK_QUESTIONS_INDEX
from app.utils.gui_utils import generate_colors, hex_to_rgb
from app.utils.inference_utils import pprint_qa
from html import escape


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

        self.examples = list(BENCHMARK_QUESTIONS_INDEX.values())

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

    @staticmethod
    def render_timeline(timeline_data):
        """
        Render the timeline visualization.
        """
        timeline_css = """
        <style>
        .timeline-container {
            position: relative;
            width: 100%;
            max-width: 100%;
            max-height: 740px;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 16px;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            align-items: stretch;
        }
        .timeline-track {
            position: relative;
            width: 100%;
            display: flex;
            flex-direction: column;
            align-items: stretch;
            padding: 0;
        }
        .timeline-line {
            position: absolute;
            top: 10px;
            bottom: 10px;
            left: 50%;
            transform: translateX(-50%);
            width: 4px;
            border-radius: 4px;
            background-color: var(--timeline-line-color, #212BF3);
            opacity: 0.4;
            z-index: 0;
        }
        .timeline-node {
            display: flex;
            align-items: center;
            justify-content: center;
            margin: clamp(24px, 5vh, 60px) 0;
            width: 100%;
            max-width: 100%;
            position: relative;
            z-index: 1;
            box-sizing: border-box;
            /* Hover scaling system */
            --node-scale: 1;
            transform: scale(var(--node-scale));
            transform-origin: center;
            transition: transform 0.35s cubic-bezier(.22,.68,.37,1), z-index 0.35s;
        }
        .timeline-node:hover {
            --node-scale: 1.08;
            z-index: 10;
        }

        .timeline-node:first-child { margin-top: 0; }
        .timeline-node:last-child { margin-bottom: 0; }

        .timeline-side {
            flex: 1 1 50%;
            min-width: 0;
            display: flex;
            align-items: center;
            position: relative;
            --branch-size: clamp(24px, 18%, 140px);
            --side-gap: clamp(8px, 2.2vw, 24px);
        }
        .timeline-center {
            width: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }
        .timeline-marker-container {
            position: absolute;
            left: 50%;
            transform: translateX(-50%);
            z-index: 2;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .timeline-marker {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background-color: var(--segment-color, #2196F3);
            border: 4px solid var(--background-fill-primary);
            box-shadow: 0 0 0 2px var(--segment-color, #2196F3),
                        0 2px 8px rgba(0, 0, 0, 0.2);
            transition: inherit;
            transform: scale(var(--node-scale));
        }
        .timeline-marker:hover {
            transform: scale(1.2);
            box-shadow: 0 0 0 3px var(--segment-color, #2196F3),
                        0 4px 12px rgba(0, 0, 0, 0.3);
        }

        .timeline-branch {
            flex: 0 0 var(--branch-size);
            width: var(--branch-size);
            height: 2px;
            border-radius: 2px;
        }
        .timeline-branch-left {
            background: linear-gradient(
                to right,
                var(--segment-color, #2196F3),
                rgba(var(--segment-color-rgb, 33,150,243), 0.2)
            );
        }
        .timeline-branch-right {
            background: linear-gradient(
                to left,
                var(--segment-color, #2196F3),
                rgba(var(--segment-color-rgb, 33,150,243), 0.2)
            );
        }

        .timeline-content-left {
            justify-content: flex-end;
            text-align: right;
            gap: var(--side-gap);
            padding-right: clamp(8px, 1.8vw, 24px);
        }
        .timeline-content-right {
            justify-content: flex-start;
            text-align: left;
            gap: var(--side-gap);
            padding-left: clamp(8px, 1.8vw, 24px);
        }
        .timeline-content-box {
            box-sizing: border-box;
            width: clamp(220px, calc(100% - var(--branch-size) - var(--side-gap) - 12px), 520px);
            max-width: calc(100% - var(--branch-size) - var(--side-gap) - 12px);
            padding: calc(16px * var(--node-scale));
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            transition: inherit;
            background-color: var(--background-fill-secondary);
            border: 1px solid var(--border-color-primary);
            position: relative;
            overflow: hidden;
            font-size: calc(0.95em * var(--node-scale));
        }
        .timeline-content-box:hover {
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
            transform: translateY(-2px);
        }
        .timeline-content-box-left { border-left: 6px solid var(--segment-color, #2196F3); }
        .timeline-content-box-left::after {
            content: '';
            position: absolute;
            top: 50%;
            right: -12px;
            transform: translateY(-50%);
            border-left: 12px solid var(--segment-color, #2196F3);
            border-top: 8px solid transparent;
            border-bottom: 8px solid transparent;
        }
        .timeline-content-box-right { border-right: 6px solid var(--segment-color, #2196F3); }
        .timeline-content-box-right::after {
            content: '';
            position: absolute;
            top: 50%;
            left: -12px;
            transform: translateY(-50%);
            border-right: 12px solid var(--segment-color, #2196F3);
            border-top: 8px solid transparent;
            border-bottom: 8px solid transparent;
        }
        .timeline-node:hover .timeline-content-box {
            box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        }
        .timeline-date {
            font-weight: bold;
            font-size: calc(1em * var(--node-scale));
            margin: 0 0 calc(12px * var(--node-scale)) 0;
            padding: calc(8px * var(--node-scale)) calc(12px * var(--node-scale));
            border-radius: 16px;
            background-color: var(--segment-color, #2196F3);
            color: white;
            box-shadow: 0 2px 6px rgba(33, 150, 243, 0.3);
            text-align: center;
            display: inline-block;
            max-width: 100%;
            overflow-wrap: anywhere;
            transition: inherit;
        }
        .timeline-content {
            line-height: 1.6;
            color: var(--body-text-color);
            font-size: 1em; /* already scaled by parent */
            overflow-wrap: anywhere;
            word-break: break-word;
            white-space: normal;
            transition: inherit;
        }
        @media (max-width: 900px) {
            .timeline-side { --branch-size: clamp(16px, 12%, 80px); }
            .timeline-content-box {
                width: clamp(200px, calc(100% - var(--branch-size) - var(--side-gap) - 8px), 480px);
            }
        }
        @media (max-width: 680px) {
            .timeline-side { --branch-size: 0px; }
            .timeline-branch { display: none; }
            .timeline-content-left, .timeline-content-right {
                padding-left: 8px;
                padding-right: 8px;
                gap: 10px;
            }
            .timeline-content-box {
                width: min(520px, 100%);
                max-width: 100%;
            }
        }
        </style>
        """

        colors = generate_colors(len(timeline_data))

        html = timeline_css + '''
        <div class="timeline-container">
            <div class="timeline-track" style="--timeline-line-color: #1e88e5;">
                <div class="timeline-line"></div>
        '''

        for i, (timespan, answer, metadata_info) in enumerate(timeline_data):

            safe_answer = escape(answer).replace('\n', '<br>')

            color = colors[i] if i < len(colors) else "#1976D2"
            rgb = hex_to_rgb(color)
            is_left = i % 2 == 0

            if is_left:
                html += f'''
                <div class="timeline-node" style="--segment-color: {color}; --segment-color-rgb: {rgb[0]}, {rgb[1]}, {rgb[2]};">
                    <div class="timeline-side timeline-content-left">
                        <div class="timeline-content-box timeline-content-box-left">
                            <div class="timeline-date">{timespan}</div>
                            <div class="timeline-content">{safe_answer}</div>
                        </div>
                        <div class="timeline-branch timeline-branch-left"></div>
                    </div>
                    <div class="timeline-center">
                        <div class="timeline-marker-container">
                            <div class="timeline-marker"></div>
                        </div>
                    </div>
                    <div class="timeline-side"></div>
                </div>
                '''
            else:
                html += f'''
                <div class="timeline-node" style="--segment-color: {color}; --segment-color-rgb: {rgb[0]}, {rgb[1]}, {rgb[2]};">
                    <div class="timeline-side"></div>
                    <div class="timeline-center">
                        <div class="timeline-marker-container">
                            <div class="timeline-marker"></div>
                        </div>
                    </div>
                    <div class="timeline-side timeline-content-right">
                        <div class="timeline-branch timeline-branch-right"></div>
                        <div class="timeline-content-box timeline-content-box-right">
                            <div class="timeline-date">{timespan}</div>
                            <div class="timeline-content">{safe_answer}</div>
                        </div>
                    </div>
                </div>
                '''

        html += '''
            </div>
        </div>
        '''

        return gr.HTML(html)

    def query_conv_agent(self, chat_history, render_state):
        """
        Modified version that populates timeline data with only the current query results.

        Returns:
            tuple: Updated timeline data, response metadata, and render state.
        """
        question = chat_history[-1][0] if chat_history else ""
        results = self.conv_agent.query_llm(query_string=question)
        pprint_qa(question=question, results=results)

        response_metadata = {}
        new_timeline_items = []

        for answer, metadata, _, (min_timestamp, max_timestamp) in results:
            start_date_str = datetime.datetime.fromtimestamp(min_timestamp).strftime('%d/%m/%Y')
            end_date_str = datetime.datetime.fromtimestamp(max_timestamp).strftime('%d/%m/%Y')
            timespan_key = f"{start_date_str} to {end_date_str}"

            if timespan_key not in response_metadata:
                response_metadata[timespan_key] = {}

            timeline_metadata = {}

            if metadata:
                for node_id, node_values in metadata.items():
                    if "metadata" in node_values and isinstance(node_values["metadata"], dict):
                        node_metadata = node_values["metadata"]
                        file_name = node_metadata.get("file_name")
                        file_path = node_metadata.get("file_path")
                        page_number = node_metadata.get("page_number")

                        if file_name and file_path and page_number is not None:
                            if file_name in response_metadata[timespan_key]:
                                response_metadata[timespan_key][file_name]["page_numbers"].append(page_number)
                            else:
                                response_metadata[timespan_key][file_name] = {
                                    "file_path": file_path,
                                    "page_numbers": [page_number]
                                }

                            if file_name in timeline_metadata:
                                timeline_metadata[file_name]["page_numbers"].append(page_number)
                            else:
                                timeline_metadata[file_name] = {
                                    "file_path": file_path,
                                    "page_numbers": [page_number]
                                }

            new_timeline_items.append((timespan_key, answer, timeline_metadata))

        timeline_data = new_timeline_items

        if render_state:
            render_state = not render_state

        gc.collect()
        return timeline_data, response_metadata, render_state

    def run(self):
        """
        Launch the GUI application.

        Creates and configures the Gradio interface, sets up the layout,
        and handles the application lifecycle.
        """

        spinner_css = """
        <style>
        .loader {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
        """

        head_style = f"""
            <style>
            @media (min-width: 1900px) {{
                .gradio-container {{
                    min-width: var(--size-full) !important;
                    min-height: 100vh !important;
                }}
            }}
            </style>
            {spinner_css}
        """

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

            with gr.Row():
                with gr.Column(scale=65):
                    timeline_data = gr.State([])

                    @gr.render(inputs=[timeline_data], triggers=[timeline_data.change])
                    def render_timeline_component(timeline_data):
                        return self.render_timeline(timeline_data)

                with gr.Column(scale=35):
                    @gr.render(inputs=[render_state, metadata],
                               triggers=[app.load, timeline_data.change])
                    def render_context(render_state, metadata):
                        """
                        Render PDF context based on conversation state.
                        Displays relevant PDF pages in the interface.
                        """
                        if render_state:
                            pdf_info = {}
                            for node_id, node_values in metadata.items():
                                file_name = node_values["file_name"]
                                file_path = node_values["file_path"]
                                page_number = node_values.get("page_number") - 1

                                if file_name not in pdf_info:
                                    pdf_info[file_name] = {"file_path": file_path, "page_numbers": [page_number]}
                                else:
                                    pdf_info[file_name]["page_numbers"].append(page_number)

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
                            for timespan, files in metadata.items():
                                with gr.Tab(label=timespan):
                                    for file_name, file_info in files.items():
                                        pdf_path = file_info["file_path"]

                                        with gr.Tab(label=file_name):
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
                                                    height=SMALL_WIN,
                                                )

                        gc.collect()

            with gr.Row():
                input_textbox = gr.Textbox(
                    placeholder="Pose-moi une question sur le projet!",
                    scale=1,
                    show_label=False,
                    interactive=True,
                    container=False
                )
                submit_btn = gr.Button("➤", variant="primary", visible=False, scale=0.05)

            loading_state = gr.State(False)

            @gr.render(inputs=[loading_state], triggers=[loading_state.change])
            def render_loading_indicator(is_loading):
                if is_loading:
                    return gr.HTML(
                        '<div style="text-align: center; padding: 20px;"><div class="loader"></div><p>Processing query...</p></div>')
                else:
                    return gr.HTML("")

            with gr.Row():
                _examples = gr.Examples(examples=self.examples, inputs=[input_textbox])

            def start_loading():
                return True

            def stop_loading():
                return False

            input_textbox.submit(
                fn=start_loading,
                outputs=[loading_state]
            ).then(
                fn=self.exec_user_query,
                inputs=[input_textbox, chat_history],
                outputs=[input_textbox, chat_history]
            ).then(
                fn=self.query_conv_agent,
                inputs=[chat_history, render_state],
                outputs=[timeline_data, metadata, render_state]
            ).then(
                fn=stop_loading,
                outputs=[loading_state]
            )

            submit_btn.click(
                fn=start_loading,
                outputs=[loading_state]
            ).then(
                fn=self.exec_user_query,
                inputs=[input_textbox, chat_history],
                outputs=[input_textbox, chat_history]
            ).then(
                fn=self.query_conv_agent,
                inputs=[chat_history, render_state],
                outputs=[timeline_data, metadata, render_state]
            ).then(
                fn=stop_loading,
                outputs=[loading_state]
            )

        try:
            if self.args.prod:
                auth_pairs = os.getenv("GRADIO_AUTH_PAIRS").split(',')
                auth_users = [tuple(pair.split(':')) for pair in auth_pairs]

                app.queue(max_size=64)
                app.launch(
                    server_port=9000,
                    auth=auth_users,
                    max_threads=32,
                    favicon_path="./app/assets/logo.png"
                )
            else:
                app.queue(max_size=4)
                app.launch(
                    inbrowser=False,
                    max_threads=8,
                    favicon_path="./app/assets/logo.png",
                    server_name="0.0.0.0",
                    server_port=7863
                )

        except (KeyboardInterrupt, SystemExit):
            app.close()
