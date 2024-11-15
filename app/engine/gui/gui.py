import datetime
import os

import fitz
import gradio as gr
from PIL import Image
from gradio_calendar import Calendar

from app.utils.app_utils import pprint_console, pprint_qa

HOME_PATH = os.path.expanduser("~")
# BIG_WIN_H, SMALL_WIN_H = 1120, 950  # 3440x1440
BIG_WIN_H, SMALL_WIN_H = 770, 665  # 1920x1080
DPI = 150


class GUI:
    def __init__(self, args, conv_agent):
        self.args = args
        self.conv_agent = conv_agent

        self.start_date = datetime.datetime(1970, 1, 1)
        self.end_date = datetime.datetime.now()

        self.chat_history = []
        self.metadata = {"default_1": {"file_name": "Home", "file_path": "./app/assets/idle_screen.pdf"}}

    def set_start_date(self, value):
        self.start_date = value

    def set_end_date(self, value):
        self.end_date = value

    def set_cutoff_percentage(self, value):
        if value > 20.0:
            gr.Warning("Going over 20% might result in performance slowdown, and/or the demo to crash.")
        self.conv_agent.cutoff_percentage = value / 100

    def exec_user_query(self, question):
        if question == 'exit' or question == 'quit':
            pprint_console("Exiting chatbot...")
            raise SystemExit

        self.chat_history.append([question, None])
        return "", self.chat_history

    def query_conv_agent(self):
        question = self.chat_history[-1][0]
        answer, self.metadata = self.conv_agent.query_llm(query_string=question,
                                                          start_date=self.start_date,
                                                          end_date=self.end_date)
        pprint_qa(question=question, answer=answer, metadata=self.metadata, dates=[self.start_date, self.end_date])

        self.chat_history[-1][1] = answer

        return self.chat_history

    def run(self):

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

        # frontend
        with gr.Blocks(title="Meeting Assistant", fill_height=True, head=head_style) as app:
            with gr.Row():
                with gr.Column():
                    chatbot = gr.Chatbot(
                        value=self.chat_history,
                        elem_id="chatbot",
                        label='Chat History',
                        placeholder="👤 🤖️"
                    )
                with gr.Column():
                    @gr.render(inputs=None, triggers=[app.load, chatbot.change])
                    def render_context():
                        pdf_info = {}
                        for node_id, node_values in self.metadata.items():
                            file_name = node_values["file_name"]
                            file_path = node_values["file_path"]
                            page_number = node_values.get("page_number", 1) - 1
                            if file_name not in pdf_info:
                                pdf_info[file_name] = {"file_path": file_path, "page_numbers": [page_number]}
                            else:
                                pdf_info[file_name]["page_numbers"].append(page_number)

                        for pdf_name, info in pdf_info.items():
                            pdf_path = info["file_path"]
                            page_numbers = list(set(info["page_numbers"]))  # Remove duplicates
                            pdf_pages = []
                            doc = fitz.open(pdf_path)
                            for pg_num in page_numbers:
                                page = doc[pg_num]
                                pix = page.get_pixmap(matrix=fitz.Matrix(DPI / 72, DPI / 72))
                                pdf_pages.append(Image.frombytes('RGB', (pix.width, pix.height), pix.samples))

                            with gr.Tab(label=pdf_name):
                                gr.Gallery(
                                    value=pdf_pages,
                                    allow_preview=True,
                                    preview=True,
                                    selected_index=0,
                                    type='pil',
                                    interactive=False
                                )
            with gr.Row():
                with gr.Column(scale=5):
                    input_prompt = gr.Textbox(
                        placeholder="Pose-moi une question sur le projet!",
                        scale=40,
                        show_label=False,
                        interactive=True,
                        container=False
                    )
                with gr.Column(scale=2):
                    with gr.Row():
                        from_calendar = Calendar(type="string",
                                                 label="Start Date",
                                                 show_label=True,
                                                 scale=8,
                                                 container=True,
                                                 min_width=25)
                        to_calendar = Calendar(type="string",
                                               label="End Date",
                                               show_label=True,
                                               scale=8,
                                               container=True,
                                               min_width=25)
                    with gr.Row():
                        slider = gr.Slider(0, 100, step=0.5,
                                           value=self.args.cutoff_percentage * 100, label="Index Leniency (%)")

            # backend
            input_prompt.submit(
                fn=self.exec_user_query,
                inputs=[input_prompt],
                outputs=[input_prompt, chatbot]
            ).success(
                fn=self.query_conv_agent,
                outputs=[chatbot]
            )

            from_calendar.input(fn=self.set_start_date, inputs=[from_calendar])
            to_calendar.input(fn=self.set_end_date, inputs=[to_calendar])
            slider.release(fn=self.set_cutoff_percentage, inputs=[slider])

        app.queue()
        try:
            app.launch(share=False,
                       inbrowser=True,
                       favicon_path="./app/assets/bpc_logo.png",
                       server_port=7862)
        except (KeyboardInterrupt, SystemExit):
            app.close()
