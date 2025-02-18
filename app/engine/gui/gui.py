import datetime
import os

import fitz
import gradio as gr
from PIL import Image

from app.utils.app_utils import pprint_console, pprint_qa

HOME_PATH = os.path.expanduser("~")
# BIG_WIN_H, SMALL_WIN_H = 1120, 950  # 3440x1440
BIG_WIN_H, SMALL_WIN_H = 740, 665  # 1920x1080
DPI = 150


class GUI:
    def __init__(self, args, conv_agent):
        self.args = args
        self.conv_agent = conv_agent

        self.total_duration = self._get_months()

        self.chat_history = []
        self.metadata = {"default_1": {"file_name": "Home",
                                       "file_path": "./app/assets/idle_screen.pdf",
                                       "page_number": 1}}

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
            "Peux-je avoir une liste remarques SECO ?"
        ]

        self.state = True

    def _get_months(self):
        start_date = datetime.datetime.fromtimestamp(self.conv_agent.start_date)
        end_date = datetime.datetime.fromtimestamp(self.conv_agent.end_date)
        return (end_date.year - start_date.year) * 12 + end_date.month - start_date.month

    def set_timestep(self, value):
        if value >= self.total_duration * 0.5:
            gr.Warning("The execution might be faster, but the result might be inaccurate.")
        elif value <= self.total_duration * 0.1:
            gr.Warning("The results will be more precise, but the execution will be slower.")

        self.conv_agent.generate_timespans(starting_month_timestamp=self.conv_agent.start_date,
                                           ending_month_timestamp=self.conv_agent.end_date,
                                           time_freq=value)

    def exec_user_query(self, question):
        if question == 'exit' or question == 'quit':
            pprint_console("Exiting chatbot...")
            raise SystemExit

        self.chat_history.append([question, None])
        return "", self.chat_history

    def query_conv_agent(self):
        question = self.chat_history[-1][0]
        results = self.conv_agent.query_llm(query_string=question)
        pprint_qa(question=question, results=results)

        self.metadata = {}

        for answer, metadata, (_s_date, _e_date) in results:
            start_date_str = datetime.datetime.fromtimestamp(_s_date).strftime('%d/%m/%Y')
            end_date_str = datetime.datetime.fromtimestamp(_e_date).strftime('%d/%m/%Y')
            timespan_key = f"{start_date_str} to {end_date_str}"

            if timespan_key not in self.metadata:
                self.metadata[timespan_key] = {}

            for node_id, node_values in metadata.items():
                file_name = node_values["file_name"]
                file_path = node_values["file_path"]
                page_number = node_values["page_number"]
                if file_name in self.metadata[timespan_key]:
                    self.metadata[timespan_key][file_name]["page_numbers"].append(page_number)
                else:
                    self.metadata[timespan_key][file_name] = {
                        "file_path": file_path,
                        "page_numbers": [page_number]
                    }

            self.chat_history.append([None, f"From {start_date_str} to {end_date_str}:"])
            self.chat_history.append([None, f"{answer}"])
            self.chat_history.append([None, f"-------"])

        self.state = False
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
        with gr.Blocks(title="Meeting Minutes Assistant",
                       fill_height=True,
                       head=head_style,
                       theme=gr.themes.Default(primary_hue=gr.themes.colors.blue)) as app:

            with gr.Row():
                with gr.Column():
                    chatbot = gr.Chatbot(
                        value=self.chat_history,
                        elem_id="chatbot",
                        label='Chat History',
                        placeholder="👤 🤖️",
                        height=BIG_WIN_H,
                    )
                with gr.Column():
                    @gr.render(triggers=[app.load, chatbot.change])
                    def render_context():
                        if self.state:
                            pdf_info = {}
                            for node_id, node_values in self.metadata.items():
                                file_name = node_values["file_name"]
                                file_path = node_values["file_path"]
                                page_number = node_values.get("page_number") - 1
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
                                        interactive=False,
                                        height=SMALL_WIN_H,
                                    )
                        else:
                            for timespan, files in self.metadata.items():
                                with gr.Tab(label=timespan):
                                    for file_name, file_info in files.items():
                                        with gr.Tab(label=file_name):
                                            pdf_path = file_info["file_path"]
                                            page_numbers = list(set(file_info["page_numbers"]))  # Remove duplicates
                                            pdf_pages = []
                                            doc = fitz.open(pdf_path)
                                            for pg_num in page_numbers:
                                                page = doc[pg_num - 1]
                                                pix = page.get_pixmap(matrix=fitz.Matrix(DPI / 72, DPI / 72))
                                                pdf_pages.append(
                                                    Image.frombytes('RGB', (pix.width, pix.height), pix.samples))

                                            gr.Gallery(
                                                value=pdf_pages,
                                                allow_preview=True,
                                                preview=True,
                                                selected_index=0,
                                                type='pil',
                                                interactive=False,
                                                height=int(SMALL_WIN_H * 0.9),
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
                        slider = gr.Slider(minimum=1, maximum=self.total_duration, step=1,
                                           value=self.args.time_freq, label="Time step (months)")
            with gr.Row():
                _examples = gr.Examples(examples=self.examples, inputs=[input_prompt])

            # backend
            input_prompt.submit(
                fn=self.exec_user_query,
                inputs=[input_prompt],
                outputs=[input_prompt, chatbot]
            ).success(
                fn=self.query_conv_agent,
                outputs=[chatbot]
            )

            slider.release(fn=self.set_timestep, inputs=[slider])

        app.queue(max_size=4)
        try:
            app.launch(share=True,
                       inbrowser=False,
                       max_threads=16,
                       favicon_path="./app/assets/bpc_logo.png",
                       server_port=7862)
        except (KeyboardInterrupt, SystemExit):
            app.close()
