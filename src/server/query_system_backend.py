from server.server_pipeline import ServerPipeline
from server.config import TOP_K

import gradio as gr
import pandas as pd

import os


class QuerySystemBackend:
    def __init__(self, server_pipeline: ServerPipeline):
        self.server_pipeline = server_pipeline
        self.top_k = TOP_K
        self.page = self.build_page()

    def build_page(self):
        
        with gr.Blocks() as page:
            with gr.Box():
                with gr.Row():
                    # gr.Image("server/src/BikeReID-logo2-s.jpg", container=False, show_download_button=False)
                    gr.Markdown("# Bicycle Re-ID")
                gr.Markdown("Input a picture of your bicycle to find its recent location.")
                
                # self.top_k = gr.Slider(5, 50, value=TOP_K, step=1, label="Number of query results")
            with gr.Box():
                gr.Markdown("## Query Image:")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(" ")
                    image_input = gr.Image(scale = 2)
                    with gr.Column(scale=1):
                        gr.Markdown(" ")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown(" ")
                    image_button = gr.Button("Query",scale=1)
                    with gr.Column(scale=2):
                        gr.Markdown(" ")
                
            with gr.Box():
                gr.Markdown("## Query Results:")
                gr.Markdown("The following search results are sorted by similarity. ")
                gr.Markdown("If the search results are empty, it may be because the bicycle in the query image was not recognized. Please try using another image.")

                ui_content=[]
                for _ in range(self.top_k):
                    with gr.Row():
                        with gr.Column(scale=1):
                            image_output = gr.Image(type="filepath", label=None, container = True)
                            ui_content.append(image_output)
                        with gr.Column(scale=2):
                            table_output = gr.DataFrame(type="pandas", label=None)
                            ui_content.append(table_output)



                #with gr.Accordion("See Details"):
                    #gr.Markdown("lorem ipsum")
                    
                
                image_button.click(fn=self.server_pipeline.query_img, inputs=image_input, outputs=ui_content, api_name="greet")

            
        return page

    def launch(self):
        self.page.launch()

    
    
