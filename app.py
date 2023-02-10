from io import BytesIO

import string
import gradio as gr
import requests
from utils import Endpoint, get_token


def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    buffered.seek(0)

    return buffered


def query_chat_api(
    image, prompt, decoding_method, temperature, len_penalty, repetition_penalty
):

    url = endpoint.url
    url = url + "/api/generate"

    headers = {
        "User-Agent": "BLIP-2 HuggingFace Space",
        "Auth-Token": get_token(),
    }

    data = {
        "prompt": prompt,
        "use_nucleus_sampling": decoding_method == "Nucleus sampling",
        "temperature": temperature,
        "length_penalty": len_penalty,
        "repetition_penalty": repetition_penalty,
    }

    image = encode_image(image)
    files = {"image": image}

    response = requests.post(url, data=data, files=files, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + response.text


def query_caption_api(
    image, decoding_method, temperature, len_penalty, repetition_penalty
):

    url = endpoint.url
    url = url + "/api/caption"

    headers = {
        "User-Agent": "BLIP-2 HuggingFace Space",
        "Auth-Token": get_token(),
    }

    data = {
        "use_nucleus_sampling": decoding_method == "Nucleus sampling",
        "temperature": temperature,
        "length_penalty": len_penalty,
        "repetition_penalty": repetition_penalty,
    }

    image = encode_image(image)
    files = {"image": image}

    response = requests.post(url, data=data, files=files, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + response.text


def postprocess_output(output):
    # if last character is not a punctuation, add a full stop
    if not output[0][-1] in string.punctuation:
        output[0] += "."

    return output


def inference_chat(
    image,
    text_input,
    decoding_method,
    temperature,
    length_penalty,
    repetition_penalty,
    history=[],
):
    text_input = text_input
    history.append(text_input)

    prompt = " ".join(history)

    output = query_chat_api(
        image, prompt, decoding_method, temperature, length_penalty, repetition_penalty
    )
    output = postprocess_output(output)
    history += output

    chat = [
        (history[i], history[i + 1]) for i in range(0, len(history) - 1, 2)
    ]  # convert to tuples of list

    return {chatbot: chat, state: history}


def inference_caption(
    image,
    decoding_method,
    temperature,
    length_penalty,
    repetition_penalty,
):
    output = query_caption_api(
        image, decoding_method, temperature, length_penalty, repetition_penalty
    )

    return output[0]


title = """<h1 align="center">BLIP-2</h1>"""
description = """Gradio demo for BLIP-2, image-to-text generation from Salesforce Research. To use it, simply upload your image, or click one of the examples to load them.
<br> <strong>Disclaimer</strong>: This is a research prototype and is not intended for production use. No data including but not restricted to text and images is collected."""
article = """<strong>Paper</strong>: <a href='https://arxiv.org/abs/2301.12597' target='_blank'>BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models</a>
<br> <strong>Code</strong>: BLIP2 is now integrated into GitHub repo: <a href='https://github.com/salesforce/LAVIS' target='_blank'>LAVIS: a One-stop Library for Language and Vision</a>
<br> <strong>🤗 `transformers` integration</strong>: You can now use `transformers` to use our BLIP-2 models! Check out the <a href='https://huggingface.co/docs/transformers/main/en/model_doc/blip-2' target='_blank'> official docs </a>
<p> <strong>Project Page</strong>: <a href='https://github.com/salesforce/LAVIS/tree/main/projects/blip2' target='_blank'> BLIP2 on LAVIS</a>
<br> <strong>Description</strong>: Captioning results from <strong>BLIP2_OPT_6.7B</strong>. Chat results from <strong>BLIP2_FlanT5xxl</strong>.
"""

endpoint = Endpoint()

examples = [
    ["house.png", "How could someone get out of the house?"],
    ["flower.jpg", "Question: What is this flower and where is it's origin? Answer:"],
    ["pizza.jpg", "What are steps to cook it?"],
    ["sunset.jpg", "Here is a romantic message going along the photo:"],
    ["forbidden_city.webp", "In what dynasties was this place built?"],
]

with gr.Blocks(
    css="""
    .message.svelte-w6rprc.svelte-w6rprc.svelte-w6rprc {font-size: 20px; margin-top: 20px}
    #component-21 > div.wrap.svelte-w6rprc {height: 600px;}
    """
) as iface:
    state = gr.State([])

    gr.Markdown(title)
    gr.Markdown(description)
    gr.Markdown(article)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil")

            # with gr.Row():
            sampling = gr.Radio(
                choices=["Beam search", "Nucleus sampling"],
                value="Beam search",
                label="Text Decoding Method",
                interactive=True,
            )

            temperature = gr.Slider(
                minimum=0.5,
                maximum=1.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature (used with nucleus sampling)",
            )

            len_penalty = gr.Slider(
                minimum=-1.0,
                maximum=2.0,
                value=1.0,
                step=0.2,
                interactive=True,
                label="Length Penalty (set to larger for longer sequence, used with beam search)",
            )

            rep_penalty = gr.Slider(
                minimum=1.0,
                maximum=5.0,
                value=1.5,
                step=0.5,
                interactive=True,
                label="Repeat Penalty (larger value prevents repetition)",
            )

        with gr.Column(scale=1.8):

            with gr.Column():
                caption_output = gr.Textbox(lines=1, label="Caption Output")
                caption_button = gr.Button(
                    value="Caption it!", interactive=True, variant="primary"
                )
                caption_button.click(
                    inference_caption,
                    [
                        image_input,
                        sampling,
                        temperature,
                        len_penalty,
                        rep_penalty,
                    ],
                    [caption_output],
                )

            gr.Markdown("""Trying prompting your input for chat; e.g. example prompt for QA, \"Question: {} Answer:\" Use proper punctuation (e.g., question mark).""")
            with gr.Row():
                with gr.Column(
                    scale=1.5, 
                ):
                    chatbot = gr.Chatbot(
                        label="Chat Output (from FlanT5)",
                    )

                # with gr.Row():
                with gr.Column(scale=1):
                    chat_input = gr.Textbox(lines=1, label="Chat Input")
                    chat_input.submit(
                        inference_chat,
                        [
                            image_input,
                            chat_input,
                            sampling,
                            temperature,
                            len_penalty,
                            rep_penalty,
                            state,
                        ],
                        [chatbot, state],
                    )

                    with gr.Row():
                        clear_button = gr.Button(value="Clear", interactive=True)
                        clear_button.click(
                            lambda: ("", [], []),
                            [],
                            [chat_input, chatbot, state],
                            queue=False,
                        )

                        submit_button = gr.Button(
                            value="Submit", interactive=True, variant="primary"
                        )
                        submit_button.click(
                            inference_chat,
                            [
                                image_input,
                                chat_input,
                                sampling,
                                temperature,
                                len_penalty,
                                rep_penalty,
                                state,
                            ],
                            [chatbot, state],
                        )

            image_input.change(
                lambda: ("", "", []),
                [],
                [chatbot, caption_output, state],
                queue=False,
            )

    examples = gr.Examples(
        examples=examples,
        inputs=[image_input, chat_input],
    )

iface.queue(concurrency_count=1, api_open=False, max_size=10)
iface.launch(enable_queue=True)
