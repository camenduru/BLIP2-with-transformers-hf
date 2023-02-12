#!/usr/bin/env python

from __future__ import annotations

import string

import gradio as gr
import PIL.Image
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration

DESCRIPTION = '# BLIP-2'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MODEL_ID_OPT_6_7B = 'Salesforce/blip2-opt-6.7b'
MODEL_ID_FLAN_T5_XXL = 'Salesforce/blip2-flan-t5-xxl'
model_dict = {
    #MODEL_ID_OPT_6_7B: {
    #    'processor':
    #    AutoProcessor.from_pretrained(MODEL_ID_OPT_6_7B),
    #    'model':
    #    Blip2ForConditionalGeneration.from_pretrained(MODEL_ID_OPT_6_7B,
    #                                                  device_map='auto',
    #                                                  load_in_8bit=True),
    #},
    MODEL_ID_FLAN_T5_XXL: {
        'processor':
        AutoProcessor.from_pretrained(MODEL_ID_FLAN_T5_XXL),
        'model':
        Blip2ForConditionalGeneration.from_pretrained(MODEL_ID_FLAN_T5_XXL,
                                                      device_map='auto',
                                                      load_in_8bit=True),
    }
}


def generate_caption(model_id: str, image: PIL.Image.Image,
                     decoding_method: str, temperature: float,
                     length_penalty: float, repetition_penalty: float) -> str:
    model_info = model_dict[model_id]
    processor = model_info['processor']
    model = model_info['model']

    inputs = processor(images=image,
                       return_tensors='pt').to(device, torch.float16)
    generated_ids = model.generate(
        pixel_values=inputs.pixel_values,
        do_sample=decoding_method == 'Nucleus sampling',
        temperature=temperature,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        max_length=50)
    result = processor.batch_decode(generated_ids,
                                    skip_special_tokens=True)[0].strip()
    return result


def answer_question(model_id: str, image: PIL.Image.Image, text: str,
                    decoding_method: str, temperature: float,
                    length_penalty: float, repetition_penalty: float) -> str:
    model_info = model_dict[model_id]
    processor = model_info['processor']
    model = model_info['model']

    inputs = processor(images=image, text=text,
                       return_tensors='pt').to(device, torch.float16)
    generated_ids = model.generate(**inputs,
                                   do_sample=decoding_method ==
                                   'Nucleus sampling',
                                   temperature=temperature,
                                   length_penalty=length_penalty,
                                   repetition_penalty=repetition_penalty)
    result = processor.batch_decode(generated_ids,
                                    skip_special_tokens=True)[0].strip()
    return result


def postprocess_output(output: str) -> str:
    if output and not output[-1] in string.punctuation:
        output += '.'
    return output


def chat(
    model_id: str,
    image: PIL.Image.Image,
    text: str,
    decoding_method: str,
    temperature: float,
    length_penalty: float,
    repetition_penalty: float,
    history_orig: list[str] = [],
    history_qa: list[str] = [],
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
    history_orig.append(text)
    text_qa = f'Question: {text} Answer:'
    history_qa.append(text_qa)
    prompt = ' '.join(history_qa)

    output = answer_question(
        model_id,
        image,
        prompt,
        decoding_method,
        temperature,
        length_penalty,
        repetition_penalty,
    )
    output = postprocess_output(output)
    history_orig.append(output)
    history_qa.append(output)

    chat_val = list(zip(history_orig[0::2], history_orig[1::2]))
    return gr.update(value=chat_val), gr.update(value=history_orig), gr.update(
        value=history_qa)


examples = [
    [
        'house.png',
        'How could someone get out of the house?',
    ],
    [
        'flower.jpg',
        'What is this flower and where is it\'s origin?',
    ],
    [
        'pizza.jpg',
        'What are steps to cook it?',
    ],
    [
        'sunset.jpg',
        'Here is a romantic message going along the photo:',
    ],
    [
        'forbidden_city.webp',
        'In what dynasties was this place built?',
    ],
]

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)

    image = gr.Image(type='pil')
    with gr.Accordion(label='Advanced settings', open=False):
        with gr.Row():
            model_id_caption = gr.Dropdown(
                label='Model ID for image captioning',
                choices=[MODEL_ID_OPT_6_7B, MODEL_ID_FLAN_T5_XXL],
                value=MODEL_ID_FLAN_T5_XXL,
                interactive=False)
            model_id_chat = gr.Dropdown(
                label='Model ID for VQA',
                choices=[MODEL_ID_OPT_6_7B, MODEL_ID_FLAN_T5_XXL],
                value=MODEL_ID_FLAN_T5_XXL,
                interactive=False)
        sampling_method = gr.Radio(
            label='Text Decoding Method',
            choices=['Beam search', 'Nucleus sampling'],
            value='Beam search',
        )
        temperature = gr.Slider(
            label='Temperature (used with nucleus sampling)',
            minimum=0.5,
            maximum=1.0,
            value=1.0,
            step=0.1,
        )
        length_penalty = gr.Slider(
            label=
            'Length Penalty (set to larger for longer sequence, used with beam search)',
            minimum=-1.0,
            maximum=2.0,
            value=1.0,
            step=0.2,
        )
        rep_penalty = gr.Slider(
            label='Repeat Penalty (larger value prevents repetition)',
            minimum=1.0,
            maximum=5.0,
            value=1.5,
            step=0.5,
        )
    with gr.Row():
        with gr.Column():
            with gr.Box():
                gr.Markdown('Image Captioning')
                caption_button = gr.Button(value='Caption it!')
                caption_output = gr.Textbox(label='Caption Output')
        with gr.Column():
            with gr.Box():
                gr.Markdown('VQA Chat')
                vqa_input = gr.Text(label='Chat Input', max_lines=1)
                with gr.Row():
                    clear_chat_button = gr.Button(value='Clear')
                    chat_button = gr.Button(value='Submit')
                chatbot = gr.Chatbot(label='Chat Output')
                history_orig = gr.State(value=[])
                history_qa = gr.State(value=[])

    gr.Examples(
        examples=examples,
        inputs=[
            image,
            vqa_input,
        ],
    )

    caption_button.click(
        fn=generate_caption,
        inputs=[
            model_id_caption,
            image,
            sampling_method,
            temperature,
            length_penalty,
            rep_penalty,
        ],
        outputs=caption_output,
    )

    chat_inputs = [
        model_id_chat,
        image,
        vqa_input,
        sampling_method,
        temperature,
        length_penalty,
        rep_penalty,
        history_orig,
    ]
    chat_outputs = [
        chatbot,
        history_orig,
        history_qa,
    ]
    vqa_input.submit(
        fn=chat,
        inputs=chat_inputs,
        outputs=chat_outputs,
    )
    chat_button.click(
        fn=chat,
        inputs=chat_inputs,
        outputs=chat_outputs,
    )
    clear_chat_button.click(
        fn=lambda: ('', [], [], []),
        inputs=None,
        outputs=[
            vqa_input,
            chatbot,
            history_orig,
            history_qa,
        ],
        queue=False,
    )
    image.change(
        fn=lambda: ('', '', [], []),
        inputs=None,
        outputs=[
            chatbot,
            caption_output,
            history_orig,
            history_qa,
        ],
        queue=False,
    )

demo.queue(max_size=10).launch()
