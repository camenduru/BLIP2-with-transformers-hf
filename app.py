#!/usr/bin/env python

from __future__ import annotations

import os
import string

import gradio as gr
import PIL.Image
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration

DESCRIPTION = '# [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)'

if (SPACE_ID := os.getenv('SPACE_ID')) is not None:
    DESCRIPTION += f'\n<p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings. <a href="https://huggingface.co/spaces/{SPACE_ID}?duplicate=true"><img style="display: inline; margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space" /></a></p>'
if not torch.cuda.is_available():
    DESCRIPTION += '\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MODEL_ID_OPT_6_7B = 'Salesforce/blip2-opt-6.7b'
MODEL_ID_FLAN_T5_XXL = 'Salesforce/blip2-flan-t5-xxl'

if torch.cuda.is_available():
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
else:
    model_dict = {}


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
        max_length=50,
        min_length=1,
        num_beams=5,
        top_p=0.9)
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
                                   repetition_penalty=repetition_penalty,
                                   max_length=30,
                                   min_length=1,
                                   num_beams=5,
                                   top_p=0.9)
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
                interactive=False,
                visible=False)
            model_id_chat = gr.Dropdown(
                label='Model ID for VQA',
                choices=[MODEL_ID_OPT_6_7B, MODEL_ID_FLAN_T5_XXL],
                value=MODEL_ID_FLAN_T5_XXL,
                interactive=False,
                visible=False)
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
                caption_button = gr.Button(value='Caption it!')
                caption_output = gr.Textbox(
                    label='Caption Output',
                    show_label=False).style(container=False)
        with gr.Column():
            with gr.Box():
                chatbot = gr.Chatbot(label='VQA Chat')
                history_orig = gr.State(value=[])
                history_qa = gr.State(value=[])
                vqa_input = gr.Text(label='Chat Input',
                                    show_label=False,
                                    max_lines=1).style(container=False)
                with gr.Row():
                    clear_chat_button = gr.Button(value='Clear')
                    chat_button = gr.Button(value='Submit')

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
        api_name='caption',
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
        history_qa,
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
        api_name='chat',
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
        api_name='clear',
    )
    image.change(
        fn=lambda: ('', [], [], []),
        inputs=None,
        outputs=[
            caption_output,
            chatbot,
            history_orig,
            history_qa,
        ],
        queue=False,
    )

demo.queue(max_size=10).launch()
