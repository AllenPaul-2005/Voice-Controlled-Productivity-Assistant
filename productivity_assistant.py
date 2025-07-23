import torch
from transformers import pipeline
import ollama
import gradio as gr
import os
from tools import TOOLS

# ========== File Operations ==========
def create_file(filename, content):
    with open(filename, 'w') as f:
        f.write(content)

def read_file(filename):
    try:
        with open(filename, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return "❌ File not found."

def delete_file(filename):
    try:
        os.remove(filename)
    except FileNotFoundError:
        return "❌ File not found."

# ========== AI Response ==========
def get_response(prompt):
    response = ollama.chat(
        model='mistral',
        messages=[{'role': 'user', 'content': prompt}],
        tools=TOOLS
    )

    tool_calls = response.get('message', {}).get('tool_calls', None)
    raw_text = response.get('message', {}).get('content', '🤖 No model response.')
    results = []
    task_table = None

    if tool_calls:
        for call in tool_calls:
            fn = call['function']['name']
            args = call['function']['arguments']

            if fn == 'create_file':
                create_file(args['filename'], args['content'])
                results.append(f"✅ File `{args['filename']}` created.")
            elif fn == 'read_file':
                content = read_file(args['filename'])
                results.append(f"📖 Read `{args['filename']}`:\n{content}")
            elif fn == 'delete_file':
                delete_file(args['filename'])
                results.append(f"🗑️ File `{args['filename']}` deleted.")
            elif fn == 'edit_file':
                create_file(args['filename'], args['content'])
                results.append(f"✏️ File `{args['filename']}` edited.")
            elif fn == 'add_task':
                results.append(f"📝 Task added: {args['task_description']}")
    else:
        results.append(f"💬 {raw_text}")

    return results, task_table

# ========== Transcription ==========
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=-1)

def transcribe_and_respond(audio_input):
    if audio_input is None:
        raise gr.Error("Please provide an audio input.")
    transcription = pipe(audio_input, generate_kwargs={"task": "transcribe"}, return_timestamps=True)["text"]
    response, task_table = get_response(transcription)
    return transcription, "\n".join(response), task_table

# ========== Gradio UI ==========
with gr.Blocks(css="custom.css", title="Voice-Controlled Productivity Assistant") as demo:
    gr.HTML("""
    <head>
        <link rel="icon" href="https://em-content.zobj.net/source/telegram/386/microphone_1f3a4.png" type="image/png">
        <style>
            .app-header {
                text-align: center;
                margin-top: 30px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .app-header h1 {
                font-size: 2.4rem;
                color: #4eaaff;
                margin-bottom: 0.4rem;
            }
            .app-header p {
                font-size: 1.1rem;
                color: #aaa;
            }
        </style>
    </head>
    <div class="app-header">
        <h1>🎙️ Voice-Controlled Productivity Assistant</h1>
        <p>✨ Use your voice to manage files, tasks, and get AI-powered replies</p>
    </div>
    """)

    with gr.Row():
        audio = gr.Audio(label="🎧 Your Voice", sources=["microphone", "upload"], type="filepath")

    with gr.Row():
        transcription = gr.Textbox(label="📝 Transcription", lines=2)
        ai_response = gr.Textbox(label="💡 AI Response", lines=6)

    with gr.Row():
        task_table = gr.Dataframe(label="📋 Task Table")

    run_btn = gr.Button("🚀 Run")
    run_btn.click(fn=transcribe_and_respond, inputs=[audio], outputs=[transcription, ai_response, task_table])

    gr.HTML("""
    <footer style="margin-top: 25px; text-align: center; font-size: 0.85rem; color: #888;">
        🌐 Powered by Whisper + Ollama + Mistral • UI by Allen
    </footer>
    """)

demo.launch()
