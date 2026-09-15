import gradio as gr

from app import orchestrator


def format_response(result):
    cites = "\n".join(f"- {c}" for c in result["citations"]) or "_none_"
    side = f"**Citations:**\n{cites}"
    return result["answer"], side


def build_demo(runner=orchestrator.run_agentic):
    with gr.Blocks(title="Agentic RAG") as demo:
        gr.Markdown("# Agentic RAG\nAsk about AI-security papers. "
                    "Answers are generated from a retrieved corpus.")
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=460)
                box = gr.Textbox(placeholder="e.g. What is an indirect prompt injection?",
                                 show_label=False)
            with gr.Column(scale=1):
                side = gr.Markdown("_Ask a question to see retrieval path + citations._")

        def respond(message, history):
            result = runner(message)
            answer, side_md = format_response(result)
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": answer},
            ]
            return history, side_md, ""

        box.submit(respond, [box, chatbot], [chatbot, side, box])
    return demo
