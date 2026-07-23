import gradio as gr

from app import graph


def format_response(result):
    badge = ("🔁 retrieval retried" if "rewrite" in result["path_trace"]
             else "✅ direct")
    cites = "\n".join(f"- {c}" for c in result["citations"]) or "_none_"
    side = f"**Path:** {badge}\n\n**Citations:**\n{cites}"
    return result["answer"], side


def build_demo(runner=graph.run):
    with gr.Blocks(title="Self-Correcting RAG") as demo:
        gr.Markdown("# Self-Correcting RAG\nAsk about AI-security papers. "
                    "Answers are grounded in a retrieved corpus with a "
                    "self-correcting (CRAG) retrieval loop.")
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
