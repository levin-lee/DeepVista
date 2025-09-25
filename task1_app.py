import os
from typing import List, Dict, Tuple
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
system_prompt = "You are a concise, helpful AI assistant. Keep replies short and clearly."


def generate_ai_reply(user_message: str, short_memory: List[Dict[str, str]]) -> str:
    """Generate a single-turn reply with minimal short-term memory.

    - short_memory: last 10 exchanges from this session in [{role, content}] format
    - Implements single-turn conversation: only current user_message is required
    """
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # Add short, bounded memory (optional, small context only)
    if short_memory:
        # keep at most last 10 items to emulate short memory
        trimmed = short_memory[-10:]
        messages.extend(trimmed)

    messages.append({"role": "user", "content": user_message})

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
        )
        return completion.choices[0].message.content or ""
    except RateLimitError:
        return "The request was too frequent. Please try again later."
    except APIConnectionError:
        return "Unable to connect to OpenAI service, please check your network."
    except APIStatusError as e:
        return f"OpenAI service error：{getattr(e, 'status_code', 'unknown')}."
    except Exception:
        return "An unknown error occurred, please try again later."


# Gradio app with multi-session support
with gr.Blocks(title="Basic AI Chatbot", theme=gr.themes.Soft(primary_hue=gr.themes.colors.indigo)) as demo:
    gr.Markdown("""
    ### Basic AI Chatbot
    """)

    # Global sessions state: {session_id: {"history": List[List[str]], "memory": List[Dict]}}
    sessions_state = gr.State(value={"default": {"history": [], "memory": []}})

    with gr.Row():
        # Left main area: Chat + Input + Controls
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(label="dialogue", height=420, bubble_full_width=False)

            with gr.Row():
                txt = gr.Textbox(label="Please input", placeholder="Please input...", lines=3, scale=5)
                with gr.Column(scale=1):
                    clear_btn = gr.Button("Clear Current Session", variant="secondary")
                    send_btn = gr.Button("Send", variant="primary")

        # Right sidebar: Add session + session list (click to switch)
        with gr.Column(scale=3):
            with gr.Row():
                new_session_btn = gr.Button("+ Add Session", variant="primary")
            session_radio = gr.Radio(choices=["default"], value="default", label="Sessions")

    def ensure_session(sessions: Dict, session_id: str) -> Dict:
        if session_id not in sessions:
            sessions[session_id] = {"history": [], "memory": []}
        return sessions

    def on_submit(user_message: str, selected_session: str, sessions: Dict) -> Tuple[str, List[List[str]], Dict]:
        user_message = (user_message or "").strip()
        if not user_message:
            # no update
            current_history = ensure_session(sessions, selected_session)[selected_session]["history"]
            return gr.update(), current_history, sessions

        sessions = ensure_session(sessions, selected_session)
        memory = sessions[selected_session]["memory"]
        reply = generate_ai_reply(user_message, memory)

        # update history and memory
        sessions[selected_session]["history"] = sessions[selected_session]["history"] + [[user_message, reply]]
        sessions[selected_session]["memory"] = (memory or []) + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": reply},
        ]
        sessions[selected_session]["memory"] = sessions[selected_session]["memory"][-10:]

        return "", sessions[selected_session]["history"], sessions

    def on_new_session(sessions: Dict) -> Tuple[gr.Radio, Dict, List[List[str]]]:
        # simple incremental naming: session-1, session-2, ...
        base = "session-"
        i = 1
        while f"{base}{i}" in sessions:
            i += 1
        new_id = f"{base}{i}"
        sessions[new_id] = {"history": [], "memory": []}
        choices = list(sessions.keys())
        return gr.Radio(choices=choices, value=new_id), sessions, []

    def on_clear(selected_session: str, sessions: Dict) -> Tuple[List[List[str]], Dict]:
        sessions = ensure_session(sessions, selected_session)
        sessions[selected_session] = {"history": [], "memory": []}
        return [], sessions

    def on_switch_session(selected_session: str, sessions: Dict) -> List[List[str]]:
        sessions = ensure_session(sessions, selected_session)
        return sessions[selected_session]["history"]

    # Bind events
    txt.submit(
        fn=on_submit,
        inputs=[txt, session_radio, sessions_state],
        outputs=[txt, chatbot, sessions_state],
    )

    send_btn.click(
        fn=on_submit,
        inputs=[txt, session_radio, sessions_state],
        outputs=[txt, chatbot, sessions_state],
    )

    new_session_btn.click(
        fn=on_new_session,
        inputs=[sessions_state],
        outputs=[session_radio, sessions_state, chatbot],
    )

    clear_btn.click(
        fn=on_clear,
        inputs=[session_radio, sessions_state],
        outputs=[chatbot, sessions_state],
    )

    session_radio.change(
        fn=on_switch_session,
        inputs=[session_radio, sessions_state],
        outputs=[chatbot],
    )

if __name__ == "__main__":
    port = os.getenv("PORT", 7860)
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", port)))
