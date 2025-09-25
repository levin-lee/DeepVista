import os
import re
from typing import List, Dict, Tuple, Optional

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT_DEFAULT = "You are a concise, helpful AI assistant. Keep replies short and clearly."
SYSTEM_PROMPT_QA = (
    "You are a helpful assistant answering based on a YouTube video's transcript. "
    "Use only the provided transcript context. Be precise and concise. If you cite, include the timestamp(s)."
)

# Strict pattern for full-url validation (matches the whole string)
YOUTUBE_URL_PATTERN = re.compile(
    r"^(?:https?://)?(?:www\.)?(?:youtube\.com/(?:watch\?v=|embed/|v/)|youtu\.be/)([A-Za-z0-9_-]{11})(?:[&#?].*)?$",
    re.IGNORECASE,
)

# Lenient pattern to find the first YouTube URL anywhere inside a sentence
YOUTUBE_URL_IN_TEXT = re.compile(
    r"(https?://(?:www\.)?(?:youtube\.com/(?:watch\?v=[A-Za-z0-9_-]{11}[^\s]*|embed/[A-Za-z0-9_-]{11}(?:[^\s]*)?|v/[A-Za-z0-9_-]{11}(?:[^\s]*)?)|youtu\.be/[A-Za-z0-9_-]{11}(?:[^\s]*)?))",
    re.IGNORECASE,
)


def is_youtube_url(text: str) -> bool:
    text = (text or "").strip()
    return bool(YOUTUBE_URL_PATTERN.match(text))


def find_first_youtube_url(text: str) -> Optional[str]:
    """Find the first YouTube URL inside arbitrary text."""
    if not text:
        return None
    m = YOUTUBE_URL_IN_TEXT.search(text)
    return m.group(1) if m else None


def extract_video_id(url: str) -> Optional[str]:
    m = YOUTUBE_URL_PATTERN.match(url.strip())
    return m.group(1) if m else None


def fetch_oembed(url: str) -> Dict:
    try:
        resp = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": url, "format": "json"},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
        return {}
    except Exception:
        return {}


def fetch_transcript(video_id: str) -> Tuple[str, List[Dict]]:
    """Return transcript_text and segments with timestamps. Try common languages then auto-generated."""
    preferred_langs = ["en", "en-US", "zh", "zh-Hans", "zh-Hant", "ja", "es"]
    ytt_api = YouTubeTranscriptApi()
    try:
        segments = ytt_api.fetch(video_id, languages=preferred_langs)
    except Exception as e:
        raise e

    # Join to text with timestamps like [mm:ss] content
    def format_ts(sec: float) -> str:
        m = int(sec // 60)
        s = int(sec % 60)
        return f"{m:02d}:{s:02d}"

    lines = []
    for seg in segments:
        start = format_ts(seg.start)
        text = seg.text.replace("\n", " ")
        lines.append(f"[{start}] {text}")

    transcript_text = "\n".join(lines)
    return transcript_text, segments


def summarize_transcript(transcript_text: str) -> str:
    prompt = (
        "Summarize the video's content in 3-4 sentences. Be concise and capture the key points.\n\n"
        "Transcript:\n" + transcript_text[:12000]
    )
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_DEFAULT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=400,
        )
        return completion.choices[0].message.content or ""
    except Exception:
        return "Failed to generate summary. Please try again later."


def answer_with_transcript(question: str, transcript_text: str) -> str:
    prompt = (
        "Answer the user's question using ONLY the transcript below. "
        "If specific parts are relevant, cite timestamps like [mm:ss].\n\n"
        f"Transcript (may be truncated):\n{transcript_text[:14000]}\n\n"
        f"Question: {question}"
    )
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_QA},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=600,
        )
        return completion.choices[0].message.content or ""
    except Exception:
        return "Failed to answer based on transcript. Please try again later."


# App UI: hide transcript panel; show video info inside chat; beautify chatbot
with gr.Blocks(title="YouTube Chatbot", theme=gr.themes.Soft(primary_hue=gr.themes.colors.indigo)) as demo:
    gr.Markdown("""### YouTube Chatbot
    """)

    # sessions: {id: {history, memory, video_info, transcript_text, transcript_segments}}
    sessions_state = gr.State(value={"default": {"history": [], "memory": [], "video_info": {}, "transcript_text": "", "transcript_segments": []}})

    with gr.Row():
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(
                label="",
                height=520,
                bubble_full_width=False,
                show_copy_button=True,
            )
            with gr.Row():
                txt = gr.Textbox(label="Paste URL or ask", placeholder="Paste YouTube URL or ask a question...", lines=3, scale=5)
                with gr.Column(scale=1):
                    clear_btn = gr.Button("Clear Current Session", variant="secondary")
                    send_btn = gr.Button("Send", variant="primary")
        with gr.Column(scale=3):
            with gr.Row():
                new_session_btn = gr.Button("+ Add Session", variant="primary")
            session_radio = gr.Radio(choices=["default"], value="default", label="Sessions")

    def ensure_session(sessions: Dict, session_id: str) -> Dict:
        if session_id not in sessions:
            sessions[session_id] = {"history": [], "memory": [], "video_info": {}, "transcript_text": "", "transcript_segments": []}
        for k, v in {"history": [], "memory": [], "video_info": {}, "transcript_text": "", "transcript_segments": []}.items():
            sessions[session_id].setdefault(k, v)
        return sessions

    def make_video_info_md(info: Dict) -> str:
        if not info:
            return ""
        title = info.get("title", "")
        author = info.get("author_name", "")
        thumbnail = info.get("thumbnail_url", "")
        lines = []
        if title:
            lines.append(f"**Title**: {title}")
        if author:
            lines.append(f"**Channel**: {author}")
        if thumbnail:
            lines.append(f"![]({thumbnail})")
        return "\n\n".join(lines)

    def handle_youtube_flow(url: str, selected_session: str, sessions: Dict) -> Tuple[str, List[List[str]], Dict]:
        sessions = ensure_session(sessions, selected_session)
        vid = extract_video_id(url)
        if not vid:
            err = "Invalid YouTube URL."
            sessions[selected_session]["history"] = sessions[selected_session]["history"] + [[url, err]]
            return "", sessions[selected_session]["history"], sessions

        # Cached transcript reuse
        """
        if sessions[selected_session].get("transcript_text"):
            if not sessions[selected_session].get("video_info"):
                info = fetch_oembed(url)
                sessions[selected_session]["video_info"] = info
            summary = summarize_transcript(sessions[selected_session]["transcript_text"])
            info_md = make_video_info_md(sessions[selected_session]["video_info"])
            bot_msg = (info_md + "\n\n" if info_md else "") + summary
            sessions[selected_session]["history"] = sessions[selected_session]["history"] + [[url, bot_msg]]
            return "", sessions[selected_session]["history"], sessions
        """

        # fetch oEmbed + transcript
        info = fetch_oembed(url)
        try:
            transcript_text, segments = fetch_transcript(vid)
        except VideoUnavailable:
            err = "The video is unavailable."
            sessions[selected_session]["history"] = sessions[selected_session]["history"] + [[url, err]]
            return "", sessions[selected_session]["history"], sessions
        except (TranscriptsDisabled, NoTranscriptFound):
            err = "Transcript is not available for this video."
            sessions[selected_session]["video_info"] = info
            info_md = make_video_info_md(info)
            bot_msg = (info_md + "\n\n" if info_md else "") + err
            sessions[selected_session]["history"] = sessions[selected_session]["history"] + [[url, bot_msg]]
            return "", sessions[selected_session]["history"], sessions
        except Exception:
            err = "Failed to retrieve transcript. Please try again later."
            sessions[selected_session]["video_info"] = info
            info_md = make_video_info_md(info)
            bot_msg = (info_md + "\n\n" if info_md else "") + err
            sessions[selected_session]["history"] = sessions[selected_session]["history"] + [[url, bot_msg]]
            return "", sessions[selected_session]["history"], sessions

        sessions[selected_session]["video_info"] = info
        sessions[selected_session]["transcript_text"] = transcript_text
        sessions[selected_session]["transcript_segments"] = segments

        summary = summarize_transcript(transcript_text)
        info_md = make_video_info_md(info)
        bot_msg = (info_md + "\n\n" if info_md else "") + summary
        sessions[selected_session]["history"] = sessions[selected_session]["history"] + [[url, bot_msg]]

        return "", sessions[selected_session]["history"], sessions

    def on_submit(user_message: str, selected_session: str, sessions: Dict) -> Tuple[str, List[List[str]], Dict]:
        user_message = (user_message or "").strip()
        sessions = ensure_session(sessions, selected_session)
        if not user_message:
            return gr.update(), sessions[selected_session]["history"], sessions

        # If the sentence contains a YouTube URL, extract and handle it first
        maybe_url = find_first_youtube_url(user_message)
        if maybe_url:
            return handle_youtube_flow(maybe_url, selected_session, sessions)

        # Otherwise, normal QA or fallback chat
        transcript_text = sessions[selected_session].get("transcript_text", "")
        if transcript_text:
            answer = answer_with_transcript(user_message, transcript_text)
        else:
            memory = sessions[selected_session].get("memory", [])
            messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT_DEFAULT}]
            if memory:
                messages.extend(memory[-10:])
            messages.append({"role": "user", "content": user_message})
            try:
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=800,
                )
                answer = completion.choices[0].message.content or ""
            except (RateLimitError, APIConnectionError, APIStatusError):
                answer = "Service error. Please try again later."
            except Exception:
                answer = "Unknown error."

        sessions[selected_session]["history"] = sessions[selected_session]["history"] + [[user_message, answer]]
        sessions[selected_session]["memory"] = (sessions[selected_session].get("memory", []) + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer},
        ])[-10:]

        return "", sessions[selected_session]["history"], sessions

    def on_new_session(sessions: Dict) -> Tuple[gr.Radio, Dict, List[List[str]]]:
        base = "session-"
        i = 1
        while f"{base}{i}" in sessions:
            i += 1
        new_id = f"{base}{i}"
        sessions[new_id] = {"history": [], "memory": [], "video_info": {}, "transcript_text": "", "transcript_segments": []}
        choices = list(sessions.keys())
        return gr.Radio(choices=choices, value=new_id), sessions, []

    def on_clear(selected_session: str, sessions: Dict) -> Tuple[List[List[str]], Dict]:
        sessions = ensure_session(sessions, selected_session)
        sessions[selected_session] = {"history": [], "memory": [], "video_info": {}, "transcript_text": "", "transcript_segments": []}
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
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
