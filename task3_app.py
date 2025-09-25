import os
import re
from typing import List, Dict, Tuple, Optional

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError
import requests
import networkx as nx
import plotly.graph_objects as go
import spacy

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT_DEFAULT = "You are a concise, helpful AI assistant. Keep replies short and clearly."
SYSTEM_PROMPT_QA = (
    "You are a helpful assistant answering based on a YouTube video's transcript. "
    "Use only the provided transcript context. Be precise and concise. If you cite, include the timestamp(s)."
)

# Strict pattern for full-url validation
YOUTUBE_URL_PATTERN = re.compile(
    r"^(?:https?://)?(?:www\.)?(?:youtube\.com/(?:watch\?v=|embed/|v/)|youtu\.be/)([A-Za-z0-9_-]{11})(?:[&#?].*)?$",
    re.IGNORECASE,
)
# Lenient pattern to find the first YouTube URL anywhere inside a sentence
YOUTUBE_URL_IN_TEXT = re.compile(
    r"(https?://(?:www\.)?(?:youtube\.com/(?:watch\?v=[A-Za-z0-9_-]{11}[^\s]*|embed/[A-Za-z0-9_-]{11}(?:[^\s]*)?|v/[A-Za-z0-9_-]{11}(?:[^\s]*)?)|youtu\.be/[A-Za-z0-9_-]{11}(?:[^\s]*)?))",
    re.IGNORECASE,
)

# Soft-match triggers for concept map
CONCEPT_TRIGGERS = [
    "dive deeper into the concepts",
    "show me the concept map",
    "visualize video concepts",
    "concept map",
    "show concepts",
    "visualize concepts",
    "deeper concepts",
]


def find_first_youtube_url(text: str) -> Optional[str]:
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
    preferred_langs = ["en", "en-US", "zh", "zh-Hans", "zh-Hant", "ja", "es"]
    ytt_api = YouTubeTranscriptApi()
    segments = ytt_api.fetch(video_id, languages=preferred_langs)

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


# -------- Concept graph utilities --------

def _nlp_extract_terms(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    # Prefer spaCy noun chunks and entities if available
    if spacy is not None:
        try:
            # Try to load small English model; if failed, fallback
            try:
                nlp = spacy.load("en_core_web_sm")  # type: ignore
            except Exception:
                nlp = spacy.blank("en")  # type: ignore
            doc = nlp(text)
            terms: List[str] = []
            # noun chunks
            if hasattr(doc, "noun_chunks"):
                for chunk in doc.noun_chunks:  # type: ignore
                    t = chunk.text.strip()
                    if 2 <= len(t) <= 64:
                        terms.append(t)
            # entities
            if hasattr(doc, "ents"):
                for ent in doc.ents:
                    t = ent.text.strip()
                    if 2 <= len(t) <= 64:
                        terms.append(t)
            # fallback to tokens if nothing found
            if not terms:
                tokens = [t.text for t in doc if not t.is_stop and t.is_alpha and 3 <= len(t.text) <= 32]
                terms.extend(tokens)
            return terms
        except Exception:
            pass
    # Very light fallback: split on non-letters and filter
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,31}", text)
    return [t.lower() for t in tokens]


def build_concept_graph(transcript_text: str) -> go.Figure:
    # Extract candidate terms
    terms = _nlp_extract_terms(transcript_text)
    if not terms:
        # fallback to words
        terms = re.findall(r"[A-Za-z][A-Za-z\-]{2,31}", transcript_text)
        terms = [t.lower() for t in terms]

    # Frequency for node size
    freq: Dict[str, int] = {}
    for t in terms:
        freq[t] = freq.get(t, 0) + 1

    # Co-occurrence within sliding window
    window = 12
    edges: Dict[Tuple[str, str], int] = {}
    for i in range(len(terms)):
        for j in range(i + 1, min(i + window, len(terms))):
            a, b = terms[i].lower(), terms[j].lower()
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            edges[key] = edges.get(key, 0) + 1

    # Build graph (trim weak nodes/edges)
    G = nx.Graph()
    for term, count in freq.items():
        if count >= 2:  # keep only repeated terms
            G.add_node(term, size=count)
    for (a, b), w in edges.items():
        if a in G and b in G and w >= 2:
            G.add_edge(a, b, weight=w)

    if G.number_of_nodes() == 0:
        # add minimal node to avoid empty plot
        G.add_node("no-concepts-found", size=1)

    # Node centrality (importance)
    try:
        centrality = nx.pagerank(G, alpha=0.85)
    except Exception:
        centrality = {n: 1.0 / max(1, G.number_of_nodes()) for n in G.nodes()}

    # Layout
    pos = nx.spring_layout(G, k=0.6, iterations=50, seed=42, weight="weight")

    # Prepare Plotly traces
    # Edges
    edge_x = []
    edge_y = []
    edge_widths = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        w = data.get("weight", 1)
        edge_widths.append(max(1.0, min(6.0, 0.6 * w)))

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="#95a5a6"),
        hoverinfo="none",
        mode="lines",
    )

    # Nodes
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    for n, data in G.nodes(data=True):
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        score = centrality.get(n, 0.1)
        size = 10 + 40 * score  # size by centrality
        node_size.append(size)
        node_text.append(f"{n}<br>importance: {score:.2f}")
        # simple color bucketing by degree
        deg = G.degree[n]
        if deg >= 6:
            node_color.append("#d35400")  # orange
        elif deg >= 3:
            node_color.append("#2980b9")  # blue
        else:
            node_color.append("#27ae60")  # green

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line=dict(width=1, color="#2c3e50"),
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        title="Concept Map",
        title_x=0.5,
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=20, r=20, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    ))

    return fig


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

    # Always fetch oEmbed + transcript (simplified for Task 3)
    info = fetch_oembed(url)
    try:
        transcript_text, segments = fetch_transcript(vid)
    except VideoUnavailable:
        err = "The video is unavailable."
        sessions[selected_session]["history"] = sessions[selected_session]["history"] + [[url, err]]
        return "", sessions[selected_session]["history"], sessions
    except (TranscriptsDisabled, NoTranscriptFound):
        err = "Transcript is not available for this video."
        bot_msg = (make_video_info_md(info) + "\n\n" if info else "") + err
        sessions[selected_session]["history"] = sessions[selected_session]["history"] + [[url, bot_msg]]
        return "", sessions[selected_session]["history"], sessions
    except Exception:
        err = "Failed to retrieve transcript. Please try again later."
        bot_msg = (make_video_info_md(info) + "\n\n" if info else "") + err
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


def ensure_session(sessions: Dict, session_id: str) -> Dict:
    if session_id not in sessions:
        sessions[session_id] = {"history": [], "memory": [], "video_info": {}, "transcript_text": "", "transcript_segments": []}
    for k, v in {"history": [], "memory": [], "video_info": {}, "transcript_text": "", "transcript_segments": []}.items():
        sessions[session_id].setdefault(k, v)
    return sessions


def is_concept_trigger(text: str) -> bool:
    t = (text or "").lower()
    return any(phrase in t for phrase in CONCEPT_TRIGGERS)


# UI
with gr.Blocks(title="YouTube Concept Bot (Task 3)", theme=gr.themes.Soft(primary_hue=gr.themes.colors.indigo)) as demo:
    gr.Markdown("""### YouTube Concept Bot (Task 3)
    - Paste a YouTube URL to load transcript and summary.
    - Ask questions about the video, or type commands like "Show me the concept map" to visualize concepts.
    """)

    sessions_state = gr.State(value={"default": {"history": [], "memory": [], "video_info": {}, "transcript_text": "", "transcript_segments": []}})

    with gr.Row():
        with gr.Column(scale=7):
            # Concept graph area (hidden until generated)
            concept_plot = gr.Plot(label="Concept Map", visible=False)

            chatbot = gr.Chatbot(
                label="",
                height=480,
                bubble_full_width=False,
                show_copy_button=True,
            )
            with gr.Row():
                txt = gr.Textbox(label="Paste URL or ask", placeholder="Ask, paste YouTube URL, or say 'show concept map'...", lines=3, scale=5)
                with gr.Column(scale=1):
                    clear_btn = gr.Button("Clear Current Session", variant="secondary")
                    send_btn = gr.Button("Send", variant="primary")
        with gr.Column(scale=3):
            with gr.Row():
                new_session_btn = gr.Button("+ Add Session", variant="primary")
            session_radio = gr.Radio(choices=["default"], value="default", label="Sessions")

    def on_submit(user_message: str, selected_session: str, sessions: Dict) -> Tuple[str, List[List[str]], Dict, gr.update]:
        user_message = (user_message or "").strip()
        sessions = ensure_session(sessions, selected_session)

        # 1) If sentence contains a YouTube URL -> handle video
        url = find_first_youtube_url(user_message)
        if url:
            txt_val, history, sessions = handle_youtube_flow(url, selected_session, sessions)
            return txt_val, history, sessions, gr.update(visible=False)

        # 2) If concept trigger and transcript exists -> build concept graph
        if is_concept_trigger(user_message):
            transcript_text = sessions[selected_session].get("transcript_text", "")
            if transcript_text:
                fig = build_concept_graph(transcript_text)
                # also drop a small message in chat
                sessions[selected_session]["history"] = sessions[selected_session]["history"] + [[user_message, "Generated a concept map based on the transcript."]]
                return "", sessions[selected_session]["history"], sessions, gr.update(value=fig, visible=True)
            else:
                sessions[selected_session]["history"] = sessions[selected_session]["history"] + [[user_message, "No transcript available yet. Paste a YouTube URL first."]]
                return "", sessions[selected_session]["history"], sessions, gr.update(visible=False)

        # 3) Otherwise -> QA or default assistant
        if not user_message:
            return gr.update(), sessions[selected_session]["history"], sessions, gr.update(visible=False)

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

        return "", sessions[selected_session]["history"], sessions, gr.update(visible=False)

    def on_new_session(sessions: Dict) -> Tuple[gr.Radio, Dict, List[List[str]], gr.update]:
        base = "session-"
        i = 1
        while f"{base}{i}" in sessions:
            i += 1
        new_id = f"{base}{i}"
        sessions[new_id] = {"history": [], "memory": [], "video_info": {}, "transcript_text": "", "transcript_segments": []}
        choices = list(sessions.keys())
        return gr.Radio(choices=choices, value=new_id), sessions, [], gr.update(visible=False)

    def on_clear(selected_session: str, sessions: Dict) -> Tuple[List[List[str]], Dict, gr.update]:
        sessions = ensure_session(sessions, selected_session)
        sessions[selected_session] = {"history": [], "memory": [], "video_info": {}, "transcript_text": "", "transcript_segments": []}
        return [], sessions, gr.update(visible=False)

    def on_switch_session(selected_session: str, sessions: Dict) -> Tuple[List[List[str]], gr.update]:
        sessions = ensure_session(sessions, selected_session)
        return sessions[selected_session]["history"], gr.update(visible=False)

    # Bind events
    txt.submit(
        fn=on_submit,
        inputs=[txt, session_radio, sessions_state],
        outputs=[txt, chatbot, sessions_state, concept_plot],
    )

    send_btn.click(
        fn=on_submit,
        inputs=[txt, session_radio, sessions_state],
        outputs=[txt, chatbot, sessions_state, concept_plot],
    )

    new_session_btn.click(
        fn=on_new_session,
        inputs=[sessions_state],
        outputs=[session_radio, sessions_state, chatbot, concept_plot],
    )

    clear_btn.click(
        fn=on_clear,
        inputs=[session_radio, sessions_state],
        outputs=[chatbot, sessions_state, concept_plot],
    )

    session_radio.change(
        fn=on_switch_session,
        inputs=[session_radio, sessions_state],
        outputs=[chatbot, concept_plot],
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 8000)))
