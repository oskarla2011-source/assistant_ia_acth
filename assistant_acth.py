from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image
from google import genai
from sentence_transformers import SentenceTransformer, util

ROOT = Path(__file__).resolve().parent
KB_PATH = ROOT / "base_connaissances_acth.json"
LOGO_PATH = ROOT / "logo_acth.png"

st.set_page_config(
    page_title="ACTH | Assistant conseil & formation",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🏗️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Palette sobre inspirée d'un univers architecture / construction / numérique.
NAVY = "#183746"
INK = "#1D2C35"
TEAL = "#008491"
MUTED = "#61727B"
PALE = "#F3F7F8"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root {{ --acth-navy:{NAVY}; --acth-ink:{INK}; --acth-teal:{TEAL}; --acth-muted:{MUTED}; --acth-pale:{PALE}; }}
html, body, [data-testid="stAppViewContainer"] {{ background:#fff !important; color:var(--acth-ink); font-family:'Manrope','DM Sans',sans-serif; }}
[data-testid="stHeader"] {{ background:transparent; }}
#MainMenu, footer {{ visibility:hidden; }}
.block-container {{ max-width:940px; padding-top:1.7rem; padding-bottom:3rem; }}
.acth-kicker {{ color:var(--acth-teal); font-size:.74rem; font-weight:800; letter-spacing:2.3px; text-transform:uppercase; margin:0 0 13px; }}
.acth-title {{ color:var(--acth-navy); font-size:clamp(2.1rem,5vw,3.35rem); font-weight:800; line-height:1.08; letter-spacing:-.045em; margin:0 0 13px; }}
.acth-title em {{ color:var(--acth-teal); font-style:normal; font-weight:600; }}
.acth-subtitle {{ color:var(--acth-muted); font-size:1rem; line-height:1.75; margin:0 0 19px; max-width:800px; }}
.acth-trust {{ border:1px solid #D6E6E8; border-left:4px solid var(--acth-teal); background:#F7FAFA; border-radius:12px; padding:13px 16px; color:#314D58; font-size:.88rem; line-height:1.65; margin:18px 0 20px; }}
[data-testid="stChatMessage"] {{ border-radius:16px !important; padding:13px 16px !important; margin-bottom:12px !important; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {{ background:#FAFCFC !important; border:1px solid #E0EAEC !important; border-left:3px solid var(--acth-teal) !important; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{ background:#F1F5F6 !important; border:1px solid #E0E8EA !important; }}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {{ color:var(--acth-ink) !important; line-height:1.72; }}
.faq-label {{ color:var(--acth-navy); font-size:.73rem; font-weight:800; letter-spacing:1.8px; text-transform:uppercase; margin:20px 0 10px; }}
.stButton > button {{ border:1px solid #D9E6E8 !important; background:white !important; color:var(--acth-navy) !important; border-radius:12px !important; min-height:47px; white-space:normal !important; font-size:.86rem !important; transition:all .16s ease; }}
.stButton > button:hover {{ border-color:var(--acth-teal) !important; background:#F2FAFA !important; color:var(--acth-navy) !important; }}
[data-testid="stChatInput"] {{ border:1.5px solid #D6E3E6 !important; border-radius:18px !important; background:#fff !important; box-shadow:0 5px 20px rgba(24,55,70,.08) !important; padding:5px 7px 5px 11px !important; outline:0 !important; transition:border-color .2s ease,box-shadow .2s ease !important; }}
[data-testid="stChatInput"]:focus-within {{ border-color:var(--acth-teal) !important; outline:0 !important; box-shadow:0 6px 22px rgba(0,132,145,.13) !important; }}
[data-testid="stChatInput"] [data-baseweb], [data-testid="stChatInput"] [data-baseweb="base-input"], [data-testid="stChatInput"] [data-baseweb="textarea"], [data-testid="stChatInput"] [aria-invalid="true"] {{ border-color:transparent !important; outline:0 !important; box-shadow:none !important; }}
[data-testid="stChatInput"] textarea, [data-testid="stChatInput"] textarea:focus {{ background:#fff !important; color:#1D2C35 !important; -webkit-text-fill-color:#1D2C35 !important; caret-color:#008491 !important; color-scheme:light !important; opacity:1 !important; font-family:'Manrope','DM Sans',sans-serif !important; font-size:.96rem !important; line-height:1.5 !important; padding:12px 9px !important; min-height:48px !important; border:0 !important; box-shadow:none !important; outline:none !important; }}
[data-testid="stChatInput"] [data-baseweb="textarea"] {{ border:0 !important; box-shadow:none !important; outline:0 !important; background:#fff !important; }}
[data-testid="stChatInput"] textarea::placeholder {{ color:#7A898F !important; -webkit-text-fill-color:#7A898F !important; opacity:1 !important; }}
[data-testid="stChatInput"] button {{ background:var(--acth-navy) !important; color:white !important; border:0 !important; border-radius:12px !important; min-width:42px !important; min-height:42px !important; box-shadow:0 3px 9px rgba(24,55,70,.18) !important; }}
[data-testid="stChatInput"] button:hover {{ background:var(--acth-teal) !important; }}
[data-testid="stChatInput"] button svg {{ color:white !important; fill:white !important; }}
.acth-thinking {{ display:flex; align-items:center; gap:11px; padding:7px 2px 10px; color:#526872; font-size:.88rem; }}
.acth-dots {{ display:inline-flex; align-items:center; gap:5px; min-width:30px; }}
.acth-dot {{ width:7px; height:7px; display:block; border-radius:50%; background:var(--acth-teal); opacity:.25; transform:translateY(0) scale(.78); animation:acth-pulse 1.05s infinite ease-in-out; }}
.acth-dot:nth-child(2) {{ animation-delay:.17s; }} .acth-dot:nth-child(3) {{ animation-delay:.34s; }}
@keyframes acth-pulse {{ 0%,60%,100% {{ opacity:.25; transform:translateY(0) scale(.78); }} 30% {{ opacity:1; transform:translateY(-3px) scale(1); }} }}
@media (prefers-reduced-motion:reduce) {{ .acth-dot {{ animation:none; opacity:.8; }} }}
.acth-footer {{ border-top:1px solid #E2EAEC; margin-top:26px; padding-top:15px; color:#64757D; font-size:.78rem; line-height:1.7; }}
@media(max-width:640px) {{ .block-container {{ padding-left:1rem; padding-right:1rem; }} }}
</style>
""", unsafe_allow_html=True)

try:
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
except Exception:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("La clé API Google est introuvable. Ajoutez GOOGLE_API_KEY dans les Secrets de Streamlit Cloud.")
    st.stop()

client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

@st.cache_resource(show_spinner="Préparation de l’assistant ACTH…")
def charger_modele():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

modele = charger_modele()

try:
    KNOWLEDGE_CHUNKS: list[dict[str, Any]] = json.loads(KB_PATH.read_text(encoding="utf-8"))
except Exception as exc:
    st.error(f"Impossible de charger la base de connaissances ACTH ({exc}). Vérifiez que base_connaissances_acth.json est dans le même dépôt que ce fichier.")
    st.stop()

try:
    LOGO_AVATAR = Image.open(LOGO_PATH).convert("RGBA") if LOGO_PATH.exists() else "🏗️"
except Exception:
    LOGO_AVATAR = "🏗️"

ACTH_SITE = "https://acth.fr/"
ACTH_FAQ = "https://acth.fr/foire-aux-questions"
ACTH_FORMATIONS = "https://acth.fr/nos-formations"
ACTH_FINANCEMENT = "https://acth.fr/financement"
ACTH_CONTACT = "https://acth.fr/contact"

SYSTEM_PROMPT = """Tu es l'assistant d'information et d'orientation de ACTH Conseil & Formation, spécialisé dans le BIM et la transition numérique des acteurs de l'architecture, de l'ingénierie et de la construction.

OBJECTIF
Aider le visiteur à comprendre la FAQ ACTH, les métiers BIM, les domaines de formation, les modalités et les contacts. Sois serviable et concret ; lorsque la personne cherche une formation, demande son objectif métier ou son niveau si cela aide vraiment à l'orienter.

RÈGLES DE FIABILITÉ
- Réponds en français et fonde les faits sur le CONTEXTE ACTH fourni ci-dessous. N'invente jamais une formation, un prix, une durée, une session, un prérequis ou une garantie de financement.
- Les dix questions-réponses issues de la FAQ officielle ont priorité. Pour une question correspondante, conserve le sens et les conditions de la réponse officielle ; ne la contredis pas.
- En matière de financement, dis qu'un OPCO/FAF peut prendre en charge tout ou partie selon les conditions. Ne promets jamais que le dossier sera accepté, ni une prise en charge à 100 % de manière générale.
- Si une information précise n'est pas dans le contexte (tarif, session, prérequis, éligibilité d'un dossier), dis-le et dirige vers la fiche officielle ou ACTH.
- Les horaires habituels visibles dans la FAQ sont 9 h–17 h, avec pause déjeuner 13 h–14 h et deux pauses de 15 minutes à 11 h et 15 h ; précise qu'il s'agit de l'information FAQ, et invite à confirmer les modalités du programme si nécessaire.
- Pour un tarif, n'utilise l'exemple de 1 500 € HT / 1 800 € TTC que pour la formation « Concevoir et piloter un projet BIM en construction » en inter-entreprises. L'intra est sur devis.
- N'affirme pas avoir contacté ACTH, réservé une place ou vérifié des disponibilités en temps réel.
- Coordonnées : 01 43 26 23 21, info@acth.fr, 15 rue Lucien Sampaix, 75010 Paris.
- Liens utiles : FAQ https://acth.fr/foire-aux-questions ; financement https://acth.fr/financement ; formations https://acth.fr/nos-formations ; contact https://acth.fr/contact .

STYLE
- Ton professionnel, accueillant, naturel, sans excès commercial.
- Réponds à la question dès la première phrase. Utilise des paragraphes courts et des puces seulement si elles clarifient la réponse.
- Quand c'est utile, termine par une question simple d'orientation. Ne termine pas systématiquement chaque message par une question.
"""

FAQ_SUGGESTIONS = [
    "Ma formation peut-elle être financée ?",
    "Quel est le rôle d’un BIM coordinateur ?",
    "Où et à quels horaires se déroulent les formations ?",
    "Quelles formations ACTH propose-t-elle en BIM ?",
]


def normaliser(texte: str) -> str:
    texte = unicodedata.normalize("NFKD", texte.lower())
    texte = "".join(ch for ch in texte if not unicodedata.combining(ch))
    texte = re.sub(r"[^a-z0-9]+", " ", texte).strip()
    return re.sub(r"\s+", " ", texte)


def contenu_chunk(chunk: dict[str, Any]) -> str:
    parts = [str(chunk.get("question", "")), str(chunk.get("answer", "")), str(chunk.get("text", ""))]
    parts.extend(str(x) for x in chunk.get("aliases", []))
    return "\n".join(p for p in parts if p.strip())

FAQ_CHUNKS = [c for c in KNOWLEDGE_CHUNKS if c.get("type") == "faq"]
CHUNK_TEXTS = tuple(contenu_chunk(c) for c in KNOWLEDGE_CHUNKS)

@st.cache_data(show_spinner=False)
def calculer_embeddings(textes: tuple[str, ...]):
    return modele.encode(list(textes), convert_to_tensor=True)

CHUNK_EMBEDDINGS = calculer_embeddings(CHUNK_TEXTS)

@st.cache_data(show_spinner=False)
def calculer_embeddings_questions(questions: tuple[str, ...]):
    return modele.encode(list(questions), convert_to_tensor=True)

FAQ_ALIASES = []
for faq in FAQ_CHUNKS:
    for candidate in [faq.get("question", ""), *faq.get("aliases", [])]:
        FAQ_ALIASES.append((normaliser(str(candidate)), faq))
FAQ_Q_EMBED_TEXTS = tuple(f"{c.get('question','')} {' '.join(c.get('aliases', []))}" for c in FAQ_CHUNKS)
FAQ_Q_EMBEDDINGS = calculer_embeddings_questions(FAQ_Q_EMBED_TEXTS)


def trouver_faq_directe(question: str) -> str | None:
    """Retourne la réponse de la FAQ telle quelle pour une question exacte ou presque exacte."""
    q = normaliser(question)
    if not q:
        return None
    for candidate, faq in FAQ_ALIASES:
        if q == candidate or SequenceMatcher(None, q, candidate).ratio() >= 0.94:
            return str(faq.get("answer", "")).strip() or None
    # La proximité sémantique aide à retrouver les FAQ si la formulation est très proche,
    # mais on évite de déclencher une FAQ sur une simple similarité faible.
    try:
        q_emb = modele.encode(question, convert_to_tensor=True)
        sims = util.cos_sim(q_emb, FAQ_Q_EMBEDDINGS)[0].tolist()
        best_i = max(range(len(sims)), key=lambda i: sims[i])
        best = FAQ_CHUNKS[best_i]
        q_terms = set(q.split())
        ref_terms = set(normaliser(str(best.get("question", ""))).split())
        overlap = len(q_terms & ref_terms) / max(1, len(q_terms))
        if sims[best_i] >= 0.86 and overlap >= 0.48:
            return str(best.get("answer", "")).strip() or None
    except Exception:
        pass
    return None


def construire_requete_recherche(question: str, messages: list[dict[str, str]]) -> str:
    mots = question.strip().split()
    courte = len(mots) <= 7 or normaliser(question).startswith(("et ", "sinon", "dans ce cas", "laquelle", "lequel", "combien", "et pour", "et si", "oui", "non", "plutot", "a distance", "en presentiel"))
    if not messages or not courte:
        return question.strip()
    recent = messages[-4:]
    contexte = [f"{('Visiteur' if m.get('role') == 'user' else 'Assistant ACTH')}: {str(m.get('content',''))[:450]}" for m in recent if m.get("content")]
    return "\n".join(contexte + [f"Question actuelle : {question.strip()}"])


def selectionner_contexte(question: str, messages: list[dict[str, str]] | None = None, max_chunks: int = 5) -> str:
    q = normaliser(question)
    query = construire_requete_recherche(question, messages or [])
    q_emb = modele.encode(query, convert_to_tensor=True)
    semantic_scores = util.cos_sim(q_emb, CHUNK_EMBEDDINGS)[0].tolist()
    terms = set(q.split())
    scored = []
    for i, chunk in enumerate(KNOWLEDGE_CHUNKS):
        kws = [normaliser(str(k)) for k in chunk.get("keywords", [])]
        keyword_score = sum(1 for kw in kws if kw and (kw in q or any(term in kw for term in terms if len(term) > 2)))
        question_norm = normaliser(str(chunk.get("question", "")))
        question_bonus = 0.16 if question_norm and (question_norm in q or q in question_norm) else 0.0
        # Score lexical borné afin qu'une longue liste de mots-clés ne monopolise pas le contexte.
        lexical = min(keyword_score / 4.0, 1.0)
        score = 0.72 * float(semantic_scores[i]) + 0.28 * lexical + question_bonus
        scored.append((score, chunk))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    selected = [c for _, c in scored[:max_chunks]]
    return "\n\n---\n\n".join(
        f"[{c.get('type','info').upper()} — {c.get('question','')}]\n{c.get('answer','') or c.get('text','')}\nSource : {c.get('source','Site officiel ACTH') }"
        for c in selected
    )


def generer_reponse(question: str, messages: list[dict[str, str]]) -> str:
    direct = trouver_faq_directe(question)
    if direct:
        return direct
    contexte = selectionner_contexte(question, messages=messages)
    historique = []
    for msg in messages[-8:]:
        role = "Visiteur" if msg.get("role") == "user" else "Assistant ACTH"
        historique.append(f"{role} : {str(msg.get('content',''))[:900]}")
    prompt = (
        f"{SYSTEM_PROMPT}\n\nCONTEXTE ACTH RETROUVÉ PAR LE MOTEUR RAG :\n{contexte}\n\n"
        "HISTORIQUE RÉCENT :\n" + "\n".join(historique) +
        f"\n\nQUESTION : {question}\nRéponse de l'assistant ACTH :"
    )
    last_error = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            answer = getattr(response, "text", None)
            if answer and answer.strip():
                return answer.strip()
            return "Je ne trouve pas de réponse suffisamment fiable dans les informations disponibles. Vous pouvez contacter ACTH au 01 43 26 23 21 ou à info@acth.fr."
        except Exception as exc:
            last_error = exc
            err = str(exc).lower()
            if attempt < 2 and ("429" in err or "503" in err or "tempor" in err or "unavailable" in err):
                time.sleep(1.4 ** attempt)
                continue
            break
    return "Le service de réponse est momentanément indisponible. Vous pouvez contacter ACTH au 01 43 26 23 21 ou à info@acth.fr."


st.markdown('<div class="acth-kicker">ACTH · CONSEIL & FORMATION</div>', unsafe_allow_html=True)
st.markdown('<h1 class="acth-title">Vos questions sur le BIM.<br><em>Nos réponses, simplement.</em></h1>', unsafe_allow_html=True)
st.markdown('<div class="acth-subtitle">Un assistant pour découvrir les formations ACTH, comprendre les métiers du BIM et obtenir les premières informations pratiques sur les modalités et le financement.</div>', unsafe_allow_html=True)
st.markdown('<div class="acth-trust">✦ <b>Conseil, formation et accompagnement de la transition numérique</b> · Réponses fondées sur la FAQ et les informations publiques d’ACTH. Pour confirmer un tarif, une session ou une prise en charge, l’équipe ACTH reste votre interlocuteur.</div>', unsafe_allow_html=True)

col_a, col_b = st.columns([5, 1])
with col_a:
    st.markdown('<div class="faq-label">Questions fréquentes</div>', unsafe_allow_html=True)
with col_b:
    if st.button("Effacer", key="clear_chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if not st.session_state.messages:
    cols = st.columns(2)
    for i, faq in enumerate(FAQ_SUGGESTIONS):
        with cols[i % 2]:
            if st.button(faq, key=f"faq_{i}", use_container_width=True):
                st.session_state.pending_question = faq
                st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=LOGO_AVATAR if message["role"] == "assistant" else "👤"):
        st.markdown(message["content"])

question = st.session_state.pop("pending_question", None)
user_input = st.chat_input("Posez votre question sur le BIM ou nos formations…")
if user_input:
    question = user_input

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)
    with st.chat_message("assistant", avatar=LOGO_AVATAR):
        loading = st.empty()
        loading.markdown(
            '<div class="acth-thinking" role="status" aria-live="polite">'
            '<span class="acth-dots"><span class="acth-dot"></span><span class="acth-dot"></span><span class="acth-dot"></span></span>'
            '<span>Je recherche les informations ACTH…</span></div>',
            unsafe_allow_html=True,
        )
        answer = generer_reponse(question, st.session_state.messages[:-1])
        loading.empty()
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

st.markdown(
    '<div class="acth-footer">Cet assistant fournit des informations générales à partir des éléments publics ACTH et ne remplace pas la confirmation par un conseiller. '
    'Pour vérifier un programme, un tarif, une session ou votre financement : <b>01 43 26 23 21</b> · '
    '<a href="mailto:info@acth.fr">info@acth.fr</a> · '
    '<a href="https://acth.fr/nos-formations" target="_blank">Formations</a> · '
    '<a href="https://acth.fr/foire-aux-questions" target="_blank">FAQ officielle</a>.</div>',
    unsafe_allow_html=True,
)
