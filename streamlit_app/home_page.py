# streamlit_app/Home.py

import sys
import os

# Add project root to Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    
import streamlit as st
import time
import json
from typing import List, Dict

# Import your core pipeline functions/modules
from app.core.claim_extractor import extract_text_from_url, extract_claim_from_text, clean_text
from app.core.query_generator import generate_queries
from app.core.web_search import web_search
from app.core.evidence_aggregator import build_evidence
from app.core.verdict_engine import compute_final_verdict

st.set_page_config(
    page_title="Fake News Fact Checker",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- UI helpers ----------
def badge_for_verdict(v: str):
    mapping = {
        "LIKELY TRUE": ("✅ Likely True", "green"),
        "LIKELY FALSE": ("⛔ Likely False", "red"),
        "MIXED / MISLEADING": ("⚠️ Mixed / Misleading", "orange"),
        "UNVERIFIED": ("❓ Unverified", "gray")
    }
    return mapping.get(v, (v, "blue"))

def format_evidence_card(ev: Dict):
    sim = ev.get("similarity", 0)
    stance = ev.get("stance", "discusses")
    stance_score = ev.get("stance_score", 0)
    url = ev.get("url", "")
    sent = ev.get("best_sentence", "") or ev.get("sentence","")
    return f"**Stance:** {stance} ({stance_score:.2f})  \n**Similarity:** {sim:.2f}  \n**Snippet:** {sent[:400]}...  \n**Source:** {url}"

# ---------- Sidebar ----------
st.sidebar.title("Settings")
max_results = st.sidebar.slider("Max search results per query", 1, 10, 6)  # Changed default from 6 to 3
use_cache = st.sidebar.checkbox("Cache search results (faster)", True)
show_raw = st.sidebar.checkbox("Show raw evidence JSON", False)

st.sidebar.markdown("---")
st.sidebar.markdown("Project: Zero-shot Live Fact Checking")
st.sidebar.markdown("Built with SBERT + Zero-shot NLI + Live Web Evidence")

# ---------- Main UI ----------
st.title("📰 Fake News Fact Checker — Live Web Evidence")
st.write("Paste a news URL or the article text below. The app will extract the main claim, search the web for supporting and refuting evidence, and show a final verdict with ranked evidence.")

with st.form("input_form"):
    url_input = st.text_input("Enter Article URL", "")
    text_input = st.text_area("Or paste article text (optional)", height=180)
    submit = st.form_submit_button("Check")

if submit:
    start_time = time.time()
    with st.spinner("Extracting claim..."):
        if url_input.strip():
            full_text = extract_text_from_url(url_input.strip())
            if not full_text:
                # fallback: try showing an error but continue if text provided
                st.warning("Could not extract text from the URL. If you have the article text, paste it in the text box.")
                full_text = text_input
        else:
            full_text = text_input

        if not full_text or len(clean_text(full_text).strip()) < 30:
            st.error("No valid text found. Provide a valid URL or paste article text.")
            st.stop()

        claim = extract_claim_from_text(full_text)
        st.markdown("#### Extracted claim")
        st.info(claim)

    # Generate queries
    with st.spinner("Generating search queries..."):
        queries = generate_queries(claim)
        st.write("**Search queries generated:**")
        st.write(queries[:8])

    # Web search
    with st.spinner("Searching the web for evidence..."):
        search_results = web_search(queries, max_results=max_results)
        if not search_results:
            st.warning("No search results returned for that claim. Try different input.")
        st.write(f"Found {len(search_results)} candidate sources (deduplicated).")

    # Progress bar for evidence building
    evidences = []
    if search_results:
        progress = st.progress(0)
        total = len(search_results)
        for idx, item in enumerate(search_results):
            try:
                # build_evidence expects list of search results and handles scraping + embedding + NLI
                # For speed, pass batches or slice; here we do one-by-one using build_evidence convenience wrapper
                # We'll reuse the aggregator in a per-item fashion to get consistent structure
                sub_evs = build_evidence(claim, [item])
                if sub_evs:
                    evidences.extend(sub_evs)
            except Exception as e:
                # Log error but continue processing other sources
                st.warning(f"⚠️ Failed to process {item.get('href', 'unknown URL')}: {str(e)[:80]}")
            finally:
                # Always update progress
                progress.progress(min(int(((idx+1)/total)*100), 100))
        progress.empty()

    # Aggregate & Verdict
    with st.spinner("Aggregating evidence and computing verdict..."):
        verdict_result = compute_final_verdict(evidences)

    # Show verdict UI
    verdict_label, color = badge_for_verdict(verdict_result["verdict"])
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"### **Final Verdict:** <span style='color:{color};font-weight:600'>{verdict_label}</span>", unsafe_allow_html=True)
        st.write(f"**Confidence:** {verdict_result['confidence']*100:.1f}%")
        st.write(f"**Net score:** {verdict_result['net_score']}")
        st.write(f"**Sources analyzed:** {len(evidences)}")
    with col2:
        # visual confidence bar
        st.metric("Confidence", f"{verdict_result['confidence']*100:.1f}%")
        # simple gauge
        st.progress(min(int(verdict_result['confidence']*100), 100))

    st.markdown("---")

    # Split evidence into supporting / refuting / neutral
    supports = [e for e in evidences if e.get("stance") == "supports"]
    refutes = [e for e in evidences if e.get("stance") == "refutes"]
    neutral = [e for e in evidences if e.get("stance") not in ("supports", "refutes")]

    # Show top evidence cards side-by-side
    st.subheader("Top Evidence")
    s_col, r_col = st.columns(2)
    with s_col:
        st.markdown("#### Supporting evidence")
        if supports:
            for ev in sorted(supports, key=lambda x: (x.get("similarity",0)*x.get("stance_score",0)), reverse=True)[:6]:
                with st.expander(ev.get("url", "source")):
                    st.markdown(format_evidence_card(ev))
                    st.markdown(f"[Open source]({ev.get('url')})")
        else:
            st.info("No strong supporting evidence found.")

    with r_col:
        st.markdown("#### Refuting evidence")
        if refutes:
            for ev in sorted(refutes, key=lambda x: (x.get("similarity",0)*x.get("stance_score",0)), reverse=True)[:6]:
                with st.expander(ev.get("url", "source")):
                    st.markdown(format_evidence_card(ev))
                    st.markdown(f"[Open source]({ev.get('url')})")
        else:
            st.info("No strong refuting evidence found.")

    st.markdown("---")
    st.subheader("Other / Neutral Evidence")
    for ev in neutral[:8]:
        with st.expander(ev.get("url", "source")):
            st.markdown(format_evidence_card(ev))
            st.markdown(f"[Open source]({ev.get('url')})")

    # Raw JSON / Download
    result_obj = {
        "claim": claim,
        "queries": queries,
        "verdict": verdict_result,
        "evidences": evidences
    }

    st.markdown("---")
    st.download_button(
        "📥 Download report (JSON)",
        data=json.dumps(result_obj, indent=2),
        file_name="factcheck_report.json",
        mime="application/json"
    )

    if show_raw:
        st.subheader("Raw evidence JSON")
        st.json(result_obj)

    st.write(f"Completed in {(time.time()-start_time):.1f}s")
