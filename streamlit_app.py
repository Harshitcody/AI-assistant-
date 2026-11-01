""" AI Pitch Assistant - Streamlit version (cloud-friendly).

The 33B model is NOT loaded inside Streamlit. Text is generated through the
Hugging Face Inference API (needs an HF token in Streamlit secrets), so the app
runs on a normal CPU host such as Streamlit Community Cloud.
"""

import json
import re
from pathlib import Path

import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Lucror AI Pitch Assistant", page_icon="💡", layout="wide")

PRODUCTS = json.loads((Path(__file__).parent / "products.json").read_text(encoding="utf-8"))
DEFAULT_MODEL = "apathy-exe/Nanbeige4.3-33B"

SYSTEM_PROMPT = (
    "You are a senior AI consultant at , a consulting company that brings the world's "
    "best AI products to large enterprises in the Middle East and India. Your guiding rule: "
    "technology only matters if it helps the client 'Increase Sales or Reduce Cost'. "
    "Be concrete, professional, concise and results-focused. Never invent client names or "
    "exact statistics; use ranges and say they are estimates to be validated in a pilot."
)


# ---------- helpers ----------
def words(text):
    return set(re.findall(r"[a-z]+", text.lower()))


def match_products(industry, problem, top_k=2):
    query = words(industry + " " + problem)
    scored = []
    for p in PRODUCTS:
        corpus = words(" ".join(p["best_for"]) + " " + p["benefit"] + " " + p["name"])
        scored.append((len(query & corpus), p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_k]]


@st.cache_resource
def get_client(model_id, token):
    return InferenceClient(model=model_id, token=token)


def ask_model(messages, max_tokens=800, temperature=0.7):
    token = st.secrets.get("HF_TOKEN", "")
    model_id = st.session_state.get("model_id", DEFAULT_MODEL)
    if st.session_state.get("demo_mode") or not token:
        return ("**Demo mode** (no HF token set or demo switched on). "
                "Add `HF_TOKEN` in Streamlit secrets to get real AI output.\n\n"
                + messages[-1]["content"][:500])
    try:
        client = get_client(model_id, token)
        resp = client.chat_completion(messages=messages, max_tokens=max_tokens,
                                      temperature=temperature)
        return resp.choices[0].message.content
    except Exception as e:  # show a readable error instead of crashing
        return (f"Model call failed: `{e}`\n\n"
                "The model may not be available on the Inference API. Try another model ID "
                "in the sidebar, or point to your own Inference Endpoint.")


# ---------- sidebar ----------
with st.sidebar:
    st.header("Settings")
    st.session_state["model_id"] = st.text_input("Model ID", DEFAULT_MODEL)
    st.session_state["demo_mode"] = st.toggle("Demo mode (no model calls)", value=False)
    st.caption("Token is read from `st.secrets['HF_TOKEN']`.")

st.title("💡  AI Pitch Assistant")
st.write("Match client problems to AI solutions that **Increase Sales or Reduce Cost**.")

tab1, tab2, tab3 = st.tabs(["Pitch Generator", "ROI Calculator", "Ask the Consultant"])

# ---------- tab 1 ----------
with tab1:
    c1, c2 = st.columns([1, 1])
    with c1:
        company = st.text_input("Client company (optional)")
        industry = st.text_input("Industry", placeholder="e.g. Retail, Banking, Telecom")
        problem = st.text_area("Main problem", placeholder="e.g. Call centre costs are rising and wait times are long")
        tone = st.selectbox("Tone", ["Professional", "Friendly", "Executive / concise"])
        go = st.button("Generate pitch", type="primary")
    if go:
        if not industry.strip() or not problem.strip():
            st.warning("Please enter the industry and the main problem.")
        else:
            products = match_products(industry, problem)
            with c2:
                st.subheader("Matched products")
                for p in products:
                    st.markdown(f"**{p['name']}** → _{p['goal']}_  \n{p['benefit']}")
            info = "\n".join(f"- {p['name']} ({p['goal']}): {p['benefit']} Typical impact: {p['typical_metrics']}"
                             for p in products)
            prompt = f"""Client company: {company or 'the client'}
Industry: {industry}
Main business problem: {problem}
Tone: {tone}

Candidate AI products from our portfolio:
{info}

Write these sections with clear headings:
1. Recommended AI solution (best fit, why, in 3 lines)
2. 60-second pitch (spoken style)
3. Expected business impact (ranges, flagged as estimates, tied to Increase Sales or Reduce Cost)
4. Top 3 likely objections with short replies
5. Cold email (subject + under 120 words)
6. Suggested 60-day pilot plan with KPIs"""
            with st.spinner("Writing the pitch..."):
                out = ask_model([{"role": "system", "content": SYSTEM_PROMPT},
                                 {"role": "user", "content": prompt}])
            st.divider()
            st.markdown(out)
            st.download_button("Download pitch (.md)", out, file_name="pitch.md")

# ---------- tab 2 ----------
with tab2:
    a, b = st.columns(2)
    calls = a.number_input("Monthly calls / tickets", min_value=0.0, value=50000.0, step=1000.0)
    cpc = a.number_input("Cost per call / ticket", min_value=0.0, value=2.5, step=0.1)
    auto = b.slider("% automated by AI", 0, 100, 40)
    sol = b.number_input("Monthly cost of AI solution", min_value=0.0, value=20000.0, step=1000.0)

    current = calls * cpc
    saved = current * auto / 100
    net = saved - sol
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current monthly cost", f"{current:,.0f}")
    m2.metric("Cost removed", f"{saved:,.0f}")
    m3.metric("Net monthly saving", f"{net:,.0f}")
    m4.metric("Annual net saving", f"{net * 12:,.0f}")
    st.bar_chart({"Amount": {"Current cost": current, "After AI": current - saved + sol}})

    if st.button("Explain for an executive"):
        summary = (f"Current monthly cost {current:,.0f}; automated {auto}%; saving {saved:,.0f}; "
                   f"solution cost {sol:,.0f}; net monthly {net:,.0f}; annual {net * 12:,.0f}.")
        with st.spinner("Thinking..."):
            st.markdown(ask_model([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Explain this ROI to a non-technical executive in 4 short bullets, "
                                            "noting these are estimates to validate in a pilot: " + summary}],
                max_tokens=300))

# ---------- tab 3 ----------
with tab3:
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    if msg := st.chat_input("Ask about AI products, pitching, objections..."):
        st.session_state.chat.append({"role": "user", "content": msg})
        with st.chat_message("user"):
            st.markdown(msg)
        with st.chat_message("assistant"):
            with st.spinner("..."):
                reply = ask_model([{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.chat,
                                  max_tokens=500)
            st.markdown(reply)
        st.session_state.chat.append({"role": "assistant", "content": reply})
