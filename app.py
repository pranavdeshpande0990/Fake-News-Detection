import streamlit as st
import joblib
import re

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

st.markdown("""
<style>
  .result-box {
      padding: 1.2rem 1.5rem;
      border-radius: 10px;
      font-size: 1.1rem;
      font-weight: 600;
      text-align: center;
  }
  .fake { background-color: #fde8e8; color: #c0392b; border: 1.5px solid #c0392b; }
  .real { background-color: #e8f8f0; color: #1e8449; border: 1.5px solid #1e8449; }
</style>
""", unsafe_allow_html=True)

STOP_WORDS = set("""
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further had hadn't has hasn't have haven't having he he'd he'll he's
her here here's hers herself him himself his how how's i i'd i'll i'm i've
if in into is isn't it it's its itself let's me more most mustn't my myself
no nor not of off on once only or other ought our ours ourselves out over
own same shan't she she'd she'll she's should shouldn't so some such than
that that's the their theirs them themselves then there there's these they
they'd they'll they're they've this those through to too under until up
very was wasn't we we'd we'll we're we've were weren't what what's when
when's where where's which while who who's whom why why's with won't would
wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())

@st.cache_resource
def load_model():
    model      = joblib.load("classifier.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(words)

def predict(text):
    cleaned    = clean_text(text)
    vector     = vectorizer.transform([cleaned])
    label      = model.predict(vector)[0]
    proba      = model.predict_proba(vector)[0]
    real_conf  = round(proba[1] * 100, 1)
    fake_conf  = round(proba[0] * 100, 1)
    return label, real_conf, fake_conf

st.title("📰 Fake News Detector")
st.caption("Enter a news article or headline to check if it is real or fake.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    if st.button("Load Fake Sample", use_container_width=True):
        st.session_state['input_text'] = (
            "SHOCKING: Secret government plan exposed by whistleblower! "
            "They have been hiding this for years. Share before it gets deleted!!"
        )

with col2:
    if st.button("Load Real Sample", use_container_width=True):
        st.session_state['input_text'] = (
            "The Federal Reserve kept interest rates unchanged at its latest meeting, "
            "citing steady inflation data and a resilient job market."
        )

news_text = st.text_area(
    label="News Article or Headline",
    value=st.session_state.get('input_text', ''),
    height=200,
    placeholder="Paste article text, headline, or social media post here...",
    key="input_text"
)

if st.button("Analyze", type="primary", use_container_width=True):

    if not news_text.strip():
        st.warning("Please enter some text before analyzing.")

    else:
        label, real_conf, fake_conf = predict(news_text)

        st.divider()

        if label == 1:
            st.markdown(
                f'<div class="result-box real">✅ REAL NEWS &nbsp;|&nbsp; Confidence: {real_conf}%</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-box fake">🚫 FAKE NEWS &nbsp;|&nbsp; Confidence: {fake_conf}%</div>',
                unsafe_allow_html=True
            )

        st.markdown("")
        st.markdown("**Probability Breakdown**")

        col_r, col_f = st.columns(2)
        col_r.metric("Real News", f"{real_conf}%")
        col_f.metric("Fake News", f"{fake_conf}%")

        st.progress(real_conf / 100, text=f"Real — {real_conf}%")
        st.progress(fake_conf / 100, text=f"Fake — {fake_conf}%")

        with st.expander("Text Statistics"):
            words = news_text.split()
            st.write(f"- **Word count:** {len(words)}")
            st.write(f"- **Exclamation marks:** {news_text.count('!')}")
            st.write(f"- **Question marks:** {news_text.count('?')}")
            caps = sum(1 for c in news_text if c.isupper())
            st.write(f"- **CAPS ratio:** {caps / max(len(news_text), 1) * 100:.1f}%")

with st.sidebar:
    st.markdown("## About")
    st.info(
        "This app uses a **Logistic Regression** model trained on the "
        "ISOT Fake News Dataset (~45,000 articles). Text is converted "
        "to TF-IDF features before classification."
    )
    st.markdown("**Model Details**")
    st.markdown("- Algorithm: Logistic Regression")
    st.markdown("- Features: TF-IDF (1–2 grams, 10k vocab)")
    st.markdown("- Accuracy: ~98%")
    st.markdown("- Dataset: ISOT (Reuters + PolitiFact)")
    st.divider()
    st.caption("Final Year Project")
