"""
Starter Streamlit app: AI-Powered Resume Analyzer & Job Fit Scorer
File: app.py

Features included in this starter:
- Upload PDF/DOCX resumes and paste a Job Description
- Extract text from PDF (PyMuPDF) and DOCX (docx2txt)
- Simple section extraction for Skills / Experience / Education
- Keyword extraction from JD using TF-IDF
- Skill matching: keyword overlap + semantic similarity (SentenceTransformers if installed)
- Simple ATS checks (presence of images/tables, contact info, length)
- Placeholder for LLM-based summary/suggestions (to plug-in LLaMA/GPT API later)

How to run:
1. Create a virtualenv and install dependencies (see requirements in README or below)
2. streamlit run app.py

"""

import streamlit as st
from io import BytesIO
import re
import tempfile
import os

# Text extraction
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import docx2txt
except Exception:
    docx2txt = None

# NLP & ML
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional: sentence-transformers for better semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception:
    sbert_model = None

nltk.download('punkt', quiet=True)

# ---------------------- Helper functions ----------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF (fitz)."""
    if fitz is None:
        return ""  # user should install PyMuPDF
    text = []
    with fitz.open(stream=file_bytes, filetype='pdf') as doc:
        for page in doc:
            text.append(page.get_text())
    return "\n".join(text)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX bytes using docx2txt."""
    if docx2txt is None:
        return ""
    # docx2txt expects a file path; write to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        text = docx2txt.process(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return text or ""


def simple_skill_extraction(text: str, common_skills=None) -> set:
    """Try to extract skills via a simple heuristics approach: look for a 'Skills' section or match against a list.
    Returns a set of lowercase skill tokens.
    """
    if common_skills is None:
        # Minimal starter set — expand for your domain
        common_skills = [
            'python','java','c++','sql','excel','machine learning','deep learning','tensorflow',
            'pytorch','nlp','docker','kubernetes','aws','azure','git','html','css','javascript',
            'react','node','spring','django','flask','rest','linux','ci/cd','tableau','power bi'
        ]

    text_lower = text.lower()

    # 1) Look for a dedicated Skills section
    m = re.search(r'(skills|technical skills|core skills)[:\n\r]+(.{10,400})', text_lower, re.I | re.S)
    skills_found = set()
    if m:
        block = m.group(2)
        # split by commas / newlines / bullets
        tokens = re.split(r'[\n,•\-|;]+', block)
        for t in tokens:
            t = t.strip()
            if len(t) < 1:
                continue
            # keep words of reasonable length
            if len(t) > 1 and len(t) < 60:
                skills_found.add(re.sub(r'[^a-z0-9+#+. ]','', t))

    # 2) Match against known skills list
    for s in common_skills:
        if s in text_lower:
            skills_found.add(s)

    # normalize
    skills_norm = set(s.strip().lower() for s in skills_found if s.strip())
    return skills_norm


def extract_contact_info(text: str) -> dict:
    """Extract simple contact info (email, phone) using regex."""
    res = {}
    emails = re.findall(r'[\w\.-]+@[\w\.-]+', text)
    phones = re.findall(r'\+?\d[\d\s\-()]{6,}\d', text)
    res['emails'] = list(dict.fromkeys(emails))
    res['phones'] = list(dict.fromkeys(phones))
    return res


def extract_top_keywords(text: str, n=10) -> list:
    """Return top n keywords from text using TF-IDF (single document)."""
    # As a simple trick, treat text as a single document in a tiny corpus of itself — use TfidfVectorizer's analyzer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
    try:
        tfidf = vectorizer.fit_transform([text])
        feature_array = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf.toarray()[0]
        top_n_idx = tfidf_scores.argsort()[::-1][:n]
        return [feature_array[i] for i in top_n_idx if tfidf_scores[i] > 0]
    except Exception:
        return []


def semantic_similarity(a: str, b: str) -> float:
    """Compute semantic similarity between two texts. Uses SBERT if available, else fall back to TF-IDF cosine.
    Returns a float in [0,1].
    """
    if sbert_model is not None:
        try:
            emb = sbert_model.encode([a, b])
            sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
            return float(sim)
        except Exception:
            pass
    # fallback TF-IDF over the two texts
    try:
        vec = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
        X = vec.fit_transform([a, b])
        sim = cosine_similarity(X[0], X[1])[0][0]
        return float(sim)
    except Exception:
        return 0.0


def ats_check(resume_bytes: bytes, resume_text: str) -> dict:
    """Very simple ATS heuristics:
    - Warn if resume contains images (binary PK content for docx or /Image in PDF)
    - Check for presence of contact info
    - Check excessive use of tables/columns by searching for many consistent multiple spaces
    - Return a dict with a score and messages
    """
    score = 100
    messages = []

    # images check (PDF)
    try:
        if resume_bytes and b'/Image' in resume_bytes[:1000000]:
            score -= 20
            messages.append('Resume contains embedded images which may break ATS parsing.')
    except Exception:
        pass

    # contact info
    contact = extract_contact_info(resume_text)
    if not contact['emails'] and not contact['phones']:
        score -= 20
        messages.append('Could not detect contact email or phone number near the top of the resume.')

    # short resume
    if len(resume_text) < 200:
        score -= 15
        messages.append('Resume seems short (<200 chars) — ensure full content was parsed.')

    # many tables / columns heuristic: lots of multiple spaces implying column layout
    if resume_text.count('  ') > 100:
        score -= 10
        messages.append('Resume appears to use column layouts or excessive spacing which can confuse ATS.')

    score = max(0, score)
    return {'score': score, 'messages': messages}

import nltk
nltk.download('punkt_tab')

def generate_basic_summary(resume_text: str, top_skills: list) -> str:
    """A tiny heuristic-based summary until you plug in an LLM.
    Keep this as a placeholder: it gives 2-3 lines summarizing experience length and key skills.
    """
    # naive heuristics
    years = None
    m = re.search(r'(\d+)\+?\s+years', resume_text.lower())
    if m:
        years = m.group(1)
    sentences = nltk.tokenize.sent_tokenize(resume_text)
    first_sent = sentences[0] if sentences else ''
    skills_brief = ', '.join(top_skills[:6])
    summary = f"{first_sent[:250]}...\nKey skills: {skills_brief}."
    if years:
        summary = f"{years}+ years experience. " + summary
    return summary

# ---------------------- Streamlit App ----------------------

st.set_page_config(page_title='Resume Analyzer', layout='wide')
st.title('📄 AI Resume Analyzer — Starter')
st.markdown('Upload a resume (PDF/DOCX) and paste a job description to get a match score, missing skills, and suggestions.')

with st.sidebar:
    st.header('Quick Notes')
    st.markdown('- This is a starter app. Integrate a powerful LLM (LLaMA/GPT) for advanced suggestions later.')
    st.markdown('- Install optional `sentence-transformers` to improve semantic matching.')
    st.markdown('- For production, sanitize and secure file handling.')

col1, col2 = st.columns([1,2])

with col1:
    uploaded_file = st.file_uploader('Upload Resume (PDF or DOCX)', type=['pdf','docx'])
    jd_text = st.text_area('Paste Job Description (or upload JD later)', height=200)
    analyze_btn = st.button('Analyze')

with col2:
    st.empty()

if analyze_btn:
    if not uploaded_file:
        st.error('Please upload a resume file.')
    elif not jd_text or len(jd_text.strip()) < 20:
        st.error('Please paste a job description (at least 20 characters).')
    else:
        bytes_data = uploaded_file.read()
        filetype = uploaded_file.type

        # Extract text
        resume_text = ''
        if filetype == 'application/pdf' or uploaded_file.name.lower().endswith('.pdf'):
            if fitz is None:
                st.warning('PyMuPDF is not installed — PDF parsing will fail. Install `pymupdf`.')
            resume_text = extract_text_from_pdf(bytes_data)
        elif uploaded_file.name.lower().endswith('.docx'):
            if docx2txt is None:
                st.warning('docx2txt not installed — DOCX parsing will fail. Install `docx2txt`.')
            resume_text = extract_text_from_docx(bytes_data)
        else:
            st.warning('Unknown file type — trying raw decode...')
            try:
                resume_text = bytes_data.decode('utf-8', errors='ignore')
            except Exception:
                resume_text = ''

        if not resume_text:
            st.error('Could not extract text from the resume. Check logs or try a simpler resume PDF/DOCX.')
        else:
            with st.spinner('Analyzing...'):
                # Extract skills
                resume_skills = simple_skill_extraction(resume_text)
                jd_keywords = extract_top_keywords(jd_text, n=30)

                # Determine keyword overlap
                jd_keywords_set = set([k.lower() for k in jd_keywords])
                overlap = resume_skills.intersection(jd_keywords_set)
                keyword_score = 0
                if jd_keywords_set:
                    keyword_score = len(overlap) / len(jd_keywords_set)

                # Semantic similarity between entire resume and JD
                sem_sim = semantic_similarity(resume_text[:5000], jd_text[:5000])

                # Combine scores (simple weighted sum)
                match_score = int(round( (0.6 * sem_sim + 0.4 * keyword_score) * 100 ))

                # Skill gap detection (which top JD keywords are missing)
                missing = sorted(list(jd_keywords_set - set([s.lower() for s in resume_skills])))[:40]

                # ATS checks
                ats = ats_check(bytes_data, resume_text)

                # Basic summary (placeholder)
                top_skills_list = sorted(list(resume_skills), key=lambda x: -len(x))[:20]
                summary = generate_basic_summary(resume_text, top_skills_list)

            # ------------------ Display results ------------------
            st.subheader('Match Overview')
            c1, c2, c3 = st.columns([1,1,1])
            c1.metric('Match Score', f"{match_score}%")
            c2.metric('Semantic Similarity (approx)', f"{sem_sim:.2f}")
            c3.metric('Keyword Overlap', f"{int(keyword_score*100)}%")

            st.subheader('Top extracted skills from resume')
            if resume_skills:
                st.write(', '.join(sorted(resume_skills)))
            else:
                st.info('No skills detected via simple heuristic. Try cleaning the resume or expand skill list in code.')

            st.subheader('Top JD Keywords')
            st.write(', '.join(jd_keywords[:30]))

            st.subheader('Missing Important JD Keywords')
            if missing:
                st.write(', '.join(missing[:30]))
            else:
                st.success('No missing keywords detected among the top JD keywords.')

            st.subheader('ATS Check')
            st.write(f"Score: {ats['score']}/100")
            for m in ats['messages']:
                st.warning(m)

            st.subheader('Profile Summary (placeholder)')
            st.write(summary)

            st.subheader('Suggested Next Steps')
            st.markdown('''
            - Integrate a stronger LLM (LLaMA / GPT) to produce better targeted suggestions and resume rewrites.
            - Expand the `common_skills` list or use a large skills ontology for more accurate matching.
            - Add a resume parser (e.g., `pyresparser` or a trained spaCy model) for robust section extraction.
            - Add an "Optimize Resume" output: produce a revised experience bullet list focused on JD keywords.
            ''')

            # Optionally: allow user to download a short JSON report
            report = {
                'match_score': match_score,
                'semantic_similarity': sem_sim,
                'keyword_overlap_percent': int(keyword_score*100),
                'resume_skills': list(resume_skills),
                'jd_keywords': jd_keywords,
                'missing_keywords': missing,
                'ats': ats,
                'summary': summary
            }

            import json
            st.download_button('Download JSON report', data=json.dumps(report, indent=2), file_name='resume_report.json')

st.markdown('---')
st.caption('Starter app by you — expand with LLM integrations, better parsers, and a database for multi-resume analytics.')
