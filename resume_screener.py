"""
Resume Screening System - Backend Module
Enhanced version with detailed comments, helper functions, and
logging for debugging and traceability.
"""



# =====================================================
# LOGGER CONFIGURATION
# =====================================================
# A logger helps in tracking system events for debugging.
logging.basicConfig(
    filename='resume_screener.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# =====================================================
# MODEL INITIALIZATION (Lazy Loading)
# =====================================================
_nlp = None
_model = None

def get_nlp():
    """
    Loads and returns the spaCy NLP model.
    This is used for Named Entity Recognition (NER)
    and tokenization required in anonymization.
    """
    global _nlp
    if _nlp is None:
        logging.info("Loading spaCy NLP model...")
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def get_model():
    """
    Loads and returns the Sentence Transformer model.
    Used for creating contextual embeddings of resumes
    and job descriptions for semantic similarity.
    """
    global _model
    if _model is None:
        logging.info("Loading Sentence Transformer model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


# =====================================================
# TEXT EXTRACTION FUNCTIONS
# =====================================================

def extract_text_from_pdf(file_obj) -> str:
    """
    Extracts text from a PDF file using PyMuPDF (fitz).
    Works with file objects uploaded via Streamlit or file paths.
    """
    text = []
    try:
        with fitz.open(stream=file_obj.read() if hasattr(file_obj, 'read') else file_obj, filetype="pdf") as doc:
            for page in doc:
                page_text = page.get_text()
                if page_text:
                    text.append(page_text)
        logging.info("Successfully extracted text from PDF.")
    except Exception as e:
        logging.error(f"Error reading PDF: {e}")
        raise ValueError(f"Error reading PDF: {e}")
    return "\n".join(text)


def extract_text_from_docx(file_obj) -> str:
    """
    Extracts text from DOCX files using docx2txt.
    Works for both uploaded file objects and file paths.
    """
    try:
        if hasattr(file_obj, 'read'):
            return docx2txt.process(file_obj)
        else:
            return docx2txt.process(file_obj)
    except Exception as e:
        logging.error(f"Error reading DOCX: {e}")
        raise ValueError(f"Error reading DOCX: {e}")


def extract_text(file_obj, filename: str) -> str:
    """
    Determines file type and extracts text accordingly.
    Supported formats: PDF, DOCX, TXT.
    """
    ext = filename.split('.')[-1].lower()

    if ext == "pdf":
        return extract_text_from_pdf(file_obj)
    elif ext in ("docx", "doc"):
        return extract_text_from_docx(file_obj)
    elif ext == "txt":
        content = file_obj.read()
        return content.decode('utf-8', errors='ignore') if isinstance(content, bytes) else content
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# =====================================================
# ANONYMIZATION MODULE
# =====================================================

def anonymize_text(text: str, remove_dates=True) -> str:
    """
    Removes personally identifiable information (PII)
    from resumes to ensure fair and unbiased screening.

    PII removed:
    - Names
    - Email addresses
    - Phone numbers
    - URLs
    - Dates
    - Gendered terms
    """
    nlp = get_nlp()
    doc = nlp(text)

    # Replace detected PERSON entities with [NAME]
    tokens = []
    person_spans = {ent.start_char: ent.end_char for ent in doc.ents if ent.label_ == "PERSON"}
    i = 0
    while i < len(text):
        if i in person_spans:
            end = person_spans[i]
            tokens.append("[NAME]")
            i = end
        else:
            tokens.append(text[i])
            i += 1

    anonymized = "".join(tokens)

    # Remove contact information and URLs
    anonymized = re.sub(r'\S+@\S+', '[EMAIL]', anonymized)
    anonymized = re.sub(r'(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{6,12}', '[PHONE]', anonymized)
    anonymized = re.sub(r'http\S+|www\.\S+', '[URL]', anonymized)

    # Remove dates and gendered terms
    if remove_dates:
        anonymized = re.sub(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4})\b', '[DATE]', anonymized)

    anonymized = re.sub(
        r'\b(male|female|man|woman|he|she|his|her)\b',
        '[GENDER]', anonymized, flags=re.IGNORECASE
    )

    logging.info("Anonymization completed successfully.")
    return anonymized


# =====================================================
# SKILL EXTRACTION MODULE
# =====================================================

SKILL_LIST = [
    "python", "java", "c++", "sql", "pandas", "numpy", "spark", "hadoop", "ml",
    "machine learning", "deep learning", "tensorflow", "pytorch", "scikit-learn",
    "nlp", "natural language processing", "data analysis", "data visualization",
    "power bi", "tableau", "aws", "azure", "docker", "kubernetes", "rest api",
    "git", "linux", "javascript", "react", "node.js", "html", "css", "flask",
    "django", "mongodb", "postgresql", "mysql", "redis", "kafka", "airflow",
    "fastapi", "excel", "statistics"
]

def extract_skills(text: str) -> List[str]:
    """
    Extracts technical skills from the resume text
    using simple pattern-based matching.

    Returns a sorted list of identified skills.
    """
    text_lower = text.lower()
    found = set()

    for skill in SKILL_LIST:
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found.add(skill)

    logging.info(f"Skills extracted: {len(found)} found.")
    return sorted(found)


# =====================================================
# RESUME PROCESSING PIPELINE
# =====================================================

def process_resume(file_obj, filename: str) -> Dict:
    """
    Processes a single resume file.
    Includes text extraction, anonymization, and skill extraction.
    """
    try:
        raw_text = extract_text(file_obj, filename)
        anonymized = anonymize_text(raw_text)
        skills = extract_skills(raw_text)

        logging.info(f"Processed resume: {filename}")
        return {
            "filename": filename,
            "raw_text": raw_text,
            "anonymized_text": anonymized,
            "skills": skills,
            "status": "success"
        }

    except Exception as e:
        logging.error(f"Error processing resume {filename}: {e}")
        return {
            "filename": filename,
            "status": "error",
            "error": str(e)
        }


def process_multiple_resumes(uploaded_files) -> pd.DataFrame:
    """
    Processes multiple uploaded resume files.
    Returns a DataFrame with processed information.
    """
    results = []
    for uploaded_file in uploaded_files:
        result = process_resume(uploaded_file, uploaded_file.name)
        results.append(result)

    df = pd.DataFrame(results)
    df = df[df['status'] == 'success'].reset_index(drop=True)

    logging.info(f"Total resumes processed successfully: {len(df)}")
    return df


# =====================================================
# EMBEDDING GENERATION AND RANKING MODULE
# =====================================================

def compute_embeddings(texts: List[str]):
    """
    Generates sentence embeddings for given text data
    using the pre-trained Sentence Transformer model.
    """
    model = get_model()
    return model.encode(texts, convert_to_tensor=True, show_progress_bar=False)


def prepare_embeddings(df_resumes: pd.DataFrame, job_desc_text: str):
    """
    Prepares embeddings for resumes and job description.
    Combines anonymized resume text with extracted skills
    for a stronger semantic context.
    """
    resume_texts = []
    for _, row in df_resumes.iterrows():
        skill_summary = " ".join(row["skills"]) if row["skills"] else ""
        combined = row["anonymized_text"][:10000] + "\n\nSkills: " + skill_summary
        resume_texts.append(combined)

    jd_anonymized = anonymize_text(job_desc_text, remove_dates=False)
    jd_skills = " ".join(extract_skills(job_desc_text))
    jd_combined = jd_anonymized[:10000] + "\n\nRequired Skills: " + jd_skills

    all_texts = resume_texts + [jd_combined]
    embeddings = compute_embeddings(all_texts)

    resume_embs = embeddings[:-1]
    jd_emb = embeddings[-1]

    logging.info("Embeddings generated successfully.")
    return resume_embs, jd_emb, jd_skills


def rank_resumes(df_resumes: pd.DataFrame, job_desc_text: str,
                 top_k=10, skill_weight=0.4, semantic_weight=0.6) -> pd.DataFrame:
    """
    Ranks resumes against the given job description
    using a combination of semantic similarity
    and exact skill matching.
    """
    if len(df_resumes) == 0:
        return pd.DataFrame()

    # Generate embeddings
    resume_embs, jd_emb, jd_skills_str = prepare_embeddings(df_resumes, job_desc_text)

    # Compute semantic similarity
    cos_scores = util.cos_sim(resume_embs, jd_emb)[:, 0].cpu().numpy()

    # Compute skill overlap scores
    jd_skills = set(extract_skills(job_desc_text))
    skill_scores = []
    for skills in df_resumes["skills"]:
        if not skills:
            skill_scores.append(0.0)
        else:
            overlap = len(set(skills).intersection(jd_skills))
            denom = max(1, len(jd_skills))
            skill_scores.append(overlap / denom)

    skill_scores = np.array(skill_scores)

    # Normalize and combine scores
    cos_min, cos_max = cos_scores.min(), cos_scores.max()
    normalized_cos = (cos_scores - cos_min) / (cos_max - cos_min) if cos_max > cos_min else cos_scores

    final_scores = semantic_weight * normalized_cos + skill_weight * skill_scores

    # Combine results into DataFrame
    df = df_resumes.copy()
    df["semantic_score"] = cos_scores
    df["skill_score"] = skill_scores
    df["final_score"] = final_scores

    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    logging.info("Candidate ranking completed successfully.")
    return df.head(top_k)

