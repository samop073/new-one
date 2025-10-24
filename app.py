11.	SAMPLE SOURCE PROGRAM 
 
11.1 APP.PY - FRONTEND 
 
""" 
Resume Screening System - Streamlit Frontend 
Run with: streamlit run app.py 
""" 
 
import streamlit as st import pandas as pd from resume_screener import (     process_multiple_resumes,     rank_resumes,     extract_skills 
) 
 
# Page config st.set_page_config(     page_title="Resume Screening System",     page_icon="📄",     layout="wide" 
) 
# Custom CSS st.markdown(""" 
    <style>     .main-header {         font-size: 2.5rem;         font-weight: bold;         color: #1f77b4;         margin-bottom: 0.5rem; 
    } 
    .sub-header {         font-size: 1.2rem;         color: #666;         margin-bottom: 2rem; 
    } 
    .metric-card {         background: #f0f2f6;         padding: 1rem;         border-radius: 0.5rem;         margin: 0.5rem 0; 
    } 
    </style> 
""", unsafe_allow_html=True) # Initialize session state if 'df_resumes' not in st.session_state:     st.session_state.df_resumes = None if 'ranked_results' not in st.session_state:     st.session_state.ranked_results = None 
 
# Header st.markdown('<div 	class="main-header">📄 	Resume 	Screening 	System</div>', unsafe_allow_html=True) st.markdown('<div class="sub-header">AI-powered resume matching with bias reduction</div>', unsafe_allow_html=True) 
 
# Sidebar with st.sidebar: 
    st.header("⚙️ Configuration") 
     
    skill_weight = st.slider(         "Skill Match Weight",         min_value=0.0,         max_value=1.0,         value=0.4,         step=0.1,         help="How much to weigh exact skill matches" ) 
 
semantic_weight = 1.0 - skill_weight     st.info(f"Semantic Match Weight: {semantic_weight:.1f}") 
     
    top_k = st.number_input(         "Number of Top Candidates",         min_value=1,         max_value=50,         value=10,         step=1 
    ) 
     
    st.markdown("---")     st.markdown("### About")     st.markdown(""" 
    This system uses: 
-	🤖 NLP for text analysis 
-	🔒 Anonymization to reduce bias 
-	🎯 Semantic + skill matching 
-	📊 Embedding-based ranking 
    """) 
 
# Main content tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "🏆 Results", "📊 Analytics"]) 
 
with tab1: 
    st.header("Upload Resumes and Job Description") 
     
    col1, col2 = st.columns(2) 
     
    with col1: 
        st.subheader("📁 Upload Resumes")         uploaded_resumes = st.file_uploader(             "Upload resume files (PDF, DOCX, TXT)", 
            type=['pdf', 'docx', 'doc', 'txt'],             accept_multiple_files=True,             help="Upload one or more resume files" 
        ) 
         
        if uploaded_resumes: 
            st.success(f"✅ {len(uploaded_resumes)} resume(s) uploaded") 
     
    with col2: 
        st.subheader("📋 Job Description")        job_desc = st.text_area( 
        "Paste the job description here",         height=200,         placeholder="Enter the full job description including required skills, experience, 
and qualifications..." 
        ) 
         
        if job_desc: 
            jd_skills = extract_skills(job_desc)             st.info(f"**Required Skills Detected:** {', '.join(jd_skills) if jd_skills else 'None'}") 
     
    st.markdown("---") 
     
    # Process button     if st.button("🚀 Process Resumes", type="primary", use_container_width=True):         if not uploaded_resumes: 
            st.error("⚠️ Please upload at least one resume")         elif not job_desc: 
            st.error("⚠️ Please enter a job description") 
        else: 
            with st.spinner("Processing resumes... This may take a minute."): 
                try: 
                    # Process resumes                     df_resumes = process_multiple_resumes(uploaded_resumes)                 if len(df_resumes) == 0: 
                    st.error("❌ No valid resumes could be processed. Check file formats.") 
                    else: 
                        st.session_state.df_resumes = df_resumes 
                         
                        # Rank resumes                         ranked = rank_resumes(                             df_resumes,                             job_desc,                             top_k=top_k,                             skill_weight=skill_weight,                             semantic_weight=semantic_weight 
                        ) 
                        st.session_state.ranked_results = ranked 
                         
                        st.success(f"✅ Successfully processed {len(df_resumes)} resumes!") 
                        st.balloons() 
                         
                except Exception as e: 
                    st.error(f"❌ Error processing resumes: {str(e)}") 
 
with tab2: 
st.header("Top Candidates") 
 
if st.session_state.ranked_results is not None and len(st.session_state.ranked_results) > 0:         results = st.session_state.ranked_results 
         
        # Summary metrics         col1, col2, col3, col4 = st.columns(4)         with col1: 
            st.metric("Total Resumes", len(st.session_state.df_resumes))         with col2: 
            st.metric("Top Candidates", len(results))         with col3: 
            avg_score = results['final_score'].mean()             st.metric("Avg Match Score", f"{avg_score:.2f}")         with col4: 
            top_score = results['final_score'].iloc[0]             st.metric("Best Match", f"{top_score:.2f}") 
         
        st.markdown("---") 
         
        # Display ranked candidates         for idx, row in results.iterrows():  	with st.expander(f"#{idx+1} - {row['filename']} (Score: {row['final_score']:.3f})", 
expanded=(idx<3)): 
                col1, col2, col3 = st.columns(3) 
                                 with col1: 
                    st.metric("Final Score", f"{row['final_score']:.3f}")                 with col2:                     st.metric("Semantic Match", f"{row['semantic_score']:.3f}")                 with col3: 
                    st.metric("Skill Match", f"{row['skill_score']:.3f}") 
                 
                st.markdown("**Skills Found:**") 
                if row['skills']: 
                    skills_display = ", ".join(row['skills'])                     st.markdown(f"_{skills_display}_") 
                else: 
                    st.markdown("_No skills detected_") 
                 
                with st.expander("View Resume Text (Anonymized)"): 
                    st.text(row['anonymized_text'][:2000] + "..." if len(row['anonymized_text']) > 2000 else row['anonymized_text']) 
         
        # Download results        st.markdown("---")    csv = results[['filename', 'final_score', 'skill_score', 'semantic_score', 
'skills']].to_csv(index=False)         st.download_button(             label="📥 Download Results (CSV)", 
            data=csv,             file_name="resume_screening_results.csv",             mime="text/csv" 
        ) 
             else: 
        st.info("👆 Upload resumes and process them in the 'Upload & Process' tab first") 
 
with tab3: 
    st.header("Analytics Dashboard") 
     
    if st.session_state.ranked_results is not None and len(st.session_state.ranked_results) > 0:         results = st.session_state.ranked_results 
         
        # Score distribution         col1, col2 = st.columns(2) 
         
        with col1: 
            st.subheader("Score Distribution") 
  	 score_data = pd.DataFrame({ 
                'Candidate': [f"#{i+1}" for i in range(len(results))], 
                'Final Score': results['final_score'], 
                'Semantic Score': results['semantic_score'], 
                'Skill Score': results['skill_score'] 
            }) 
            st.bar_chart(score_data.set_index('Candidate')) 
         
        with col2: 
            st.subheader("Top Skills Across Candidates") 
            all_skills = []             for skills in results['skills']:                 all_skills.extend(skills) 
                         if all_skills: 
                skill_counts = pd.Series(all_skills).value_counts().head(10)                 st.bar_chart(skill_counts)             else: 
                st.info("No skills detected in top candidates") 
         
        # Detailed stats         st.markdown("---")         st.subheader("Statistical Summary")     stats_df = results[['final_score', 'semantic_score', 'skill_score']].describe()        st.dataframe(stats_df, use_container_width=True) 
             else: 
        st.info("👆 Process resumes first to see analytics") 
 
# Footer st.markdown("---") st.markdown(""" 
<div style='text-align: center; color: #666; padding: 1rem;'> 
    Built with Streamlit | NLP-powered Resume Screening System 
</div> 
""", unsafe_allow_html=True) 
 
 
 



















11.1  RESUME_SCREENER.PY - BACKEND 
 
""" 
Resume Screening System - Backend Module 
Refactored for Streamlit compatibility 
""" 
 
import os import re from typing import List, Dict, Tuple import pandas as pd import numpy as np import fitz  # PyMuPDF import docx2txt import spacy from sentence_transformers import SentenceTransformer, util 
 
# Initialize models (cached) 
_nlp = None 
_model = None 
 
def get_nlp():     global _nlp     if _nlp is None: 
        _nlp = spacy.load("en_core_web_sm")     return _nlp 
 
def get_model():     global _model     if _model is None:    _model = SentenceTransformer("all-MiniLM-L6-v2") return _model 
# ============ TEXT EXTRACTION ============ 
def extract_text_from_pdf(file_obj) -> str:     """Extract text from PDF file object or path"""     text = []     try: 
        with fitz.open(stream=file_obj.read() if hasattr(file_obj, 'read') else file_obj, filetype="pdf") as doc:             for page in doc: 
                page_text = page.get_text()                 if page_text: 
                    text.append(page_text)     except Exception as e:         raise ValueError(f"Error reading PDF: {e}")     return "\n".join(text) 
 
def extract_text_from_docx(file_obj) -> str:     """Extract text from DOCX file object or path""" 
    try:         if hasattr(file_obj, 'read'): 
            return docx2txt.process(file_obj) 
        else: 
      return docx2txt.process(file_obj)  except Exception as e: 
        raise ValueError(f"Error reading DOCX: {e}") 
 
def extract_text(file_obj, filename: str) -> str:     """Extract text from uploaded file"""     ext = filename.split('.')[-1].lower() 
     
    if ext == "pdf": 
        return extract_text_from_pdf(file_obj)     elif ext in ("docx", "doc"): 
        return extract_text_from_docx(file_obj)     elif ext == "txt": 
        content = file_obj.read()         return content.decode('utf-8', errors='ignore') if isinstance(content, bytes) else content 
    else: 
        raise ValueError(f"Unsupported file type: {ext}") 
 
# ============ ANONYMIZATION ============ 
def anonymize_text(text: str, remove_dates=True) -> str: 
    """Remove PII to reduce bias"""     nlp = get_nlp()     doc = nlp(text) 
# Replace PERSON entities 
   tokens = []     person_spans = {ent.start_char: ent.end_char for ent in doc.ents if ent.label_ == 
"PERSON"} 
         i = 0     while i < len(text):         if i in person_spans: 
            end = person_spans[i]             tokens.append("[NAME]") 
            i = end         else: 
            tokens.append(text[i])             i += 1 
     
    anonymized = "".join(tokens) 
     
    # Remove contact info     anonymized = re.sub(r'\S+@\S+', '[EMAIL]', anonymized)     anonymized = re.sub(r'(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{6,12}', '[PHONE]', anonymized)     anonymized = re.sub(r'http\S+|www\.\S+', '[URL]', anonymized) 
     
if remove_dates: 
     anonymized = re.sub(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4})\b', '[DATE]', 
anonymized) 
     
    # Remove gendered words     anonymized = re.sub(r'\b(male|female|man|woman|he|she|his|her)\b', '[GENDER]', anonymized, flags=re.IGNORECASE) 
     
    return anonymized 
 
# ============ SKILL EXTRACTION ============ 
SKILL_LIST = [ 
    "python", "java", "c++", "sql", "pandas", "numpy", "spark", "hadoop", "ml", "machine learning", 
    "deep learning", "tensorflow", "pytorch", "scikit-learn", "nlp", "natural language processing", 
    "data analysis", "data visualization", "power bi", "tableau", "aws", "azure", "docker", 
"kubernetes", 
    "rest api", "git", "linux", "javascript", "react", "node.js", "html", "css", "flask", 
"django", 
    "mongodb", "postgresql", "mysql", "redis", "kafka", "airflow", "fastapi", "excel", 
"statistics" 
] 
 
def extract_skills(text: str) -> List[str]: 
"""Extract skills from text using pattern matching""" text_lower = text.lower()    found = set() 
     
    for skill in SKILL_LIST: 
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower): 
            found.add(skill) 
     
    return sorted(found) 
 
# ============ RESUME PROCESSING ============ 
def process_resume(file_obj, filename: str) -> Dict: 
    """Process a single resume file""" 
    try: 
        raw_text = extract_text(file_obj, filename)         anonymized = anonymize_text(raw_text)         skills = extract_skills(raw_text) 
         
        return { 
            "filename": filename, 
            "raw_text": raw_text, 
 	        "anonymized_text": anonymized, 
        "skills": skills, 
        "status": "success" 
    } 
except Exception as e:     return { 
            "filename": filename, 
            "status": "error", 
            "error": str(e) 
        } 
 
def process_multiple_resumes(uploaded_files) -> pd.DataFrame: 
    """Process multiple uploaded resume files"""     results = [] 
     
    for uploaded_file in uploaded_files: 
        result = process_resume(uploaded_file, uploaded_file.name)         results.append(result) 
     
    df = pd.DataFrame(results)     # Filter out errors     df = df[df['status'] == 'success'].reset_index(drop=True)     return df 
 
# ============ EMBEDDING & RANKING ============ def compute_embeddings(texts: List[str]):     """Compute sentence embeddings"""     model = get_model()     return model.encode(texts, convert_to_tensor=True, show_progress_bar=False) 
 
def prepare_embeddings(df_resumes: pd.DataFrame, job_desc_text: str): 
    """Prepare embeddings for resumes and job description"""     resume_texts = [] 
     
    for _, row in df_resumes.iterrows(): 
        skill_summary = " ".join(row["skills"]) if row["skills"] else ""         # Combine anonymized text with skills (cap at 10k chars)         combined = row["anonymized_text"][:10000] + "\n\nSkills: " + skill_summary         resume_texts.append(combined) 
     
    # Prepare job description     jd_anonymized = anonymize_text(job_desc_text, remove_dates=False)     jd_skills = " ".join(extract_skills(job_desc_text))     jd_combined = jd_anonymized[:10000] + "\n\nRequired Skills: " + jd_skills 
     
    all_texts = resume_texts + [jd_combined]  	embeddings = compute_embeddings(all_texts) resume_embs = embeddings[:-1] jd_emb = embeddings[-1] 
 
return resume_embs, jd_emb, jd_skills 
 
def rank_resumes(df_resumes: pd.DataFrame, job_desc_text: str,                   top_k=10, skill_weight=0.4, semantic_weight=0.6) -> pd.DataFrame:     """Rank resumes against job description""" 
     
    if len(df_resumes) == 0:         return pd.DataFrame() 
     
    # Compute embeddings     resume_embs, jd_emb, jd_skills_str = prepare_embeddings(df_resumes, job_desc_text) 
     
    # Semantic similarity scores     cos_scores = util.cos_sim(resume_embs, jd_emb)[:, 0].cpu().numpy() 
     
    # Skill overlap scores     jd_skills = set(extract_skills(job_desc_text))     skill_scores = [] 
     
    for skills in df_resumes["skills"]: 
    if not skills: 
        skill_scores.append(0.0)     else: 
        overlap = len(set(skills).intersection(jd_skills))             denom = max(1, len(jd_skills))             skill_scores.append(overlap / denom) 
     
    skill_scores = np.array(skill_scores) 
     
    # Normalize semantic scores to 0-1     cos_min, cos_max = cos_scores.min(), cos_scores.max()     cos_range = cos_max - cos_min     if cos_range > 0: 
        normalized_cos = (cos_scores - cos_min) / cos_range 
    else: 
        normalized_cos = cos_scores 
     
    # Combined score     final_scores = semantic_weight * normalized_cos + skill_weight * skill_scores 
     
    # Add scores to dataframe  	df = df_resumes.copy() df["semantic_score"] = cos_scores df["skill_score"] = skill_scores df["final_score"] = final_scores 
 
# Sort and return top K     df = df.sort_values("final_score", ascending=False).reset_index(drop=True)     return df.head(top_k) 
