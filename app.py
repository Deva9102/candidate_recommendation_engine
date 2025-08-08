# Imports

import streamlit as st

from logic.resume_parser import parse_resume, clean_text, extract_metadata_with_bert
from logic.embedding import generate_embeddings
from logic.ranking import rank_candidates
from logic.summary import generate_universal_insights

# Page setup

st.set_page_config(page_title="Candidate Recommendation Engine", layout="wide")

st.title("Candidate Recommendation Engine")
st.write("Upload a job description and candidate resumes. Get the top relevant candidates with AI powered insights.")

# Sidebar for instructions and features

with st.sidebar:

    st.markdown("## How to Use ? ")
    st.markdown(
        """
    1. **Enter Job Description**: Paste the complete job posting  
    2. **Upload Resumes**: Choose PDF/TXT files or paste text  
    3. **Get AI Rankings**: Click 'Find Best Candidates'  
    4. **Review Results**: See top matches with explanations""")

    st.markdown("---")

    st.markdown("## Features: ")
    st.markdown(
           """
                
    - **Quick candidate ranking** Find top matches in seconds
    - **Clear summaries** Understand why candidates fit
    - **AI insights** Get detailed analysis of scores
    - **User-friendly interface** Easy to use with minimal setup
    - **Secure** All data processed locally, no external servers
    - **Support Multiple Formats**: PDF and text support
    - **Detailed Breakdowns**: See exactly why candidates match
    """
    )

# Job Description Input
st.subheader(" Step 1: Enter Job Description")
job_description = st.text_area(
    "Paste the job description here:",
    height=200,
    placeholder="Enter the complete job description including requirements, skills, and responsibilities...",
)

# Resume Input
# For both pdf and text files.

st.subheader(" Step 2: Upload Candidate Resumes")

upload_mode = st.radio(
    "Choose input method:",
    ["Upload PDF/TXT Files", "Paste Resume Text"],
    horizontal=True,
)

uploaded_files = None
resume_bulk_text = ""

if upload_mode == "Upload PDF/TXT Files":
    uploaded_files = st.file_uploader(
        "Upload candidate resumes:",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload multiple PDF or TXT resume files",
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} files uploaded successfully.")
else:
    resume_bulk_text = st.text_area(
        "Paste resumes separated by '---':",
        height=300,
        placeholder="""Harry Styles
Software Engineer with 5 years experience in Python and React...

---

James Bond
Data Scientist with expertise in Machine Learning and Python...""",
    )

# Find Best Candidates
# Parse resumes, generate embeddings, and rank candidates

if st.button("Find Best Candidates", type="primary"):
    if not job_description.strip():
        st.error("Error : Please enter a job description.")
        st.stop()

    has_candidates = False
    if upload_mode == "Upload PDF/TXT Files" and uploaded_files:
        has_candidates = True
    elif upload_mode == "Paste Resume Text" and resume_bulk_text.strip():
        has_candidates = True

    if not has_candidates:
        st.error("Error : Please provide candidate resumes.")
        st.stop()

    with st.spinner("Analyzing candidates..."):
        candidate_texts = {}
        candidate_metadata = {}

        if upload_mode == "Upload PDF/TXT Files":
            progress_bar = st.progress(0)
            for i, file in enumerate(uploaded_files):
                progress_bar.progress((i + 1) / len(uploaded_files))
                try:
                    parsed = parse_resume(file)
                except Exception as e:
                    st.warning(f"Skipping {getattr(file, 'name', 'file')} due to parse error: {e}")
                    continue

                if "error" not in parsed and parsed.get("text", "").strip():
                    name = parsed.get("name") or f"Candidate_{i+1}"
                    if name == "Unknown":
                        name = f"Candidate_{i+1}"
                    candidate_texts[name] = parsed["text"]
                    candidate_metadata[name] = {
                        "name": parsed.get("name", name),
                        "email": parsed.get("email", "N/A"),
                        "phone": parsed.get("phone", "N/A"),
                    }

        else:
            raw_resumes = resume_bulk_text.split("---")
            for i, text in enumerate(raw_resumes):
                if text.strip():
                    cleaned = clean_text(text)
                    try:
                        metadata = extract_metadata_with_bert(cleaned)
                    except Exception:
                        metadata = {"name": f"Candidate_{i+1}", "email": "N/A", "phone": "N/A"}

                    name = metadata.get("name") or f"Candidate_{i+1}"
                    if name == "Unknown":
                        name = f"Candidate_{i+1}"
                    candidate_texts[name] = cleaned
                    candidate_metadata[name] = metadata

        if not candidate_texts:
            st.error("Error : No valid resumes could be processed.")
            st.stop()

        # Generate embeddings and compute similarities
        st.write("Finding best matches ...")
        names = list(candidate_texts.keys())
        texts = list(candidate_texts.values())

        all_texts = [job_description] + texts
        embeddings = generate_embeddings(all_texts)
        jd_embedding = embeddings[0]
        resume_embeddings = embeddings[1:]

        similarity_results = rank_candidates(
            jd_embedding,
            resume_embeddings,
            job_description,
            texts,
        )

        if isinstance(similarity_results, tuple):
            similarity_scores, score_breakdown = similarity_results
        else:
            similarity_scores = similarity_results
            score_breakdown = {"base_scores": similarity_scores}

        # Ensure breakdown arrays exist and align; fall back to zeros
        N = len(texts)
        base_scores = list(score_breakdown.get("base_scores", [0.0] * N))
        keyword_scores = list(score_breakdown.get("keyword_scores", [0.0] * N))
        experience_scores = list(score_breakdown.get("experience_scores", [0.0] * N))
        education_scores = list(score_breakdown.get("education_scores", [0.0] * N))


        # Create dictionary for each candidate with metadata and scores
        # Sort candidates by overall score

        candidate_rows = []
        for i, (name, text, total) in enumerate(zip(names, texts, similarity_scores)):

            candidate_rows.append(
                {
                    "name": name,
                    "text": text,
                    "score": float(total),
                    "base": float(base_scores[i] if i < len(base_scores) else 0.0),
                    "keyword": float(keyword_scores[i] if i < len(keyword_scores) else 0.0),
                    "exp_score": float(experience_scores[i] if i < len(experience_scores) else 0.0),
                    "edu_score": float(education_scores[i] if i < len(education_scores) else 0.0),
                    "email": candidate_metadata.get(name, {}).get("email", "N/A"),
                    "phone": candidate_metadata.get(name, {}).get("phone", "N/A"),
                }
            )
        sorted_rows = sorted(candidate_rows, key=lambda r: r["score"], reverse=True)[:10]


    # Dsiplay top candidates with extracted metadata and scores

    st.subheader(" Top Matching Candidates : ")

    for rank, row in enumerate(sorted_rows, 1):
        name = row["name"]
        score = row["score"]

        with st.container():
            st.markdown(f"### #{rank} - {name}")
            col1, col2, col3 = st.columns([2, 1, 2])

            with col1:
                st.markdown("#### Contact Info:")
                st.write(f"**Email:** {row.get('email', 'N/A')}")
                st.write(f"**Phone:** {row.get('phone', 'N/A')}")

            with col2:
                if score >= 0.6:
                    delta_color, tier = "normal", "Strong Fit"
                elif score >= 0.3:
                    delta_color, tier = "inverse", "Moderate Fit"
                else:
                    delta_color, tier = "off", "Weak Fit"

                st.metric("Overall Match", f"{score:.1%}", delta=tier, delta_color=delta_color)
                st.progress(min(score, 1.0))

                with st.popover(" Score Details"):
                    st.write(f"**Semantic:** {row['base']:.3f}")
                    st.write(f"**Keywords:** {row['keyword']:.3f}")
                    st.write(f"**Experience:** {row['exp_score']:.3f}")
                    st.write(f"**Education:** {row['edu_score']:.3f}")

            with col3:
                with st.expander(" Analysis", expanded=False):
                    try:
                        insights = generate_universal_insights(
                            name, 
                            row['text'], 
                            job_description, 
                            row
                        )
                        st.markdown(insights)
                        
                        st.markdown("---")
                        if score >= 0.7:
                            st.success(" **Recommend for interview** - Strong alignment")
                        elif score >= 0.5:
                            st.info("**Consider for interview** - Good potential fit")
                        elif score >= 0.3:
                            st.warning(" **Secondary choice** - May need development")
                        else:
                            st.error(" **Not recommended** - Poor requirements fit")
                            
                    except Exception as e:
                        st.write("**Quick Assessment:**")
                        if score >= 0.6:
                            st.write("Strong candidate - multiple alignment factors")
                        elif score >= 0.3:
                            st.write("Moderate fit - some relevant experience")
                        else:
                            st.write("Limited alignment with job requirements")
                        
                        st.write(f"Overall Score: {score:.1%}")
            st.divider()


    #  Gives summary of analysis 
    st.subheader(" Analysis Summary")

    if similarity_scores is not None and len(similarity_scores) > 0:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Candidates", len(candidate_rows))

        with col2:
            avg_score = sum(r["score"] for r in candidate_rows) / max(len(candidate_rows), 1)
            st.metric("Average Match", f"{avg_score:.1%}")

        with col3:
            best_score = max((r["score"] for r in candidate_rows), default=0.0)
            st.metric("Best Match", f"{best_score:.1%}")

        with col4:
            good_matches = sum(1 for r in candidate_rows if r["score"] >= 0.6)
            st.metric("Strong Matches", f"{good_matches}/{len(candidate_rows)}")
    else:
        st.info("No similarity scores to summarize.")
