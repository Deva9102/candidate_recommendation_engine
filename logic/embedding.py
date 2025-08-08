# Imports

from sentence_transformers import SentenceTransformer
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Tries to load all-mpnet-base-v2 model for embeddings.
# If it fails, falls back to a all-MiniLM-L6-v2 model ( smaller but still effective).
try:
    _model = SentenceTransformer("all-mpnet-base-v2")
    print("Loaded all-mpnet-base-v2 model")
except:
    # Fallback to the smaller model if memory issues
    _model = SentenceTransformer("all-MiniLM-L6-v2") 
    print("Loaded all-MiniLM-L6-v2 model (fallback)")

# Function to generate embeddings for a list of texts.
# Preprocesses the texts and returns vector embeddings.

def generate_embeddings(text_list, batch_size=32):
    processed_texts = []
    for text in text_list:
        cleaned = preprocess_text_for_embedding(text)
        processed_texts.append(cleaned)
    
    embeddings = _model.encode(
        processed_texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=batch_size,
        normalize_embeddings=True,
        device='cpu'
    )
    
    return embeddings

# Function to preprocess text for embedding.
# Cleans up whitespace, truncates if too long, and ensures it is suitable for embedding.

def preprocess_text_for_embedding(text, max_tokens=384):

    if not text or not text.strip():
        return "No content provided"
        
    text = text.strip()

    import re
    text = re.sub(r'\s+', ' ', text)

    estimated_tokens = len(text) / 4
    if estimated_tokens > max_tokens:
        truncate_chars = int(max_tokens * 4 * 0.9)
        text = text[:truncate_chars] + "..."
    
    return text

# Function to compute similarity matrix for embeddings.
# Returns a similarity matrix.

def compute_similarity_matrix(embeddings):

    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(embeddings)

# Function to get model information.

def get_model_info():

    return {
        "model_name": _model._modules['0'].auto_model.name_or_path,
        "max_seq_length": _model.max_seq_length,
        "embedding_dimension": _model.get_sentence_embedding_dimension()
    }


# Function to generate job description embeddings.
# Weights important sections more for better matching.

def generate_job_description_embedding(jd_text):

    important_sections = extract_key_jd_sections(jd_text)

    if important_sections:
        enhanced_text = jd_text + " " + " ".join(important_sections)
    else:
        enhanced_text = jd_text
        
    return generate_embeddings([enhanced_text])[0]

# Function to generate resume embeddings.

def generate_resume_embedding(resume_text):

    key_sections = extract_key_resume_sections(resume_text)
    
    if key_sections:
        enhanced_text = resume_text + " " + " ".join(key_sections)
    else:
        enhanced_text = resume_text
        
    return generate_embeddings([enhanced_text])[0]


# Function to extract key sections from job descriptions.
# Focuses on requirements, qualifications, and skills.

def extract_key_jd_sections(jd_text):

    import re
    key_sections = []

    req_patterns = [
        r'(?:requirements?|qualifications?|skills?)[:\-\s]*(.*?)(?=\n\s*\n|\n\s*[A-Z]|$)',
        r'(?:must have|required|essential)[:\-\s]*(.*?)(?=\n\s*\n|\n\s*[A-Z]|$)',
        r'(?:experience with|proficiency in)[:\-\s]*(.*?)(?=\n\s*\n|\n\s*[A-Z]|$)'
    ]
    
    for pattern in req_patterns:
        matches = re.findall(pattern, jd_text, re.IGNORECASE | re.DOTALL)
        key_sections.extend([match.strip() for match in matches])
    
    return key_sections[:3]

# Function to extract key sections from resumes.
# Focuses on skills, experience, and projects.

def extract_key_resume_sections(resume_text):

    import re
    key_sections = []
    
    section_patterns = [
        r'(?:skills?|technologies?|expertise)[:\-\s]*(.*?)(?=\n\s*\n|\n\s*[A-Z]|$)',
        r'(?:experience|employment|work history)[:\-\s]*(.*?)(?=\n\s*\n|\n\s*[A-Z]|$)',
        r'(?:projects?|achievements?)[:\-\s]*(.*?)(?=\n\s*\n|\n\s*[A-Z]|$)'
    ]
    
    for pattern in section_patterns:
        matches = re.findall(pattern, resume_text, re.IGNORECASE | re.DOTALL) 
        key_sections.extend([match.strip() for match in matches])
    
    return key_sections[:3]