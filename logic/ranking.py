#Imports

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
from datetime import date
from calendar import month_abbr

# Function to rank candidates.
# Uses semantic similarity, keyword matching, experience level, and education.
# Returns final scores and a breakdown of scoring factors.

def rank_candidates(jd_embedding, resume_embeddings, job_description, resume_texts):

    # 1. Semantic similarity - 40% 
    jd_embedding = jd_embedding.reshape(1, -1)
    semantic_scores = cosine_similarity(jd_embedding, resume_embeddings)[0]
    
    # 2. Keyword match - 30%
    keyword_scores = [compute_keyword_score(job_description, resume) 
                     for resume in resume_texts]
    
    # 3. Experience - 20%
    experience_scores = [compute_experience_score(job_description, resume) 
                        for resume in resume_texts]
    
    # 4. Education - 10%
    education_scores = [compute_education_score(job_description, resume) 
                       for resume in resume_texts]
    
    print(f"Raw semantic scores: {semantic_scores}")
    print(f"Raw keyword scores: {keyword_scores}")
    print(f"Raw experience scores: {experience_scores}")
    print(f"Raw education scores: {education_scores}")
    
    # Use raw scores for small datasets
    if len(resume_texts) <= 3:
        semantic_scores = np.clip(semantic_scores, 0, 1)
        keyword_scores = np.clip(keyword_scores, 0, 1)
        experience_scores = np.clip(experience_scores, 0, 1)
        education_scores = np.clip(education_scores, 0, 1)
    else:
        semantic_scores = smart_normalize(semantic_scores)
        keyword_scores = smart_normalize(keyword_scores) 
        experience_scores = np.clip(experience_scores, 0, 1)
        education_scores  = np.clip(education_scores, 0, 1)
    
    # Weighted combination
    final_scores = []
    for i in range(len(resume_texts)):
        weighted_score = (
            0.40 * semantic_scores[i] +  
            0.30 * keyword_scores[i] + 
            0.20 * experience_scores[i] + 
            0.10 * education_scores[i]
        )
        final_scores.append(weighted_score)
    
    print(f"Final weighted scores: {final_scores}")
    
    score_breakdown = {
        "base_scores": semantic_scores.tolist() if isinstance(semantic_scores, np.ndarray) else semantic_scores,
        "keyword_scores": keyword_scores,
        "experience_scores": experience_scores, 
        "education_scores": education_scores
    }
    
    return final_scores, score_breakdown

# Helpers for different types of months, year and dates.

MONTHS = {m.lower(): i for i, m in enumerate(month_abbr) if m} 
 
MONTHS.update({
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6, 
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
    "sept":9,"sep":9
})

DATE_RANGE_RE = re.compile(r"""
    (?P<m1>[A-Za-z]{3,9})?\s*\.?\s*(?P<y1>20\d{2}|19\d{2})
    \s*[-–—]\s*
    (?P<m2>present|current|now|[A-Za-z]{3,9})?\.?\s*(?P<y2>20\d{2}|19\d{2})?
""", re.IGNORECASE | re.VERBOSE)

PHRASE_YEARS_RE = re.compile(r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)?', re.I)

# Function to parse month token into a month number.
# Returns number for month or returns None for "present", "current", or "now".

def _parse_month_token(mtok: str):
    if not mtok: return 1
    mtok = mtok.strip().lower()
    if mtok in ("present","current","now"): return None
    return MONTHS.get(mtok, 1)

# Function to convert year and month into a date object.

def _as_date(y: int, m: int|None) -> date:
    if m is None:
        today = date.today()
        return date(today.year, today.month, 1)
    return date(int(y), int(m), 1)

# Function to merge overlapping or adjacent date intervals.

def _merge_intervals(intervals):
    if not intervals: return []
    intervals.sort()
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ls, le = merged[-1]
        contiguous = (s.year == le.year and s.month <= le.month + 1) or (
            s.year == le.year + 1 and le.month == 12 and s.month == 1
        )
        if s <= le or contiguous:
            if e > le:
                merged[-1] = (ls, e)
        else:
            merged.append((s, e))
    return merged

def _months_between(s: date, e: date) -> int:
    return (e.year - s.year) * 12 + (e.month - s.month) + 1

# Function to extract years of experience from text.
# Handles ranges like '2018–2021', 'Jan 2019 – Present'.
# Merges overlapping or adjacent ranges.
# Returns float years or None if no valid experience found.

def extract_years_experience(text: str):
    t = text or ""
    intervals = []
    for m in DATE_RANGE_RE.finditer(t):
        y1 = m.group("y1"); y2 = m.group("y2")
        m1 = _parse_month_token(m.group("m1"))
        m2 = _parse_month_token(m.group("m2"))
        if not y1:
            continue
        start = _as_date(int(y1), m1 or 1)
        end = _as_date(int(y2) if y2 else date.today().year, m2)
        if end >= start:
            intervals.append((start, end))

    if intervals:
        merged = _merge_intervals(intervals)
        months = sum(_months_between(s, e) for s, e in merged)
        return round(months / 12.0, 1)

    yrs = [int(x) for x in PHRASE_YEARS_RE.findall(t)]
    return float(max(yrs)) if yrs else None

# Common stopwords.

STOPWORDS = set("""
a an and are as at be by for from has have in is it its of on or that the to with will shall
""".split())

SECTION_HDRS = re.compile(
    r'(?im)^(requirements?|qualifications?|skills?|responsibilities?|about you|what you\'ll do)\s*[:\-]*\s*$'
)

# Split text into sentences.

def _simple_sentences(text):
    return re.split(r'(?m)(?:\n\s*[-•*]\s+|[.!?]\s+)', text)

# Tokenize text into words, removing stopwords and short tokens.
def _tokenize(text):
    # keep words (letters, digits, hyphen, slash), drop tiny tokens, lowercase
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9/\-]{1,}", text.lower())
    return [t for t in toks if t not in STOPWORDS and len(t) > 2]

# Function to build multi-word phrases from text.
# Returns a list of phrases.

def _phrases(text):
    phrases = []

    for m in re.finditer(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})", text):
        span = m.group(0).strip()
        if len(span.split()) >= 2:
            phrases.append(span.lower())
    words = _tokenize(text)
    bigrams = [" ".join(t) for t in zip(words, words[1:])]
    trigrams = [" ".join(t) for t in zip(words, words[1:], words[2:])]

    bg_counts = Counter(bigrams)
    tg_counts = Counter(trigrams)

    phrases += [p for p,c in bg_counts.items() if c >= 2]
    phrases += [p for p,c in tg_counts.items() if c >= 2]

    return phrases

# Function to extract key terms and phrases from job description.
def _extract_jd_terms(jd_text, top_k=30):
    text = jd_text.strip()

    lines = jd_text.splitlines()
    boosted_chunks = []
    i = 0
    while i < len(lines):
        if SECTION_HDRS.match(lines[i]):
            chunk = []
            i += 1
            while i < len(lines) and lines[i].strip() and not SECTION_HDRS.match(lines[i]):
                chunk.append(lines[i])
                i += 1
            if chunk:
                boosted_chunks.append("\n".join(chunk))
        else:
            i += 1

    boosted_text = " ".join(boosted_chunks) if boosted_chunks else jd_text

    base_tokens = _tokenize(jd_text)
    boost_tokens = _tokenize(boosted_text) * 2 

    word_counts = Counter(base_tokens) + Counter(boost_tokens)

    ph_all = Counter(_phrases(jd_text))
    ph_boosted = Counter(_phrases(boosted_text))
    phrase_counts = ph_all + ph_boosted

    candidates = {}

    for w, c in word_counts.items():
        if re.search(r"[a-z0-9]", w):
            candidates[w] = candidates.get(w, 0.0) + c

    for p, c in phrase_counts.items():
        length_bonus = 1.0 + 0.25 * (len(p.split()) - 1)
        candidates[p] = candidates.get(p, 0.0) + c * length_bonus

    top_terms = dict(Counter(candidates).most_common(top_k))
    total_weight = sum(top_terms.values()) or 1.0

    return top_terms, total_weight

# Helper to check if a phrase is in lowercase text.
def _contains_phrase(haystack_lower, needle_lower):
    return re.search(rf"(?<!\w){re.escape(needle_lower)}(?!\w)", haystack_lower) is not None

# Function to compute keyword score based on JD terms coverage in resume.
# Returns a normalized score between 0 and 1.

def compute_keyword_score(job_description, resume_text):

    jd_terms, total_weight = _extract_jd_terms(job_description)
    if not jd_terms:
        jd_set = set(_tokenize(job_description))
        rez_set = set(_tokenize(resume_text))
        if not jd_set:
            return 0.6 
        return min(len(jd_set & rez_set) / max(len(jd_set), 1), 1.0)

    rez_lower = resume_text.lower()

    covered = 0.0
    for term, w in jd_terms.items():
        if " " in term:
            hit = _contains_phrase(rez_lower, term)
        else:
            hit = re.search(rf"(?<!\w){re.escape(term)}(?!\w)", rez_lower) is not None
        if hit:
            covered += w

    score = covered / total_weight
    return float(max(0.0, min(score * 1.05, 1.0)))

# Helper to match section headers.

SECTION_HDRS = re.compile(
    r'(?im)^(requirements?|qualifications?|skills?|responsibilities?|about you|what you\'ll do)\s*[:\-]*\s*$'
)

# Function to extract a specific section from text based on headers.

def _extract_section(text, headers_regex, max_chars=4000):
    lines = text.splitlines()
    chunks = []
    i = 0
    hdr = re.compile(headers_regex, re.IGNORECASE | re.MULTILINE)
    while i < len(lines):
        if hdr.match(lines[i].strip()):
            i += 1
            block = []
            while i < len(lines) and lines[i].strip() and not hdr.match(lines[i].strip()):
                block.append(lines[i])
                i += 1
            if block:
                chunks.append("\n".join(block))
        else:
            i += 1
    section = "\n\n".join(chunks).strip()
    if not section:
        section = text
    return section[:max_chars]

# Coverage calculation from JD terms.
# Returns a score between 0 and 1.

def _coverage_from_terms(jd_terms, total_weight, target_text):
    if not jd_terms:
        return 0.6
    rez = target_text.lower()
    covered = 0.0
    for term, w in jd_terms.items():
        if " " in term:
            hit = re.search(rf"(?<!\w){re.escape(term)}(?!\w)", rez) is not None
        else:
            hit = re.search(rf"(?<!\w){re.escape(term)}(?!\w)", rez) is not None
        if hit:
            covered += w
    score = covered / (total_weight or 1.0)
    return float(max(0.0, min(score * 1.05, 1.0)))

# Relevance factor based on JD terms coverage in target text.

def _relevance_factor(jd_text, target_text):
    jd_terms, total_weight = _extract_jd_terms(jd_text)
    return _coverage_from_terms(jd_terms, total_weight, target_text)

# Function to score experience match between JD and resume.
# Returns a score between 0 and 1.

def compute_experience_score(job_description, resume_text):
    jd_exp = extract_years_experience(job_description)
    exp_section = _extract_section(
        resume_text, r"^(experience|employment|work history|professional experience|relevant experience)\s*[:\-]?$"
    )
    rez_exp = extract_years_experience(exp_section)

    if jd_exp is None and rez_exp is None:
        base = 0.6
    elif jd_exp is None and rez_exp is not None:
        base = min(0.85, 0.6 + min(rez_exp, 10) * 0.025)
    elif jd_exp is not None and rez_exp is None:
        base = 0.3
    else:
        if rez_exp >= jd_exp:
            excess_bonus = min((rez_exp - jd_exp) * 0.05, 0.15)
            base = min(1.0, 0.75 + excess_bonus)
        else:
            gap = jd_exp - rez_exp
            if gap <= 1:
                base = 0.65
            elif gap <= 2:
                base = 0.45
            else:
                base = max(0.15, 0.45 - (gap - 2) * 0.1)

    rel = _relevance_factor(job_description, exp_section)
    scaled = base * (0.35 + 0.65 * rel)
    if rel < 0.2:
        scaled = min(scaled, 0.40)

    return float(max(0.1, min(scaled, 1.0)))

# Function to score education match between JD and resume.
# Returns a score between 0 and 1.

def compute_education_score(job_description, resume_text):
    jd_edu = extract_education_requirements(job_description.lower())
    edu_section = _extract_section(
        resume_text, r"^(education|academic background|qualifications|certifications|training|licenses)\s*[:\-]?$"
    )
    rez_edu = extract_education_achievements(edu_section.lower())

    if not jd_edu['required'] and not jd_edu['preferred']:
        base = 0.7
    else:
        base = 0.5
        if jd_edu['degree_required']:
            base += 0.3 if rez_edu['has_degree'] else -0.2
        if jd_edu['fields'] and rez_edu['fields']:
            if bool(set(jd_edu['fields']).intersection(set(rez_edu['fields']))):
                base += 0.2
        if rez_edu['certifications'] and jd_edu['certifications']:
            if bool(set(jd_edu['certifications']).intersection(set(rez_edu['certifications']))):
                base += 0.15

    rel = _relevance_factor(job_description, edu_section)
    scaled = base * (0.30 + 0.70 * rel)
    if jd_edu['degree_required'] and rel < 0.2:
        scaled = min(scaled, 0.45)

    return float(max(0.1, min(scaled, 1.0)))

# Detects year in text.

DATE_RE = re.compile(r'\b(20\d{2}|19\d{2})\b')
YEAR_PHRASES = [
    r'(\d+)\s*[\+\-–—]?\s*years?\s*(?:of\s*)?(?:experience|exp)?',
    r'over\s*(\d+)\s*years?',
    r'more than\s*(\d+)\s*years?',
    r'(\d+)\s*y(?:rs?)?\b'
]

# Helper to extract tenure from dates in text.

def _tenure_from_dates(text):
    years = sorted({int(y) for y in DATE_RE.findall(text)})
    if not years:
        return None
    # naive heuristic: span from earliest to latest, capped at 40
    earliest, latest = years[0], max(years[-1], datetime.now().year)
    span = max(0, min(40, latest - earliest))
    return span if span > 0 else None

# Function to extract education requirements from job description.
def extract_education_requirements(jd_text):
    """Extract education requirements from job description."""
    return {
        'degree_required': bool(re.search(r'\b(?:bachelor|master|degree|phd)\b', jd_text)),
        'fields': re.findall(r'\b(?:computer science|engineering|mathematics|cs)\b', jd_text),
        'certifications': re.findall(r'\b(?:aws|azure|certified|certification)\b', jd_text),
        'required': True,
        'preferred': False
    }

# Function to extract education achievements from resume.

def extract_education_achievements(resume_text):
    """Extract education achievements from resume."""
    return {
        'has_degree': bool(re.search(r'\b(?:bachelor|master|degree|phd|university|college)\b', resume_text)),
        'fields': re.findall(r'\b(?:computer science|engineering|mathematics|cs)\b', resume_text),
        'certifications': re.findall(r'\b(?:aws|azure|certified|certification)\b', resume_text)
    }

# Function to extract years of experience from text.

def extract_years_experience(text):
    """
    More robust experience extraction.
    """
    patterns = [
        r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
        r'experience.*?(\d+)\+?\s*years?',
        r'(\d+)\+?\s*years?\s*(?:in|with|of)',
        r'(\d+)\+?\s*yrs?\s*(?:experience|exp)',
        r'over\s*(\d+)\s*years?',
        r'more than\s*(\d+)\s*years?'
    ]
    
    years_found = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        years_found.extend([int(match) for match in matches])
    
    if years_found:
        return max(years_found)
    return None

# Normalization for scores.

def smart_normalize(scores):

    scores = np.array(scores, dtype=float)
    
    if len(scores) <= 1:
        return scores
    
    min_score = scores.min()
    max_score = scores.max()
    
    if abs(max_score - min_score) < 1e-6:
        return np.clip(scores, 0.0, 1.0)

    normalized = (scores - min_score) / (max_score - min_score)
    normalized = 0.2 + normalized * 0.8
    
    return normalized