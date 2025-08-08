# Imports

import re
from collections import Counter
from datetime import date
from calendar import month_abbr

# Helper to extract tenure from dates in text.
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

# Generic noise terms
GENERIC_TERMS = {
    "experience","experiences","skills","skill","team","teams","work","works","project","projects",
    "customer","customers","client","clients","service","services","role","roles","background"
}

# Function to extract years of experience from text.
def _parse_month_token(mtok: str):
    if not mtok:
        return 1
    mtok = mtok.strip().lower()
    if mtok in ("present","current","now"):
        return None  # special: means "today"
    return MONTHS.get(mtok, 1)

# Function to convert year and month to a date object.
def _as_date(y: int, m: int|None) -> date:
    if m is None:
        today = date.today()
        return date(today.year, today.month, 1)
    return date(int(y), int(m), 1)

# Function to merge overlapping date intervals.
def _merge_intervals(intervals):
    if not intervals:
        return []
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

# Function to calculate months between two dates.
def _months_between(s: date, e: date) -> int:
    return (e.year - s.year) * 12 + (e.month - s.month) + 1

# Function to extract years of experience from text.
def _fmt_years(y):
    if y is None:
        return "—"
    return f"{round(y,1)}y" if y < 8 else f"{int(round(y))}y"

# Explains why a candidate is a good fit for a job.
class UniversalInsightEngine:


    def __init__(self):
        self.patterns = {
            'leadership': [
                r'\b(?:led|managed|supervised|directed|coordinated|oversaw)\b',
                r'\b(?:team lead|manager|supervisor|director|head of)\b',
                r'\b(?:managed team|led project|supervised staff)\b'
            ],
            'growth_learning': [
                r'\b(?:certified|certification|course|training|workshop|seminar)\b',
                r'\b(?:learned|developed|improved|enhanced|upgraded)\b',
                r'\b(?:continuous learning|professional development|upskilling)\b'
            ],
            'achievement': [
                r'\b(?:achieved|accomplished|delivered|exceeded|improved)\b',
                r'\b(?:award|recognition|promoted|successful)\b',
                r'\b(?:\d+%\s*(?:increase|improvement|growth|reduction))\b'
            ],
            'collaboration': [
                r'\b(?:collaborated|partnered|worked with|cross-functional)\b',
                r'\b(?:team player|stakeholder|client facing|customer)\b'
            ],
            'technical_depth': [
                r'\b(?:expert|proficient|advanced|specialist|experienced)\b',
                r'\b(?:deep knowledge|extensive experience|mastery)\b'
            ],
            'project_delivery': [
                r'\b(?:delivered|completed|implemented|launched|deployed)\b',
                r'\b(?:project|initiative|program|solution|system)\b'
            ]
        }

    def generate_insights(self, candidate_data, job_data, scores):
        insights = []

        # Skills
        skills_insight = self._analyze_skills_alignment_universal(
            candidate_data['text'], job_data['text'], scores
        )
        if skills_insight:
            insights.append(skills_insight)

        # Experience
        exp_insight = self._analyze_experience_fit_universal(
            candidate_data['text'], job_data['text'], scores
        )
        if exp_insight:
            insights.append(exp_insight)

        # Professional qualities
        qualities_insight = self._analyze_professional_qualities(
            candidate_data['text'], scores
        )
        if qualities_insight:
            insights.append(qualities_insight)

        # Content depth
        depth_insight = self._analyze_content_depth(
            candidate_data['text'], job_data['text'], scores
        )
        if depth_insight:
            insights.append(depth_insight)

        # Overall fit
        fit_insight = self._analyze_overall_fit(scores)
        if fit_insight:
            insights.append(fit_insight)

        return insights

    # Functions to analyze different aspects of the candidadate.

    def _analyze_skills_alignment_universal(self, resume_text, jd_text, scores):
        jd_terms = self._extract_important_terms(jd_text, top_k=30)
        rez_terms = self._extract_important_terms(resume_text, top_k=40)

        overlap = set(jd_terms.keys()) & set(rez_terms.keys())
        overlap_score = len(overlap) / max(len(jd_terms), 1)

        ranked = sorted(overlap, key=lambda t: jd_terms.get(t, 0), reverse=True)
        top_matches = [t for t in ranked if t not in GENERIC_TERMS and len(t) > 3][:3]

        if overlap_score >= 0.4 and top_matches:
            return {'type': 'strength', 'title': 'Strong Domain Alignment',
                    'detail': f"Key matching areas: {', '.join(top_matches)}"}
        elif overlap_score >= 0.2 and top_matches:
            return {'type': 'neutral', 'title': 'Moderate Domain Overlap',
                    'detail': f"Relevant overlap in: {', '.join(top_matches)}"}
        else:
            return {'type': 'caution', 'title': 'Limited Domain Overlap',
                    'detail': "Resume focus differs; may require domain onboarding"}
        
    # Function to analyze experience fit with job description.

    def _analyze_experience_fit_universal(self, resume_text, jd_text, scores):

        jd_seniority = self._assess_seniority_level(jd_text)
        rez_seniority = self._assess_seniority_level(resume_text)

        jd_years = extract_years_experience(jd_text)
        rez_years = extract_years_experience(resume_text)
        if rez_years is not None and jd_years is not None:
            if rez_years >= jd_years:
                return {
                    'type': 'strength',
                    'title': 'Experience Requirement Met',
                    'detail': 'Meets or exceeds the required years of experience'
                }
            elif (jd_years - rez_years) <= 1.0:
                return {
                    'type': 'neutral',
                    'title': 'Close Experience Match',
                    'detail': 'Slight shortfall; close to the required experience'
                }
            else:
                return {
                    'type': 'caution',
                    'title': 'Experience Gap',
                    'detail': 'Does not meet the required years of experience'
                }

        if rez_seniority is not None and jd_seniority is not None:
            levels = ['junior', 'mid-level', 'senior', 'executive']
            if rez_seniority >= jd_seniority:
                return {
                    'type': 'strength',
                    'title': 'Appropriate Seniority Level',
                    'detail': f"{levels[rez_seniority].title()} level matches the role’s seniority"
                }
            else:
                return {
                    'type': 'caution',
                    'title': 'Seniority Gap',
                    'detail': 'Overall seniority appears below the role’s level'
                }

        return {
            'type': 'neutral',
            'title': 'Experience Unclear',
            'detail': 'Years of experience are not clearly specified'
        }


    # Function to analyze professional qualities based on patterns.

    def _analyze_professional_qualities(self, resume_text, scores):
        counts = {}
        for quality, patterns in self.patterns.items():
            c = 0
            for pat in patterns:
                c += len(re.findall(pat, resume_text, re.IGNORECASE))
            if c > 0:
                counts[quality] = c

        if len(counts) >= 2:
            top = sorted(counts.items(), key=lambda x: -x[1])[:2]
            names = [q.replace('_', ' ').title() for q, _ in top]
            detail = (f"Shows {', '.join(names[:-1])} and {names[-1]} experience"
                      if len(names) > 1 else f"Shows {names[0]} experience")
            return {'type': 'strength', 'title': 'Strong Professional Profile', 'detail': detail}
        elif len(counts) == 1:
            q = next(iter(counts.keys()))
            name = q.replace('_', ' ').title()
            return {'type': 'neutral', 'title': f'{name} Background',
                    'detail': f"Demonstrates {name.lower()} capabilities"}

    def _analyze_content_depth(self, resume_text, jd_text, scores):
        resume_length = len(resume_text.split())
        indicators = [
            r'\b\d+\s*(?:years?|months?)\b',
            r'\b\d+%\b',
            r'\b\$\d[\d,]*\b',
            r'\b\d+\s*(?:people|team|staff|members)\b',
        ]
        specificity = sum(len(re.findall(p, resume_text, re.IGNORECASE)) for p in indicators)

        if specificity >= 5:
            return {'type': 'strength', 'title': 'Detailed Professional Background',
                    'detail': 'Resume provides specific metrics and concrete examples'}
        elif resume_length < 100:
            return {'type': 'caution', 'title': 'Limited Detail Provided',
                    'detail': 'Resume is brief — may need more information for full assessment'}

    # Function to analyze overall fit based on scores.

    def _analyze_overall_fit(self, scores):
        total_score = scores.get('score', 0.0)
        base_score = scores.get('base', 0.0)
        keyword_score = scores.get('keyword', 0.0)
        exp_score = scores.get('exp_score', 0.0)

        strong, weak = [], []
        if base_score >= 0.6: strong.append('semantic alignment')
        elif base_score < 0.3: weak.append('semantic match')

        if keyword_score >= 0.6: strong.append('keyword relevance')
        elif keyword_score < 0.3: weak.append('keyword alignment')

        if exp_score >= 0.6: strong.append('experience level')
        elif exp_score < 0.3: weak.append('experience match')

        if total_score >= 0.7:
            return {'type': 'strength', 'title': 'Excellent Overall Match',
                    'detail': f"High scores across {len(strong)} key areas: {', '.join(strong)}"}
        elif weak:
            return {'type': 'caution', 'title': 'Areas for Consideration',
                    'detail': f"Lower alignment in: {', '.join(weak[:2])}"}

    # Helper functions for extracting important terms and assessing seniority level.

    def _extract_important_terms(self, text, top_k=20):
        """TF-like term frequency (domain-agnostic)."""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', (text or "").lower())
        stop = {
            'the','and','for','are','but','not','you','all','can','had','her','was','one','our','out','day','get',
            'has','him','his','how','its','may','new','now','old','see','two','who','boy','did','man','way','she',
            'use','your','said','each','make','most','over','such','very','what','with','have','from','they','know',
            'want','been','good','much','some','time','will','when','come','here','just','like','long','many','take',
            'than','them','well','were'
        }
        toks = [w for w in words if w not in stop and len(w) > 3]
        counts = Counter(toks)
        return dict(counts.most_common(top_k))

    def _assess_seniority_level(self, text):
        """0=junior, 1=mid, 2=senior, 3=executive; None if unknown."""
        tl = (text or "").lower()
        executive_terms = ['director','vp','vice president','ceo','cto','head of','chief']
        senior_terms = ['senior','lead','principal','architect','specialist']
        mid_terms = ['mid','intermediate','analyst','associate']
        junior_terms = ['junior','entry','trainee','intern','assistant']
        if any(t in tl for t in executive_terms): return 3
        if any(t in tl for t in senior_terms): return 2
        if any(t in tl for t in mid_terms): return 1
        if any(t in tl for t in junior_terms): return 0
        return None


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


# Format analysis for web.

def format_insights_for_streamlit(insights):
    if not insights:
        return "No specific insights generated."
    icon_map = {'strength': '✅', 'caution': '⚠️', 'concern': '❌', 'neutral': 'ℹ️'}
    lines = []
    for ins in insights:
        icon = icon_map.get(ins.get('type'), '•')
        title = ins.get('title', 'Insight')
        detail = ins.get('detail', '')
        lines.append(f"{icon} **{title}**: {detail}")
    return "\n\n".join(lines)


# Wrapper 
def generate_universal_insights(candidate_name, candidate_text, job_description, scores):
    engine = UniversalInsightEngine()
    candidate_data = {'text': candidate_text, 'name': candidate_name}
    job_data = {'text': job_description}
    insights = engine.generate_insights(candidate_data, job_data, scores)
    return format_insights_for_streamlit(insights)
