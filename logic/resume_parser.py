#Imports

# For PDF processing
import fitz 
import re 
from transformers import pipeline


# Try to load BERT NER Model , fallsback to QA model if NER fails , else warning prints.
# This is used for extracting names, emails, and phone numbers from resumes.

try:
    ner_pipeline = pipeline(
        "ner", 
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english",
        aggregation_strategy="simple"
    )

    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    
except Exception as e:
    print(f"Warning: Could not load BERT models: {e}")
    ner_pipeline = None
    qa_pipeline = None

# Function to parse resumes 
# If it's a PDF, it extracts text using fitz.
# If it's a TXT file, it reads the content directly.
# It cleans the text and extracts metadata using BERT and regex.
# Returns a dictionary with the cleaned text and metadata.

def parse_resume(file):
    """
    Extract clean text and metadata from a resume file (PDF or TXT).
    Returns a dict with 'text', 'name', 'email', 'phone', 'location'.
    """
    filename = file.name.lower()

    try:
        if filename.endswith(".pdf"):
            raw_text = extract_text_from_pdf(file)
        elif filename.endswith(".txt"):
            raw_text = extract_text_from_txt(file)
        else:
            raise ValueError("Unsupported file format. Please upload a .pdf or .txt file.")

        cleaned_text = clean_text(raw_text)
        metadata = extract_metadata_with_bert(cleaned_text)

        return {
            "text": cleaned_text,
            **metadata
        }

    except Exception as e:
        return {
            "text": "",
            "name": "Error",
            "email": "N/A",
            "phone": "N/A",
            "error": str(e)
        }

# Function to extract text from PDF using fitz.
# Reads each page and concatenates the text.
# Returns the full text as a single string.

def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

# Function to extract text from TXT files
# Reads the file content and decodes it to UTF-8.
# Strips whitespace and returns the cleaned text.

def extract_text_from_txt(file):
    return file.read().decode("utf-8").strip()

# Function to clean text by removing unwanted characters and formatting.

def clean_text(text):
    # Remove common footer/page patterns
    text = re.sub(r'Page\s*\d+(\s*of\s*\d+)?', '', text, flags=re.IGNORECASE)
    
    # Remove bullet symbols
    text = re.sub(r'[•·◦–]', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# Function to extract name using BERT NER and fallback methods.
# Tries to use BERT NER, then a question-answering approach, and finally a heuristic method.
# Returns the  name found or "Unknown" if no valid name is found.

def extract_name_with_bert(text):
    names = []
    
    # BERT NER
    if ner_pipeline:
        try:
            text_sample = text[:500]
            entities = ner_pipeline(text_sample)
            
            for entity in entities:
                if entity['entity_group'] == 'PER' and entity['score'] > 0.9:
                    names.append(entity['word'].strip())
        except:
            pass
    
    # QA
    if qa_pipeline and not names:
        try:
            result = qa_pipeline(
                question="What is the person's full name?", 
                context=text[:800]
            )
            if result['score'] > 0.5:
                names.append(result['answer'].strip())
        except:
            pass
    
    # Heuristics
    if not names:
        names.extend(extract_name_heuristic(text))
    
    # Clean and validate names
    valid_names = []
    for name in names:
        cleaned_name = clean_name(name)
        if is_valid_name(cleaned_name):
            valid_names.append(cleaned_name)
    
    return valid_names[0] if valid_names else "Unknown"


# Function to extract name using heuristic methods.
# Looks for capitalized words in the first few lines of the text by leaving out common keywords.
# Returns a list of potential names.

def extract_name_heuristic(text):
    """
    Heuristic-based name extraction as fallback.
    """
    lines = text.split('\n')
    potential_names = []
    for line in lines[:5]:
        line = line.strip()
        if line and not any(keyword in line.lower() for keyword in 
                           ['resume', 'cv', 'curriculum', 'vitae', '@', 'phone', 'email', 'address']):
        
            words = line.split()
            if 2 <= len(words) <= 4 and all(word[0].isupper() for word in words):
                potential_names.append(line)
    
    return potential_names

# Function to clean extracted names by removing titles and extra whitespace.

def clean_name(name):
    titles = ['mr', 'mrs', 'ms', 'dr', 'prof', 'professor', 'sir', 'madam']
    words = name.split()
    cleaned_words = [word for word in words if word.lower().rstrip('.') not in titles]
    
    return ' '.join(cleaned_words).strip()

# Function to check if a name is valid.

def is_valid_name(name):
    if not name or len(name) < 2:
        return False
    
    # Should not contain numbers, emails, or phone patterns
    if re.search(r'\d{3,}|@|\+\d', name):
        return False
    
    # Should have at least one letter
    if not re.search(r'[a-zA-Z]', name):
        return False
    
    words = name.split()
    
    # Should have 2-4 words typically
    if len(words) < 1 or len(words) > 4:
        return False
    
    return True


# Function to extract metadata using BERT.

def extract_metadata_with_bert(text):

    # Extract name using BERT
    name = extract_name_with_bert(text)
    
    # Email extraction using regex
    email_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
    email = email_match.group() if email_match else "N/A"
    
    # Phone number extraction using regex
    phone_patterns = [
        r'\((\d{3})\)\s*(\d{3})-(\d{4})', 
        r'(\d{3})-(\d{3})-(\d{4})',     
        r'(\d{3})\.(\d{3})\.(\d{4})', 
        r'(\d{3})\s+(\d{3})\s+(\d{4})',  
        r'\+?1?[-.\s]?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})', 
        r'(\d{10})'
    ]
    
    phone = "N/A"
    for pattern in phone_patterns:
        match = re.search(pattern, text)
        if match:
            groups = match.groups()
            if len(groups) == 3:
                phone = f"({groups[0]}) {groups[1]}-{groups[2]}"
            elif len(groups) == 1 and len(groups[0]) == 10:
                num = groups[0]
                phone = f"({num[:3]}) {num[3:6]}-{num[6:]}"
            break
    
    return {
        "name": name,
        "email": email,
        "phone": phone
    }
