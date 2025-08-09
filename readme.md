Candidate Recommendation Engine : 

    An app that ranks resumes against job description and explains why the candidate is a good fit.

1. Approach : 

    - Resumes are parsed with PyMuPdf and cleaned.
    - Name, email and phone number are extracted using BERT NER with fallback options.
    - Job description and resumes are embedded using all-mpnet-base-v2 (Sentence Transformers).
    - Cosine similarity is comouted to give the base fit score.
    - Extract weighted JD terms from key sections and measures their presence in resumes.
    - Extract years of experience from date ranges and phrases, compares with JD requirements, and scores accordingly.
    - Identify degrees, fields of study, and certifications in resumes and aligns them with JD expectations.
    - Combine semantic, keyword, experience, and education scores into an overall match percentage.
    - Generate summaries highlighting skill match, experience match, and why the candidate is a good fit.

2. Assumptions : 

    - The model assumes all the content is in english.
    - Works best with clear section headers. Noisy or scanned PDFs may affect extraction.
    - One JD is evaluated at a time against multiple resumes.
    - All processing is local, no data is stored between sessions.
    - Image pdfs may reduce accuracy.
    - Seperate candidates resume by '---'

3. Tech Stack and Models : 

    - Frontend :  Streamlit
    - Embedding Model (primary): sentence-transformers/all-mpnet-base-v2
    - Embedding Model (fallback): sentence-transformers/all-MiniLM-L6-v2
    - NER (names/locations): dbmdz/bert-large-cased-finetuned-conll03-english (Transformers pipeline)
    - QA backup for name extraction: distilbert-base-cased-distilled-squad (Transformers pipeline)-
    
    - Libraries:
        * sentence-transformers, transformers, torch
        * scikit-learn (cosine similarity)
        * numpy
        * PyMuPDF (fitz) for PDF text extraction
        * re (regex) for parsing (emails, phones, dates, education)
    
4. How to run : 

    - Create a virtual environment : 
        * python -m venv .venv
        * source .venv/bin/activate
        * Windows : .venv\Scripts\activate
    - Install dependencies : 
        pip install -r requirements.txt
    - Run the app : 
        streamlit run app.py

5. Note : 

    - The application deployed on Streamlit Cloud runs successfully with the following requirements on requirements.txt
    - The package versions in requirements.txt have been tested and verified to work on all computers when running the hosted Streamlit website.
    - If you download the project files from GitHub and run them locally, package versions may need adjustment depending on your OS, Python version, or hardware.
    - If you encounter issues with requirements and imports, please :
        * use Python 3.11 or later
        * Upgrade pip, setuptools, and wheel before installation - pip install --upgrade pip setuptools wheel
        * Installing the dependencies with - pip install -r requirements.txt
        * If errors persist, adjust specific package versions to match your local environment.
