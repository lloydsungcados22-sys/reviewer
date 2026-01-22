"""
PH Criminology Exam Reviewer - Streamlit Web App
==================================================
A comprehensive exam reviewer with PDF upload, question generation,
free limits, paywall, premium codes, and admin panel.

Features:
- PDF text extraction and question generation (LLM or rule-based)
- Free limit: 15 questions
- Paywall: ₱50 via GCash
- Premium codes: 100 questions
- Admin panel for code generation and receipt validation
"""

import os
import sqlite3
import hashlib
import secrets
import string
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import json
import io

import streamlit as st
import pandas as pd

# PDF processing
fitz_module = None
pdfplumber_module = None
try:
    import fitz as fitz_module  # PyMuPDF
    PDF_AVAILABLE = True
    PDF_LIB = "fitz"
except ImportError:
    fitz_module = None
    try:
        import pdfplumber as pdfplumber_module
        PDF_AVAILABLE = True
        PDF_LIB = "pdfplumber"
    except ImportError:
        PDF_AVAILABLE = False
        PDF_LIB = None
        pdfplumber_module = None

# Word document processing
Document_class = None
try:
    from docx import Document as Document_class
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    Document_class = None

# LLM for question generation (optional)
try:
    import openai
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

# Page config
st.set_page_config(
    page_title="PH Criminology Exam Reviewer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
FREE_QUESTION_LIMIT = 15
PREMIUM_QUESTION_LIMIT = 115  # 15 free + 100 additional = 115 total
PAYMENT_AMOUNT = 50
GCASH_NUMBER = "0927 159 5709"
GCASH_NAME = "M**K L***D S."
RECEIPT_EMAIL = "criminologysupp@gmail.com"

# ============================================================================
# DATABASE SETUP
# ============================================================================

def init_database():
    """Initialize SQLite database for codes and receipts"""
    db_path = os.path.join("data", "reviewer.db")
    os.makedirs("data", exist_ok=True)
    
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    
    # Premium codes table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS premium_codes (
            code TEXT PRIMARY KEY,
            status TEXT DEFAULT 'active',
            expiry_date TEXT,
            max_uses INTEGER DEFAULT 1,
            uses_left INTEGER DEFAULT 1,
            created_at TEXT,
            created_by TEXT
        )
    """)
    
    # Code usage tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS code_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT,
            used_at TEXT,
            session_id TEXT,
            FOREIGN KEY (code) REFERENCES premium_codes(code)
        )
    """)
    
    # Payment receipts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS payment_receipts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT,
            email TEXT,
            gcash_reference TEXT,
            receipt_filename TEXT,
            status TEXT DEFAULT 'Pending',
            submitted_at TEXT,
            reviewed_at TEXT,
            reviewed_by TEXT,
            notes TEXT
        )
    """)
    
    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            access_level TEXT DEFAULT 'Free',
            questions_answered INTEGER DEFAULT 0,
            created_at TEXT,
            last_login TEXT,
            premium_code_used TEXT,
            is_admin INTEGER DEFAULT 0
        )
    """)
    
    # PDF management table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pdf_resources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            filepath TEXT,
            is_premium_only INTEGER DEFAULT 0,
            use_for_ai_generation INTEGER DEFAULT 1,
            uploaded_at TEXT,
            uploaded_by TEXT,
            description TEXT
        )
    """)
    
    conn.commit()
    return conn

# Initialize database
db_conn = init_database()

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        "user_email": None,
        "user_logged_in": False,
        "user_access_level": "Free",
        "questions_answered": 0,
        "premium_active": False,
        "premium_code_used": None,
        "current_questions": [],
        "current_question_index": 0,
        "answers": {},
        "score": 0,
        "pdf_text": "",
        "pdf_name": "",
        "admin_logged_in": False,
        "uploaded_pdfs": [],
        "selected_pdf": None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================================================================
# PDF PROCESSING
# ============================================================================

def extract_text_from_docx(docx_file) -> Tuple[str, str]:
    """Extract text from Word document"""
    try:
        if not DOCX_AVAILABLE or Document_class is None:
            return "", ""
        
        # Get filename
        if hasattr(docx_file, 'name'):
            filename = docx_file.name
        elif isinstance(docx_file, str):
            filename = os.path.basename(docx_file)
        else:
            filename = "unknown.docx"
        
        if isinstance(docx_file, str):
            # File path
            doc = Document_class(docx_file)
        else:
            # File object
            docx_file.seek(0)
            doc = Document_class(io.BytesIO(docx_file.read()))
        
        # Extract text from all paragraphs
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        
        # Clean text
        text = " ".join(text.split())
        return text, filename
    except ImportError:
        return "", ""
    except Exception as e:
        st.error(f"Error extracting Word document: {str(e)}")
        return "", ""

def extract_text_from_pdf(pdf_file) -> Tuple[str, str]:
    """Extract text from PDF file"""
    try:
        if not PDF_AVAILABLE:
            return "", ""
        
        # Get filename
        if hasattr(pdf_file, 'name'):
            filename = pdf_file.name
        elif isinstance(pdf_file, str):
            filename = os.path.basename(pdf_file)
        else:
            filename = "unknown.pdf"
        
        if PDF_LIB == "pdfplumber" and pdfplumber_module:
            # pdfplumber can handle file objects or paths
            if isinstance(pdf_file, str):
                with pdfplumber_module.open(pdf_file) as pdf:
                    text = "\n".join([page.extract_text() or "" for page in pdf.pages])
            else:
                # File object - need to reset position
                pdf_file.seek(0)
                with pdfplumber_module.open(pdf_file) as pdf:
                    text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        elif PDF_LIB == "fitz" and fitz_module:
            # PyMuPDF (fitz)
            if isinstance(pdf_file, str):
                # File path
                pdf_doc = fitz_module.open(pdf_file)
            else:
                # File object
                pdf_file.seek(0)
                pdf_doc = fitz_module.open(stream=pdf_file.read(), filetype="pdf")
            text = "\n".join([page.get_text() for page in pdf_doc])
            pdf_doc.close()
        else:
            return "", ""
        
        # Clean text
        text = " ".join(text.split())
        return text, filename
    except Exception as e:
        st.error(f"Error extracting PDF: {str(e)}")
        return "", ""

def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """Split text into chunks for processing"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# ============================================================================
# QUESTION GENERATION
# ============================================================================

def generate_questions_llm(text: str, difficulty: str, num_questions: int, question_types: List[str]) -> List[Dict]:
    """Generate questions using OpenAI API if available"""
    if not LLM_AVAILABLE:
        return []
    
    # Try to get API key from secrets or environment
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
    except Exception:
        # Secrets file doesn't exist, try environment variable
        api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        return []
    
    try:
        openai.api_key = api_key
        client = openai.OpenAI(api_key=api_key)
        
        prompt = f"""Generate {num_questions} {difficulty} level exam questions based on the following criminology review material.

Text content:
{text[:4000]}  # Limit context

Difficulty: {difficulty}
Question types: {', '.join(question_types)}

For each question, provide:
- question: The question text
- type: One of {question_types}
- options: List of 4 options (for MCQ) or ['True', 'False'] (for True/False) or empty list (for identification)
- correct_answer: The correct answer
- explanation: Brief explanation

Return as JSON array of question objects.
"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )
        
        # Parse response (simplified - would need proper JSON parsing)
        # For now, return empty and fall back to rule-based
        return []
    except Exception as e:
        st.warning(f"LLM generation failed: {str(e)}. Using rule-based generation.")
        return []

def generate_questions_rule_based(text: str, difficulty: str, num_questions: int, question_types: List[str]) -> List[Dict]:
    """Generate questions using rule-based approach"""
    chunks = chunk_text(text, chunk_size=500)
    if not chunks:
        return []
    
    questions = []
    words = text.split()
    
    # Simple question generation based on text patterns
    for i in range(min(num_questions, len(chunks) * 2)):
        chunk = chunks[i % len(chunks)]
        chunk_words = chunk.split()
        
        if len(chunk_words) < 5:
            continue
        
        q_type = secrets.choice(question_types) if question_types else "MCQ"
        
        if q_type == "True/False":
            # Create True/False question
            question_text = f"True or False: {chunk[:200]}"
            questions.append({
                "question": question_text,
                "type": "True/False",
                "options": ["True", "False"],
                "correct_answer": "True",  # Simplified
                "explanation": chunk[:150]
            })
        
        elif q_type == "Identification":
            # Create fill-in-the-blank
            if len(chunk_words) > 3:
                blank_word = chunk_words[2]
                question_text = chunk.replace(blank_word, "_____", 1)
                questions.append({
                    "question": question_text[:300],
                    "type": "Identification",
                    "options": [],
                    "correct_answer": blank_word,
                    "explanation": chunk[:150]
                })
        
        else:  # MCQ
            # Create multiple choice
            question_text = f"What is described in the following: {chunk[:150]}?"
            correct = chunk_words[0] if chunk_words else "Option A"
            options = [correct, "Option B", "Option C", "Option D"]
            secrets.SystemRandom().shuffle(options)
            
            questions.append({
                "question": question_text,
                "type": "MCQ",
                "options": options,
                "correct_answer": correct,
                "explanation": chunk[:150]
            })
        
        if len(questions) >= num_questions:
            break
    
    return questions[:num_questions]

def generate_default_dummy_questions(difficulty: str, num_questions: int, question_types: List[str]) -> List[Dict]:
    """Generate default dummy questions when no PDF/document is loaded"""
    # Pre-defined criminology questions pool
    dummy_questions_pool = [
        {
            "question": "What is the primary purpose of the Revised Penal Code in the Philippines?",
            "type": "MCQ",
            "options": ["To define criminal offenses and penalties", "To establish court procedures", "To regulate police operations", "To manage prison systems"],
            "correct_answer": "To define criminal offenses and penalties",
            "explanation": "The Revised Penal Code (Act No. 3815) is the main criminal law that defines offenses and prescribes penalties."
        },
        {
            "question": "True or False: A person is presumed innocent until proven guilty beyond reasonable doubt.",
            "type": "True/False",
            "options": ["True", "False"],
            "correct_answer": "True",
            "explanation": "This is a fundamental principle in criminal law - the presumption of innocence."
        },
        {
            "question": "What does 'nullum crimen, nulla poena sine lege' mean?",
            "type": "Identification",
            "options": [],
            "correct_answer": "No crime, no punishment without law",
            "explanation": "This Latin maxim means that no act is a crime unless it is clearly defined and penalized by law."
        },
        {
            "question": "Which of the following is NOT an element of a crime?",
            "type": "MCQ",
            "options": ["Human being", "Criminal act or omission", "Criminal intent", "Police report"],
            "correct_answer": "Police report",
            "explanation": "The elements of a crime are: human being, criminal act/omission, criminal intent, and the act must be punishable by law."
        },
        {
            "question": "True or False: Attempted felony is punishable under the Revised Penal Code.",
            "type": "True/False",
            "options": ["True", "False"],
            "correct_answer": "True",
            "explanation": "Attempted felony is one of the stages of a crime and is punishable, though with a lesser penalty than consummated felony."
        },
        {
            "question": "What is the maximum penalty for crimes punishable by reclusion perpetua?",
            "type": "MCQ",
            "options": ["20 years and 1 day to 40 years", "Life imprisonment", "6 years and 1 day to 12 years", "12 years and 1 day to 20 years"],
            "correct_answer": "20 years and 1 day to 40 years",
            "explanation": "Reclusion perpetua ranges from 20 years and 1 day to 40 years of imprisonment."
        },
        {
            "question": "Identify the term: The body of law that regulates the apprehension and prosecution of persons accused of crimes.",
            "type": "Identification",
            "options": [],
            "correct_answer": "Criminal procedure",
            "explanation": "Criminal procedure refers to the method prescribed by law for the apprehension and prosecution of persons accused of crimes."
        },
        {
            "question": "True or False: The accused has the right to remain silent and to have counsel during custodial investigation.",
            "type": "True/False",
            "options": ["True", "False"],
            "correct_answer": "True",
            "explanation": "This is a constitutional right under the Miranda doctrine and Philippine Constitution."
        },
        {
            "question": "What is the standard of proof required in criminal cases?",
            "type": "MCQ",
            "options": ["Preponderance of evidence", "Beyond reasonable doubt", "Substantial evidence", "Clear and convincing evidence"],
            "correct_answer": "Beyond reasonable doubt",
            "explanation": "Beyond reasonable doubt is the highest standard of proof required in criminal cases."
        },
        {
            "question": "True or False: A warrant of arrest is required for all arrests.",
            "type": "True/False",
            "options": ["True", "False"],
            "correct_answer": "False",
            "explanation": "Warrantless arrests are allowed in certain circumstances, such as when a crime is committed in the presence of the arresting officer."
        },
        {
            "question": "What is the term for a crime that is both a felony and a violation?",
            "type": "MCQ",
            "options": ["Complex crime", "Compound crime", "Continuing crime", "Composite crime"],
            "correct_answer": "Complex crime",
            "explanation": "A complex crime is a single act that constitutes two or more grave or less grave felonies, or an offense and a violation."
        },
        {
            "question": "Identify: The principle that no person shall be held to answer for a criminal offense without due process of law.",
            "type": "Identification",
            "options": [],
            "correct_answer": "Due process",
            "explanation": "Due process ensures that no person is deprived of life, liberty, or property without following proper legal procedures."
        },
        {
            "question": "True or False: Double jeopardy prevents a person from being tried twice for the same offense.",
            "type": "True/False",
            "options": ["True", "False"],
            "correct_answer": "True",
            "explanation": "Double jeopardy is a constitutional protection that prevents multiple prosecutions for the same offense."
        },
        {
            "question": "What is the minimum age of criminal responsibility in the Philippines?",
            "type": "MCQ",
            "options": ["12 years old", "15 years old", "18 years old", "9 years old"],
            "correct_answer": "15 years old",
            "explanation": "Under Republic Act No. 9344, as amended, the minimum age of criminal responsibility is 15 years old."
        },
        {
            "question": "True or False: Self-defense is a justifying circumstance that exempts a person from criminal liability.",
            "type": "True/False",
            "options": ["True", "False"],
            "correct_answer": "True",
            "explanation": "Self-defense is a justifying circumstance that exempts a person from criminal liability when all its elements are present."
        }
    ]
    
    # Filter by difficulty and question types
    selected_questions = []
    available_types = question_types if question_types else ["MCQ", "True/False", "Identification"]
    
    # Shuffle and select questions
    import random
    random.shuffle(dummy_questions_pool)
    
    for q in dummy_questions_pool:
        if q["type"] in available_types:
            selected_questions.append(q)
            if len(selected_questions) >= num_questions:
                break
    
    # If we need more questions, repeat with different variations
    while len(selected_questions) < num_questions:
        for q in dummy_questions_pool:
            if q["type"] in available_types and len(selected_questions) < num_questions:
                # Create a variation
                new_q = q.copy()
                new_q["question"] = q["question"] + " (Question " + str(len(selected_questions) + 1) + ")"
                selected_questions.append(new_q)
    
    return selected_questions[:num_questions]

def generate_questions(text: str, difficulty: str, num_questions: int, question_types: List[str]) -> List[Dict]:
    """Generate questions using LLM if available, else rule-based, or default dummy questions"""
    # If no text provided, use default dummy questions
    if not text or len(text.strip()) < 50:
        return generate_default_dummy_questions(difficulty, num_questions, question_types)
    
    # Try LLM first
    llm_questions = generate_questions_llm(text, difficulty, num_questions, question_types)
    if llm_questions:
        return llm_questions
    
    # Fall back to rule-based
    rule_based = generate_questions_rule_based(text, difficulty, num_questions, question_types)
    if rule_based:
        return rule_based
    
    # Final fallback to dummy questions
    return generate_default_dummy_questions(difficulty, num_questions, question_types)

# ============================================================================
# PREMIUM CODE MANAGEMENT
# ============================================================================

def generate_premium_code(length: int = 12) -> str:
    """Generate a random premium code"""
    alphabet = string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def create_premium_codes(num_codes: int, length: int, expiry_days: Optional[int], max_uses: int, created_by: str) -> List[str]:
    """Create premium codes in database"""
    codes = []
    cursor = db_conn.cursor()
    
    for _ in range(num_codes):
        code = generate_premium_code(length)
        expiry = None
        if expiry_days:
            expiry = (datetime.now() + timedelta(days=expiry_days)).isoformat()
        
        cursor.execute("""
            INSERT INTO premium_codes (code, status, expiry_date, max_uses, uses_left, created_at, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (code, "active", expiry, max_uses, max_uses, datetime.now().isoformat(), created_by))
        
        codes.append(code)
    
    db_conn.commit()
    return codes

def validate_premium_code(code: str) -> Tuple[bool, str]:
    """Validate a premium code"""
    cursor = db_conn.cursor()
    cursor.execute("""
        SELECT status, expiry_date, uses_left, max_uses
        FROM premium_codes
        WHERE code = ?
    """, (code,))
    
    result = cursor.fetchone()
    if not result:
        return False, "Code not found"
    
    status, expiry_date, uses_left, max_uses = result
    
    if status != "active":
        return False, "Code is inactive"
    
    if expiry_date:
        expiry = datetime.fromisoformat(expiry_date)
        if datetime.now() > expiry:
            return False, "Code has expired"
    
    if uses_left <= 0:
        return False, "Code has no uses left"
    
    return True, "Valid"

def use_premium_code(code: str, session_id: str):
    """Mark a premium code as used"""
    cursor = db_conn.cursor()
    
    # Decrement uses_left
    cursor.execute("""
        UPDATE premium_codes
        SET uses_left = uses_left - 1
        WHERE code = ?
    """, (code,))
    
    # Record usage
    cursor.execute("""
        INSERT INTO code_usage (code, used_at, session_id)
        VALUES (?, ?, ?)
    """, (code, datetime.now().isoformat(), session_id))
    
    db_conn.commit()

# ============================================================================
# USER AUTHENTICATION & MANAGEMENT
# ============================================================================

def login_user(email: str) -> Tuple[bool, Dict]:
    """Login user by email, create if doesn't exist"""
    cursor = db_conn.cursor()
    email = email.strip().lower()
    
    # Check if user exists
    cursor.execute("""
        SELECT email, access_level, questions_answered, premium_code_used, is_admin
        FROM users
        WHERE email = ?
    """, (email,))
    
    result = cursor.fetchone()
    
    if result:
        # User exists, update last login
        cursor.execute("""
            UPDATE users
            SET last_login = ?
            WHERE email = ?
        """, (datetime.now().isoformat(), email))
        db_conn.commit()
        
        return True, {
            "email": result[0],
            "access_level": result[1],
            "questions_answered": result[2],
            "premium_code_used": result[3],
            "is_admin": bool(result[4])
        }
    else:
        # Create new user with Free access
        cursor.execute("""
            INSERT INTO users (email, access_level, questions_answered, created_at, last_login, is_admin)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (email, "Free", 0, datetime.now().isoformat(), datetime.now().isoformat(), 0))
        db_conn.commit()
        
        return True, {
            "email": email,
            "access_level": "Free",
            "questions_answered": 0,
            "premium_code_used": None,
            "is_admin": False
        }

def update_user_access_level(email: str, access_level: str):
    """Update user access level (Free or Premium)"""
    cursor = db_conn.cursor()
    cursor.execute("""
        UPDATE users
        SET access_level = ?
        WHERE email = ?
    """, (access_level, email))
    db_conn.commit()

def update_user_questions_answered(email: str, count: int):
    """Update user's questions answered count"""
    cursor = db_conn.cursor()
    cursor.execute("""
        UPDATE users
        SET questions_answered = questions_answered + ?
        WHERE email = ?
    """, (count, email))
    db_conn.commit()
    
    # Also update session state
    cursor.execute("""
        SELECT questions_answered FROM users WHERE email = ?
    """, (email,))
    result = cursor.fetchone()
    if result:
        st.session_state.questions_answered = result[0]

def get_user_info(email: str) -> Optional[Dict]:
    """Get user information"""
    cursor = db_conn.cursor()
    cursor.execute("""
        SELECT email, access_level, questions_answered, premium_code_used, is_admin
        FROM users
        WHERE email = ?
    """, (email,))
    
    result = cursor.fetchone()
    if result:
        return {
            "email": result[0],
            "access_level": result[1],
            "questions_answered": result[2],
            "premium_code_used": result[3],
            "is_admin": bool(result[4])
        }
    return None

# ============================================================================
# PDF RESOURCE MANAGEMENT
# ============================================================================

def save_pdf_resource(filename: str, filepath: str, is_premium_only: bool, use_for_ai: bool, uploaded_by: str, description: str = ""):
    """Save PDF resource to database"""
    cursor = db_conn.cursor()
    cursor.execute("""
        INSERT INTO pdf_resources (filename, filepath, is_premium_only, use_for_ai_generation, uploaded_at, uploaded_by, description)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (filename, filepath, 1 if is_premium_only else 0, 1 if use_for_ai else 0, 
          datetime.now().isoformat(), uploaded_by, description))
    db_conn.commit()

def get_pdf_resources(premium_only: Optional[bool] = None) -> List[Dict]:
    """Get PDF resources, optionally filtered by premium status"""
    cursor = db_conn.cursor()
    if premium_only is not None:
        cursor.execute("""
            SELECT id, filename, filepath, is_premium_only, use_for_ai_generation, uploaded_at, description
            FROM pdf_resources
            WHERE is_premium_only = ?
            ORDER BY uploaded_at DESC
        """, (1 if premium_only else 0,))
    else:
        cursor.execute("""
            SELECT id, filename, filepath, is_premium_only, use_for_ai_generation, uploaded_at, description
            FROM pdf_resources
            ORDER BY uploaded_at DESC
        """)
    
    results = cursor.fetchall()
    return [
        {
            "id": r[0],
            "filename": r[1],
            "filepath": r[2],
            "is_premium_only": bool(r[3]),
            "use_for_ai_generation": bool(r[4]),
            "uploaded_at": r[5],
            "description": r[6]
        }
        for r in results
    ]

def update_pdf_resource(resource_id: int, is_premium_only: Optional[bool] = None, use_for_ai: Optional[bool] = None):
    """Update PDF resource settings"""
    cursor = db_conn.cursor()
    updates = []
    params = []
    
    if is_premium_only is not None:
        updates.append("is_premium_only = ?")
        params.append(1 if is_premium_only else 0)
    
    if use_for_ai is not None:
        updates.append("use_for_ai_generation = ?")
        params.append(1 if use_for_ai else 0)
    
    if updates:
        params.append(resource_id)
        cursor.execute(f"""
            UPDATE pdf_resources
            SET {', '.join(updates)}
            WHERE id = ?
        """, params)
        db_conn.commit()

def delete_pdf_resource(resource_id: int):
    """Delete PDF resource"""
    cursor = db_conn.cursor()
    cursor.execute("DELETE FROM pdf_resources WHERE id = ?", (resource_id,))
    db_conn.commit()

# ============================================================================
# PAYMENT RECEIPT MANAGEMENT
# ============================================================================

def save_payment_receipt(name: str, email: str, reference: str, filename: str):
    """Save payment receipt to database"""
    cursor = db_conn.cursor()
    cursor.execute("""
        INSERT INTO payment_receipts (full_name, email, gcash_reference, receipt_filename, status, submitted_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (name, email, reference, filename, "Pending", datetime.now().isoformat()))
    db_conn.commit()

# ============================================================================
# UI COMPONENTS - PNP THEME
# ============================================================================

def inject_pnp_theme_css():
    """Inject PNP/tactical theme CSS"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Poppins:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', 'Poppins', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main background - Navy to Charcoal gradient */
    .stApp {
        background: linear-gradient(180deg, #1a2332 0%, #0f1419 100%);
        color: #e0e0e0;
    }
    
    /* Header banner */
    .header-banner {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d4a6b 50%, #1a2332 100%);
        padding: 1.5rem 2rem;
        border-radius: 0;
        border-bottom: 3px solid #d4af37;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        margin-bottom: 2rem;
    }
    
    .header-banner h1 {
        color: #ffffff;
        font-weight: 800;
        font-size: 2rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    
    .header-banner p {
        color: #d4af37;
        font-weight: 600;
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }
    
    .header-badge {
        display: inline-block;
        background: rgba(212, 175, 55, 0.2);
        border: 2px solid #d4af37;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        color: #d4af37;
        font-weight: 700;
        font-size: 0.85rem;
        margin-left: 1rem;
    }
    
    /* Sidebar - Gunmetal panel */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a2332 0%, #0f1419 100%);
        border-right: 2px solid #2d4a6b;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #e0e0e0;
    }
    
    /* Cards/Panels */
    .pnp-card {
        background: rgba(30, 58, 95, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid #d4af37;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .pnp-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(212, 175, 55, 0.3);
    }
    
    .pnp-card h3 {
        color: #d4af37;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 1.2rem;
        margin-top: 0;
        border-bottom: 2px solid #d4af37;
        padding-bottom: 0.5rem;
    }
    
    /* Badges */
    .badge-easy {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.4);
    }
    
    .badge-average {
        background: linear-gradient(135deg, #ffc107 0%, #ff9800 100%);
        color: #1a2332;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(255, 193, 7, 0.4);
    }
    
    .badge-difficult {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(220, 53, 69, 0.4);
    }
    
    .badge-premium {
        background: linear-gradient(135deg, #d4af37 0%, #f4d03f 100%);
        color: #1a2332;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(212, 175, 55, 0.4);
    }
    
    /* Buttons - Navy with gold hover */
    .stButton>button {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d4a6b 100%);
        color: #ffffff;
        border: 2px solid #d4af37;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #2d4a6b 0%, #3d5a7b 100%);
        border-color: #f4d03f;
        box-shadow: 0 6px 20px rgba(212, 175, 55, 0.5);
        transform: translateY(-2px);
    }
    
    /* Alert panels */
    .alert-restricted {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        border: 3px solid #ff6b6b;
        border-radius: 12px;
        padding: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(220, 53, 69, 0.5);
    }
    
    .alert-premium {
        background: linear-gradient(135deg, #d4af37 0%, #f4d03f 100%);
        border: 3px solid #d4af37;
        border-radius: 12px;
        padding: 2rem;
        color: #1a2332;
        text-align: center;
        box-shadow: 0 8px 32px rgba(212, 175, 55, 0.5);
    }
    
    /* Progress bar */
    .progress-container {
        background: #1a2332;
        border-radius: 10px;
        height: 20px;
        overflow: hidden;
        border: 1px solid #2d4a6b;
        margin: 1rem 0;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #d4af37 0%, #f4d03f 100%);
        border-radius: 10px;
        transition: width 0.6s ease;
        box-shadow: 0 0 10px rgba(212, 175, 55, 0.6);
    }
    
    /* System status box */
    .system-status {
        background: rgba(30, 58, 95, 0.8);
        border: 1px solid #2d4a6b;
        border-radius: 12px;
        padding: 1rem;
        margin-top: 2rem;
    }
    
    .system-status h4 {
        color: #d4af37;
        font-weight: 700;
        text-transform: uppercase;
        margin-top: 0;
        font-size: 0.9rem;
        border-bottom: 1px solid #2d4a6b;
        padding-bottom: 0.5rem;
    }
    
    .system-status p {
        color: #e0e0e0;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }
    
    /* Text colors */
    .stMarkdown {
        color: #e0e0e0;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(30, 58, 95, 0.6);
        color: #d4af37;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """Render PNP-style header banner"""
    if st.session_state.user_logged_in:
        mode = st.session_state.user_access_level.upper()
    else:
        mode = "PREMIUM" if st.session_state.premium_active else "FREE"
    badge_html = f'<span class="header-badge">MODE: {mode}</span>'
    
    st.markdown(f"""
    <div class="header-banner">
        <h1>🛡️ PH CRIMINOLOGY EXAM REVIEWER {badge_html}</h1>
        <p>PNP-Style Tech Reviewer Console</p>
    </div>
    """, unsafe_allow_html=True)

def render_premium_lock(feature_name: str, upgrade_message: str = "Upgrade to Premium to unlock this feature"):
    """Render a premium lock indicator"""
    if st.session_state.user_access_level != "Premium":
        st.markdown(f"""
        <div style="background: rgba(220, 53, 69, 0.2); border: 2px solid #dc3545; border-radius: 12px; padding: 1.5rem; text-align: center; margin: 1rem 0;">
            <h3 style="color: #dc3545; margin: 0 0 0.5rem 0;">🔒 {feature_name}</h3>
            <p style="color: #e0e0e0; margin: 0;">{upgrade_message}</p>
            <p style="color: #d4af37; margin: 0.5rem 0 0 0; font-weight: 600;">Go to Premium Access or Payment page to upgrade</p>
        </div>
        """, unsafe_allow_html=True)
        return True
    return False

def render_card(title: str, content_html: str):
    """Render a PNP-style card"""
    st.markdown(f"""
    <div class="pnp-card">
        <h3>{title}</h3>
        <div>{content_html}</div>
    </div>
    """, unsafe_allow_html=True)

def render_badge(text: str, color: str):
    """Render a difficulty badge"""
    badge_class = f"badge-{color.lower()}"
    return f'<span class="{badge_class}">{text}</span>'

# Inject theme
inject_pnp_theme_css()
render_header()

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

with st.sidebar:
    st.markdown("### 🧭 NAVIGATION")
    
    page = st.radio(
        "Select Page",
        ["🏠 Home", "📄 Upload Reviewer", "🧠 Practice Exam", "🔑 Premium Access", "💳 Payment", "🛠️ Admin Panel"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # User Status
    if st.session_state.user_logged_in:
        st.markdown("""
        <div class="system-status">
            <h4>USER STATUS</h4>
            <p>👤 Email: {}</p>
            <p>🔑 Access: {}</p>
            <p>📊 Questions: {}/{}</p>
            <p>📄 PDF Loaded: {}</p>
        </div>
        """.format(
            st.session_state.user_email[:20] + "..." if len(st.session_state.user_email) > 20 else st.session_state.user_email,
            st.session_state.user_access_level,
            st.session_state.questions_answered,
            PREMIUM_QUESTION_LIMIT if st.session_state.user_access_level == "Premium" else FREE_QUESTION_LIMIT,
            "Yes" if st.session_state.pdf_text else "No"
        ), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="system-status">
            <h4>SYSTEM STATUS</h4>
            <p>📄 PDFs Loaded: {}</p>
            <p>❓ Questions Generated: {}</p>
            <p>📊 Remaining: {}/{}</p>
            <p>🔑 Premium: {}</p>
        </div>
        """.format(
            "Yes" if st.session_state.pdf_text else "No",
            len(st.session_state.current_questions),
            st.session_state.questions_answered,
            PREMIUM_QUESTION_LIMIT if st.session_state.premium_active else FREE_QUESTION_LIMIT,
            "ON" if st.session_state.premium_active else "OFF"
        ), unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("For review purposes only. Verify with latest PH laws.")

# ============================================================================
# PAGE: HOME
# ============================================================================

if page == "🏠 Home":
    st.markdown("# 🏠 Welcome to PH Criminology Exam Reviewer")
    
    # User Login Section
    if not st.session_state.user_logged_in:
        render_card("🔐 Login Required", """
        <p>Please login with your email address to access the reviewer.</p>
        <p><strong>New users:</strong> Enter your email to automatically create a Free account.</p>
        """)
        
        with st.form("login_form"):
            email_input = st.text_input("Email Address", placeholder="your.email@example.com", help="Enter your email to login or create an account")
            login_submitted = st.form_submit_button("🔓 Login / Sign Up", type="primary", use_container_width=True)
            
            if login_submitted:
                if email_input and "@" in email_input and "." in email_input.split("@")[1]:
                    success, user_info = login_user(email_input)
                    if success:
                        st.session_state.user_logged_in = True
                        st.session_state.user_email = user_info["email"]
                        st.session_state.user_access_level = user_info["access_level"]
                        st.session_state.questions_answered = user_info["questions_answered"]
                        st.session_state.premium_active = (user_info["access_level"] == "Premium")
                        st.session_state.premium_code_used = user_info.get("premium_code_used")
                        if user_info.get("is_admin"):
                            st.session_state.admin_logged_in = True
                        
                        st.success(f"✅ Welcome, {user_info['email']}! Access Level: {user_info['access_level']}")
                        st.rerun()
                    else:
                        st.error("❌ Login failed. Please try again.")
                else:
                    st.error("❌ Please enter a valid email address.")
        
        st.stop()  # Stop here if not logged in
    
    # User is logged in - show dashboard
    user_info = get_user_info(st.session_state.user_email)
    if user_info:
        st.session_state.user_access_level = user_info["access_level"]
        st.session_state.questions_answered = user_info["questions_answered"]
        st.session_state.premium_active = (user_info["access_level"] == "Premium")
    
    # User profile card
    access_badge = "🔑 PREMIUM" if st.session_state.user_access_level == "Premium" else "🆓 FREE"
    render_card(f"👤 User Profile - {access_badge}", f"""
    <p><strong>Email:</strong> {st.session_state.user_email}</p>
    <p><strong>Access Level:</strong> {st.session_state.user_access_level}</p>
    <p><strong>Questions Answered:</strong> {st.session_state.questions_answered}</p>
    """)
    
    if st.button("🚪 Logout", type="secondary"):
        st.session_state.user_logged_in = False
        st.session_state.user_email = None
        st.session_state.user_access_level = "Free"
        st.session_state.admin_logged_in = False
        st.rerun()
    
    st.markdown("---")
    
    render_card("📋 About This App", """
    <p>This application helps you prepare for the Philippine Criminology Exam by:</p>
    <ul>
        <li>📄 Uploading or selecting reviewer PDFs</li>
        <li>🧠 Generating practice questions automatically</li>
        <li>📊 Tracking your progress and scores</li>
        <li>🔑 Unlocking premium features with codes</li>
    </ul>
    """)
    
    render_card("🎯 Getting Started", """
    <ol>
        <li><strong>Upload/Select Document:</strong> Go to "Upload Reviewer" page and select or upload your review materials</li>
        <li><strong>Generate Questions:</strong> Choose difficulty level and question types</li>
        <li><strong>Practice:</strong> Answer questions and track your progress</li>
        <li><strong>Upgrade:</strong> Unlock 100 questions with premium code or payment</li>
    </ol>
    """)
    
    render_card("⚠️ Important Notice", """
    <p><strong>For review purposes only.</strong> Always verify with official references and latest PH laws.</p>
    <p>This app generates questions based on uploaded PDF content. Always cross-reference with official materials.</p>
    """)
    
    # Quick stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Questions Answered", st.session_state.questions_answered)
    with col2:
        st.metric("Current Mode", st.session_state.user_access_level.upper())
    with col3:
        max_q = PREMIUM_QUESTION_LIMIT if st.session_state.user_access_level == "Premium" else FREE_QUESTION_LIMIT
        remaining = max_q - st.session_state.questions_answered
        st.metric("Remaining", max(0, remaining))

# ============================================================================
# PAGE: UPLOAD REVIEWER
# ============================================================================

elif page == "📄 Upload Reviewer":
    st.markdown("# 📄 Upload / Select Reviewer Document")
    
    # Check if user is logged in
    if not st.session_state.user_logged_in:
        st.warning("⚠️ Please login first on the Home page to access this feature.")
        st.stop()
    
    # Option 1: Admin-managed PDFs
    render_card("📚 Admin-Managed PDFs", """
    <p>PDFs uploaded by administrators. Premium PDFs are marked with 🔒.</p>
    """)
    
    pdf_resources = get_pdf_resources()
    if pdf_resources:
        for pdf in pdf_resources:
            col1, col2 = st.columns([3, 1])
            with col1:
                premium_label = "🔒 Premium Only" if pdf['is_premium_only'] else "🆓 Free"
                ai_label = "🤖 AI Enabled" if pdf['use_for_ai_generation'] else ""
                st.write(f"**{pdf['filename']}** - {premium_label} {ai_label}")
                if pdf.get('description'):
                    st.caption(pdf['description'])
            with col2:
                if pdf['is_premium_only'] and st.session_state.user_access_level != "Premium":
                    st.button("🔒 Premium Only", disabled=True, key=f"lock_{pdf['id']}")
                else:
                    if os.path.exists(pdf['filepath']):
                        with open(pdf['filepath'], "rb") as f:
                            file_data = f.read()
                            st.download_button(
                                "📥 Download",
                                data=file_data,
                                file_name=pdf['filename'],
                                mime="application/pdf",
                                key=f"dl_{pdf['id']}"
                            )
                    else:
                        st.error("File not found.")
    else:
        st.info("No admin-managed PDFs available yet.")
    
    st.markdown("---")
    
    # Option 2: Use sample PDFs
    render_card("📚 Sample Reviewers", """
    <p>Use built-in sample PDFs for practice.</p>
    """)
    
    sample_pdfs = []
    sample_dir = "sample_pdfs"
    if os.path.exists(sample_dir):
        sample_pdfs = [f for f in os.listdir(sample_dir) if f.lower().endswith('.pdf')]
    
    if sample_pdfs:
        selected_sample = st.selectbox("Select Sample PDF", ["None"] + sample_pdfs)
        if selected_sample != "None":
            sample_path = os.path.join(sample_dir, selected_sample)
            text, name = extract_text_from_pdf(sample_path)
            if text:
                st.session_state.pdf_text = text
                st.session_state.pdf_name = name
                st.success(f"✅ Loaded: {name}")
    else:
        st.info("No sample PDFs found. Please upload your own document or run `python create_sample_pdfs.py` to generate sample PDFs.")
    
    st.markdown("---")
    
    # Option 2: Upload PDF or Word Document
    render_card("📤 Upload Your Document", """
    <p>Upload your own reviewer PDF or Word document files.</p>
    """)
    
    uploaded_file = st.file_uploader("Choose PDF or Word document", type=["pdf", "docx", "doc"], help="Upload your criminology reviewer PDF or Word document")
    
    if uploaded_file:
        file_ext = uploaded_file.name.lower().split('.')[-1] if uploaded_file.name else ""
        
        if file_ext == "pdf":
            text, name = extract_text_from_pdf(uploaded_file)
        elif file_ext in ["docx", "doc"]:
            text, name = extract_text_from_docx(uploaded_file)
            if not text and not name:
                # Error already shown in extract_text_from_docx
                pass
        else:
            st.error("Unsupported file type. Please upload PDF or Word document (.docx)")
            text, name = "", ""
        
        if text:
            st.session_state.pdf_text = text
            st.session_state.pdf_name = name
            st.success(f"✅ Successfully loaded: {name}")
            
            # Preview
            with st.expander("📖 Document Preview (First 1000 characters)"):
                st.text(text[:1000] + "..." if len(text) > 1000 else text)
            
            # Stats
            word_count = len(text.split())
            char_count = len(text)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Word Count", f"{word_count:,}")
            with col2:
                st.metric("Character Count", f"{char_count:,}")
    
    st.markdown("---")
    
    # Option 3: Use default dummy questions
    render_card("🎲 Use Default Questions", """
    <p>Practice with pre-loaded criminology questions without uploading a document.</p>
    """)
    
    if st.button("🎯 Use Default Questions", type="secondary", use_container_width=True):
        st.session_state.pdf_text = "DEFAULT_DUMMY_QUESTIONS"  # Marker for dummy questions
        st.session_state.pdf_name = "Default Criminology Questions"
        st.success("✅ Default questions enabled! You can now go to Practice Exam page.")
        st.info("💡 Default questions will be used automatically when generating practice exams.")

# ============================================================================
# PAGE: PRACTICE EXAM
# ============================================================================

elif page == "🧠 Practice Exam":
    st.markdown("# 🧠 Practice Exam")
    
    # Check if user is logged in
    if not st.session_state.user_logged_in:
        st.warning("⚠️ Please login first on the Home page to access Practice Exam.")
        st.stop()
    
    # Check if PDF is loaded or default questions enabled
    if not st.session_state.pdf_text:
        st.info("💡 No document loaded. Default dummy questions will be used for practice.")
        # Allow continuing with dummy questions
    
    # Check question limit based on user access level
    max_questions = PREMIUM_QUESTION_LIMIT if st.session_state.user_access_level == "Premium" else FREE_QUESTION_LIMIT
    remaining = max_questions - st.session_state.questions_answered
    
    if remaining <= 0:
        # Show paywall
        st.markdown("""
        <div class="alert-restricted">
            <h2>🚫 ACCESS RESTRICTED</h2>
            <p style="font-size: 1.2rem; margin: 1rem 0;">You reached the Free Limit (15 Questions).</p>
            <p>To continue practicing, please:</p>
            <ol style="text-align: left; display: inline-block; margin: 1rem auto;">
                <li>Pay ₱50 via GCash to <strong>{GCASH_NUMBER} ({GCASH_NAME})</strong></li>
                <li>Send receipt to <strong>{RECEIPT_EMAIL}</strong></li>
                <li>Or enter a Premium Code</li>
            </ol>
            <p style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.9;">Premium access will be activated after receipt validation.</p>
        </div>
        """.format(GCASH_NUMBER=GCASH_NUMBER, GCASH_NAME=GCASH_NAME, RECEIPT_EMAIL=RECEIPT_EMAIL), unsafe_allow_html=True)
        
        st.markdown("### 📍 Next Steps")
        st.info("👆 Use the sidebar navigation to go to **💳 Payment** or **🔑 Premium Access** pages.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background: rgba(30, 58, 95, 0.6); border-radius: 12px; border: 2px solid #d4af37;">
                <h3>💳 Payment Option</h3>
                <p>Go to Payment page in sidebar</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background: rgba(30, 58, 95, 0.6); border-radius: 12px; border: 2px solid #d4af37;">
                <h3>🔑 Premium Code</h3>
                <p>Go to Premium Access page in sidebar</p>
            </div>
            """, unsafe_allow_html=True)
        st.stop()
    
    # Question generation form
    if not st.session_state.current_questions:
        render_card("⚙️ Generate Questions", """
        <p>Configure your practice exam settings.</p>
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            difficulty = st.selectbox("Difficulty Level", ["Easy", "Average", "Difficult"])
            num_questions = st.number_input("Number of Questions", min_value=1, max_value=min(remaining, 50), value=min(10, remaining))
        
        with col2:
            question_types = st.multiselect(
                "Question Types",
                ["MCQ", "True/False", "Identification"],
                default=["MCQ", "True/False"]
            )
        
        if st.button("🎯 Generate Questions", type="primary", use_container_width=True):
            if not question_types:
                st.error("Please select at least one question type.")
            else:
                with st.spinner("Generating questions..."):
                    # Use empty string if default dummy questions marker
                    text_for_generation = "" if st.session_state.pdf_text == "DEFAULT_DUMMY_QUESTIONS" else st.session_state.pdf_text
                    questions = generate_questions(
                        text_for_generation,
                        difficulty,
                        num_questions,
                        question_types
                    )
                    if questions:
                        st.session_state.current_questions = questions
                        st.session_state.current_question_index = 0
                        st.session_state.answers = {}
                        st.session_state.exam_completed = False
                        st.success(f"✅ Generated {len(questions)} questions!")
                        st.rerun()
                    else:
                        st.error("Failed to generate questions. Please try again.")
    
    # Display current question
    if st.session_state.current_questions:
        questions = st.session_state.current_questions
        current_idx = st.session_state.current_question_index
        
        if current_idx < len(questions):
            q = questions[current_idx]
            
            # Progress
            progress = (current_idx + 1) / len(questions)
            st.markdown(f"""
            <div style="margin: 1rem 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="color: #d4af37; font-weight: 600;">TRAINING PROGRESS</span>
                    <span style="color: #d4af37; font-weight: 700;">Question {current_idx + 1}/{len(questions)}</span>
                </div>
                <div class="progress-container">
                    <div class="progress-bar" style="width: {progress * 100}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Question card
            render_card(f"Question {current_idx + 1}", f"""
            <p style="font-size: 1.1rem; margin-bottom: 1rem;">{q['question']}</p>
            """)
            
            # Answer options
            if q['type'] == "MCQ":
                answer = st.radio("Select your answer:", q['options'], key=f"q_{current_idx}")
            elif q['type'] == "True/False":
                answer = st.radio("Select your answer:", ["True", "False"], key=f"q_{current_idx}")
            else:  # Identification
                answer = st.text_input("Enter your answer:", key=f"q_{current_idx}")
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("✅ Submit Answer", type="primary"):
                    st.session_state.answers[current_idx] = answer
                    st.session_state.current_question_index += 1
                    # Only count when exam is fully completed
                    if st.session_state.current_question_index >= len(questions):
                        if "exam_completed" not in st.session_state or not st.session_state.get("exam_completed", False):
                            st.session_state.questions_answered += len(questions)
                            # Update user's question count in database
                            if st.session_state.user_logged_in and st.session_state.user_email:
                                update_user_questions_answered(st.session_state.user_email, len(questions))
                            st.session_state.exam_completed = True
                    st.rerun()
            
            with col2:
                if st.button("⏭️ Skip", type="secondary"):
                    st.session_state.current_question_index += 1
                    # Only count when exam is fully completed
                    if st.session_state.current_question_index >= len(questions):
                        if "exam_completed" not in st.session_state or not st.session_state.get("exam_completed", False):
                            st.session_state.questions_answered += len(questions)
                            # Update user's question count in database
                            if st.session_state.user_logged_in and st.session_state.user_email:
                                update_user_questions_answered(st.session_state.user_email, len(questions))
                            st.session_state.exam_completed = True
                    st.rerun()
        else:
            # Show results
            st.markdown("# 📊 Exam Results")
            
            score = 0
            total = len(questions)
            wrong_answers = []
            
            for idx, q in enumerate(questions):
                user_answer = st.session_state.answers.get(idx, "Not answered")
                correct = q['correct_answer']
                is_correct = str(user_answer).strip().lower() == str(correct).strip().lower()
                
                if is_correct:
                    score += 1
                else:
                    wrong_answers.append({
                        "question": q['question'],
                        "your_answer": user_answer,
                        "correct_answer": correct,
                        "explanation": q.get('explanation', '')
                    })
            
            # Score display
            percentage = (score / total * 100) if total > 0 else 0
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem; background: rgba(30, 58, 95, 0.6); border-radius: 16px; border: 2px solid #d4af37;">
                <h2 style="color: #d4af37; font-size: 3rem; margin: 0;">{score}/{total}</h2>
                <p style="color: #e0e0e0; font-size: 1.5rem; margin: 0.5rem 0;">{percentage:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Performance Insights
            st.markdown("### 📈 Performance Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("✅ Correct", score)
            with col2:
                st.metric("❌ Incorrect", total - score)
            with col3:
                st.metric("📊 Accuracy", f"{percentage:.1f}%")
            
            # Performance message
            if percentage >= 90:
                perf_msg = "🌟 Outstanding! You're excelling in your preparation!"
                perf_color = "#28a745"
            elif percentage >= 75:
                perf_msg = "👍 Great job! You're on the right track!"
                perf_color = "#20c997"
            elif percentage >= 60:
                perf_msg = "💪 Good start! Keep practicing to improve!"
                perf_color = "#ffc107"
            else:
                perf_msg = "📚 Keep studying! Review the materials and try again!"
                perf_color = "#dc3545"
            
            st.markdown(f"""
            <div style="background: rgba(30, 58, 95, 0.6); border-left: 4px solid {perf_color}; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <p style="color: {perf_color}; font-weight: 600; font-size: 1.1rem; margin: 0;">{perf_msg}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Review wrong answers
            if wrong_answers:
                st.markdown("### ❌ Review Wrong Answers")
                for i, wa in enumerate(wrong_answers):
                    with st.expander(f"Question {i+1}: {wa['question'][:50]}..."):
                        st.write(f"**Your Answer:** {wa['your_answer']}")
                        st.write(f"**Correct Answer:** {wa['correct_answer']}")
                        st.write(f"**Explanation:** {wa['explanation']}")
            
            # Check if Free user reached 15 questions
            is_free_user = st.session_state.user_access_level == "Free"
            reached_free_limit = st.session_state.questions_answered >= FREE_QUESTION_LIMIT
            
            if is_free_user and reached_free_limit:
                st.markdown("---")
                # Access Restriction Card with Premium Upsell
                st.markdown("""
                <div style="background: linear-gradient(135deg, #dc3545 0%, #c82333 100%); border: 3px solid #ff6b6b; border-radius: 16px; padding: 2.5rem; text-align: center; margin: 2rem 0; box-shadow: 0 8px 32px rgba(220, 53, 69, 0.5);">
                    <h2 style="color: white; font-size: 2rem; margin: 0 0 1rem 0;">🚫 ACCESS RESTRICTED</h2>
                    <p style="color: white; font-size: 1.2rem; margin: 1rem 0;">You've completed your free 15 questions!</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div style="background: linear-gradient(135deg, #d4af37 0%, #f4d03f 100%); border: 3px solid #d4af37; border-radius: 16px; padding: 2rem; margin: 1rem 0; box-shadow: 0 8px 32px rgba(212, 175, 55, 0.5);">
                    <h2 style="color: #1a2332; font-size: 1.8rem; margin: 0 0 1rem 0;">🔑 UNLOCK PREMIUM ACCESS</h2>
                    <p style="color: #1a2332; font-size: 1.1rem; margin: 0.5rem 0; font-weight: 600;">Get 100+ additional questions and advanced features:</p>
                    <ul style="text-align: left; color: #1a2332; font-size: 1rem; margin: 1rem auto; display: inline-block;">
                        <li>✅ 100+ additional practice questions</li>
                        <li>✅ Access to advanced situational scenarios</li>
                        <li>✅ Better preparation with deeper insights</li>
                        <li>✅ Real-world readiness training</li>
                        <li>✅ Full access to all premium PDF resources</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔓 Unlock Premium Access", type="primary", use_container_width=True):
                        st.session_state.page = "🔑 Premium Access"
                        st.rerun()
                with col2:
                    if st.button("💳 Go to Payment", type="primary", use_container_width=True):
                        st.session_state.page = "💳 Payment"
                        st.rerun()
            
            # Reset or new set (only if not at limit)
            if not (is_free_user and reached_free_limit):
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Generate New Set", type="primary", use_container_width=True):
                        st.session_state.current_questions = []
                        st.session_state.current_question_index = 0
                        st.session_state.answers = {}
                        st.session_state.exam_completed = False
                        st.rerun()
                with col2:
                    if st.button("🏠 Back to Home", use_container_width=True):
                        st.session_state.current_questions = []
                        st.session_state.current_question_index = 0
                        st.session_state.answers = {}
                        st.session_state.exam_completed = False
                        st.rerun()
            else:
                st.info("💡 Upgrade to Premium to continue practicing with more questions!")

# ============================================================================
# PAGE: PREMIUM ACCESS
# ============================================================================

elif page == "🔑 Premium Access":
    st.markdown("# 🔑 Premium Access")
    
    render_card("🔓 Unlock Premium Features", """
    <p>Enter a valid premium code to unlock <strong>100 additional questions</strong> (115 total questions).</p>
    <p>Premium features include:</p>
    <ul>
        <li>✅ 100+ additional practice questions</li>
        <li>✅ Access to advanced situational scenarios</li>
        <li>✅ Full access to all premium PDF resources</li>
        <li>✅ Better preparation with deeper insights</li>
    </ul>
    <p>Premium codes are generated by administrators.</p>
    """)
    
    code_input = st.text_input("Enter Premium Code:", placeholder="ABCD1234EFGH", help="Enter your premium access code")
    
    if st.button("🔓 Activate Premium", type="primary", use_container_width=True):
        if code_input:
            is_valid, message = validate_premium_code(code_input.strip().upper())
            if is_valid:
                st.session_state.premium_active = True
                st.session_state.premium_code_used = code_input.strip().upper()
                use_premium_code(code_input.strip().upper(), st.session_state.get("session_id", "default"))
                
                # Update user access level in database
                if st.session_state.user_logged_in and st.session_state.user_email:
                    update_user_access_level(st.session_state.user_email, "Premium")
                    cursor = db_conn.cursor()
                    cursor.execute("""
                        UPDATE users
                        SET premium_code_used = ?
                        WHERE email = ?
                    """, (code_input.strip().upper(), st.session_state.user_email))
                    db_conn.commit()
                    st.session_state.user_access_level = "Premium"
                
                st.success("✅ Premium access activated! You now have access to 115 total questions (15 free + 100 premium).")
                st.balloons()
            else:
                st.error(f"❌ {message}")
        else:
            st.error("Please enter a premium code.")
    
    if st.session_state.premium_active or st.session_state.user_access_level == "Premium":
        st.markdown("""
        <div class="alert-premium">
            <h2>✅ PREMIUM ACCESS ACTIVE</h2>
            <p>You have access to 115 total questions (15 free + 100 premium)!</p>
            <p>Code used: <strong>{}</strong></p>
        </div>
        """.format(st.session_state.premium_code_used or "Admin Activated"), unsafe_allow_html=True)

# ============================================================================
# PAGE: PAYMENT
# ============================================================================

elif page == "💳 Payment":
    st.markdown("# 💳 Payment & Receipt Upload")
    
    render_card("💰 Payment Instructions", f"""
    <h3 style="color: #d4af37; margin-top: 0;">Payment Method:</h3>
    <ul>
        <li>Pay <strong>₱{PAYMENT_AMOUNT}</strong> via GCash</li>
        <li>GCash Number: <strong>{GCASH_NUMBER}</strong></li>
        <li>Account Name: <strong>{GCASH_NAME}</strong></li>
    </ul>
    
    <h3 style="color: #d4af37; margin-top: 1rem;">📩 Receipt Submission:</h3>
    <p>Email your payment receipt to: <strong>{RECEIPT_EMAIL}</strong></p>
    
    <div style="background: rgba(212, 175, 55, 0.2); border-left: 4px solid #d4af37; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
        <p style="color: #d4af37; font-weight: 600; margin: 0;">⚠️ Important: Premium access will be activated after receipt validation.</p>
        <p style="color: #e0e0e0; margin: 0.5rem 0 0 0; font-size: 0.9rem;">Please allow 12-24 hours for processing after submitting your receipt.</p>
    </div>
    """)
    
    with st.form("payment_form"):
        st.markdown("### 📝 Payment Details")
        full_name = st.text_input("Full Name *", placeholder="Juan dela Cruz")
        email = st.text_input("Email Address *", placeholder="juan@example.com")
        gcash_ref = st.text_input("GCash Reference Number (Optional)", placeholder="Reference number from GCash")
        receipt_file = st.file_uploader("Upload Receipt (Image/PDF)", type=["png", "jpg", "jpeg", "pdf"])
        
        submitted = st.form_submit_button("📤 Submit Payment Request", type="primary", use_container_width=True)
        
        if submitted:
            if not full_name or not email:
                st.error("Please fill in all required fields (marked with *)")
            elif not receipt_file:
                st.warning("⚠️ Receipt upload is recommended for faster processing.")
                if st.button("Submit Anyway"):
                    save_payment_receipt(full_name, email, gcash_ref, "")
                    st.success("✅ Payment request submitted! Please send receipt to " + RECEIPT_EMAIL)
            else:
                # Save receipt file
                receipt_dir = "data/receipts"
                os.makedirs(receipt_dir, exist_ok=True)
                receipt_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{receipt_file.name}"
                receipt_path = os.path.join(receipt_dir, receipt_filename)
                
                with open(receipt_path, "wb") as f:
                    f.write(receipt_file.getbuffer())
                
                save_payment_receipt(full_name, email, gcash_ref, receipt_filename)
                st.success("✅ Payment request and receipt submitted! We'll review it within 12-24 hours.")

# ============================================================================
# PAGE: ADMIN PANEL
# ============================================================================

elif page == "🛠️ Admin Panel":
    st.markdown("# 🛠️ Admin Command Center")
    
    # Admin login
    if not st.session_state.admin_logged_in:
        render_card("🔐 Admin Login", """
        <p>Enter admin password to access the command center.</p>
        """)
        
        admin_password = st.text_input("Admin Password", type="password", key="admin_pw")
        
        if st.button("🔓 Login", type="primary", use_container_width=True):
            # Try to get password from secrets, environment, or use default
            correct_password = "banban1231"  # Default
            try:
                correct_password = st.secrets.get("ADMIN_PASSWORD", os.environ.get("ADMIN_PASSWORD", "banban1231"))
            except Exception:
                # Secrets file doesn't exist, try environment variable or use default
                correct_password = os.environ.get("ADMIN_PASSWORD", "banban1231")
            
            if admin_password == correct_password:
                st.session_state.admin_logged_in = True
                st.success("✅ Admin access granted!")
                st.rerun()
            else:
                st.error("❌ Invalid password.")
        st.stop()
    
    # Admin logged in - show tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔑 Generate Codes", "📋 Code Database", "💳 Receipt Validation", "📄 PDF Management", "👥 User Management"])
    
    with tab1:
        st.markdown("### 🔑 Generate Premium Codes")
        
        with st.form("generate_codes"):
            col1, col2 = st.columns(2)
            with col1:
                num_codes = st.number_input("Number of Codes", min_value=1, max_value=100, value=5)
                code_length = st.number_input("Code Length", min_value=8, max_value=20, value=12)
            with col2:
                expiry_days = st.number_input("Expiry (Days, 0 = No expiry)", min_value=0, max_value=365, value=30)
                max_uses = st.number_input("Max Uses per Code", min_value=1, max_value=100, value=1)
            
            if st.form_submit_button("🎯 Generate Codes", type="primary", use_container_width=True):
                expiry = expiry_days if expiry_days > 0 else None
                codes = create_premium_codes(num_codes, code_length, expiry, max_uses, "admin")
                st.session_state.generated_codes = codes
                st.success(f"✅ Generated {len(codes)} premium codes!")
                st.rerun()
        
        # Display codes and download button outside form
        if "generated_codes" in st.session_state and st.session_state.generated_codes:
            codes_df = pd.DataFrame({"Code": st.session_state.generated_codes})
            st.dataframe(codes_df, use_container_width=True, hide_index=True)
            
            # Export CSV
            csv = codes_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download Codes as CSV",
                data=csv,
                file_name=f"premium_codes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with tab2:
        st.markdown("### 📋 Code Database")
        
        cursor = db_conn.cursor()
        cursor.execute("""
            SELECT code, status, expiry_date, max_uses, uses_left, created_at
            FROM premium_codes
            ORDER BY created_at DESC
        """)
        
        codes_data = cursor.fetchall()
        
        if codes_data:
            df = pd.DataFrame(codes_data, columns=["Code", "Status", "Expiry", "Max Uses", "Uses Left", "Created At"])
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Search and filter
            search_code = st.text_input("🔍 Search Code", placeholder="Enter code to search")
            if search_code:
                filtered = df[df["Code"].str.contains(search_code.upper(), na=False)]
                st.dataframe(filtered, use_container_width=True, hide_index=True)
        else:
            st.info("No codes generated yet.")
    
    with tab3:
        st.markdown("### 💳 Receipt Validation")
        
        cursor = db_conn.cursor()
        cursor.execute("""
            SELECT id, full_name, email, gcash_reference, receipt_filename, status, submitted_at
            FROM payment_receipts
            ORDER BY submitted_at DESC
        """)
        
        receipts_data = cursor.fetchall()
        
        if receipts_data:
            df = pd.DataFrame(receipts_data, columns=["ID", "Name", "Email", "GCash Ref", "Receipt", "Status", "Submitted"])
            
            # Filter by status
            status_filter = st.selectbox("Filter by Status", ["All", "Pending", "Approved", "Rejected"])
            if status_filter != "All":
                df = df[df["Status"] == status_filter]
            
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Approve/Reject actions
            st.markdown("### ⚡ Actions")
            receipt_id = st.number_input("Receipt ID to Update", min_value=1)
            new_status = st.selectbox("New Status", ["Pending", "Approved", "Rejected"])
            admin_notes = st.text_area("Admin Notes")
            
            if st.button("💾 Update Status", type="primary"):
                cursor.execute("""
                    UPDATE payment_receipts
                    SET status = ?, reviewed_at = ?, reviewed_by = ?, notes = ?
                    WHERE id = ?
                """, (new_status, datetime.now().isoformat(), "admin", admin_notes, receipt_id))
                db_conn.commit()
                st.success(f"✅ Receipt #{receipt_id} updated to {new_status}")
                st.rerun()
        else:
            st.info("No payment receipts submitted yet.")
    
    with tab4:
        st.markdown("### 📄 PDF Resource Management")
        
        # Upload new PDF
        st.markdown("#### 📤 Upload New PDF")
        with st.form("upload_pdf_admin"):
            uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf", "docx", "doc"], key="admin_pdf_upload")
            is_premium_only = st.checkbox("Premium Only", value=False, help="Only Premium users can download this PDF")
            use_for_ai = st.checkbox("Use for AI Question Generation", value=True, help="Include this PDF in AI question generation")
            description = st.text_area("Description (optional)", placeholder="Brief description of this PDF")
            
            if st.form_submit_button("📤 Upload PDF", type="primary", use_container_width=True):
                if uploaded_pdf:
                    # Save file
                    pdf_dir = "data/admin_pdfs"
                    os.makedirs(pdf_dir, exist_ok=True)
                    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_pdf.name}"
                    filepath = os.path.join(pdf_dir, filename)
                    
                    with open(filepath, "wb") as f:
                        f.write(uploaded_pdf.getbuffer())
                    
                    save_pdf_resource(filename, filepath, is_premium_only, use_for_ai, st.session_state.user_email or "admin", description)
                    st.success(f"✅ PDF uploaded successfully: {filename}")
                    st.rerun()
                else:
                    st.error("Please select a PDF file to upload.")
        
        st.markdown("---")
        
        # List PDFs
        st.markdown("#### 📋 Manage PDFs")
        pdf_resources = get_pdf_resources()
        
        if pdf_resources:
            for pdf in pdf_resources:
                with st.expander(f"📄 {pdf['filename']} - {'🔒 Premium' if pdf['is_premium_only'] else '🆓 Free'} - {'🤖 AI Enabled' if pdf['use_for_ai_generation'] else '❌ AI Disabled'}"):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**Description:** {pdf.get('description', 'No description')}")
                        st.write(f"**Uploaded:** {pdf['uploaded_at']}")
                        st.write(f"**Uploaded by:** {pdf.get('uploaded_by', 'Unknown')}")
                    with col2:
                        premium_toggle = st.checkbox("Premium Only", value=pdf['is_premium_only'], key=f"premium_{pdf['id']}")
                        ai_toggle = st.checkbox("Use for AI", value=pdf['use_for_ai_generation'], key=f"ai_{pdf['id']}")
                        
                        if st.button("💾 Update", key=f"update_{pdf['id']}"):
                            update_pdf_resource(pdf['id'], premium_toggle, ai_toggle)
                            st.success("✅ Updated!")
                            st.rerun()
                        
                        if st.button("🗑️ Delete", key=f"delete_{pdf['id']}"):
                            if os.path.exists(pdf['filepath']):
                                os.remove(pdf['filepath'])
                            delete_pdf_resource(pdf['id'])
                            st.success("✅ Deleted!")
                            st.rerun()
        else:
            st.info("No PDFs uploaded yet.")
    
    with tab5:
        st.markdown("### 👥 User Management")
        
        # Get all users
        cursor = db_conn.cursor()
        cursor.execute("""
            SELECT email, access_level, questions_answered, created_at, last_login
            FROM users
            ORDER BY created_at DESC
        """)
        
        users_data = cursor.fetchall()
        
        if users_data:
            df = pd.DataFrame(users_data, columns=["Email", "Access Level", "Questions Answered", "Created At", "Last Login"])
            
            # Search
            search_email = st.text_input("🔍 Search User by Email", placeholder="Enter email to search")
            if search_email:
                df = df[df["Email"].str.contains(search_email.lower(), na=False, case=False)]
            
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.markdown("#### ⚡ Update User Access")
            
            col1, col2 = st.columns(2)
            with col1:
                user_email_update = st.text_input("User Email", placeholder="user@example.com")
                new_access_level = st.selectbox("New Access Level", ["Free", "Premium"], key="user_access_update")
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("💾 Update Access Level", type="primary", use_container_width=True):
                    if user_email_update:
                        update_user_access_level(user_email_update.lower(), new_access_level)
                        st.success(f"✅ User {user_email_update} access updated to {new_access_level}")
                        st.rerun()
                    else:
                        st.error("Please enter a user email.")
        else:
            st.info("No users registered yet.")
    
    with tab4:
        st.markdown("### 📄 PDF Resource Management")
        
        # Upload new PDF
        st.markdown("#### 📤 Upload New PDF")
        with st.form("upload_pdf_admin"):
            uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf", "docx", "doc"], key="admin_pdf_upload")
            is_premium_only = st.checkbox("Premium Only", value=False, help="Only Premium users can download this PDF")
            use_for_ai = st.checkbox("Use for AI Question Generation", value=True, help="Include this PDF in AI question generation")
            description = st.text_area("Description (optional)", placeholder="Brief description of this PDF")
            
            if st.form_submit_button("📤 Upload PDF", type="primary", use_container_width=True):
                if uploaded_pdf:
                    # Save file
                    pdf_dir = "data/admin_pdfs"
                    os.makedirs(pdf_dir, exist_ok=True)
                    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_pdf.name}"
                    filepath = os.path.join(pdf_dir, filename)
                    
                    with open(filepath, "wb") as f:
                        f.write(uploaded_pdf.getbuffer())
                    
                    save_pdf_resource(filename, filepath, is_premium_only, use_for_ai, st.session_state.user_email or "admin", description)
                    st.success(f"✅ PDF uploaded successfully: {filename}")
                    st.rerun()
                else:
                    st.error("Please select a PDF file to upload.")
        
        st.markdown("---")
        
        # List PDFs
        st.markdown("#### 📋 Manage PDFs")
        pdf_resources = get_pdf_resources()
        
        if pdf_resources:
            for pdf in pdf_resources:
                with st.expander(f"📄 {pdf['filename']} - {'🔒 Premium' if pdf['is_premium_only'] else '🆓 Free'} - {'🤖 AI Enabled' if pdf['use_for_ai_generation'] else '❌ AI Disabled'}"):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**Description:** {pdf.get('description', 'No description')}")
                        st.write(f"**Uploaded:** {pdf['uploaded_at']}")
                        st.write(f"**Uploaded by:** {pdf.get('uploaded_by', 'Unknown')}")
                    with col2:
                        premium_toggle = st.checkbox("Premium Only", value=pdf['is_premium_only'], key=f"premium_{pdf['id']}")
                        ai_toggle = st.checkbox("Use for AI", value=pdf['use_for_ai_generation'], key=f"ai_{pdf['id']}")
                        
                        if st.button("💾 Update", key=f"update_{pdf['id']}"):
                            update_pdf_resource(pdf['id'], premium_toggle, ai_toggle)
                            st.success("✅ Updated!")
                            st.rerun()
                        
                        if st.button("🗑️ Delete", key=f"delete_{pdf['id']}"):
                            if os.path.exists(pdf['filepath']):
                                os.remove(pdf['filepath'])
                            delete_pdf_resource(pdf['id'])
                            st.success("✅ Deleted!")
                            st.rerun()
        else:
            st.info("No PDFs uploaded yet.")
    
    with tab5:
        st.markdown("### 👥 User Management")
        
        # Get all users
        cursor = db_conn.cursor()
        cursor.execute("""
            SELECT email, access_level, questions_answered, created_at, last_login
            FROM users
            ORDER BY created_at DESC
        """)
        
        users_data = cursor.fetchall()
        
        if users_data:
            df = pd.DataFrame(users_data, columns=["Email", "Access Level", "Questions Answered", "Created At", "Last Login"])
            
            # Search
            search_email = st.text_input("🔍 Search User by Email", placeholder="Enter email to search")
            if search_email:
                df = df[df["Email"].str.contains(search_email.lower(), na=False, case=False)]
            
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.markdown("#### ⚡ Update User Access")
            
            col1, col2 = st.columns(2)
            with col1:
                user_email_update = st.text_input("User Email", placeholder="user@example.com")
                new_access_level = st.selectbox("New Access Level", ["Free", "Premium"], key="user_access_update")
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("💾 Update Access Level", type="primary", use_container_width=True):
                    if user_email_update:
                        update_user_access_level(user_email_update.lower(), new_access_level)
                        st.success(f"✅ User {user_email_update} access updated to {new_access_level}")
                        st.rerun()
                    else:
                        st.error("Please enter a user email.")
        else:
            st.info("No users registered yet.")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 1rem;">
    <p><strong>For review purposes only.</strong> Always verify with official references and latest PH laws and regulations.</p>
    <p>© 2024 PH Criminology Exam Reviewer</p>
</div>
""", unsafe_allow_html=True)


