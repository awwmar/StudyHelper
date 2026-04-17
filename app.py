import os
import shutil
import io
import json
import random
import string
from functools import wraps
from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
from dotenv import load_dotenv
from supabase import create_client, Client
from authlib.integrations.flask_client import OAuth

load_dotenv()

# Service Imports
from services.client import supabase
from agents.system1_ingestion import system1 as ingestion_service
from services.vector import vector_service
from services.llm import llm_service
from services.selector import selector_agent
from agents.system2_query import system2
from services.db import DBService
from reportlab.pdfgen import canvas


app = Flask(__name__)

app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-123")
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- OAuth Setup ---
oauth = OAuth(app)
CONF_URL = "https://accounts.google.com/.well-known/openid-configuration"
oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url=CONF_URL,
    client_kwargs={"scope": "openid email profile"},
)

# Global Progress Tracker
upload_progress = {}


# --- Legacy helpers removed (replaced by services) ---


# --- Auth Decorator ---
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"success": False, "error": "Login required"}), 401
        return f(*args, **kwargs)
    return wrapper

# --- Helper: Get Vector Store ---
def get_vectorstore(doc_name):
    persist_directory = f"./vector_db/{doc_name.replace(' ', '_').lower()}"
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

def generate_unique_code(length=6):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

# --- Auth Routes ---
# --- Auth Routes ---
@app.route("/auth/google")
def google_auth_start():
    role = request.args.get('role', 'student') # Default to student if not specified
    session['pending_role'] = role
    redirect_uri = url_for("google_auth_callback", _external=True)
    return oauth.google.authorize_redirect(redirect_uri)

@app.route("/google/auth")
def google_auth_callback():
    token = oauth.google.authorize_access_token()
    user_info = oauth.google.userinfo()

    user_id = user_info["sub"] # Google Unique ID
    email = user_info.get("email")
    name = user_info.get("name")
    
    # Check if user exists first to preserve their original role
    # If not, user will be created with pending_role
    try:
        existing = supabase.table("users").select("*").eq("id", user_id).execute()
        
        current_role = session.get('pending_role', 'student')
        
        # Hardcode Admin for specific email
        if email == "f20230518@pilani.bits-pilani.ac.in":
            current_role = "admin"
        
        profile_completed = False
        
        if existing.data:
            # User exists, keep their role (unless forced admin)
            if email != "f20230518@pilani.bits-pilani.ac.in":
                current_role = existing.data[0].get('role', current_role)
            profile_completed = existing.data[0].get('profile_completed', False)
        
        user_data = {
            "id": user_id,
            "email": email,
            "name": name,
            "role": current_role
        }
        
        if current_role == "admin":
             user_data["is_verified"] = True # Admins auto-verified
        
        # upsert: insert if doesn't exist, update if it does
        supabase.table("users").upsert(user_data).execute()

        session["user_id"] = user_id
        session["user_email"] = email
        session["user_name"] = name
        session["user_role"] = current_role
        
        # Redirect to registration if profile not completed
        if not profile_completed:
            return redirect(url_for("register"))
        
        return redirect(url_for("index"))
    except Exception as e:
        print(f"Auth Error: {e}")
        return redirect(url_for("login_page", error="Auth Failed"))

@app.route("/login")
def login_page():
    if "user_id" in session:
        return redirect(url_for("index"))
    return render_template('login.html')

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))

@app.route("/register")
@login_required
def register():
    role = session.get("user_role", "student")
    return render_template('registration.html', role=role)

@app.route("/complete-registration", methods=["POST"])
@login_required
def complete_registration():
    try:
        data = request.json
        user_id = session.get("user_id")
        role = session.get("user_role")
        
        update_data = {
            "profile_completed": True,
            "username": data.get("username"),
            "avatar_url": data.get("avatar_url"),
            "bio": data.get("bio")
        }
        
        if role == "teacher":
            update_data["institute"] = data.get("institute")
            update_data["subject"] = data.get("subject")
            update_data["years_of_experience"] = int(data.get("years_of_experience", 0))
            update_data["is_verified"] = False  # Requires admin verification
            
            # Create verification request
            supabase.table("teacher_verifications").insert({
                "user_id": user_id,
                "status": "pending"
            }).execute()
        else:  # student
            update_data["school_name"] = data.get("school_name")
            update_data["grade_level"] = data.get("grade_level")
        
        supabase.table("users").update(update_data).eq("id", user_id).execute()
        
        return jsonify({"success": True})
    except Exception as e:
        print(f"Error in complete_registration: {e}")
        error_msg = str(e)
        if "users_username_key" in error_msg or "23505" in error_msg:
            return jsonify({"success": False, "error": "Username already taken. Please choose another."}), 400
        return jsonify({"success": False, "error": error_msg}), 500

@app.route("/profile", methods=["GET"])
@login_required
def get_profile():
    user_id = session.get("user_id")
    user_data = supabase.table("users").select("*").eq("id", user_id).execute()
    
    if user_data.data:
        return jsonify({"success": True, "profile": user_data.data[0]})
    return jsonify({"success": False, "error": "User not found"}), 404

@app.route("/update-profile", methods=["POST"])
@login_required
def update_profile():
    try:
        data = request.json
        user_id = session.get("user_id")
        role = session.get("user_role")
        
        update_data = {}
        
        # Common fields
        keys = ["username", "avatar_url", "bio", "social_links"]
        for k in keys:
            if k in data:
                update_data[k] = data[k]
        
        if role == "teacher":
            if "institute" in data:
                update_data["institute"] = data["institute"]
            if "subject" in data:
                update_data["subject"] = data["subject"]
            if "years_of_experience" in data:
                try:
                    update_data["years_of_experience"] = int(data["years_of_experience"])
                except (ValueError, TypeError):
                    pass # Ignore invalid values
        else:  # student
            if "school_name" in data:
                update_data["school_name"] = data["school_name"]
            if "grade_level" in data:
                update_data["grade_level"] = data["grade_level"]
        
        if update_data:
            supabase.table("users").update(update_data).eq("id", user_id).execute()
        
        return jsonify({"success": True})
    except Exception as e:
        print(f"Error in update_profile: {e}")
        error_msg = str(e)
        if "users_username_key" in error_msg or "23505" in error_msg:
            return jsonify({"success": False, "error": "Username already taken. Please choose another."}), 400
        return jsonify({"success": False, "error": error_msg}), 500

@app.route('/')
def index():
    return render_template('index.html', user=session.get("user_name"), role=session.get("user_role"))

@app.route('/upload-progress/<path:filename>')
@login_required
def get_upload_progress(filename):
    user_id = session.get("user_id")
    # Logic to match upload_pdf's naming
    doc_name = os.path.splitext(filename)[0]
    folder_name = doc_name.replace(' ', '_').lower()
    progress_key = f"{user_id}_{folder_name}"
    
    data = upload_progress.get(progress_key, {'status': 'waiting', 'percent': 0})
    return jsonify(data)

# --- 1. UPLOAD & CHUNK ---
# --- 1. UPLOAD & CHUNK ---
@app.route('/upload', methods=['POST'])
@login_required
def upload_pdf():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        raw_name = request.form.get('book_name') or file.filename
        doc_name = os.path.splitext(raw_name)[0] or "untitled"
        
        # Consistent naming for ID
        # Note: In new system we track by filename in DB, but frontend might expect 'book_name' to be same
        
        user_id = session.get("user_id")
        progress_key = f"{user_id}_{doc_name.replace(' ', '_').lower()}"
        
        # Callback for progress
        def progress_callback(curr, total, msg):
            percent = int((curr / total) * 100)
            upload_progress[progress_key] = {"status": msg, "percent": percent}

        upload_progress[progress_key] = {"status": "Starting Upload...", "percent": 5}

        # Run Ingestion
        ingestion_service.process_document(file.stream, doc_name, user_id, progress_callback)
        
        return jsonify({'success': True, 'book_name': doc_name})
    except Exception as e:
        print(e)
        upload_progress[f"{user_id}_{doc_name.replace(' ', '_').lower()}"] = {"status": "Failed", "percent": 0}
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get-topics', methods=['POST'])
@login_required
def get_topics():
    data = request.json
    book_name = data.get('book_name')
    # Book name is essentially the filename without extension (or strictly tracked)
    # The IngestionService uses doc_name as filename. 
    
    topics = DBService.get_topics_by_filename(book_name)
    topics.sort()
    # Frontend expects objects with 'title'
    topics_formatted = [{"title": t} for t in topics]
    
    return jsonify({"success": True, "topics": topics_formatted})

@app.route('/generate-questions', methods=['POST'])
@login_required
def generate_questions():
    try:
        data = request.json
        # book_name = data.get('book_name') # Not strictly needed if we search global index, 
        # but if we want to filter by document, we need to implement that in vector_service.
        # Current vector_service.search searches EVERYTHING in index.
        # For now, let's assume global search is acceptable or we add filtering later.
        
        topics = data.get('topics', [])
        q_type = data.get('question_type', 'conceptual')
        num_questions = data.get('num_questions', 5)

        # Instruction
        type_instruction = ""
        if q_type == 'mixed':
            type_instruction = "Create a mix of Multiple Choice Questions (MCQ), Fill in the Blanks, and Conceptual/Short Answer questions."
        else:
            type_instruction = f"Generate {q_type} questions."

        # System 2 Orchestrator Pipeline
        questions_list = system2.generate_quiz(
            topics=topics,
            difficulty="Medium", 
            num_questions=num_questions,
            q_type=q_type,
            type_instruction=type_instruction
        )
        
        response_content = "\n\n".join(questions_list)
        return jsonify({'questions': response_content})
    except Exception as e:
        print(f"Error in generate_questions: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/generate-test', methods=['POST'])
@login_required
def generate_test():
    try:
        data = request.json
        # book_name = data.get('book_name')
        topics = data.get('topics', [])
        difficulty = data.get('difficulty', 'Medium')
        num_questions = int(data.get('num_questions', 5))
        question_type = data.get('question_type', 'mixed') 
        
        # Question Types
        if question_type == 'mcq':
            type_instruction = "Generate only Multiple Choice Questions (MCQ)."
        elif question_type == 'conceptual':
            type_instruction = "Generate only Conceptual/Short Answer questions."
        elif question_type == 'analytical':
            type_instruction = "Generate Analytical / Case Study questions that require critical thinking."
        elif question_type == 'fill':
            type_instruction = "Generate Fill in the Blanks questions."
        elif question_type == 'bool':
            type_instruction = "Generate True/False questions."
        elif question_type == 'mcq_conceptual':
            type_instruction = "Generate a mix of Multiple Choice Questions (MCQ) and Conceptual/Short Answer questions."
        elif question_type == 'mixed':
            type_instruction = "Generate a mix of Multiple Choice (MCQ), Fill in the Blanks, True/False, and Conceptual questions."
        else:
            type_instruction = "Generate a mix of question types."

        # System 2 Orchestrator Pipeline
        questions_list = system2.generate_quiz(
            topics=topics,
            difficulty=difficulty,
            num_questions=num_questions,
            q_type=question_type,
            type_instruction=type_instruction
        )

        save_db = data.get('save', True)
        test_id = None
        
        if save_db:
            user_id = session.get("user_id")
            test_record = supabase.table("tests").insert({
                "user_id": user_id,
                "book_name": data.get('book_name', 'Mixed'),
                "difficulty": difficulty,
                "total_questions": len(questions_list),
                "score": 0 
            }).execute()
            test_id = test_record.data[0]['id']

        return jsonify({'success': True, 'questions': questions_list, 'test_id': test_id})

    except Exception as e:
        print(f"Error in generate_test: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/start-assignment', methods=['POST'])
@login_required
def start_assignment():
    try:
        data = request.json
        assignment_id = data.get('assignment_id')
        password_attempt = data.get('password')
        location_lat = data.get('location_lat')
        location_long = data.get('location_long')
        
        user_id = session.get("user_id")
        
        # 1. Enforce Location
        if location_lat is None or location_long is None:
             return jsonify({'success': False, 'error': 'Location access is required to start this test.'}), 400

        # 2. Check for Existing Attempts
        user_role = session.get("user_role")
        existing_attempt = supabase.table("tests").select("id").eq("assignment_id", assignment_id).eq("user_id", user_id).execute()
        if existing_attempt.data and user_role != "teacher":
            return jsonify({'success': False, 'error': 'You have already attempted this assignment.'}), 403

        # 3. Fetch Assignment
        assign_res = supabase.table("class_assignments").select("*").eq("id", assignment_id).execute()
        if not assign_res.data:
            return jsonify({'success': False, 'error': 'Assignment not found'}), 404
            
        assignment = assign_res.data[0]
        
        # 4. Check Password
        if assignment.get('password'):
            if assignment.get('password') != password_attempt:
                return jsonify({'success': False, 'error': 'Incorrect Password'}), 403
                
        # 5. Create Test Record
        test_res = supabase.table("tests").insert({
            "user_id": user_id,
            "assignment_id": assignment_id,
            "book_name": assignment.get('title'), 
            "difficulty": assignment.get('difficulty'),
            "total_questions": len(assignment.get('questions', [])),
            "score": 0,
            "location_lat": location_lat,
            "location_long": location_long
        }).execute()
        
        test_id = test_res.data[0]['id']
        
        # 6. Populate Questions
        questions = assignment.get('questions', [])
        if isinstance(questions, str):
            try:
                questions = json.loads(questions)
            except:
                questions = [questions]
                
        for q in questions:
            supabase.table("test_questions").insert({
                "test_id": test_id,
                "question": str(q),
                "user_answer": "",
                "ai_evaluation": "Pending",
                "score": 0
            }).execute()
            
        return jsonify({
            'success': True, 
            'test_id': test_id,
            'questions': questions,
            'duration': assignment.get('duration_minutes')
        })
        
    except Exception as e:
        print(f"Start Assignment Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# --- 2. RAG & EVALUATION ---

@app.route('/submit-answer', methods=['POST'])
@login_required
def submit_answer():
    data = request.json
    test_id = data.get('test_id')
    question = data.get('question')
    answer = data.get('user_answer')
    
    # Check if exists to update
    existing = supabase.table("test_questions").select("*").eq("test_id", test_id).eq("question", question).execute()
    
    if existing.data:
        supabase.table("test_questions").update({
            "user_answer": answer
        }).eq("id", existing.data[0]['id']).execute()
    else:
        supabase.table("test_questions").insert({
            "test_id": test_id,
            "question": question,
            "user_answer": answer,
            "ai_evaluation": "Pending",
            "score": 0
        }).execute()
        
    return jsonify({'success': True})

@app.route('/finish-test', methods=['POST'])
@login_required
def finish_test():
    data = request.json
    test_id = data.get('test_id')
    
    answers_response = supabase.table("test_questions").select("*").eq("test_id", test_id).execute()
    answers = answers_response.data
    
    # Build batch for parallel evaluation
    eval_items = [
        {"question": row['question'], "student_answer": row['user_answer'] or ""}
        for row in answers
    ]
    
    # Parallel evaluation — all answers evaluated concurrently
    eval_results = llm_service.evaluate_batch(eval_items, max_workers=5)
    
    total_score = 0
    
    for row, response in zip(answers, eval_results):
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                eval_data = json.loads(json_match.group(0))
            else:
                eval_data = json.loads(response)
                
            score = int(eval_data.get('score', 0))
            score = min(score, 10)  # Cap at 10
            
            feedback = eval_data.get('feedback', '')
            correctness = str(eval_data.get('correctness', '')).title()
            issues = eval_data.get('issues', [])
            issues_str = f" | Issues: {', '.join(issues)}" if issues else ""
            
            formatted_eval = f"Score: {score}/10 | Correctness: {correctness} | Feedback: {feedback}{issues_str}"
        except Exception as e:
            print(f"DEBUG JSON Parse Error: {e}")
            score = 0
            formatted_eval = response
            
        supabase.table("test_questions").update({
            "ai_evaluation": formatted_eval,
            "score": score
        }).eq("id", row['id']).execute()
        
        total_score += score

    supabase.table("tests").update({
        "score": total_score
    }).eq("id", test_id).execute()
    
    return jsonify({'success': True, 'total_score': total_score})

@app.route('/get-test-details', methods=['POST'])
@login_required
def get_test_details():
    test_id = request.json.get('test_id')
    
    test_meta = supabase.table("tests").select("*").eq("id", test_id).execute()
    if not test_meta.data:
        return jsonify({'success': False, 'error': 'Test not found'}), 404
        
    questions_data = supabase.table("test_questions").select("*").eq("test_id", test_id).execute()
    
    return jsonify({
        'success': True, 
        'test_info': test_meta.data[0],
        'questions': questions_data.data
    })

@app.route("/my-results", methods=["GET"])
@login_required
def my_results():
    user_id = session.get("user_id")
    # Fetch tests taken by this user
    resp = supabase.table("tests").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
    return jsonify({"success": True, "results": resp.data})

@app.route("/my-assignments", methods=["GET"])
@login_required
def my_assignments():
    try:
        user_id = session.get("user_id")
        
        # 1. Get classes user is member of
        members = supabase.table("class_members").select("class_id").eq("user_id", user_id).execute()
        class_ids = [m['class_id'] for m in members.data]
        
        if not class_ids:
             return jsonify({'success': True, 'assignments': []})
             
        # 2. Get assignments for these classes
        # linking to classrooms to get class name
        res = supabase.table("class_assignments").select("*, classrooms(name)").in_("class_id", class_ids).order("due_date", desc=False).execute()
        
        return jsonify({'success': True, 'assignments': res.data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route("/teacher/results", methods=["GET"])
@login_required
def teacher_results():
    # Verify user is a teacher
    if session.get("user_role") != "teacher":
        return jsonify({"success": False, "error": "Unauthorized"}), 403
    
    # Return all results with user info
    resp = supabase.table("tests").select("*, users(name, email, school_name, grade_level)").order("created_at", desc=True).execute()
    return jsonify({"success": True, "results": resp.data})

@app.route("/teacher/submit-review", methods=["POST"])
@login_required
def submit_teacher_review():
    try:
        # Verify user is a teacher
        if session.get("user_role") != "teacher":
            return jsonify({"success": False, "error": "Unauthorized"}), 403
        
        data = request.json
        teacher_id = session.get("user_id")
        test_question_id = data.get("test_question_id")
        review_text = data.get("review_text")
        
        # Check if review already exists
        existing = supabase.table("teacher_reviews").select("*").eq("test_question_id", test_question_id).eq("teacher_id", teacher_id).execute()
        
        if existing.data:
            # Update existing review
            supabase.table("teacher_reviews").update({
                "review_text": review_text,
                "reviewed_at": "now()"
            }).eq("id", existing.data[0]["id"]).execute()
        else:
            # Insert new review
            supabase.table("teacher_reviews").insert({
                "test_question_id": test_question_id,
                "teacher_id": teacher_id,
                "review_text": review_text
            }).execute()
        
        return jsonify({"success": True})
    except Exception as e:
        print(f"Error in submit_teacher_review: {e}")
        error_msg = str(e)
        if "users_username_key" in error_msg or "23505" in error_msg:
            return jsonify({"success": False, "error": "Username already taken. Please choose another."}), 400
        return jsonify({"success": False, "error": error_msg}), 500

@app.route("/get-reviews/<test_id>", methods=["GET"])
@login_required
def get_reviews(test_id):
    try:
        # Get all questions for this test
        questions = supabase.table("test_questions").select("id").eq("test_id", test_id).execute()
        
        if not questions.data:
            return jsonify({"success": True, "reviews": []})
        
        question_ids = [q["id"] for q in questions.data]
        
        # Get all reviews for these questions
        reviews = supabase.table("teacher_reviews").select("*, users(name)").in_("test_question_id", question_ids).execute()
        
        return jsonify({"success": True, "reviews": reviews.data})
    except Exception as e:
        print(f"Error in get_reviews: {e}")
        error_msg = str(e)
        if "users_username_key" in error_msg or "23505" in error_msg:
            return jsonify({"success": False, "error": "Username already taken. Please choose another."}), 400
        return jsonify({"success": False, "error": error_msg}), 500

@app.route("/teacher/reevaluate-test", methods=["POST"])
@login_required
def reevaluate_test():
    try:
        # Verify user is a teacher
        if session.get("user_role") != "teacher":
            return jsonify({"success": False, "error": "Unauthorized"}), 403

        test_id = request.form.get("test_id")
        instruction_type = request.form.get("instruction_type") # 'text' or 'file'
        
        reference_context = ""
        
        if instruction_type == 'text':
            reference_context = request.form.get("text_content", "")
        elif instruction_type == 'file':
            if 'file' not in request.files:
                return jsonify({"success": False, "error": "No file uploaded"}), 400
            file = request.files['file']
            if file.filename == '':
                return jsonify({"success": False, "error": "No file selected"}), 400
            
            # Extract text from PDF using fitz
            try:
                import fitz
                doc = fitz.open(stream=file.read(), filetype="pdf")
                text = ""
                for page in doc:
                    text += page.get_text()
                reference_context = text
            except Exception as e:
                return jsonify({"success": False, "error": f"Failed to extract text from PDF: {str(e)}"}), 500
        
        if not reference_context:
             return jsonify({"success": False, "error": "No reference content provided"}), 400

        # Get all questions for this test
        questions_res = supabase.table("test_questions").select("*").eq("test_id", test_id).execute()
        questions = questions_res.data
        
        # Get Teacher Reviews
        reviews_res = supabase.table("teacher_reviews").select("*").in_("test_question_id", [q['id'] for q in questions]).execute()
        reviews_map = {r['test_question_id']: r['review_text'] for r in reviews_res.data}
        
        total_score = 0
        
        for row in questions:
            q = row['question']
            a = row['user_answer']
            t_review = reviews_map.get(row['id'])
            
            # Re-evaluate
            # Using the new method in LLMService with Teacher Review
            response = llm_service.evaluate_answer(q, a, reference_context, teacher_review=t_review)
            
            # Parse JSON Score and Feedback
            try:
                import re
                import json
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    eval_data = json.loads(json_match.group(0))
                else:
                    eval_data = json.loads(response)
                    
                score = int(eval_data.get('score', 0))
                score = min(score, 10)  # Cap at 10
                
                feedback = eval_data.get('feedback', '')
                correctness = str(eval_data.get('correctness', '')).title()
                issues = eval_data.get('issues', [])
                issues_str = f" | Issues: {', '.join(issues)}" if issues else ""
                
                formatted_eval = f"Score: {score}/10 | Correctness: {correctness} | Feedback: {feedback}{issues_str}"
            except Exception as e:
                print(f"DEBUG JSON Parse Error: {e}")
                score = 0
                formatted_eval = response
            
            # Update Question Record
            supabase.table("test_questions").update({
                "ai_evaluation": formatted_eval,
                "score": score
            }).eq("id", row['id']).execute()
            
            total_score += score
            
        # Update Test Total Score
        supabase.table("tests").update({
            "score": total_score
        }).eq("id", test_id).execute()
        
        return jsonify({"success": True, "total_score": total_score})

    except Exception as e:
        print(f"Error in reevaluate_test: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# --- Social Features ---
@app.route("/create-group", methods=["POST"])
@login_required
def create_group():
    try:
        user_id = session.get("user_id")
        data = request.json
        name = data.get("name")
        desc = data.get("description", "")
        
        code = generate_unique_code()
        
        # Create group
        group_res = supabase.table("study_groups").insert({
            "name": name,
            "description": desc,
            "code": code,
            "created_by": user_id
        }).execute()
        
        group_id = group_res.data[0]["id"]
        
        # Add creator as member
        supabase.table("group_members").insert({
            "group_id": group_id,
            "user_id": user_id
        }).execute()
        
        return jsonify({"success": True, "group": group_res.data[0]})
    except Exception as e:
        error_msg = str(e)
        if "users_username_key" in error_msg or "23505" in error_msg:
            return jsonify({"success": False, "error": "Username already taken. Please choose another."}), 400
        return jsonify({"success": False, "error": error_msg}), 500

@app.route("/join-group", methods=["POST"])
@login_required
def join_group():
    try:
        user_id = session.get("user_id")
        code = request.json.get("code")
        
        # Find group
        group_res = supabase.table("study_groups").select("*").eq("code", code).execute()
        if not group_res.data:
            return jsonify({"success": False, "error": "Invalid Code"}), 404
            
        group_id = group_res.data[0]["id"]
        
        # Check membership
        member_check = supabase.table("group_members").select("*").eq("group_id", group_id).eq("user_id", user_id).execute()
        if member_check.data:
             return jsonify({"success": False, "error": "Already a member"}), 400
             
        # Add member
        supabase.table("group_members").insert({
            "group_id": group_id,
            "user_id": user_id
        }).execute()
        
        return jsonify({"success": True, "group": group_res.data[0]})
    except Exception as e:
        error_msg = str(e)
        if "users_username_key" in error_msg or "23505" in error_msg:
            return jsonify({"success": False, "error": "Username already taken. Please choose another."}), 400
        return jsonify({"success": False, "error": error_msg}), 500

@app.route("/create-class", methods=["POST"])
@login_required
def create_class():
    if session.get("user_role") != "teacher":
        return jsonify({"success": False, "error": "Unauthorized"}), 403
        
    try:
        user_id = session.get("user_id")
        data = request.json
        name = data.get("name")
        subject = data.get("subject", "")
        
        code = generate_unique_code()
        
        # Create class
        class_res = supabase.table("classrooms").insert({
            "name": name,
            "subject": subject,
            "code": code,
            "teacher_id": user_id
        }).execute()
        
        return jsonify({"success": True, "class": class_res.data[0]})
    except Exception as e:
        error_msg = str(e)
        if "users_username_key" in error_msg or "23505" in error_msg:
            return jsonify({"success": False, "error": "Username already taken. Please choose another."}), 400
        return jsonify({"success": False, "error": error_msg}), 500

@app.route("/join-class", methods=["POST"])
@login_required
def join_class():
    try:
        user_id = session.get("user_id")
        code = request.json.get("code")
        
        # Find class
        class_res = supabase.table("classrooms").select("*").eq("code", code).execute()
        if not class_res.data:
            return jsonify({"success": False, "error": "Invalid Code"}), 404
            
        class_id = class_res.data[0]["id"]
        
        # Check membership
        member_check = supabase.table("class_members").select("*").eq("class_id", class_id).eq("user_id", user_id).execute()
        if member_check.data:
             return jsonify({"success": False, "error": "Already a member"}), 400
             
        # Add member
        supabase.table("class_members").insert({
            "class_id": class_id,
            "user_id": user_id
        }).execute()
        
        return jsonify({"success": True, "class": class_res.data[0]})
    except Exception as e:
        error_msg = str(e)
        if "users_username_key" in error_msg or "23505" in error_msg:
            return jsonify({"success": False, "error": "Username already taken. Please choose another."}), 400
        return jsonify({"success": False, "error": error_msg}), 500

@app.route("/get-community", methods=["GET"])
@login_required
def get_community():
    try:
        user_id = session.get("user_id")
        role = session.get("user_role")
        
        response = {"groups": [], "classes": []}
        
        # Get Groups
        groups = supabase.table("group_members").select("group_id, study_groups(*)").eq("user_id", user_id).execute()
        response["groups"] = [g['study_groups'] for g in groups.data if g['study_groups']]
        
        # Get Classes
        if role == "teacher":
            classes = supabase.table("classrooms").select("*").eq("teacher_id", user_id).execute()
            response["classes"] = classes.data
        else:
            classes = supabase.table("class_members").select("class_id, classrooms(*)").eq("user_id", user_id).execute()
            response["classes"] = [c['classrooms'] for c in classes.data if c['classrooms']]
            
        return jsonify({"success": True, "data": response})
    except Exception as e:
        error_msg = str(e)
        if "users_username_key" in error_msg or "23505" in error_msg:
            return jsonify({"success": False, "error": "Username already taken. Please choose another."}), 400
        return jsonify({"success": False, "error": error_msg}), 500


# --- Get Group Members ---
@app.route("/get-group-members/<group_id>", methods=["GET"])
@login_required
def get_group_members(group_id):
    try:
        members = supabase.table("group_members").select("user_id, users(name, email, avatar_url, username)").eq("group_id", group_id).execute()
        return jsonify({"success": True, "members": members.data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# --- Teacher Verification Admin Route ---
@app.route("/get-pending-teachers", methods=["GET"])
@login_required
def get_pending_teachers():
    try:
        current_user_id = session.get("user_id")
        # Check admin
        current_user = supabase.table("users").select("role").eq("id", current_user_id).execute()
        if not current_user.data or current_user.data[0].get("role") != "admin":
            return jsonify({"success": False, "error": "Unauthorized"}), 403
            
        # Fetch pending from teacher_verifications linked with users
        # Supabase join: teacher_verifications(..., users(...))
        # Note: Supabase-py might need specific syntax for foreign tables
        # or we just fetch verifications and then users.
        # Let's try direct select with join if relations are set up, otherwise simple approach.
        
        # Assuming relations exist:
        pending = supabase.table("teacher_verifications").select("*, users(name, email, institute)").eq("status", "pending").execute()
        return jsonify({"success": True, "teachers": pending.data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/verify-teacher/<user_id>", methods=["POST"])
@login_required
def verify_teacher(user_id):
    try:
        current_user_id = session.get("user_id")
        current_user = supabase.table("users").select("role").eq("id", current_user_id).execute()
        if not current_user.data or current_user.data[0].get("role") != "admin":
            return jsonify({"success": False, "error": "Admin access required"}), 403
        
        # Update User
        supabase.table("users").update({"is_verified": True}).eq("id", user_id).execute()
        # Update Verification Request
        supabase.table("teacher_verifications").update({"status": "approved"}).eq("user_id", user_id).execute()
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    vectorstore = get_vectorstore(data['book_name'])
    retriever = vectorstore.as_retriever()
    template = "Context: {context}\nQuestion: {question}"
    prompt = ChatPromptTemplate.from_template(template)
    chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    return jsonify({'success': True, 'answer': chain.invoke(data['question'])})



@app.route('/teacher/assignment-responses/<assignment_id>', methods=['GET'])
@login_required
def teacher_assignment_responses(assignment_id):
    try:
        user_id = session.get("user_id")
        # Verify ownership
        assign = supabase.table("class_assignments").select("teacher_id").eq("id", assignment_id).execute()
        if not assign.data or assign.data[0]['teacher_id'] != user_id:
             return jsonify({'success': False, 'error': 'Unauthorized'}), 403
             
        # Fetch tests linked to assignment
        # Join with users to get names
        res = supabase.table("tests").select("*, users(name, email)").eq("assignment_id", assignment_id).order("score", desc=True).execute()
        
        return jsonify({'success': True, 'responses': res.data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def draw_wrapped_text(canvas_obj, text, x, y, max_width, font_size):
    """Draw text with word wrapping, returns new y position."""
    canvas_obj.setFont("Helvetica", font_size)
    
    # Calculate approximate chars per line (assuming ~6 pixels per char at size 11)
    chars_per_line = int(max_width / (font_size * 0.5))
    
    words = text.split(' ')
    line = ""
    
    for word in words:
        test_line = line + word + " "
        if len(test_line) > chars_per_line:
            if line:
                canvas_obj.drawString(x, y, line.strip())
                y -= font_size + 3
            line = word + " "
        else:
            line = test_line
    
    # Draw remaining text
    if line.strip():
        canvas_obj.drawString(x, y, line.strip())
        y -= font_size + 3
    
    return y

def create_pdf_buffer(questions, title, institute, date, duration, marks):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)
    width, height = 595.27, 841.89  # A4 size in points
    
    # --- Header ---
    y = height - 50
    
    # Institute Name (Bold, Centered)
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(width / 2, y, institute)
    y -= 20
    
    # Semester/Year (Optional default)
    c.setFont("Helvetica", 11)
    y -= 18
    
    # Exam Title (Bold, Centered)
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(width / 2, y, title)
    y -= 18
    
    # Closed Book Part
    c.setFont("Helvetica-Bold", 11)
    c.drawCentredString(width / 2, y, "Closed Book Part")
    y -= 30
    
    # Metadata Line 1
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Date: {date}")
    c.drawRightString(width - 50, y, f"Max Marks: {marks}")
    y -= 15
    
    # Metadata Line 2
    c.drawString(50, y, f"Max Duration: {duration}")
    y -= 10
    
    # Horizontal Line
    c.line(50, y, width - 50, y)
    y -= 30
    
    # --- Questions ---
    c.setFont("Helvetica", 11)
    
    # Handle if questions is string or list
    q_list = []
    if isinstance(questions, str):
        try:
            q_list = json.loads(questions)
        except:
            q_list = questions.split('\n')
    else:
        q_list = questions
    
    import re
    
    for i, q in enumerate(q_list):
        q_text = str(q).strip()
        if not q_text: 
            continue
        
        # Clean up special/non-printable characters
        q_text = q_text.replace('\u25a0', '').replace('\u25aa', '').replace('\u25ae', '')
        q_text = ''.join(char for char in q_text if ord(char) < 128 or char in 'àáâãäåèéêëìíîïòóôõöùúûüýÿñç')
        
        # Add numbering if not present
        if not q_text[0].isdigit():
            q_text = f"{i+1}. {q_text}"
        
        # Check if MCQ - look for options pattern like [Options: A, B, C, D] or (a) (b) etc.
        options_match = re.search(r'\[Options?:\s*(.+?)\]', q_text, re.IGNORECASE)
        mcq_pattern = re.search(r'([a-dA-D][\)\.])\s+', q_text)
        
        if options_match:
            # Extract question part and options
            main_q = q_text[:options_match.start()].strip()
            options_str = options_match.group(1)
            options = [opt.strip() for opt in options_str.split(',')]
            
            # Draw main question with wrapping
            y = draw_wrapped_text(c, main_q, 50, y, width - 100, 11)
            y -= 5
            
            # Draw options on separate lines
            c.setFont("Helvetica", 10)
            for j, opt in enumerate(options):
                option_label = chr(65 + j)  # A, B, C, D...
                option_text = f"    ({option_label}) {opt}"
                c.drawString(60, y, option_text)
                y -= 14
            
            c.setFont("Helvetica", 11)
            y -= 10
            
        elif mcq_pattern:
            # Pattern like "Question? A) opt1 B) opt2 C) opt3 D) opt4"
            # Try to split on option markers
            parts = re.split(r'([a-dA-D][\)\.])\s+', q_text)
            if len(parts) > 1:
                main_q = parts[0].strip()
                y = draw_wrapped_text(c, main_q, 50, y, width - 100, 11)
                y -= 5
                
                c.setFont("Helvetica", 10)
                for j in range(1, len(parts), 2):
                    if j+1 < len(parts):
                        option_text = f"    {parts[j]} {parts[j+1].strip()}"
                        c.drawString(60, y, option_text[:80])  # Limit length
                        y -= 14
                
                c.setFont("Helvetica", 11)
                y -= 10
            else:
                y = draw_wrapped_text(c, q_text, 50, y, width - 100, 11)
                y -= 20
        else:
            # Regular question - use wrapping
            y = draw_wrapped_text(c, q_text, 50, y, width - 100, 11)
            y -= 20
        
        # New page if needed
        if y < 80:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 11)
    
    # Footer
    c.setFont("Helvetica-Oblique", 10)
    c.drawCentredString(width / 2, 40, "---------------- ALL THE BEST ----------------")
    
    c.save()
    buffer.seek(0)
    return buffer

@app.route('/download-questions', methods=['POST'])
def download_questions():
    try:
        data = request.json
        questions = data.get('questions', [])
        filename = data.get('filename', 'questions.pdf')
        
        institute = data.get('institute', "BIRLA INSTITUTE OF TECHNOLOGY AND SCIENCE, PILANI")
        title = data.get('title', "Comprehensive Exam")
        date = data.get('date', "")
        duration = data.get('duration', "")
        marks = data.get('marks', "")
        
        buffer = create_pdf_buffer(questions, title, institute, date, duration, marks)
        
        return send_file(buffer, as_attachment=True, download_name=filename, mimetype='application/pdf')
        
    except Exception as e:
        print(f"PDF Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/generate-assignment-pdf', methods=['POST'])
@login_required
def generate_assignment_pdf():
    try:
        data = request.json
        assignment_id = data.get('assignment_id')
        
        # Fetch assignment
        assign_res = supabase.table("class_assignments").select("*").eq("id", assignment_id).execute()
        if not assign_res.data:
            return jsonify({'success': False, 'error': 'Assignment not found'}), 404
        
        a = assign_res.data[0]
        
        title = a.get('title', 'Assignment')
        institute = a.get('institute_name') or "BIRLA INSTITUTE OF TECHNOLOGY AND SCIENCE, PILANI"
        date = a.get('due_date') or ""
        duration = str(a.get('duration_minutes') or "") + " mins"
        marks = str(a.get('total_marks') or "")
        questions = a.get('questions', [])
        
        buffer = create_pdf_buffer(questions, title, institute, date, duration, marks)
        
        return send_file(buffer, as_attachment=True, download_name=f"{title.replace(' ', '_')}.pdf", mimetype='application/pdf')
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500



@app.route("/list-books", methods=["GET"])
def list_books():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"books": []})
        
    try:
        docs = DBService.get_documents_by_user(user_id)
        # Return filenames/book_names
        books = [d['filename'] for d in docs]
        return jsonify({"books": books})
    except Exception as e:
        print(f"Error listing books: {e}")
        return jsonify({"books": []})

# --- Classroom Management Routes ---

@app.route("/get-class-details/<class_id>", methods=["GET"])
@login_required
def get_class_details(class_id):
    try:
        user_id = session.get("user_id")
        role = session.get("user_role")

        # 1. Get Class Info
        class_res = supabase.table("classrooms").select("*, users(name)").eq("id", class_id).execute()
        if not class_res.data:
            return jsonify({"success": False, "error": "Class not found"}), 404
        
        class_info = class_res.data[0]
        
        # 2. Authorization Check
        if role == "teacher":
             if class_info["teacher_id"] != user_id:
                  return jsonify({"success": False, "error": "Unauthorized"}), 403
        else:
             # Check if member
             mem_check = supabase.table("class_members").select("*").eq("class_id", class_id).eq("user_id", user_id).execute()
             if not mem_check.data:
                  return jsonify({"success": False, "error": "Not a member"}), 403

        # 3. Get Members
        members = supabase.table("class_members").select("user_id, users(name, email, avatar_url, username)").eq("class_id", class_id).execute()
        
        # 4. Get Assignments
        assignments = supabase.table("class_assignments").select("*").eq("class_id", class_id).order("created_at", desc=True).execute()

        return jsonify({
            "success": True, 
            "class_info": class_info,
            "members": members.data,
            "assignments": assignments.data
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/remove-class-member", methods=["POST"])
@login_required
def remove_class_member():
    try:
        if session.get("user_role") != "teacher":
             return jsonify({"success": False, "error": "Unauthorized"}), 403
             
        data = request.json
        class_id = data.get("class_id")
        user_id_to_remove = data.get("user_id")
        
        # Verify Teacher owns class
        teacher_id = session.get("user_id")
        class_check = supabase.table("classrooms").select("*").eq("id", class_id).eq("teacher_id", teacher_id).execute()
        if not class_check.data:
             return jsonify({"success": False, "error": "Unauthorized class access"}), 403
             
        # Remove
        supabase.table("class_members").delete().eq("class_id", class_id).eq("user_id", user_id_to_remove).execute()
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/create-assignment", methods=["POST"])
@login_required
def create_assignment():
    try:
        if session.get("user_role") != "teacher":
             return jsonify({"success": False, "error": "Unauthorized"}), 403

        data = request.json
        class_id = data.get("class_id")
        teacher_id = session.get("user_id")
        title = data.get("title")
        topics = data.get("topics") # JSON list
        questions = data.get("questions") # JSON list
        difficulty = data.get("difficulty")
        password = data.get("password")
        duration = data.get("duration")
        marks = data.get("marks")
        institute = data.get("institute")
        due_date = data.get("due_date")
        
        # Handle empty date
        if not due_date:
            due_date = None

        # Check ownership
        class_check = supabase.table("classrooms").select("*").eq("id", class_id).eq("teacher_id", teacher_id).execute()
        if not class_check.data:
             return jsonify({"success": False, "error": "Unauthorized"}), 403

        # Insert Assignment
        supabase.table("class_assignments").insert({
            "class_id": class_id,
            "teacher_id": teacher_id,
            "title": title,
            "topics": topics, 
            "questions": questions, 
            "difficulty": difficulty,
            "due_date": due_date,
            "password": password,
            "duration_minutes": int(duration) if duration else None,
            "total_marks": int(marks) if marks else None,
            "institute_name": institute
        }).execute()

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500



@app.route("/teacher/assignments", methods=["GET"])
@login_required
def teacher_assignments():
    try:
        if session.get("user_role") != "teacher":
             return jsonify({"success": False, "error": "Unauthorized"}), 403
             
        user_id = session.get("user_id")
        
        # Fetch assignments created by him
        res = supabase.table("class_assignments").select("*, classrooms(name)").eq("teacher_id", user_id).order("created_at", desc=True).execute()
        
        return jsonify({"success": True, "assignments": res.data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# --- Admin Panel Routes ---

def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
             return jsonify({"success": False, "error": "Login required"}), 401
        
        # Check if actually admin
        # Optimization: Store is_admin in session, but for security verify against DB or relying on session set at login
        if session.get("user_role") != "admin":
             return jsonify({"success": False, "error": "Admin access required"}), 403
        return f(*args, **kwargs)
    return wrapper

@app.route("/admin/users", methods=["GET"])
@admin_required
def admin_get_users():
    try:
        users = supabase.table("users").select("*").order("created_at", desc=True).execute()
        return jsonify({"success": True, "users": users.data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/admin/delete-user/<user_id>", methods=["DELETE"])
@admin_required
def admin_delete_user(user_id):
    try:
        # Supabase cascade should handle details, but let's be sure
        supabase.table("users").delete().eq("id", user_id).execute()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/admin/classes", methods=["GET"])
@admin_required
def admin_get_classes():
    try:
        # Fetch classes with teacher details
        classes = supabase.table("classrooms").select("*, users(name, email)").order("created_at", desc=True).execute()
        return jsonify({"success": True, "classes": classes.data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/admin/delete-class/<class_id>", methods=["DELETE"])
@admin_required
def admin_delete_class(class_id):
    try:
        supabase.table("classrooms").delete().eq("id", class_id).execute()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/admin/resources", methods=["GET"])
@admin_required
def admin_get_resources():
    try:
        # Use DB to get documents
        docs = DBService.get_all_documents()
        # Map to expected format
        books = [{"id": d['id'], "name": d['filename']} for d in docs]
        return jsonify({"success": True, "resources": books})
    except Exception as e:
         return jsonify({"success": False, "error": str(e)}), 500

@app.route("/admin/delete-resource/<resource_id>", methods=["DELETE"])
@admin_required
def admin_delete_resource(resource_id):
    try:
        DBService.delete_document(resource_id)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


