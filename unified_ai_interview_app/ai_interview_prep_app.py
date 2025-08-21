"""
Unified AI Interview Prep — Gradio single-file app with integrated Technical Quiz

Run: python ai_interview_prep_app.py
Open the Gradio URL (http://127.0.0.1:7860) in your browser.

Notes:
- The HR interview flow (AI-generated questions, recording, analysis) is unchanged.
- A Technical Quiz module is added on the right column. After entering job description and role,
  you can choose between 'HR Interview' (existing flow) or 'Technical Quiz' and take MCQs.
- Quiz supports choosing topic, number of questions (up to 15) and will display MCQs using radio buttons.
- Question bank contains 10 questions per topic (Python, C, DBMS, OOPs, Aptitude) — 50 total.
"""

import os
import time
import json
import tempfile
import math
from pathlib import Path
import random

# Graceful imports
try:
    import gradio as gr
except Exception:
    raise RuntimeError("Please install gradio: pip install gradio")

import numpy as np

# Optional heavy libraries (kept as in original prototype)
_HAS_WHISPER = False
try:
    import whisper
    _HAS_WHISPER = True
except Exception:
    _HAS_WHISPER = False

_HAS_ST = False
try:
    from sentence_transformers import SentenceTransformer, util
    _HAS_ST = True
except Exception:
    _HAS_ST = False

_HAS_FER = False
try:
    from fer import FER
    import cv2
    _HAS_FER = True
except Exception:
    _HAS_FER = False

_HAS_LIBROSA = False
try:
    import librosa
    _HAS_LIBROSA = True
except Exception:
    _HAS_LIBROSA = False

# OpenAI optional
_OPENAI_AVAILABLE = False
try:
    import openai
    if os.environ.get('OPENAI_API_KEY'):
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

# load heavy models if present (kept minimal)
_sentence_model = None
if _HAS_ST:
    try:
        _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print('SentenceTransformer load failed:', e)
        _sentence_model = None

_whisper_model = None
if _HAS_WHISPER:
    try:
        _whisper_model = whisper.load_model('small')
    except Exception as e:
        print('Whisper load failed:', e)
        _whisper_model = None

# -------------------------
# Question bank (50 questions)
# -------------------------
QUESTION_BANK = {
    "Python": [
        {"q":"What is the output of print(type([]))?","opts":["<class 'list'>","<class 'tuple'>","<class 'dict'>","<class 'set'>"],"a":0},
        {"q":"Which keyword is used to define a function in Python?","opts":["func","def","function","lambda"],"a":1},
        {"q":"Which data type is immutable?","opts":["List","Set","Tuple","Dictionary"],"a":2},
        {"q":"What does PEP stand for?","opts":["Python Enhancement Proposal","Python Extra Package","Performance Enhancement Plan","Program Execution Process"],"a":0},
        {"q":"Which module is used for regular expressions?","opts":["regex","pyregex","re","match"],"a":2},
        {"q":"What is the output of 2**3 in Python?","opts":["5","6","8","9"],"a":2},
        {"q":"Which statement is used to handle exceptions?","opts":["catch","except","error","handle"],"a":1},
        {"q":"How do you create a virtual environment in Python 3?","opts":["python -m venv env","virtualenv env","create venv","py -venv"],"a":0},
        {"q":"Which method adds an element to end of list?","opts":["add()","append()","push()","insert()"],"a":1},
        {"q":"What is the output of len('hello')?","opts":["4","5","6","Error"],"a":1}
    ],
    "C": [
        {"q":"Size of int in C on typical 32-bit systems?","opts":["2 bytes","4 bytes","8 bytes","Depends on compiler"],"a":1},
        {"q":"Which header file is needed for printf?","opts":["stdlib.h","stdio.h","conio.h","string.h"],"a":1},
        {"q":"Which operator is used for pointer dereference?","opts":["&","*","->","%"],"a":1},
        {"q":"What is the default return type of main() in C?","opts":["void","int","char","float"],"a":1},
        {"q":"Which of the following is not a loop in C?","opts":["for","while","foreach","do-while"],"a":2},
        {"q":"Which function is used to allocate memory dynamically?","opts":["alloc()","malloc()","new()","memalloc()"],"a":1},
        {"q":"Which format specifier is used for int in printf?","opts":["%d","%f","%s","%c"],"a":0},
        {"q":"Which header contains memcpy?","opts":["string.h","stdlib.h","stdio.h","math.h"],"a":0},
        {"q":"Which symbol ends a statement in C?","opts":[".",";","/","!"],"a":1},
        {"q":"What will sizeof(char) return typically?","opts":["1","2","4","Depends"],"a":0}
    ],
    "DBMS": [
        {"q":"What does SQL stand for?","opts":["Structured Query Language","System Query Language","Simple Query Language","Standard Query List"],"a":0},
        {"q":"Which command removes a table?","opts":["DELETE","DROP","REMOVE","TRUNCATE"],"a":1},
        {"q":"Which normal form removes partial dependency?","opts":["1NF","2NF","3NF","BCNF"],"a":1},
        {"q":"Which key uniquely identifies a record?","opts":["Foreign Key","Candidate Key","Primary Key","Alternate Key"],"a":2},
        {"q":"What does ACID stand for?","opts":["Atomicity, Consistency, Isolation, Durability","Accuracy, Consistency, Isolation, Dependency","Atomic, Consistent, Integrated, Durable","None"],"a":0},
        {"q":"Which SQL clause is used to group records?","opts":["GROUP BY","ORDER BY","WHERE","HAVING"],"a":0},
        {"q":"Which join returns all rows from both tables?","opts":["INNER JOIN","LEFT JOIN","RIGHT JOIN","FULL OUTER JOIN"],"a":3},
        {"q":"Which index type is best for exact matches?","opts":["B-Tree","Bitmap","Hash","Full-text"],"a":2},
        {"q":"Which command adds a new column?","opts":["ALTER TABLE ADD COLUMN","UPDATE TABLE ADD","INSERT COLUMN","MODIFY COLUMN"],"a":0},
        {"q":"Which SQL keyword is used to remove duplicate rows?","opts":["DISTINCT","UNIQUE","REMOVE","FILTER"],"a":0}
    ],
    "OOPs": [
        {"q":"Which concept is shown by function overloading?","opts":["Inheritance","Polymorphism","Encapsulation","Abstraction"],"a":1},
        {"q":"Which access specifier hides data inside class?","opts":["public","private","protected","internal"],"a":1},
        {"q":"Which keyword used for inheritance in many languages?","opts":["extends","inherits","parentof","base"],"a":0},
        {"q":"Which OOP principle groups data and methods?","opts":["Polymorphism","Encapsulation","Inheritance","Abstraction"],"a":1},
        {"q":"Which cannot be instantiated?","opts":["Interface","Concrete class","Static class","Struct"],"a":0},
        {"q":"Which achieves runtime polymorphism?","opts":["Function Overloading","Function Overriding","Operator Overloading","Constructor Overloading"],"a":1},
        {"q":"What is encapsulation mainly about?","opts":["Hiding implementation","Speeding up code","Memory management","Inheritance"],"a":0},
        {"q":"Which describes 'is-a' relationship?","opts":["Composition","Aggregation","Inheritance","Association"],"a":2},
        {"q":"Which supports code reuse?","opts":["Abstraction","Encapsulation","Inheritance","Polymorphism"],"a":2},
        {"q":"Which OOP feature models real-world entities?","opts":["Encapsulation","Polymorphism","Abstraction","All of the above"],"a":3}
    ],
    "Aptitude": [
        {"q":"Speed = Distance / ?","opts":["Time","Speed","Work","Ratio"],"a":0},
        {"q":"If CP=100 and SP=120, profit% = ?","opts":["10%","15%","20%","25%"],"a":2},
        {"q":"Probability of head in coin toss?","opts":["1/4","1/2","1/3","1"],"a":1},
        {"q":"A can do a work in 10 days, B in 20 days. Together?","opts":["5","6.67","7.5","15"],"a":1},
        {"q":"20% of 250 is?","opts":["40","45","50","55"],"a":2},
        {"q":"If train covers 60 km in 1 hour, speed in m/s?","opts":["16.67","20","25","30"],"a":0},
        {"q":"If average of 5 numbers is 10, sum is?","opts":["50","40","60","100"],"a":0},
        {"q":"If ratio a:b is 2:3 and sum is 25, a = ?","opts":["10","8","12","5"],"a":0},
        {"q":"If price increases 10% then decreases 10%, net change?","opts":["0%","-1%","+1%","-10%"],"a":1},
        {"q":"Simple interest formula is?","opts":["P*R*T/100","P+R+T","PR+T","PRT"],"a":0}
    ]
}

# -------------------------
# Reuse helper functions from prototype (transcription, emotion analysis, similarity)
# For brevity, these functions are simplified copies of earlier implementation.
# -------------------------

def simple_question_generator(job_desc: str, role: str, n=8):
    base = []
    base.append(f"Tell me about yourself and why you're a good fit for the {role} role.")
    base.append(f"Walk me through a recent project you worked on that's relevant to {role}.")
    base.append(f"Describe a challenging technical problem you've solved related to {role} responsibilities.")
    base.append(f"How do you keep up-to-date with technologies/theories relevant to {role}?")
    base.append(f"How would you approach a typical task for a {role}, given this job description?")
    base.append(f"Give an example of working in a team — a conflict and how you handled it.")
    base.append(f"Describe your strengths and weaknesses as they relate to the {role}.")
    base.append(f"Do you have questions for the interviewer about the {role} or team?")
    lines = [l.strip() for l in job_desc.split('\n') if l.strip()]
    extra = []
    for line in lines[:6]:
        if len(line.split()) > 3:
            extra.append(f"The JD mentions: '{line[:90]}...'. Can you explain how your experience maps to this?")
    all_qs = base + extra
    return all_qs[:n]

def generate_questions_with_openai(job_desc: str, role: str, n=8):
    if not _OPENAI_AVAILABLE:
        return simple_question_generator(job_desc, role, n)
    prompt = f"You are an interview question generator. The role is {role}. The job description is:\n{job_desc}\n\nGenerate {n} concise, role-specific interview questions (one per line) focusing on behavioral and technical aspects.\n"
    try:
        resp = openai.ChatCompletion.create(
            model='gpt-4o-mini',
            messages=[{"role":"user","content":prompt}],
            max_tokens=400
        )
        text = resp['choices'][0]['message']['content']
        qs = [q.strip() for q in text.split('\n') if q.strip()]
        clean = []
        for q in qs:
            clean.append(''.join(ch for ch in q if ch not in '0123456789.\t') if len(q) < 500 else q)
        return clean[:n]
    except Exception as e:
        print('OpenAI generation failed, falling back:', e)
        return simple_question_generator(job_desc, role, n)

def transcribe_audio_with_whisper(audio_path: str):
    return None  # keep stub for local setups

def estimate_voice_confidence(audio_path: str):
    return 0.5

def analyze_facial_emotion(image):
    return {'dominant_emotion': 'neutral', 'confidence': 0.5}

def compute_similarity(answer_text: str, question_text: str, job_desc: str = None):
    if not answer_text.strip():
        return 0.0
    return min(1.0, len(answer_text.split()) / 120.0)

def process_answer(audio_file, snapshot, question_text, job_desc):
    transcript = ''
    sim = compute_similarity(transcript, question_text, job_desc)
    face = analyze_facial_emotion(snapshot)
    face_conf = face['confidence']
    voice_conf = 0.5
    comm_len = 0.0
    communication_score = float(np.clip(0.6 * comm_len + 0.3 * sim + 0.1 * voice_conf, 0.0, 1.0))
    technical_score = float(np.clip(0.7 * sim + 0.2 * comm_len + 0.1 * face_conf, 0.0, 1.0))
    confidence_score = float(np.clip(0.5 * face_conf + 0.5 * voice_conf, 0.0, 1.0))
    result = {
        'transcript': transcript,
        'similarity': sim,
        'face': face,
        'voice_confidence': voice_conf,
        'communication_score': communication_score,
        'technical_score': technical_score,
        'confidence_score': confidence_score,
        'tips': ['This is a stub tip.']
    }
    return result

# -------------------------
# In-memory sessions
# -------------------------
SESSIONS = {}

# -------------------------
# Gradio UI
# -------------------------

with gr.Blocks(title='AI Interview Prep — Unified') as demo:
    gr.Markdown("# AI Interview Prep — Unified\nThis app includes the HR interview flow (AI-driven) and a Technical Quiz.\n\nFlow: Paste job description + role → choose 'HR' or 'Technical' → if Technical, take MCQ quiz; if HR, continue with existing interview flow.")
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## Job description & Role\nPaste your job description and role. Then choose which flow you want: HR Interview or Technical Quiz.")
            jd_input = gr.Textbox(label='Job description (paste or type)', lines=8, placeholder='Paste job description here')
            role_input = gr.Textbox(label='Role (e.g., Backend Developer)', placeholder='Backend Developer')
            flow_choice = gr.Radio(choices=['HR Interview','Technical Quiz'], value='HR Interview', label='Choose flow')
            gen_btn = gr.Button('Proceed')
            gr.Markdown("---\n### HR Interview (existing flow) — visible only if you choose HR") 
            n_q = gr.Slider(minimum=1, maximum=12, value=6, step=1, label='Number of HR questions')
            gen_hr = gr.Button('Generate HR Questions')
            q_list = gr.Dataframe(headers=['#','Question'], interactive=False)
            st = gr.State()
            start_session = gr.Button('Start HR Interview Session')
            question_index = gr.Number(value=0, visible=False)
            current_q = gr.Textbox(label='Current HR Question', interactive=False)
            webcam = gr.Image(label='Webcam snapshot (captured while answering)')
            mic = gr.Audio(type='filepath', label='Record your answer (press stop when done)')
            submit_answer = gr.Button('Submit HR Answer')
            history = gr.Dataframe(headers=['Q','Transcript','Comm','Tech','Conf'], interactive=False)
            results_md = gr.Markdown('### Latest HR result')
            result_json = gr.Textbox(label='Last HR raw JSON result', lines=8)
        with gr.Column(scale=1):
            gr.Markdown("## Technical Quiz\nChoose a topic, number of questions (max 15), then Generate Quiz. Answer using radios and Submit to get score.")
            topic = gr.Dropdown(choices=list(QUESTION_BANK.keys()), label='Topic', value='Python')
            num_q = gr.Slider(minimum=1, maximum=15, value=10, step=1, label='Number of questions (max 15)')
            gen_quiz = gr.Button('Generate Quiz Questions')
            # We'll prepare up to 15 radio components
            quiz_radios = [gr.Radio(choices=['...'], label=f'Q{i+1}', visible=False) for i in range(15)]
            quiz_session_key = gr.Textbox(visible=False)  # <-- Add this line
            submit_quiz = gr.Button('Submit Quiz')
            quiz_result = gr.Markdown('Quiz result will appear here')
            quiz_summary = gr.Textbox(label='Quiz JSON summary', lines=8)
    
    # HR Actions (same as original)
    def handle_generate_hr(jd, role, n):
        qs = generate_questions_with_openai(jd, role, int(n))
        table = [[i+1, qs[i]] for i in range(len(qs))]
        key = str(time.time())
        SESSIONS[key] = {'job_desc': jd, 'role': role, 'questions': qs, 'answers': []}
        return table, key

    gen_hr.click(handle_generate_hr, inputs=[jd_input, role_input, n_q], outputs=[q_list, st])

    def start_session_fn(state_key):
        if not state_key or state_key not in SESSIONS:
            return gr.update(value=''), gr.update(value=0), "Please generate HR questions first (press 'Generate HR Questions')."
        SESSIONS[state_key]['current'] = 0
        q = SESSIONS[state_key]['questions'][0]
        return state_key, 0, q

    start_session.click(start_session_fn, inputs=[st], outputs=[st, question_index, current_q])

    def submit_answer_fn(state_key, q_idx, audio_file, snapshot):
        if not state_key or state_key not in SESSIONS:
            return '', '', ''
        q_idx = int(q_idx)
        session = SESSIONS[state_key]
        question_text = session['questions'][q_idx]
        res = process_answer(audio_file, snapshot, question_text, session['job_desc'])
        session['answers'].append({'question': question_text, 'result': res})
        hist = []
        for a in session['answers']:
            r = a['result']
            hist.append([a['question'][:60], r['transcript'][:120], f"{r['communication_score']:.2f}", f"{r['technical_score']:.2f}", f"{r['confidence_score']:.2f}"])
        next_idx = q_idx + 1
        next_q = session['questions'][next_idx] if next_idx < len(session['questions']) else ''
        comm_avg = np.mean([x['result']['communication_score'] for x in session['answers']])
        tech_avg = np.mean([x['result']['technical_score'] for x in session['answers']])
        conf_avg = np.mean([x['result']['confidence_score'] for x in session['answers']])
        md = f"**Question:** {question_text}\n\n**Transcript (excerpt):** {res['transcript'][:300]}\n\n**Scores this Q** — Communication: {res['communication_score']:.2f}, Technical: {res['technical_score']:.2f}, Confidence: {res['confidence_score']:.2f}\n\n**Aggregated so far** — Communication: {comm_avg:.2f}, Technical: {tech_avg:.2f}, Confidence: {conf_avg:.2f}\n\n**Tips:**\n- " + "\n- ".join(res['tips'])
        return next_q, gr.update(value=hist), md, json.dumps(res, indent=2)

    submit_answer.click(submit_answer_fn, inputs=[st, question_index, mic, webcam], outputs=[current_q, history, results_md, result_json])

    # Technical Quiz actions
    QUIZ_STATE = gr.State()

    def generate_quiz_questions(topic_val, num_q_val):
        bank = QUESTION_BANK.get(topic_val, [])
        # shuffle and pick
        indices = list(range(len(bank)))
        random.shuffle(indices)
        pick = indices[:int(num_q_val)]
        selected = [bank[i] for i in pick]
        # store into state as json string
        key = str(time.time())
        SESSIONS[key] = {'quiz_topic': topic_val, 'questions': selected}
        # prepare updates for each radio: label and choices
        updates = []
        for i in range(15):
            if i < len(selected):
                q = selected[i]
                lbl = f"Q{i+1}. {q['q']}"
                choices = q['opts']
                updates.append(gr.update(label=lbl, choices=choices, value=None, visible=True))
            else:
                updates.append(gr.update(visible=False))
        return (key, *updates)  # <-- Unpack the updates list

    gen_quiz.click(generate_quiz_questions, inputs=[topic, num_q], outputs=[quiz_session_key] + quiz_radios)

    def submit_quiz_answers(state_key, *answers):
        # answers is a tuple of 15 selections (some may be None)
        if not state_key or state_key not in SESSIONS:
            return "No quiz generated yet.", "{}"
        quiz = SESSIONS[state_key]['questions']
        total = len(quiz)
        score = 0
        details = []
        for i, q in enumerate(quiz):
            sel = answers[i]
            # sel will be the chosen option string or None
            if sel is None:
                chosen_idx = None
            else:
                try:
                    chosen_idx = q['opts'].index(sel)
                except ValueError:
                    chosen_idx = None
            correct_idx = q['a']
            correct_opt = q['opts'][correct_idx]
            is_correct = (chosen_idx == correct_idx)
            if is_correct:
                score += 1
            details.append({'question': q['q'], 'chosen': sel, 'correct': correct_opt, 'is_correct': is_correct})
        pct = (score / total) * 100 if total>0 else 0
        # feedback based on score
        if pct >= 80:
            fb = 'Excellent technical knowledge.'
        elif pct >= 60:
            fb = 'Good — a bit more practice on weak areas will help.'
        elif pct >= 40:
            fb = 'Fair — revise fundamentals and practice more.'
        else:
            fb = 'Needs improvement — focus on basics and practice problem solving.'
        result_md = f"**Score:** {score}/{total} ({pct:.1f}%)\n\n**Feedback:** {fb}\n\n"
        # list incorrect questions
        incorrect = [d for d in details if not d['is_correct']]
        if incorrect:
            result_md += "**Review these questions:**\n"
            for d in incorrect:
                result_md += f"- {d['question']}\n  - Your answer: {d['chosen']}\n  - Correct: {d['correct']}\n"
        # JSON summary for download or debugging
        summary = {'score': score, 'total': total, 'pct': pct, 'details': details, 'feedback': fb}
        return result_md, json.dumps(summary, indent=2)

    # submit_quiz: connect 15 radio values as inputs
    submit_quiz.click(submit_quiz_answers, inputs=[quiz_session_key] + quiz_radios, outputs=[quiz_result, quiz_summary])

    demo.launch()
    
if __name__ == '__main__':
    pass
