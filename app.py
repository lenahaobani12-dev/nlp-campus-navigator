import streamlit as st
import pickle
import re
import os
import tempfile
import subprocess

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gtts import gTTS
import speech_recognition as sr

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

responses_smart = {
    "admin_office": {
        "keywords": {
            "admission":   "The Admission Office is in the Main Administration Building, Ground Floor.",
            "register":    "You can register for courses at the Registration Office near the main entrance.",
            "enroll":      "Enrollment services are available at the Administration Building.",
            "student id":  "Student ID services are at the Student Affairs Office, Building 1.",
            "transcript":  "You can request transcripts from the Academic Records Office, Floor 1.",
            "graduation":  "Graduation applications are submitted at the Registrar Office.",
            "scholarship": "The Scholarship Office is in the Administration Building, Floor 2.",
            "financial":   "The Financial Aid Office is in the Administration Building, Floor 2.",
            "advisor":     "Academic advisors are available in the Academic Affairs Office.",
            "student":     "The Student Affairs Office is located in the main building."
        },
        "default": "Please visit the Administration Building for all registration and enrollment services."
    },
    "facility_location": {
        "keywords": {
            "library":   "The Library is located in Building 12 on the main campus.",
            "cafeteria": "The cafeteria and food court are near the Student Center.",
            "food":      "You can find food options near the Student Center.",
            "eat":       "The cafeteria is located near the Student Center.",
            "gym":       "The gym is in the Sports Complex on the east side of campus.",
            "health":    "The health center is located near the main gate.",
            "print":     "Printing services are available in the library and student center.",
            "mosque":    "The campus mosque is located in the central area near Building 5.",
            "prayer":    "The prayer room is in Building 3, Ground Floor.",
            "atm":       "ATM machines are located near the main entrance and the Student Center.",
            "bookstore": "The bookstore is in the Student Center, Ground Floor.",
            "coffee":    "There is a coffee shop near the library entrance.",
            "clinic":    "The campus medical clinic is near the main gate.",
            "study":     "Quiet study rooms are available in the library, floors 2 and 3."
        },
        "default": "Facilities are distributed around the main campus. Check the campus map for details."
    },
    "building_location": {
        "keywords": {
            "engineering": "The Engineering Building is on the west side of campus, near Gate 3.",
            "science":     "The Science Building is in the central campus area.",
            "arts":        "The Faculty of Arts is in Building D, east side of campus.",
            "it":          "The IT Building is near the library, Building 10.",
            "computer":    "The Computer Science Building is Block C, near the library.",
            "nursing":     "The Nursing College is in Building 7, south campus.",
            "pharmacy":    "The Pharmacy Building is near the health center.",
            "business":    "The Business School is in Building E, near Gate 2.",
            "education":   "The College of Education is in Building F.",
            "main":        "The main building is located near the main entrance.",
            "research":    "The Research Center is in Building 15, north campus."
        },
        "default": "Academic buildings are labeled A-F. Please refer to the campus map for the exact location."
    },
    "classroom_location": {
        "keywords": {
            "room":       "Please check your timetable for the exact room number and floor.",
            "class":      "Your class location is listed in your course schedule.",
            "lecture":    "Lecture halls are on floors 1-3 of the academic buildings.",
            "auditorium": "The auditorium is in Building B, Ground Floor.",
            "lab":        "Labs are located on floor 2 of the science and engineering buildings.",
            "exam":       "Exam halls are in Buildings A and B. Check your exam schedule for the room.",
            "studio":     "Studios are in the Arts and Design Building, floor 3.",
            "simulation": "The nursing simulation lab is in Building 7, floor 2.",
            "robotics":   "The robotics lab is in the Engineering Building, floor 3."
        },
        "default": "Please check your schedule for classroom details or ask at the information desk."
    },
    "parking_location": {
        "keywords": {
            "visitor":    "Visitor parking is available at Gate 2.",
            "staff":      "Staff parking is near Gate 1.",
            "faculty":    "Faculty parking is reserved near Gate 1 and Gate 3.",
            "gate":       "Parking is available near Gate 1 and Gate 4.",
            "motorcycle": "Motorcycle parking is near Gate 3.",
            "car":        "Student parking is near Gate 4.",
            "disability": "Disability parking spaces are available near all main entrances.",
            "bus":        "The campus bus stop is near the main entrance, Gate 1.",
            "permit":     "Parking permits are issued from the Security Office near Gate 1.",
            "electric":   "Electric vehicle charging stations are in parking zone B."
        },
        "default": "Parking lots are available near Gate 1 and Gate 4. Student parking is near Gate 4."
    }
}

def smart_response(intent, user_input):
    user_input = user_input.lower()
    intent_data = responses_smart[intent]
    for keyword, response in intent_data["keywords"].items():
        if keyword in user_input:
            return response
    return intent_data["default"]

intent_labels = {
    "admin_office":      "Admin Office",
    "facility_location": "Facility",
    "building_location": "Building",
    "classroom_location":"Classroom",
    "parking_location":  "Parking"
}

def run_pipeline(user_input):
    cleaned  = clean_text(user_input)
    vec      = tfidf.transform([cleaned])
    intent   = model.predict(vec)[0]
    proba    = model.predict_proba(vec).max() * 100
    response = smart_response(intent, user_input)
    label    = intent_labels.get(intent, intent)
    return cleaned, intent, label, proba, response

def speak_response(text):
    tts = gTTS(text=text, lang="en")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    st.audio(tmp.name, format="audio/mp3", autoplay=True)

def show_pipeline(user_input, cleaned, label, proba, response):
    st.markdown(f"""
    <div style="background:#f0faf6;border:1px solid #9FE1CB;border-radius:10px;padding:14px;margin:10px 0;font-family:sans-serif;">
      <div style="display:flex;gap:10px;align-items:center;margin:6px 0;font-size:14px;">
        <span style="background:#1D9E75;color:white;border-radius:50%;width:24px;height:24px;display:inline-flex;align-items:center;justify-content:center;font-size:12px;font-weight:bold;">1</span>
        Speech Recognition &nbsp;→&nbsp; <b>{user_input}</b>
      </div>
      <div style="display:flex;gap:10px;align-items:center;margin:6px 0;font-size:14px;">
        <span style="background:#1D9E75;color:white;border-radius:50%;width:24px;height:24px;display:inline-flex;align-items:center;justify-content:center;font-size:12px;font-weight:bold;">2</span>
        Preprocessing &nbsp;→&nbsp; <b>{cleaned}</b>
      </div>
      <div style="display:flex;gap:10px;align-items:center;margin:6px 0;font-size:14px;">
        <span style="background:#1D9E75;color:white;border-radius:50%;width:24px;height:24px;display:inline-flex;align-items:center;justify-content:center;font-size:12px;font-weight:bold;">3</span>
        Intent Detection &nbsp;→&nbsp; <b>{label}</b> &nbsp;
        <span style="background:#fff3cd;color:#856404;padding:2px 8px;border-radius:20px;font-size:12px;">{proba:.1f}%</span>
      </div>
      <div style="display:flex;gap:10px;align-items:center;margin:6px 0;font-size:14px;">
        <span style="background:#1D9E75;color:white;border-radius:50%;width:24px;height:24px;display:inline-flex;align-items:center;justify-content:center;font-size:12px;font-weight:bold;">4</span>
        Database Lookup &nbsp;→&nbsp; <b>Response retrieved</b>
      </div>
      <div style="display:flex;gap:10px;align-items:center;margin:6px 0;font-size:14px;">
        <span style="background:#1D9E75;color:white;border-radius:50%;width:24px;height:24px;display:inline-flex;align-items:center;justify-content:center;font-size:12px;font-weight:bold;">5</span>
        Voice Output &nbsp;→&nbsp; <b>Audio generated</b>
      </div>
      <div style="background:white;border:1px solid #dee2e6;border-radius:8px;padding:12px 16px;margin-top:10px;font-size:15px;color:#333;">
        💬 <b>Response:</b> {response}
      </div>
    </div>
    """, unsafe_allow_html=True)

st.set_page_config(page_title="PNU Campus Navigator", page_icon="🎓", layout="centered")

st.markdown("""
<div style="background:#1D9E75;color:white;padding:16px 20px;border-radius:12px;text-align:center;margin-bottom:20px;">
  <h2 style="margin:0;">🎓 PNU Campus Navigator</h2>
  <p style="margin:4px 0 0;opacity:0.85;">Voice-Activated Virtual Assistant · CAI 350</p>
</div>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

tab1, tab2 = st.tabs(["💬 Text Input", "🎙️ Voice Input"])

with tab1:
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask about a location on campus:", placeholder="e.g. Where is the library?")
        submitted  = st.form_submit_button("Send")

    st.markdown("**Quick questions:**")
    cols  = st.columns(5)
    quick = ["Where is the library?", "Where can I park?", "Admission office?", "Where is room 101?", "Where is the cafeteria?"]
    for i, q in enumerate(quick):
        if cols[i].button(q, key=f"q{i}", use_container_width=True):
            user_input = q
            submitted  = True

    if submitted and user_input:
        cleaned, intent, label, proba, response = run_pipeline(user_input)
        st.session_state.history.append((user_input, response, label, proba, cleaned))
        show_pipeline(user_input, cleaned, label, proba, response)
        speak_response(response)

    if st.session_state.history:
        st.markdown("---")
        st.markdown("**Chat History:**")
        for q, r, lbl, conf, _ in reversed(st.session_state.history):
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.write(r)
                st.caption(f"{lbl} · {conf:.1f}%")

with tab2:
    st.markdown("Click **Start Recording**, speak your question, then click **Stop Recording**.")
    st.markdown("---")

    if st.button("🎙️ Start Recording", type="primary", use_container_width=True):
        with st.spinner("Listening... speak now!"):
            try:
                recognizer = sr.Recognizer()
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source, duration=1)
                    st.info("Listening... speak your question now!")
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)

                text = recognizer.recognize_google(audio, language="en-US")
                st.success(f'You said: "{text}"')

                cleaned, intent, label, proba, response = run_pipeline(text)
                st.session_state.history.append((text, response, label, proba, cleaned))
                show_pipeline(text, cleaned, label, proba, response)
                speak_response(response)

            except sr.WaitTimeoutError:
                st.warning("No speech detected. Please try again.")
            except sr.UnknownValueError:
                st.error("Could not understand. Please speak clearly.")
            except sr.RequestError:
                st.error("Speech recognition unavailable. Check your internet.")
            except Exception as e:
                st.error(f"Microphone error: {str(e)}")
                st.info("If microphone does not work, upload a voice file below.")

    st.markdown("---")
    st.markdown("**Or upload a voice file:**")
    audio_file = st.file_uploader("Upload voice file", type=["wav", "mp3", "m4a", "ogg"])

    if audio_file:
        st.audio(audio_file)
        with st.spinner("Recognizing speech..."):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.write(audio_file.read())
            tmp.close()
            wav_path = tmp.name.replace(".wav", "_conv.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp.name, "-vn",
                 "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", wav_path],
                capture_output=True
            )
            recognizer = sr.Recognizer()
            try:
                with sr.AudioFile(wav_path) as source:
                    audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data, language="en-US")
                st.success(f'You said: "{text}"')
                cleaned, intent, label, proba, response = run_pipeline(text)
                st.session_state.history.append((text, response, label, proba, cleaned))
                show_pipeline(text, cleaned, label, proba, response)
                speak_response(response)
            except sr.UnknownValueError:
                st.error("Could not understand the audio. Please try again.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                if os.path.exists(tmp.name): os.unlink(tmp.name)
                if os.path.exists(wav_path): os.unlink(wav_path)