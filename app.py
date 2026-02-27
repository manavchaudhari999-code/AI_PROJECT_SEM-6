import streamlit as st
from vision.image_router import detect_image_type
from llm.llm_loader import load_llm
from rag.retriever import load_retriever
from vision.xray_analyzer import analyze_xray
from utils.prompts import MEDICAL_PROMPT

# Initialize models
llm = load_llm()
retriever = load_retriever()

st.warning(
    "⚠️ This system is for educational and decision-support purposes only. "
    "It does NOT provide medical diagnosis. Always consult a qualified healthcare professional."
)

st.title("🩺 RAG Medical Assistant")

symptoms = st.text_area("Enter symptoms")
xray = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

evaluation_mode = st.checkbox("Enable Evaluation Mode")

# ---------------- Symptom Scoring ----------------
def symptom_score(symptoms_text):
    score = 0
    symptoms_text = symptoms_text.lower()

    if "fever" in symptoms_text:
        score += 0.2
    if "breath" in symptoms_text or "shortness" in symptoms_text:
        score += 0.3
    if "productive cough" in symptoms_text:
        score += 0.2
    if "chest pain" in symptoms_text:
        score += 0.1
    if "cough" in symptoms_text:
        score += 0.1

    return min(score, 1.0)


if st.button("Analyze"):

    # ---------- RAG Retrieval ----------
    context = ""
    if symptoms:
        docs = retriever.invoke(symptoms)
        context = "\n".join([d.page_content for d in docs]) if docs else ""

    # ---------- Image Handling ----------
    image_type = "No image provided"
    image_confidence = 0.0
    xray_score = None
    xray_interpretation = "No imaging provided"

    if xray is not None:
        with open("temp_image.jpg", "wb") as f:
            f.write(xray.read())

        image_type, image_confidence = detect_image_type("temp_image.jpg")

        if "x-ray" in image_type or "radiograph" in image_type:
            xray_score = analyze_xray("temp_image.jpg")

            if xray_score < 0.3:
                xray_interpretation = "Low likelihood of pneumonia on chest X-ray"
            elif xray_score < 0.6:
                xray_interpretation = "Indeterminate chest X-ray findings"
            else:
                xray_interpretation = "High likelihood of pneumonia on chest X-ray"
        else:
            xray_interpretation = (
                f"The uploaded image appears to be a {image_type}. "
                "No specialized diagnostic model available for this type."
            )

    # ---------- Probability Fusion ----------
    symptom_prob = symptom_score(symptoms)

    if xray_score is not None:
        final_probability = 0.7 * xray_score + 0.3 * symptom_prob
    else:
        final_probability = symptom_prob

    if final_probability >= 0.7:
        final_confidence = "High"
    elif final_probability >= 0.4:
        final_confidence = "Moderate"
    else:
        final_confidence = "Low"

    # ---------- Dynamic Confidence Explanation ----------
    if xray_score is None:
        confidence_explanation = (
            "Confidence based primarily on symptom severity due to absence of imaging."
        )
    else:
        confidence_explanation = (
            f"Confidence derived from weighted fusion of imaging probability "
            f"({xray_score:.2f}) and symptom severity score ({symptom_prob:.2f})."
        )

    # ---------- Discordance Detection ----------
    discordance_note = ""

    if xray_score is not None:
        if xray_score < 0.3 and symptom_prob > 0.6:
            discordance_note = (
                "Clinical–imaging discordance detected: strong symptoms with weak imaging findings."
            )
        elif xray_score > 0.7 and symptom_prob < 0.3:
            discordance_note = (
                "Imaging–clinical discordance detected: strong imaging findings with mild symptoms."
            )

    # ---------- Prompt ----------
    prompt = f"""
{MEDICAL_PROMPT}

Patient symptoms:
{symptoms}

Image interpretation:
{xray_interpretation}

Combined pneumonia probability:
{final_probability:.2f}

Confidence level:
{final_confidence}

Relevant medical literature:
{context}
"""

    response = llm.invoke(prompt)

    if hasattr(response, "content"):
        response = response.content

    # ---------- Hospital Style Report ----------
    st.subheader("🏥 Clinical Decision Support Report")

    st.markdown(f"""
**Patient Symptoms:**  
{symptoms}

**Detected Image Type:**  
{image_type} (Confidence: {image_confidence:.2f})

**Imaging Interpretation:**  
{xray_interpretation}

**Combined Pneumonia Probability:**  
{final_probability:.2f} ({final_confidence})

**Confidence Explanation:**  
{confidence_explanation}

**Discordance Analysis:**  
{discordance_note if discordance_note else "No major discordance detected."}

---

### 🧠 AI Clinical Assessment
{response}

---
*This report is generated for educational and decision-support purposes only.*
""")

    # ---------- Evaluation Mode ----------
    if evaluation_mode:
        st.subheader("📊 Internal Evaluation Metrics")
        st.write(f"Imaging Probability: {xray_score if xray_score else 'N/A'}")
        st.write(f"Symptom Score: {symptom_prob:.2f}")
        st.write(f"Final Probability: {final_probability:.2f}")
        st.write(f"Confidence Level: {final_confidence}")
