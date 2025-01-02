import os
import streamlit as st
import PyPDF2
from typing import Dict, Optional
from groq import Groq

# Streamlit Page Configurations
st.set_page_config(
    layout="wide",
    page_title="AI Sentiment Analyzer",
    initial_sidebar_state="expanded"
)

# Sidebar Configuration
st.sidebar.image("p2.PNG", width=200)
st.sidebar.image("p1.png", width=300)
st.sidebar.title("Configuration Options")
st.sidebar.subheader("1. Select AI Model")

SUPPORTED_MODELS = {
    "Llama 3 8B": "llama3-8b-8192",
    "Llama 3.2 1B (Preview)": "llama-3.2-1b-preview",
    "Llama 3 70B": "llama3-70b-8192",
    "Mixtral 8x7B": "mixtral-8x7b-32768",
    "Gemma 2 9B": "gemma2-9b-it",
    "Llama 3.2 11B Vision (Preview)": "llama-3.2-11b-vision-preview",
    "Llama 3.2 11B Text (Preview)": "llama-3.2-11b-text-preview",
    "Llama 3.1 8B Instant (Text-Only Workloads)": "llama-3.1-8b-instant",
    "Llama 3.2 90B Vision (Preview)": "llama-3.2-90b-vision-preview",
    "Llama 3.1 70B Versatile": "llama-3.1-70b-versatile",
    "Llama 3.3 70B SpecDec": "llama-3.3-70b-specdec",
    "Llama 3.3 70B Versatile": "llama-3.3-70b-versatile",
}

selected_model = st.sidebar.selectbox("Choose an AI Model", list(SUPPORTED_MODELS.keys()))

st.sidebar.subheader("2. Adjust Temperature")
temperature = st.sidebar.slider("Set Temperature", min_value=0.0, max_value=1.0, value=0.7)

st.sidebar.subheader("3. Upload PDF Document")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

st.sidebar.subheader("4. Select Analysis Prompt")
PROMPT_TYPES = {
    "Summarization Prompt": "You are an expert analyst. I am [mention the problem you’re facing in detail with context]. Analyze the attached document thoroughly and provide a concise summary. Ensure the summary captures the main points, key arguments, supporting evidence, and the conclusions drawn by the author. Maintain the original context and intent while keeping the summary brief. I want you to [mention how you want the output in detail with examples].",
    "Sentiment Analysis Prompt": "You are an expert analyst. I am [mention the problem you’re facing in detail with context]. Read the attached document and perform a sentiment analysis. Determine if the overall sentiment is positive, negative, or neutral. Identify specific sentences or paragraphs that exemplify this sentiment. Additionally, explain how the choice of words, tone, and context contribute to the identified sentiment. I want you to [mention how you want the output in detail with examples].",
    "Key Themes and Topics Identification Prompt": "You are an expert analyst. I am [mention the problem you’re facing in detail with context]. Examine the attached document and identify the primary themes and topics discussed. List each theme or topic along with a brief explanation. Provide relevant excerpts from the text that illustrate each identified theme or topic. Ensure that all significant themes and topics are covered comprehensively. I want you to [mention how you want the output in detail with examples].",
    "Tone and Style Analysis Prompt": "You are an expert analyst. I am [mention the problem you’re facing in detail with context]. Analyze the tone and style of the attached document. Describe the tone (e.g., formal, informal, persuasive, informative) and the style (e.g., academic, conversational, technical). Provide multiple examples from the text to support your analysis. Additionally, explain how the tone and style contribute to the overall effectiveness of the document. I want you to [mention how you want the output in detail with examples].",
    "Argument Analysis Prompt": "You are an expert business coach. I am [mention the problem you are facing in detail with context]. Develop a comprehensive branding strategy for my new digital product in the [specific niche]. Include ideas for the product name that reflect its value and appeal to the target audience, logo design concepts that align with the brand identity, brand voice and tone guidelines, and key messaging that highlights the product’s unique benefits and differentiators. Check my attached document for [mention reference]. I want you to [mention how you want the output in detail with examples].",
    "Fact-Checking Prompt": "You are an expert analyst. I am [mention the problem you’re facing in detail with context]. Analyze the attached document and identify all factual claims made by the author. Verify the accuracy of these claims using reliable external sources. For each claim, provide a brief report on its validity, including citations from your sources. Highlight any discrepancies or confirmations found during the verification process. I want you to [mention how you want the output in detail with examples].",
    "Comparison and Contrast Prompt": "You are an expert analyst. I am [mention the problem you’re facing in detail with context]. Compare and contrast the key points and perspectives presented in the attached document. Highlight the similarities and differences in their arguments, evidence, and conclusions. For each comparison point, provide specific excerpts from both documents. Summarize the overall comparative analysis. I want you to [mention how you want the output in detail with examples].",
    "Persuasive Techniques Analysis Prompt": "You are an expert analyst. I am [mention the problem you’re facing in detail with context]. Examine the attached document and identify any persuasive techniques used by the author, such as appeals to emotion, logical reasoning, or use of evidence. For each technique identified, provide specific examples from the text and assess their effectiveness. Additionally, explain how these techniques contribute to the overall persuasive power of the document. I want you to [mention how you want the output in detail with examples].",
    "Structure and Organization Analysis Prompt": "You are an expert analyst. I am [mention the problem you’re facing in detail with context]. Analyze the structure and organization of the attached document. Describe how the document is structured, including sections such as introduction, body, and conclusion. Evaluate the effectiveness of this structure in conveying the author’s message. Provide examples from the text that illustrate the organization and flow of information. I want you to [mention how you want the output in detail with examples].",
    "Critical Analysis Prompt": "You are an expert analyst. I am [mention the problem you’re facing in detail with context]. Perform a critical analysis of the attached document. Identify any biases, assumptions, or gaps in the author’s reasoning. Provide specific examples from the text to support your critique. Additionally, suggest ways in which the document could be improved to address these issues. Summarize your overall critical assessment. I want you to [mention how you want the output in detail with examples].",
}

selected_prompt_type = st.sidebar.radio("Select a Prompt Type", list(PROMPT_TYPES.keys()))
selected_prompt_text = PROMPT_TYPES[selected_prompt_type]

st.sidebar.subheader("5. Customize Selected Prompt")
custom_prompt = st.sidebar.text_area("Edit Prompt", value=selected_prompt_text, height=300)

# Main Section
st.title("AI Sentiment Analyzer")
st.write("Analyze your PDF document using advanced AI prompts. Results will be displayed below.")

# Function to Extract Text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

if uploaded_file:
    with st.expander("Extracted PDF Text"):
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.text_area("Preview of Extracted Text", pdf_text, height=500)

    if st.button("Analyze Document"):
        st.write("Analyzing the document...")
        if not pdf_text.strip():
            st.warning("No text extracted from the PDF. Please check the document.")
        else:
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                st.error("GROQ_API_KEY is not set. Please configure your environment variables.")
            else:
                client = Groq(api_key=groq_api_key)

                try:
                    response = client.chat.completions.create(
                        model=SUPPORTED_MODELS[selected_model],
                        messages=[
                            {"role": "system", "content": custom_prompt},
                            {"role": "user", "content": pdf_text},
                        ],
                        temperature=temperature,
                        max_tokens=1500,
                    )
                    analysis_result = response.choices[0].message.content.strip()
                    
                    # Display the result
                    st.subheader("Analysis Result")
                    st.write(analysis_result)

                except Exception as e:
                    st.error(f"Error during analysis: {e}")

else:
    st.info("built by dw - Please upload a PDF document to start the analysis.")
