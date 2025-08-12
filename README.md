# Resume-Analyzer
📄 Resume Analyzer – NLP Streamlit App
An interactive Resume Analyzer built with Python and Streamlit, designed to parse resumes, extract key skills, and match them to job descriptions. Perfect for HR teams, recruiters, and job seekers who want quick, AI-assisted resume insights.

🚀 Features
📂 Upload Resumes in PDF or DOCX format

🔍 Automatic Text Extraction using pdfplumber and python-docx

🧠 NLP-based Skill Extraction with nltk and scikit-learn

🔗 Job Description Matching using semantic similarity

📊 Match Percentage display for quick decision-making

🌐 Works locally or in Google Colab

📦 Installation
Run Locally
Clone the repository

bash
Copy
Edit
git clone https://github.com/yourusername/resume-analyzer.git
cd resume-analyzer
Create a virtual environment

bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run Streamlit app

bash
Copy
Edit
streamlit run app.py
Run in Google Colab
Open the project in Google Colab.

Install dependencies:

python
Copy
Edit
!pip install -r requirements.txt
Run Streamlit in Colab:

python
Copy
Edit
!pip install pyngrok
!streamlit run app.py & npx localtunnel --port 8501
Click the tunnel link generated to open your app.

📂 Project Structure
lua
Copy
Edit
resume-analyzer/
│-- app.py                  # Main Streamlit app
│-- requirements.txt        # Project dependencies
│-- README.md               # Project documentation
│-- sample_resumes/         # Example resumes
│-- utils/                  # Helper functions
🛠 Tech Stack
Python 3.9+

Streamlit – Frontend web app

NLTK – NLP processing

scikit-learn – Skill matching

pdfplumber / python-docx – Resume parsing

spaCy – Advanced NLP (optional)

📌 Future Improvements
✅ AI-powered resume ranking

✅ Multiple job description matching

✅ Keyword highlighting in resumes
