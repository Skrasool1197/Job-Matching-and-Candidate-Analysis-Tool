# Job Matching and Candidate Analysis Tool

This tool uses Streamlit, Pinecone, and Google Generative AI to analyze job descriptions, candidate resumes, and interview videos to identify the best job-candidate match. The system processes job descriptions, resumes, and video content, then merges the results to evaluate how well a candidate aligns with a particular role.

## Features

- **Job Matching**: Compares candidate resumes and videos against job descriptions to find the best match.
- **LLM Analysis**: Utilizes Google Generative AI for structured analysis of resumes and videos.
- **Similarity Search**: Uses Pinecone to store job descriptions and retrieve the closest match based on the candidate's profile.

---

## Setup Instructions

### Prerequisites

Ensure that you have:
- Python 3.8 or above
- A Google GenAI API key
- A Pinecone API key and an active Pinecone account

### Installation

1. **Clone the repository**:
   ```bash
   [git clone https://github.com/your-repo/job-matching-tool.git
   cd job-matching-tool](https://github.com/Skrasool1197/Job-Matching-and-Candidate-Analysis-Tool.git)

1. **Install dependencies**:
   ```bash
    pip install -r requirements.txt

- Dependencies include:

streamlit - for the web interface
langchain-core - for managing documents
pydantic - for defining structured response models
PyMuPDF - for PDF processing
google-generativeai - for accessing Google Generative AI
pinecone-client - for interacting with Pinecone

