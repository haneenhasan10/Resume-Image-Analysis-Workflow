# Resume-Image-Analysis-Workflow

This project is a simple AI workflow application built with Streamlit and Google Gemini Vision API. It allows the user to upload multiple resume images, extract structured candidate information, apply Python-based qualification logic, and display the final results in a table.

The project follows a workflow style similar to AI automation platforms, where an input is passed into an AI model, converted into structured output, processed with logic, and then displayed as a final result.

# Project Goal 

The goal of this project is to analyze resume images using a Vision-Language Model and convert unstructured visual information into structured JSON data.

The workflow then uses Python logic to calculate candidate experience and determine whether the candidate is:

Qualified

Unqualified

Unknown

This project demonstrates how AI workflows are built step by step instead of only generating text output.


# Workflow Structure 

Resume Image Upload --> Gemini Vision Analysis --> Structured JSON Extraction
--> Python Experience Calculation and Decision Logic --> Final Results Table and Summary


# Technologies Used

Python

Streamlit

Google Gemini Vision API

Pandas

Pillow

google-genai

python-dotenv

JSON



# Required Packages
Install the required packages with:

bash: pip install streamlit pandas pillow google-genai python-dotenv


# Setup and How to Run

1. Clone or download the project
Place the project files inside one folder.

3. Create a .env file
Create a file named .env in the same folder as app.py.
Add your Gemini API key:
env
GEMINI_API_KEY=your_api_key_here

4. Run the Streamlit app 

bash: streamlit run app.py

After that, the application will open in your browser


# Python Decision Logic

The model does not directly decide whether the candidate is qualified.
Instead, Python calculates experience from the extracted date ranges and then applies this rule:

If experience_years == "unknown" -> Qualification = Unknown

Else if experience_years >= 2 -> Qualification = Qualified

Else -> Qualification = Unqualified


# Final Output Table

- The application displays a table with:

Candidate Name

Experience

Skills

Education

Confidence

Qualification

- It also shows summary statistics:
  
Total resumes

Qualified

Unqualified

Unknown

# Error Handling

The application handles common issues such as:

invalid JSON returned by the model

API errors

missing fields

unsupported image formats

corrupted image files

missing API key

User-friendly messages are displayed in the Streamlit interface.



# Prompt Used


PROMPT = """
Analyze this resume image and extract ONLY these fields in valid JSON:

{
  "candidate_name": string,
  "experience_dates": [string],
  "skills": [string],
  "education": string,
  "confidence": "High" | "Medium" | "Low"
}

Instructions:
- Return one JSON object only.
- Return only valid JSON.
- Do not include markdown.
- Do not include explanations.
- Do not include notes.
- Do not include extra text.
- experience_dates must include only visible employment date ranges found in the resume.
- Preserve the date text as closely as possible to the resume.
- skills must be an array of strings.
- If no experience dates are found, return an empty array.
- Extract and include ALL educational degrees and certificates mentioned in the text. Do not omit or skip any background education, even if there are multiple entries.
- If a field is missing, use:
  - candidate_name: "unknown"
  - skills: []
  - education: "unknown"
  - confidence: "Low"

Example:
{
  "candidate_name": "John Smith",
  "experience_dates": [
    "June 2018-present",
    "September 2016-May 2018"
  ],
  "skills": [
    "Python",
    "SQL",
    "Project Management"
  ],
  "education": "Bachelor of Computer Science",
  "confidence": "High"
}

Example when experience dates are missing:
{
  "candidate_name": "John Smith",
  "experience_dates": [],
  "skills": [
    "Python",
    "SQL"
  ],
  "education": "Bachelor of Computer Science",
  "confidence": "Medium"
}
""".strip()



# Example Input and Output

- Example Input

A resume image containing:

candidate name

work experience date ranges

skills

education

 
- Example Output: A table contains candidates info with qualification decision
  *See interface_images folder 

# What AI Helped Me With

AI was used to help with:
planning the workflow structure
improving the prompt format
generating the first version of the Streamlit code
suggesting better error handling
improving experience date extraction logic
improving the README structure


# What I Changed, Fixed, or Improved Myself
 I improved the project by:
- changing the idea from simple resume screening to a clearer workflow-based analysis project
- separating AI extraction from Python decision logic
- improving the prompt
- changing experience extraction from direct number prediction to date-range extraction
- improving the skills display format
- testing the app with multiple resume images
- refining the final results structure


# What I Learned from This Task

From this task, I learned that building AI applications is not only about calling a model. The most important part is designing the workflow around the model.
I learned how to:
turn image input into structured output
separate AI extraction from rule-based logic
validate and clean model output
build a small end-to-end workflow using Streamlit
This task helped me understand how different parts of an AI workflow connect together in a practical application.


