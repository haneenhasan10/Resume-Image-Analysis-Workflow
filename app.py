import io
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

APP_TITLE = "Resume Image Analysis"
APP_DESCRIPTION = (
    "Upload multiple resume images, extract structured candidate data using Gemini Vision, "
    "calculate experience using Python logic, and display final results in a table."
)

SUPPORTED_TYPES = {"image/png", "image/jpeg", "image/jpg"}

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

MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def get_gemini_client() -> genai.Client:
    """Create and return Gemini client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing. Please add it to your .env file.")
    return genai.Client(api_key=api_key)


def validate_uploaded_file(uploaded_file) -> bool:
    """Validate uploaded file type."""
    return uploaded_file.type in SUPPORTED_TYPES


def open_image(uploaded_file) -> Image.Image:
    """Open uploaded image safely."""
    try:
        return Image.open(uploaded_file)
    except UnidentifiedImageError as exc:
        raise ValueError("Unsupported or corrupted image file.") from exc


def extract_json_text(response_text: str) -> str:
    """Extract clean JSON text."""
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    return cleaned


def normalize_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize model response."""
    normalized = {
        "candidate_name": "unknown",
        "experience_dates": [],
        "skills": [],
        "education": "unknown",
        "confidence": "Low",
    }

    normalized["candidate_name"] = str(data.get("candidate_name", "unknown")).strip() or "unknown"

    experience_dates = data.get("experience_dates", [])
    if isinstance(experience_dates, list):
        normalized["experience_dates"] = [str(item).strip() for item in experience_dates if str(item).strip()]
    else:
        normalized["experience_dates"] = []

    skills = data.get("skills", [])
    if isinstance(skills, list):
        normalized["skills"] = [str(skill).strip() for skill in skills if str(skill).strip()]
    else:
        normalized["skills"] = []

    normalized["education"] = str(data.get("education", "unknown")).strip() or "unknown"

    confidence = str(data.get("confidence", "Low")).strip().title()
    if confidence not in {"High", "Medium", "Low"}:
        confidence = "Low"
    normalized["confidence"] = confidence

    return normalized


def parse_month_year(text: str) -> Optional[Tuple[int, int]]:
    """Parse month and year from text."""
    text = text.strip().lower()
    text = text.replace("|", " ").replace(",", " ")
    text = re.sub(r"\s+", " ", text)

    month_year_pattern = r"([a-zA-Z]+)\s+(\d{4})"
    year_only_pattern = r"^(\d{4})$"

    month_match = re.search(month_year_pattern, text)
    if month_match:
        month_name = month_match.group(1).lower()
        year = int(month_match.group(2))
        if month_name in MONTHS:
            return MONTHS[month_name], year

    year_match = re.search(year_only_pattern, text)
    if year_match:
        return 1, int(year_match.group(1))

    return None


def parse_date_token(token: str) -> Optional[Tuple[int, int]]:
    """Parse one side of a date range."""
    token = token.strip().lower()

    if token in {"present", "current", "now", "today"}:
        now = datetime.now()
        return now.month, now.year

    return parse_month_year(token)


def calculate_month_difference(start: Tuple[int, int], end: Tuple[int, int]) -> int:
    """Calculate inclusive month difference."""
    start_month, start_year = start
    end_month, end_year = end
    return (end_year - start_year) * 12 + (end_month - start_month)


def parse_date_range(date_range: str) -> Optional[int]:
    """Parse a work date range and return months of experience."""
    cleaned = date_range.strip()
    cleaned = cleaned.replace("|", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)

    parts = re.split(r"\s*-\s*", cleaned)
    if len(parts) != 2:
        return None

    start = parse_date_token(parts[0])
    end = parse_date_token(parts[1])

    if not start or not end:
        return None

    months = calculate_month_difference(start, end)
    if months < 0:
        return None

    return months


def calculate_experience_years(experience_dates: List[str]) -> Union[int, str]:
    """Calculate total experience years from extracted date ranges."""
    total_months = 0
    valid_found = False

    for date_range in experience_dates:
        months = parse_date_range(date_range)
        if months is not None:
            total_months += months
            valid_found = True

    if not valid_found:
        return "unknown"

    years = total_months // 12
    return years


def get_qualification(experience_years: Union[int, str]) -> str:
    """Apply qualification rule."""
    if experience_years == "unknown":
        return "Unknown"
    if isinstance(experience_years, int) and experience_years >= 2:
        return "Qualified"
    return "Unqualified"


def analyze_resume_image(client: genai.Client, uploaded_file) -> Dict[str, Any]:
    """Analyze one resume image."""
    image = open_image(uploaded_file)

    image_bytes_io = io.BytesIO()
    image_format = image.format if image.format in {"PNG", "JPEG"} else "PNG"
    image.save(image_bytes_io, format=image_format)
    image_bytes = image_bytes_io.getvalue()

    mime_type = "image/png" if image_format == "PNG" else "image/jpeg"

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            PROMPT,
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
        ),
    )

    if not response.text:
        raise ValueError("Empty response received from Gemini API.")

    raw_json = extract_json_text(response.text)

    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON returned by the model.") from exc

    required_fields = {
        "candidate_name",
        "experience_dates",
        "skills",
        "education",
        "confidence",
    }

    missing_fields = required_fields - set(parsed.keys())
    if missing_fields:
        raise ValueError(f"Missing fields in model output: {', '.join(sorted(missing_fields))}")

    normalized = normalize_response(parsed)
    normalized["experience_years"] = calculate_experience_years(normalized["experience_dates"])
    normalized["qualification"] = get_qualification(normalized["experience_years"])
    normalized["raw_json"] = parsed

    return normalized


def build_results_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build results table."""
    rows = []

    for item in results:
        rows.append(
            {
                "Candidate Name": item["candidate_name"],
                "Experience": item["experience_years"],
                "Skills": "<br>".join(item["skills"]) if item["skills"] else "None",
                "Education": item["education"],
                "Confidence": item["confidence"],
                "Qualification": item["qualification"],
            }
        )

    return pd.DataFrame(rows)


def show_summary_metrics(df: pd.DataFrame) -> None:
    """Display summary metrics."""
    total_resumes = len(df)
    qualified = (df["Qualification"] == "Qualified").sum()
    unqualified = (df["Qualification"] == "Unqualified").sum()
    unknown = (df["Qualification"] == "Unknown").sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Resumes", total_resumes)
    col2.metric("Qualified", int(qualified))
    col3.metric("Unqualified", int(unqualified))
    col4.metric("Unknown", int(unknown))


def main() -> None:
    """Main app."""
    st.set_page_config(page_title="Resume Image Analysis", page_icon="📄", layout="wide")

    st.title(APP_TITLE)
    st.write(APP_DESCRIPTION)

    uploaded_files = st.file_uploader(
        "Upload resume images (PNG or JPG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    show_json = st.checkbox("Show extracted JSON", value=False)

    if st.button("Analyze"):
        if not uploaded_files:
            st.warning("Please upload at least one resume image.")
            return

        try:
            client = get_gemini_client()
        except ValueError as error:
            st.error(str(error))
            return

        valid_files = []
        for file in uploaded_files:
            if not validate_uploaded_file(file):
                st.error(f"{file.name}: Unsupported image format.")
                continue
            valid_files.append(file)

        if not valid_files:
            st.warning("No valid image files available for analysis.")
            return

        results = []
        raw_outputs = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for index, uploaded_file in enumerate(valid_files, start=1):
            status_text.info(f"Analyzing {uploaded_file.name}...")

            try:
                result = analyze_resume_image(client, uploaded_file)
                results.append(result)
                raw_outputs.append(
                    {
                        "file_name": uploaded_file.name,
                        "json_output": result["raw_json"],
                    }
                )
            except ValueError as error:
                st.error(f"{uploaded_file.name}: {error}")
            except Exception as error:
                st.error(f"{uploaded_file.name}: API error - {error}")

            progress_bar.progress(index / len(valid_files))

        status_text.empty()

        if not results:
            st.warning("No resumes were successfully analyzed.")
            return

        df = build_results_dataframe(results)

        st.subheader("Results Table")
        st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)

        st.subheader("Summary")
        show_summary_metrics(df)

        if show_json:
            st.subheader("Extracted JSON")
            for item in raw_outputs:
                with st.expander(item["file_name"]):
                    st.json(item["json_output"])


if __name__ == "__main__":
    main()
