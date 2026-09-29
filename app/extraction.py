"""Structured field extraction from resume text using Gemini."""

import os

from google import genai
from google.genai import types

from app.config import GEMINI_MODEL
from app.schemas import CandidateFields
from app.taxonomy import ALL_SKILLS, SKILLS_TAXONOMY, normalize_skills

_client: genai.Client | None = None

_STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        # Fail fast on stalled requests instead of hanging forever (ms).
        _client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=120_000))
    return _client


def _taxonomy_block() -> str:
    lines = []
    for category, skills in SKILLS_TAXONOMY.items():
        lines.append(f"{category}:")
        lines.extend(f"  - {s}" for s in skills)
    return "\n".join(lines)


SYSTEM_INSTRUCTION = f"""You extract structured data from job applicant resumes for airport roles.

Return these fields:
- skills: a list of skills the candidate demonstrably has. You MUST choose ONLY from the
  taxonomy below, copying each skill name exactly. Do not invent skills. Include a skill
  only if the resume gives clear evidence for it.
- current_job_title: the candidate's most recent / current role title.
- past_job_titles: earlier role titles, most recent first, excluding the current one.
- years_experience: total years of relevant work experience as a whole number. Use a stated
  total if the resume gives one; otherwise sum the dated roles. Use 0 if unknown.
- location_city: city where the candidate lives or is based.
- location_region: the US state, spelled out in full (e.g. "Texas", not "TX").

If a field is genuinely unknown, use an empty string or empty list.

SKILLS TAXONOMY:
{_taxonomy_block()}
"""


def normalize_state(region: str) -> str:
    """Expand a postal abbreviation to the full state name; pass other values through."""
    region = region.strip()
    return _STATE_NAMES.get(region.upper(), region)


def extract_fields(resume: str) -> CandidateFields:
    response = get_client().models.generate_content(
        model=GEMINI_MODEL,
        contents=f"RESUME:\n\n{resume}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=CandidateFields,
            temperature=0.0,
        ),
    )
    fields: CandidateFields = response.parsed
    if fields is None:
        fields = CandidateFields.model_validate_json(response.text)
    fields.skills = normalize_skills(fields.skills)
    fields.location_region = normalize_state(fields.location_region)
    fields.location_city = fields.location_city.strip()
    fields.current_job_title = fields.current_job_title.strip()
    fields.past_job_titles = [t.strip() for t in fields.past_job_titles if t.strip()]
    return fields
