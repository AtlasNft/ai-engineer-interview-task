import numpy as np
from fastapi import FastAPI, HTTPException

from app import db, embeddings, extraction
from app.ingest import build_database
from app.schemas import (
    AddCandidateRequest,
    Candidate,
    CreateDbRequest,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.taxonomy import ALL_SKILLS, SKILLS_TAXONOMY


app = FastAPI(title="Airport Applicant Semantic Search")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/taxonomy")
def taxonomy():
    return {"categories": SKILLS_TAXONOMY, "all_skills": ALL_SKILLS}


@app.post("/db/create")
def create_db(req: CreateDbRequest):
    try:
        return build_database(reset=req.reset)
    except FileNotFoundError as exc:
        raise HTTPException(500, str(exc)) from exc


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    rows, matrix = db.query_candidates(
        skills=req.skills,
        current_job_title=req.current_job_title,
        past_job_titles=req.past_job_titles,
        min_years_experience=req.min_years_experience,
        max_years_experience=req.max_years_experience,
        location_city=req.location_city,
        location_region=req.location_region,
    )
    if not rows:
        return SearchResponse(results=[], total_matching_filters=0)

    query_vec = embeddings.embed_query(req.query)
    scores = matrix @ query_vec  # embeddings are L2-normalised, so dot == cosine
    order = np.argsort(-scores, kind="stable")[: req.top_k]
    results = [SearchResult(**rows[i], score=float(scores[i])) for i in order]
    return SearchResponse(results=results, total_matching_filters=len(rows))


@app.post("/candidates", response_model=Candidate, status_code=201)
def add_candidate(req: AddCandidateRequest):
    if not req.resume.strip():
        raise HTTPException(422, "resume must not be empty")
    try:
        fields = extraction.extract_fields(req.resume)
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc

    db.init_db(reset=False)
    row = {"name": req.name.strip(), "resume": req.resume, **fields.model_dump()}
    [candidate_id] = db.insert_candidates([row], embeddings.embed_documents([req.resume]))
    return Candidate(id=candidate_id, **row)


@app.get("/candidates/{candidate_id}", response_model=Candidate)
def get_candidate(candidate_id: int):
    row = db.get_candidate(candidate_id)
    if row is None:
        raise HTTPException(404, "candidate not found")
    return Candidate(**row)
