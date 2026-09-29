from pydantic import BaseModel, Field


class CandidateFields(BaseModel):
    skills: list[str] = Field(default_factory=list)
    current_job_title: str
    past_job_titles: list[str] = Field(default_factory=list)
    years_experience: int = Field(ge=0)
    location_city: str
    location_region: str


class Candidate(CandidateFields):
    id: int
    name: str
    resume: str


class CreateDbRequest(BaseModel):
    reset: bool = True


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    skills: list[str] | None = None
    current_job_title: str | None = None
    past_job_titles: list[str] | None = None
    min_years_experience: int | None = Field(default=None, ge=0)
    max_years_experience: int | None = Field(default=None, ge=0)
    location_city: str | None = None
    location_region: str | None = None


class SearchResult(Candidate):
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total_matching_filters: int


class AddCandidateRequest(BaseModel):
    name: str
    resume: str
