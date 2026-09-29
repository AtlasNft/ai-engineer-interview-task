"""Generate data/notes.json: recruiter notes for 50 of the committed candidates.

- Each chosen candidate gets 1-5 notes, weighted heavily toward 1.
- ~70% of notes are noise (system events, missed calls, outbound emails about roles).
- ~30% are useful: Gemini writes a short note containing a concrete fact about the candidate,
  grounded in their profile.
- A few candidates additionally get a phone-screen transcript in which the recruiter asks about
  their background, skills, and experience; answers are grounded in the resume.

Candidate IDs match the DB build order: data/resumes.json index + 1.

Usage: uv run python scripts/generate_notes.py
"""

import json
import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google.genai import types  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from app.config import DATA_DIR, GEMINI_MODEL, RESUMES_PATH  # noqa: E402
from app.extraction import get_client  # noqa: E402

SEED = 7
N_CANDIDATES = 50
NOTE_COUNT_WEIGHTS = {1: 45, 2: 25, 3: 16, 4: 9, 5: 5}
GARBAGE_RATIO = 0.70
N_TRANSCRIPTS = 6
NOW = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)
WINDOW_DAYS = 540
NOTES_PATH = DATA_DIR / "notes.json"

RECRUITERS = ["Dana Whitfield", "Marcus Oyelaran", "Priya Raman", "Tom Becker", "Lena Ortiz"]
AIRPORTS = ["DFW", "ATL", "ORD", "DEN", "LAX", "SEA", "PHX", "MCO", "CLT", "EWR", "BOS", "MSP"]
EMPLOYERS = ["Delta", "United", "American", "Southwest", "SkyWest", "Envoy", "Swissport", "Menzies", "the airport authority"]

GARBAGE_TEMPLATES = [
    "Called but no answer",
    "Called, went to voicemail",
    "LVM",
    "No answer x2",
    "Left voicemail, will try again tomorrow",
    "Application updated by JobHire API",
    "Profile synced from JobHire",
    "Status changed to Screening",
    "Status changed to Sourced",
    "Moved to pipeline: {airport} Q{q}",
    "Tagged: {airport}-{role_slug}",
    "Resume parsed successfully",
    "Duplicate profile merged (ref {ref})",
    "{ref}",
    "{ref} {ref2}",
    "Ticket #{ref}",
    "ATS import batch {ref}",
    "Bulk email sent: {airport} {role} openings",
    "Email sent: Hi {first}, we have a {role} opening with {employer} at {airport} that I think could be a fit. Let me know if you'd like to hear more. Thanks, {recruiter}",
    "Email sent: Hi {first}, following up on my note last week about the {role} position at {airport}. Happy to jump on a quick call if you're interested. Best, {recruiter}",
    "Email sent: Hi {first}, {employer} has opened several {role} positions for their {airport} base. Apply via the link in this email or reply and I'll walk you through it. {recruiter}",
    "Email sent: Hi {first}, just checking in. We're still hiring {role}s at {airport}, let me know if timing has changed. {recruiter}",
    "Email sent: reminder about the {airport} hiring event on the {day}th. {recruiter}",
    "Sent job link",
    "Sent follow-up",
    "Reminder set for next week",
    "Added to {airport} talent list",
    "Re-engaged via campaign {ref}",
    "Text sent: Hi {first}, this is {recruiter} about the {role} role at {airport}. Good time to chat?",
    "Text sent, no reply",
    "Calendar invite sent",
    "Calendar invite bounced",
    "-",
    "n/a",
    "ok",
]


class GeneratedNote(BaseModel):
    candidate_index: int
    note: str


class NoteBatch(BaseModel):
    notes: list[GeneratedNote]


USEFUL_PROMPT = """You are writing internal recruiter notes about airport-industry candidates, jotted down
right after a phone screen, interview, or reference call. Each note must record NEW information the
recruiter learned from talking to the candidate that is NOT on their resume.

STRICT RULES:
- Do NOT mention, praise, or summarise any skill, certification, aircraft type, or employer that
  already appears in the profile. The reader has the resume; the note must add something.
- The fact must be concrete and actionable. Pick ONE per note, varying across candidates:
  hourly/salary expectation with a number; notice period or earliest start date; whether they
  will relocate or commute and to where; shift or schedule limits (nights, weekends, childcare);
  the real reason they are leaving; a pending certification, medical, or background-check issue;
  a competing offer with a company and a deadline; a reference-check outcome; a red flag from the
  screen (no-show, vague answers, salary mismatch); a preferred base or airport; a visa or
  authorisation detail; a family or commute constraint; interview availability windows.
- Write 1-2 sentences, informal recruiter shorthand is fine. Use the candidate's first name or
  no name at all. Do not invent a different name.
Examples of GOOD notes:
  "Wants $38/hr min, currently on $34. Can start 2 weeks after offer."
  "Won't relocate — wife's job is in Phoenix, only interested in PHX or Mesa."
  "Screen went badly, kept dodging why he left SkyWest. Park for now."
  "Has a verbal offer from Menzies at ORD, needs our answer by Friday."
Examples of BAD notes (do not write these):
  "Holds an A&P license and is strong on Boeing airframes."
  "Experienced with Sabre and conflict resolution."
Return one entry per candidate_index.

CANDIDATES:
{candidates}
"""


class Transcript(BaseModel):
    candidate_index: int
    transcript: str


class TranscriptBatch(BaseModel):
    transcripts: list[Transcript]


TRANSCRIPT_PROMPT = """Write a realistic transcript of a 4-6 minute recruiter phone screen for each candidate below.
The recruiter ({recruiter}) is asking about the candidate's background, skills, and experience.
Requirements:
- Format as alternating lines: "Recruiter: ..." and "{first}: ..." (use the candidate's first name),
  separated by newlines. 10-16 exchanges total.
- The recruiter should ask about: how they got into the role, their current duties, specific
  skills/certifications and how they use them, a past role from the resume, and what they are
  looking for next. Include one or two natural follow-up questions.
- The candidate's answers must be consistent with the resume (employers, cities, years, skills)
  and can add small plausible detail, but must not contradict it.
- Conversational, with a little filler ("yeah", "so", "right") but not padded.
Return one entry per candidate_index, with the transcript as a single string containing newlines.

CANDIDATES:
{candidates}
"""


def transcripts(client, batch: list[tuple[int, dict]]) -> dict[int, str]:
    facts = "\n\n".join(
        f"candidate_index: {idx}\nfirst name: {c['name'].split()[0]}\nRESUME:\n{c['resume']}" for idx, c in batch
    )
    rng = random.Random(SEED + 2)
    for attempt in range(4):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=TRANSCRIPT_PROMPT.format(
                    recruiter=rng.choice(RECRUITERS), first="{first}", candidates=facts
                ).replace("{first}", "<first name>"),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", response_schema=TranscriptBatch, temperature=0.9
                ),
            )
            parsed: TranscriptBatch = resp.parsed or TranscriptBatch.model_validate_json(resp.text)
            out = {t.candidate_index: t.transcript.strip() for t in parsed.transcripts}
            missing = [i for i, _ in batch if i not in out or out[i].count("Recruiter:") < 5]
            if missing:
                raise ValueError(f"missing/short transcripts for {missing}")
            return out
        except Exception as exc:  # noqa: BLE001
            print(f"  transcript batch failed ({exc}); retrying", file=sys.stderr)
    raise RuntimeError("giving up on transcript batch")


def pick_count(rng: random.Random) -> int:
    counts, weights = zip(*NOTE_COUNT_WEIGHTS.items())
    return rng.choices(counts, weights=weights, k=1)[0]


def garbage_note(rng: random.Random, cand: dict) -> str:
    t = rng.choice(GARBAGE_TEMPLATES)
    return t.format(
        first=cand["name"].split()[0],
        role=cand["current_job_title"],
        role_slug=cand["current_job_title"].lower().replace(" ", "-")[:12],
        airport=rng.choice(AIRPORTS),
        employer=rng.choice(EMPLOYERS),
        recruiter=rng.choice(RECRUITERS),
        ref=str(rng.randint(10**9, 10**11)),
        ref2=str(rng.randint(10**7, 10**9)),
        q=rng.randint(1, 4),
        day=rng.randint(1, 28),
    )


def useful_notes(client, batch: list[tuple[int, dict]]) -> dict[int, str]:
    facts = "\n\n".join(
        f"candidate_index: {idx}\n"
        + json.dumps(
            {k: c[k] for k in ("name", "current_job_title", "years_experience", "location_city", "location_region")},
            indent=1,
        )
        for idx, c in batch
    )
    for attempt in range(4):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=USEFUL_PROMPT.format(candidates=facts),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", response_schema=NoteBatch, temperature=1.0
                ),
            )
            parsed: NoteBatch = resp.parsed or NoteBatch.model_validate_json(resp.text)
            out = {n.candidate_index: n.note.strip() for n in parsed.notes if n.note.strip()}
            by_idx = dict(batch)
            for i in list(out):
                low = out[i].lower()
                if any(sk.lower() in low for sk in by_idx[i]["skills"]):
                    print(f"  rejecting note for {i}: repeats a listed skill", file=sys.stderr)
                    del out[i]
            missing = [i for i, _ in batch if i not in out]
            if missing:
                raise ValueError(f"missing notes for {missing}")
            return out
        except Exception as exc:  # noqa: BLE001
            print(f"  batch failed ({exc}); retrying", file=sys.stderr)
    raise RuntimeError("giving up on useful-notes batch")


def main() -> None:
    rng = random.Random(SEED)
    candidates = json.loads(RESUMES_PATH.read_text())
    chosen = sorted(rng.sample(range(len(candidates)), N_CANDIDATES))

    # Plan every note first so the useful ones can be generated in batches.
    plan: list[dict] = []  # {idx, candidate_id, timestamp, useful}
    for idx in chosen:
        n = pick_count(rng)
        stamps = sorted(NOW - timedelta(days=rng.uniform(0, WINDOW_DAYS), minutes=rng.randint(0, 1439)) for _ in range(n))
        for ts in stamps:
            plan.append({"idx": idx, "candidate_id": idx + 1, "timestamp": ts, "useful": rng.random() >= GARBAGE_RATIO})

    # Force the useful share close to 30% overall.
    target_useful = round(len(plan) * (1 - GARBAGE_RATIO))
    useful_slots = [p for p in plan if p["useful"]]
    other_slots = [p for p in plan if not p["useful"]]
    rng.shuffle(other_slots)
    while len(useful_slots) < target_useful and other_slots:
        s = other_slots.pop(); s["useful"] = True; useful_slots.append(s)
    while len(useful_slots) > target_useful:
        s = useful_slots.pop(); s["useful"] = False

    client = get_client()
    useful_text: dict[int, list[str]] = {}
    # One Gemini note per (candidate, slot); ask for as many distinct notes as slots per candidate.
    slots_by_idx: dict[int, int] = {}
    for p in plan:
        if p["useful"]:
            slots_by_idx[p["idx"]] = slots_by_idx.get(p["idx"], 0) + 1
    requests: list[tuple[int, dict]] = []
    for idx, k in slots_by_idx.items():
        requests += [(idx, candidates[idx])] * k
    # Each batch contains at most one slot per candidate so candidate_index stays unique.
    rounds: list[list[tuple[int, dict]]] = []
    remaining = list(requests)
    while remaining:
        seen: set[int] = set(); this_round: list[tuple[int, dict]] = []; rest: list[tuple[int, dict]] = []
        for r in remaining:
            (this_round if r[0] not in seen else rest).append(r); seen.add(r[0])
        rounds.append(this_round); remaining = rest
    for rnd in rounds:
        for start in range(0, len(rnd), 10):
            batch = rnd[start : start + 10]
            print(f"generating {len(batch)} useful notes")
            for idx, note in useful_notes(client, batch).items():
                useful_text.setdefault(idx, []).append(note)

    notes = []
    for p in plan:
        cand = candidates[p["idx"]]
        text = useful_text[p["idx"]].pop(0) if p["useful"] else garbage_note(rng, cand)
        notes.append({"candidate_id": p["candidate_id"], "timestamp": p["timestamp"].strftime("%Y-%m-%dT%H:%M:%SZ"), "note": text})
    # Phone-screen transcripts for a few of the chosen candidates. Separate RNG stream so the
    # notes above are unaffected.
    trng = random.Random(SEED + 1)
    t_idx = trng.sample(chosen, N_TRANSCRIPTS)
    print(f"generating {N_TRANSCRIPTS} call transcripts")
    for idx, text in transcripts(client, [(i, candidates[i]) for i in t_idx]).items():
        ts = NOW - timedelta(days=trng.uniform(0, WINDOW_DAYS), minutes=trng.randint(0, 1439))
        notes.append({
            "candidate_id": idx + 1,
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "note": f"Phone screen transcript ({ts.strftime('%b %d')}):\n{text}",
        })

    notes.sort(key=lambda n: (n["candidate_id"], n["timestamp"]))
    NOTES_PATH.write_text(json.dumps(notes, indent=2) + "\n")
    useful_n = sum(p["useful"] for p in plan)
    print(f"wrote {len(notes)} notes for {N_CANDIDATES} candidates ({useful_n} useful, {len(notes) - useful_n - N_TRANSCRIPTS} noise, {N_TRANSCRIPTS} transcripts) to {NOTES_PATH}")


if __name__ == "__main__":
    main()
