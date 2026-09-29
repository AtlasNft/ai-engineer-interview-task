"""Skills taxonomy for airport roles.

The add-candidate endpoint asks Gemini to pick skills exclusively from this
list, and anything outside it is dropped during post-validation.
"""

SKILLS_TAXONOMY: dict[str, list[str]] = {
    "Flight Operations": [
        "ATP Certificate",
        "Commercial Pilot License",
        "Type Rating B737",
        "Type Rating A320",
        "Type Rating B777",
        "Type Rating E175",
        "Instrument Rating",
        "Multi-Engine Rating",
        "Crew Resource Management",
        "Flight Planning",
        "Fuel Management",
        "Flight Dispatch Certificate",
        "Weather Analysis",
        "Line Check Airman",
    ],
    "Cabin Crew": [
        "FAA Cabin Safety Certification",
        "CPR and First Aid",
        "Emergency Evacuation Procedures",
        "In-Flight Service",
        "Passenger Announcements",
        "Cabin Safety Demonstrations",
        "Galley Management",
        "Unruly Passenger De-escalation",
        "Duty-Free Sales",
    ],
    "Air Traffic Control": [
        "FAA ATC Certification",
        "Tower Control",
        "Ground Control",
        "Radar Approach Control",
        "En Route Control",
        "STARS",
        "ERAM",
        "Aircraft Separation Standards",
        "Runway Incursion Prevention",
        "Flight Data Processing",
    ],
    "Aircraft Maintenance": [
        "A&P License",
        "Avionics Troubleshooting",
        "Hydraulic Systems",
        "Turbine Engine Maintenance",
        "Boeing Airframe Maintenance",
        "Airbus Airframe Maintenance",
        "Non-Destructive Testing",
        "FAA Part 145 Compliance",
        "Line Maintenance",
        "Heavy Maintenance Checks",
        "Sheet Metal Repair",
        "Maintenance Logbook Documentation",
        "Composite Repair",
    ],
    "Ground Operations": [
        "Aircraft Marshalling",
        "Baggage Handling",
        "Aircraft Deicing",
        "Ground Support Equipment Operation",
        "Aircraft Fueling",
        "Pushback Operations",
        "Cargo Loading",
        "Weight and Balance",
        "Lavatory and Water Service",
        "Tug and Belt Loader Operation",
        "Forklift Certification",
    ],
    "Customer Service": [
        "Sabre",
        "Amadeus",
        "Passenger Check-In",
        "Gate Operations",
        "Boarding Procedures",
        "Irregular Operations Rebooking",
        "Lost Baggage Claims",
        "Ticketing",
        "Conflict Resolution",
        "Special Assistance Passengers",
        "Lounge Hospitality",
    ],
    "Security and Airport Operations": [
        "TSA Screening",
        "X-Ray Image Interpretation",
        "SIDA Badge",
        "Hazmat Handling",
        "FAA Part 139 Airport Operations",
        "NOTAM Management",
        "Airfield Inspection",
        "Wildlife Hazard Management",
        "Emergency Response Coordination",
        "Access Control",
        "Snow and Ice Control",
    ],
    "General": [
        "Radio Communication",
        "Shift Work",
        "Bilingual Spanish",
        "Safety Management Systems",
        "OSHA Compliance",
        "Team Leadership",
        "Training and Mentoring",
        "Microsoft Office",
        "Incident Reporting",
        "Customer Service Excellence",
    ],
}

ALL_SKILLS: list[str] = [s for skills in SKILLS_TAXONOMY.values() for s in skills]
CATEGORIES: list[str] = list(SKILLS_TAXONOMY.keys())
_LOWER_LOOKUP = {s.lower(): s for s in ALL_SKILLS}


def normalize_skills(skills: list[str]) -> list[str]:
    """Keep only skills present in the taxonomy (case-insensitive), preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for s in skills:
        canonical = _LOWER_LOOKUP.get(s.strip().lower())
        if canonical and canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
    return out
