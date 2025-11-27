# field_schema.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal, Optional, Dict, Any

AnswerType = Literal["string", "bool", "date", "number", "enum"]

@dataclass
class FieldSpec:
    id: str
    question: str
    answer_type: AnswerType
    enum_values: Optional[List[str]] = None
    required: bool = False

@dataclass
class ExtractionSchema:
    application_fields: List[FieldSpec]
    building_fields: List[FieldSpec]

def default_schema() -> ExtractionSchema:
    return ExtractionSchema(
        application_fields=[
            FieldSpec(
                id="application_reference",
                question="What is the planning application reference number?",
                answer_type="string",
                required=True,
            ),
            FieldSpec(
                id="application_type",
                question="What is the type of planning application (e.g. full planning, variation of condition, listed building)?",
                answer_type="string",
                required=True,
            ),
            FieldSpec(
                id="is_retrospective",
                question="Is the application retrospective (for works already carried out)?",
                answer_type="bool",
                required=False,
            ),
            FieldSpec(
                id="application_date",
                question="On which date was the application submitted or validated?",
                answer_type="date",
                required=False,
            ),
            FieldSpec(
                id="applicant_name",
                question="What is the applicant's or applicant company’s name?",
                answer_type="string",
                required=False,
            ),
            FieldSpec(
                id="applicant_email",
                question="What is the applicant's email address, if stated?",
                answer_type="string",
                required=False,
            ),
            FieldSpec(
                id="authority_name",
                question="Which local planning authority is this application submitted to?",
                answer_type="string",
                required=False,
            ),
            FieldSpec(
                id="site_address",
                question="What is the site address (street, town/city, postcode)?",
                answer_type="string",
                required=True,
            ),
        ],
        building_fields=[
            FieldSpec(
                id="building_use_type",
                question="What is the primary use or proposed use of the building (e.g. residential, office, mixed-use)?",
                answer_type="string",
                required=False,
            ),
            FieldSpec(
                id="building_class",
                question="What is the building class or category, if stated (e.g. residential building class, building category)?",
                answer_type="string",
                required=False,
            ),
            FieldSpec(
                id="number_of_storeys",
                question="How many storeys does the building have (if stated)?",
                answer_type="number",
                required=False,
            ),
            FieldSpec(
                id="gross_floor_area_m2",
                question="What is the gross floor area of the building in square meters, if stated?",
                answer_type="number",
                required=False,
            ),
            FieldSpec(
                id="building_height_m",
                question="What is the height of the building in meters, if stated?",
                answer_type="number",
                required=False,
            ),
        ],
    )

