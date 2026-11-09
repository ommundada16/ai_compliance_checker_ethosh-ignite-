from pydantic import BaseModel, Field
from typing import Literal, List, Union, Dict, Optional

class Finding(BaseModel):
    violating_statement: str
    guideline_clause: str
    category: str
    severity: Literal["Low", "Medium", "High", "Critical"]
    explanation: str
    confidence: float = Field(ge=0, le=1)
    suggested_correction: str
    source_quote: Optional[str] = None   # verbatim excerpt from the document text
    page_number: Optional[int] = None    # 1-indexed page the issue was found on

class AuditReport(BaseModel):
    document_name: str
    findings: List[Finding]
    readiness_score: float
    summary: Union[str, Dict]
    pages: Optional[List[Dict]] = None  # list of {page_num, text} for document view
