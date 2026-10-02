from pydantic import BaseModel
from typing import Literal

class GuardrailResult(BaseModel):
    decision: Literal["allow", "block"]
    reason: str