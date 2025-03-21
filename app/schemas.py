from pydantic import BaseModel
from typing import List,Optional

# Pydantic Models
class LineStringModel(BaseModel):
    type: str
    coordinates: List[List[float]]
class RequestBody(BaseModel):
    line: List[Optional[LineStringModel]]  # Allows None values inside the list
