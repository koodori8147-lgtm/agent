import base64
import pymupdf

from typing import Literal
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.messages import HumanMessage

class PdfReview(BaseModel):
    song_match: Literal["yes", "no", "unknown"]
    artist_match: Literal["yes", "no", "unknown"]
    instrument_match: Literal["yes", "no", "unknown"]
    score_present: Literal["yes", "no", "unknown"]