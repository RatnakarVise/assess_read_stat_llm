from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import os, json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

# --- Load environment ---
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
if langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

app = FastAPI(title="OSS Note 1803189 - READ TABLE / SORT Consistency Checker")

# ---- Strict input models ----
class ReadItem(BaseModel):
    table: str
    target_type: str
    target_name: str
    used_fields: List[str]
    suggested_fields: List[str]
    suggested_statement: Optional[str] = None

    @field_validator("used_fields", mode="before")
    @classmethod
    def clean_used_fields(cls, v):
        return [x for x in v if x]

    @field_validator("suggested_statement", mode="before")
    @classmethod
    def clean_suggested_statement(cls, v):
        return v if v is not None and v != "" else None


class NoteContext(BaseModel):
    pgm_name: Optional[str] = None
    inc_name: Optional[str] = None
    type: Optional[str] = None
    name: Optional[str] = None
    read_usage: List[ReadItem] = Field(default_factory=list)


# ---- Summarizer ----
def summarize_context(ctx: NoteContext) -> dict:
    return {
        "unit_program": ctx.pgm_name,
        "unit_include": ctx.inc_name,
        "unit_type": ctx.type,
        "unit_name": ctx.name,
        "read_usage": [item.model_dump() for item in ctx.read_usage]
    }


# ---- LangChain Prompt ----
SYSTEM_MSG = "You are a precise ABAP reviewer familiar with SAP performance note 1803189. Respond in strict JSON only."

USER_TEMPLATE = """
You are evaluating ABAP code that uses `READ TABLE ... BINARY SEARCH` and `SORT` statements.

Known risk:
- According to SAP Note 1803189, performance and correctness issues arise if the `SORT` fields do not fully cover the `READ TABLE WITH KEY ... BINARY SEARCH` fields.
- If fields mismatch, the search result may be wrong or undefined.

Your tasks:
1) Provide a concise **assessment**:
   - Risk: incorrect results or short dumps if READ uses keys not covered by SORT.
   - Impact: business logic errors, failed lookups, or incorrect data retrieval.
   - Recommend: ensure SORT includes all fields used in READ WITH KEY before binary search.

2) Provide an actionable **LLM remediation prompt**:
   - Reference program/include/type/name.
   - Identify the `SORT` and `READ TABLE` usage.
   - Ensure SORT BY fields match all READ TABLE WITH KEY fields (order and completeness).
   - Suggest corrected SORT statements.
   - Require JSON output with keys: original_code_snippet, remediated_code_snippet, changes[].

Return ONLY strict JSON:
{{
  "assessment": "<concise impact of mismatch between SORT and READ>",
  "llm_prompt": "<prompt for LLM code fixer>"
}}

Unit metadata:
- Program: {pgm_name}
- Include: {inc_name}
- Unit type: {type}
- Unit name: {name}

System context:
{context_json}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MSG),
    ("user", USER_TEMPLATE),
])

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
parser = JsonOutputParser()
chain = prompt | llm | parser


def llm_assess(ctx: NoteContext):
    ctx_json = json.dumps(summarize_context(ctx), ensure_ascii=False, indent=2)
    return chain.invoke({
        "context_json": ctx_json,
        "pgm_name": ctx.pgm_name,
        "inc_name": ctx.inc_name,
        "type": ctx.type,
        "name": ctx.name
    })


@app.post("/assess-read-table")
def assess_note_context(ctxs: List[NoteContext]):
    results = []
    for ctx in ctxs:
        try:
            llm_result = llm_assess(ctx)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

        results.append({
            "pgm_name": ctx.pgm_name,
            "inc_name": ctx.inc_name,
            "type": ctx.type,
            "name": ctx.name,
            "code": "",  # keep ABAP code outside response
            "assessment": llm_result.get("assessment", ""),
            "llm_prompt": llm_result.get("llm_prompt", "")
        })

    return results


@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}
