from fastapi import FastAPI
from pydantic import BaseModel
from pipeline import graph

app = FastAPI(title="Research Report Generator")


class ReportRequest(BaseModel):
    topic: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate(req: ReportRequest):
    initial = {
        "topic": req.topic, "research": "", "draft": "",
        "critique": "", "approved": False, "revisions": 0,
    }
    final = graph.invoke(initial)
    return {
        "topic": req.topic,
        "report": final["draft"],
        "revisions": final["revisions"],
        "approved": final["approved"],
        "critique": final["critique"],
    }
