from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from bio_agent import rag_agent, system_prompt

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_agent(query: Query):
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query.question)
    ]
    result = rag_agent.invoke({"messages": messages})
    return {"answer": result['messages'][-1].content}
