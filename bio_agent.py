from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini", temperature = 0,max_tokens=1500) # I want to minimize hallucination - temperature = 0 makes the model output more deterministic 

# Our Embedding Model - has to also be compatible with the LLM
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)


pdf_path = "bio.pdf"


# Safety measure I have put for debugging purposes :)
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path) # This loads the PDF

# Checks if the PDF is there
try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

# Chunking Process
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


pages_split = text_splitter.split_documents(pages) # We now apply this to our pages

persist_directory = r"C:\Vaibhav\LangGraph_Book\LangGraphCourse\Agents"
collection_name = "bio"

# If our collection does not exist in the directory, we create using the os command
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)


try:
    # Here, we actually create the chroma database using our embeddigns model
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Created ChromaDB vector store!")
    
except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise


# Now we create our retriever 
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1} # K is the amount of chunks to return
)

@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns information from the Biology textbook.
    """

    docs = retriever.invoke(query)

    if not docs:
        return "لم أجد أي معلومة مرتبطة في كتاب الأحياء."

    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    
    return "\n\n".join(results)


tools = [retriever_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    """Check if we need retrieval based on tool calls + question complexity."""
    last_message = state['messages'][-1]

    # Extract user question (HumanMessage content)
    user_question = None
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break

    # If question is simple → don't continue to retriever
    if user_question and is_simple_question(user_question):
        return False

    # Otherwise, continue if tool calls exist
    return hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0




def is_simple_question(user_input: str) -> bool:
    """
    Simple heuristic to detect direct/short questions.
    Returns True if question can be answered without retrieval.
    """
    # Short questions with 1–2 words or direct "ما معنى" / "عرف" / "ايه هو"
    keywords = ["ما معنى", "ايه هو", "ما هو", "عرف", "define"]
    if any(kw in user_input for kw in keywords):
        return True
    if len(user_input.split()) < 6:  # very short question
        return True
    return False


system_prompt = """
أنت وكيل ذكي متخصص فقط في مجال علم الأحياء. تعتمد إجاباتك على المعلومات الواردة في الكتاب المقدم لك، ولا تخرج عن هذا السياق. يجب أن تجيب دائمًا باللغة العربية الفصحى، لكن إذا استخدم المستخدم العامية المصرية يمكنك الرد بنفس الطريقة مع الحفاظ على الدقة العلمية.

إذا كان السؤال مباشرًا جدًا (مثل معنى كلمة أو تعريف واضح أو معلومة أساسية) فأجب مباشرة بإجابة جيدة وكاملة دون الرجوع إلى الكتاب. أما إذا كان السؤال يتطلّب شرحًا أو تفاصيل أعمق، فاعتمد على محتوى الكتاب لتوليد إجابة دقيقة ومتكاملة.

لا تجب على أي سؤال لا يخص علم الأحياء أو لا يمكن ربطه بالكتاب.
"""


tools_dict = {our_tool.name: our_tool for our_tool in tools} # Creating a dictionary of our tools

# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state['messages'])
    # if not any(isinstance(m, SystemMessage) for m in messages):
    #     messages = [SystemMessage(content=system_prompt)] + messages################################

    message = llm.invoke(messages)
    return {'messages': [message]}

# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
    
        if t['name'] not in tools_dict:  # invalid tool
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, please retry with a valid tool."
        else:  # valid tool
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
    
        # ✅ Always append a ToolMessage
        results.append(
            ToolMessage(
                tool_call_id=t['id'],
                name=t['name'],
                content=str(result)
            )
        )


    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()


def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        messages = [
            SystemMessage(content=system_prompt),  # ✅ only once
            HumanMessage(content=user_input)
        ]

        result = rag_agent.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)




#running_agent()