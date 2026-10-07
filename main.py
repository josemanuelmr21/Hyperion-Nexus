# =============================================================================
# Hyperion Agent - agentic microservice for the HyperAI IDE
#
# Architecture overview:
#   IDE frontend --POST /chat {user_id, text}--> FastAPI --> LangGraph agent
#   The agent talks to an OpenAI-compatible LLM, queries a RAG index built from
#   the HyperAI documentation, reads/validates files through the IDE backend
#   and controls the IDE editor by emitting actions over Server-Sent Events.
#
# Requirements covered:
#   1. Microservice with /chat on port 8000, answers + IDE actions via SSE
#   2. Guardrails (system prompt rules)
#   3. RAG over the HyperAI documentation
#   4. Per-user memory (checkpointer keyed by user_id)
#   5. Human-in-the-Loop for state-changing actions (delete_file, edit_file, delete_folder)
# =============================================================================
import json
import os
import re

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import Language
# Official LangChain middleware that pauses the graph BEFORE a protected tool runs.
from langchain.agents.middleware import HumanInTheLoopMiddleware
# Command(resume=...) is how a paused graph is resumed with the user's decision.
from langgraph.types import Command

from helpers import read_file, validate_file


load_dotenv()

API_KEY = os.environ.get("API_KEY", "")
BASE_URL = "https://legion1.di.uoa.gr/v1"
MODEL = "llama3.1"

# Absolute path to the docs folder: the RAG keeps working regardless of the
# directory the container (or the developer) starts the process from.
DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")

# =============================================================================
# RAG PIPELINE (Load -> Chunk -> Embed -> Store -> Retrieve)
# =============================================================================

print("Initializing HyperAI vector database...")
# Placeholder: if the pipeline fails, the agent still starts, just without the
# RAG tool (graceful degradation instead of a crashed service).
search_hyperai_docs = None

try:
    # LOAD
    # The loader walks the docs directory and reads every .docx file.
    loader = DirectoryLoader(
    DOCS_DIR,
    glob="**/*.docx",
    loader_cls=Docx2txtLoader
)
    docx_docs = loader.load()
    
    # CHUNK
    # The splitter is configured to cut text into 1000-character blocks.
    # A 200-character overlap is added to prevent cutting important sentences or context in half.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docx_docs)

    # IDE TUTORIAL (plain-text Markdown): documents the real HyperAI DSL
    # (Native Apps / Device Apps) so generated YAML follows the HyperAI format.
    # Own try/except: if the file is missing, the .docx RAG keeps working.
    try:
        tutorial_loader = DirectoryLoader(
            DOCS_DIR,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        # Markdown-aware splitter with bigger chunks, so tables and YAML
        # blocks are not cut in half.
        md_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN, chunk_size=1500, chunk_overlap=200
        )
        splits += md_splitter.split_documents(tutorial_loader.load())
    except Exception as e:
        print(f"WARNING: IDE tutorial not loaded: {e}")
    
    # EMBED (Remote execution using the hackathon's endpoint)
    # 'drop_params' tells the proxy to ignore parameters the embedding model
    # does not support, avoiding request errors.
    embeddings = OpenAIEmbeddings(
        model="mxbai-embed-large",
        base_url=BASE_URL,
        api_key=API_KEY,
        model_kwargs={"extra_body": {"drop_params": True}},
    )
    
    # STORE
    # A temporary database is created in memory holding all the text chunks and their corresponding vectors.
    vectorstore = InMemoryVectorStore.from_documents(splits, embeddings)
    
    # A retriever is set up to search the database and return only the top 4 most relevant text chunks per query.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # CREATE TOOL
    # The retriever is wrapped as a standard LangChain tool so the agent decides
    # when to consult the documentation. The docstring is what the LLM reads to
    # decide whether to call it.
    @tool
    def search_hyperai_docs(query: str) -> str:
        """Search the official HyperAI and HyperAI IDE documentation. It contains: HyperAI concepts and architecture, how to use and deploy apps in the IDE (quick start, IDE guide), the DSL for application profiles (Native Apps and Device Apps, with every field), and ready-made YAML examples. Use it before answering any question about HyperAI or the IDE, and before writing a YAML profile."""
        docs = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in docs)
    
    print(f"RAG initialized successfully. {len(splits)} chunks loaded.")

except Exception as e:
    print(f"CRITICAL ERROR INITIALIZING RAG: {e}")

# =============================================================================
# LLM
# =============================================================================

llm = ChatOpenAI(
    model=MODEL,
    base_url=BASE_URL,
    api_key=API_KEY,
    max_completion_tokens=2048,
    # temperature=0 makes tool selection deterministic. With a small model such
    # as llama3.1 this greatly reduces random "tool vs. plain answer" mistakes.
    temperature=0, 
    streaming = True # Added 'streaming=True' so the LLM can reply token by token
)

app = FastAPI(title="Hyperion Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_id: str  # a UUID automatically generated by the IDE — use it to keep per-user session memory
    text: str  # the text the user typed in the chat

# =============================================================================
# ACTION TOOLS (control the IDE via SSE)
# These tools do not touch the filesystem themselves: the real payload is sent
# to the IDE frontend by the SSE loop (see generate_reply), which listens to
# the "on_tool_start" event. The return value is only feedback for the LLM.
# =============================================================================

@tool
def create_file(path: str, content: str) -> str:
    """Creates a new file or overwrites an existing one in the workspace. Use this to write code or YAML files."""
    # The actual execution payload is sent via SSE in the main event loop.
    return f"File {path} successfully sent to the IDE for creation."

@tool
def delete_file(path: str) -> str:
    """Deletes a file from the workspace. You can provide the full path or just the file name."""
    return f"File {path} successfully sent to the IDE for deletion."

@tool
def edit_file(path: str, content: str) -> str:
    """Replaces the entire content of an existing file and opens it in the editor."""
    return f"File {path} successfully sent to the IDE for editing."

@tool
def create_folder(path: str) -> str:
    """Creates a folder in the workspace. The path is relative to the workspace root (never absolute, never '..')."""
    return f"Folder {path} successfully sent to the IDE for creation."

@tool
def delete_folder(path: str) -> str:
    """Deletes a folder from the workspace. You can provide the full path or just the folder name."""
    return f"Folder {path} successfully sent to the IDE for deletion."

# =============================================================================
# READ TOOLS (query the IDE backend, no side effects, no confirmation needed)
# Errors are returned as text so the LLM can explain them to the user instead
# of the whole request crashing.
# =============================================================================

@tool
async def read_workspace_file(path: str) -> str:
    """Reads the content of a file in the workspace. Use this to inspect code before modifying it."""
    try:
        content = await read_file(path)
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
async def validate_workspace_file(path: str) -> str:
    """Validates a file against Native Apps or Device Apps rules. Read or validate files if requested."""
    try:
        report = await validate_file(path)
        return json.dumps(report)
    except Exception as e:
        return f"Error validating file: {str(e)}"

# Group all available tools
tools = [
    create_file,
    delete_file,
    edit_file,
    create_folder,
    delete_folder,
    read_workspace_file,
    validate_workspace_file,
]

# The RAG tool is added only if the vector database was built successfully.
if search_hyperai_docs is not None:
    tools.append(search_hyperai_docs)

# =============================================================================
# AGENT AND MEMORY SETUP
# =============================================================================

# MemorySaver stores the conversation state in RAM, one thread per user_id.
# It is also REQUIRED by the Human-in-the-Loop: a paused graph needs a
# checkpoint to be resumed later.
memory = MemorySaver()

# SYSTEM PROMPT

system_prompt = """
You are Hyperion, the AI assistant integrated into the HyperAI IDE.
You help with programming, IDE workspace management, and the HyperAI/Hyperion project.

For general conversation, just reply naturally. NEVER mention tools, tool calls, functions, or whether one is needed. NEVER say "there is no function".

Rules:
1. Always reply in the language of the user's LAST message (English -> English, Spanish -> Spanish), including refusals.

2. ALWAYS ALLOWED, never decline: greetings, thanks, and any question about the conversation or about the user (their name, what they asked before, what you did). Answer these from the chat history, without tools.

3. Decline requests whose SUBJECT is unrelated to programming, the IDE or HyperAI (weather, sports, cooking, jokes, trivia), even if they use technical words such as "a pizza recipe in Kubernetes terms". Decline in one short sentence.

4. Create, edit or delete files and folders ONLY when the user explicitly asks.

5 Use search_hyperai_docs when the answer requires factual information specific to HyperAI, Hyperion, or the IDE, including its architecture, components, DSLs, application profiles, deployment process, configuration, validation rules, or documented workflows.
Do not use search_hyperai_docs for general programming knowledge or general conversation unless the question specifically asks how it applies to HyperAI.
If the retrieved documentation does not contain enough information to answer the question, say that the information could not be found in the documentation.

6. When creating HyperAI-specific files or configuration, first retrieve the relevant HyperAI documentation and use it as the source of truth. Do not substitute generic standards or conventions for the HyperAI specification.

7. When documentation is retrieved, prefer the documented HyperAI format over general knowledge. For example, if HyperAI defines its own YAML schema, use that schema instead of a generic Kubernetes, Docker Compose, or other external schema.

8. To delete or edit a file, or to delete a folder, call the tool directly. The system asks the user for confirmation, so NEVER ask for it yourself.

9. Before edit_file on an existing file, call read_workspace_file first and send the FULL updated content.

10. Never write JSON or tool names in your text.

11. If search_hyperai_docs returns nothing relevant, say you could not find it in the documentation. NEVER invent a "hypothetical" answer.

Examples (no tool needed):
User: hi
Assistant: Hello! How can I help you with the IDE or HyperAI?
User: my name is Ana
Assistant: Nice to meet you, Ana! How can I help you with the IDE or HyperAI?
User: What's my name?
Assistant: Your name is Ana.
User: Hola, me llamo Ana
Assistant: Hola Ana, ¿en qué puedo ayudarte con el IDE o con HyperAI?
User: ¿Cómo me llamo?
Assistant: Te llamas Ana.
User: ¿Qué tiempo hace hoy?
Assistant: Solo puedo ayudarte con programación, el IDE o HyperAI.
User: recipe of pizza in Kubernetes terms
Assistant: I can only help with programming, the IDE or HyperAI.
"""

# The agent executor connects LLM + tools + memory. Unlike a direct LLM call it
# can loop (think -> act -> observe -> reply).
# HUMAN-IN-THE-LOOP: the middleware intercepts ONLY delete_file, edit_file and delete_folder
# (the state-changing actions that destroy or overwrite data). The graph pauses
# before they run; read tools and RAG keep running without interruption.
# 'approve'/'reject' are the only decisions allowed, so the user cannot alter
# the arguments of the action.

agent_executor = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "delete_file": {"allowed_decisions": ["approve", "reject"]},
                "edit_file": {"allowed_decisions": ["approve", "reject"]},
                "delete_folder": {"allowed_decisions": ["approve", "reject"]},
            }
        )
    ],
    checkpointer=memory,
)

# =============================================================================
# HUMAN-IN-THE-LOOP HELPERS
# Translate the user's chat message into a decision the graph understands, and
# build the confirmation question that is shown in the chat.
# =============================================================================

# Words that count as approval. Both languages are accepted because the IDE
# users may write in either. Matching is EXACT (see is_approval).

AFFIRMATIVES = {
    # English
    "yes", "y", "yeah", "yep", "ok", "okay", "confirm", "sure",
    "go ahead", "proceed", "do it", "approved",
    # Spanish
    "si", "sí", "vale", "confirmo", "confirmar", "adelante",
    "claro", "dale", "procede", "hazlo", "aprobado",
}


def is_approval(text: str) -> bool:
    """Approve ONLY when the answer is clearly affirmative."""
    # Lowercase and strip punctuation so "SÍ!!!" and "yes." are accepted.
    cleaned = re.sub(r"[^\w\sáéíóúüñ]", "", text.lower()).strip()
    # Exact match on purpose: "yes, but delete another file" must NOT approve.
    return cleaned in AFFIRMATIVES


def build_decisions(text: str, action_requests: list) -> list:
    """One decision per pending action, in the same order (required by LangGraph)."""
    if is_approval(text):
        decision = {"type": "approve"}
    else:
        # Fail-safe: anything that is not a clear "yes" cancels the action.
        # The user's text is forwarded so the model can react to a new request.
        decision = {
            "type": "reject",
            "message": (
                f"The user did NOT approve the action and replied: '{text}'. "
                "Do not retry the rejected action. If the reply asks for something else, do that instead."
    ),
}
    return [decision for _ in action_requests]


def describe_actions(action_requests: list) -> str:
    """Build the confirmation question shown to the user in the chat."""
    # Generated by code (not by the LLM) so the question is always accurate,
    # consistent and cannot be altered by the model.
    lines = []
    for action in action_requests:
        path = action["args"].get("path", "?")
        if action["name"] == "delete_file":
            lines.append(f"- Delete `{path}`")
        elif action["name"] == "edit_file":
            lines.append(f"- Replace the entire content of `{path}`")
        elif action["name"] == "delete_folder":
            lines.append(f"- Delete the folder `{path}` and its contents")
        else:
            lines.append(f"- {action['name']} on `{path}`")
    return (
        "I need your confirmation before continuing:\n"
        + "\n".join(lines)
        + "\n\nReply **yes** to confirm or **no** to cancel."
    )

# =============================================================================
# SSE LOOP: EVENT INTERCEPTOR
# =============================================================================

async def generate_reply(request: ChatRequest):
    # 'thread_id' binds the memory to the specific user_id. This gives each user their own session history.
    config = {"configurable": {"thread_id": request.user_id}}

    # Is there a paused action waiting for this user's answer?
    state = await agent_executor.aget_state(config)

    if state.interrupts:
        # The graph is paused: this message is the answer to the confirmation
        # question, not a new conversation turn.
        action_requests = state.interrupts[0].value["action_requests"]
        decisions = build_decisions(request.text, action_requests)
        graph_input = Command(resume={"decisions": decisions})
    else:
        # Normal turn: format the user's input as a message list for LangGraph.
        graph_input = {"messages": [{"role": "user", "content": request.text}]}

    # astream_events lets us stream the final response back to the IDE even
    # when the agent is using tools under the hood.
    async for event in agent_executor.astream_events(graph_input, config=config, version="v2"):
        kind = event["event"]

        # The LLM generates standard response text
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            
            # Skip chunks that carry internal tool invocations, avoiding Tool Leakage.
            if chunk.tool_call_chunks:
                continue
            
            # Ensure the chunk has text content before yielding it via SSE.
            if chunk.content and isinstance(chunk.content, str):
                yield f"data: {json.dumps({'response': chunk.content})}\n\n"

        # An action tool is about to run (IDE control).
        # Thanks to the HITL middleware, for delete_file/edit_file this event
        # only fires AFTER the user approved, so the IDE never acts early.
        elif kind == "on_tool_start":
            tool_name = event["name"]
            tool_args = event["data"].get("input", {})

            # Only the tools that require the IDE to do something visually.
            if tool_name in ["create_file", "edit_file", "delete_file", "create_folder", "delete_folder"]:

                # Build the exact payload documented by the IDE frontend
                action_payload = {"action": tool_name}
                action_payload.update(tool_args)
                yield f"data: {json.dumps(action_payload)}\n\n"

    # The stream ended: check whether the graph stopped because it is waiting
    # for confirmation. If so, ask the user in the chat; the answer will arrive
    # in the next POST /chat and be handled by the branch at the top.
    state = await agent_executor.aget_state(config)
    if state.interrupts:
        question = describe_actions(state.interrupts[0].value["action_requests"])
        yield f"data: {json.dumps({'response': question})}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(generate_reply(request), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
