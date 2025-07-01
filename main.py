from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain.schema import HumanMessage, AIMessage
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pymongo import MongoClient
import redis
import json
import argparse
import uuid
import sys

# --- Setup: Redis (Short-Term Memory) ---
redis_client = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)

# --- Setup: MongoDB (Long-Term Memory + Static Config) ---
mongo_client = MongoClient("mongodb://127.0.0.1:27017")
mongo_db = mongo_client["chat_memory"]
threads_collection = mongo_db["threads"]
system_prompts_collection = mongo_db["system_prompts"]

# --- Setup: Vector DB (Qdrant) ---
client = QdrantClient(url="http://127.0.0.1:6333")

if not client.get_collection("rag"):
    client.create_collection(
        collection_name="rag",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    print("Created new collection: rag")

vector_store = QdrantVectorStore(
    client=client,
    collection_name="rag",
    embedding=OpenAIEmbeddings(),
)

# --- Helper: Load System Prompt ---
def load_system_prompt(agent_id):
    doc = system_prompts_collection.find_one({"agent_id": agent_id})
    if doc is not None:
        print(f"[AGENT_MNG]{json.dumps(doc, default=str, indent=2)}")
        return doc["prompt"]
    else:
        print("[AGENT_MNG]No document found.")
        raise ValueError(f"Agent {agent_id} not found in system prompts collection.")
    

# --- Node: Add Message ---
def add_message(state: dict) -> dict:
    print(f"Adding message to state... {json.dumps(state)}")
    
    # Safety check for empty state
    if not state or not isinstance(state, dict):
        print(f"WARNING: Received invalid state: {state}")
        # Create a new state object if none was provided
        state = {
            "thread_id": "fallback_thread",
            "agent_id": "default",
            "input": "No input provided"
        }
        print(f"Created fallback state: {json.dumps(state)}")
    
    try:
        thread_id = state["thread_id"]
        user_message = state["input"]
        history = redis_client.get(thread_id)
        history = json.loads(history) if history else []
        history.append({"type": "human", "content": user_message})
        redis_client.set(thread_id, json.dumps(history), ex=3600)
        state["messages"] = history
    except KeyError as e:
        print(f"KeyError in add_message: {e}")
        print(f"State keys: {list(state.keys())}")
        # Provide default values
        state["messages"] = [{"type": "human", "content": state.get("input", "No input")}]
    except Exception as e:
        print(f"Error in add_message: {e}")
        # Provide default values
        state["messages"] = [{"type": "human", "content": "Error occurred processing message"}]
        
    return state

# --- Node: Retrieve Docs ---
def retrieve_docs(state: dict) -> dict:
    query = state.get("input", "")
    agent_id = state.get("agent_id", "")
    
    try:
        if vector_store is not None:
            # Create filter for agent-specific documents
            filter_condition = None
            if agent_id:
                from qdrant_client.http import models as rest
                filter_condition = rest.Filter(
                    should=[
                        rest.FieldCondition(
                            key="metadata.agent_id",
                            match=rest.MatchValue(value=agent_id)
                        ),
                        rest.FieldCondition(
                            key="metadata.agent_id",
                            match=rest.MatchValue(value="global")
                        )
                    ]
                )
            
            # Different approach for performing similarity search
            # Check which method signature is supported
            try:
                # First try with direct filter parameter (newer API)
                docs = vector_store.similarity_search(query, k=3, filter=filter_condition)
            except TypeError:
                try:
                    # Try with metadata filter approach (alternative API)
                    if filter_condition:
                        # Convert to dict-based metadata filter
                        metadata_filter = {"$or": [
                            {"agent_id": agent_id},
                            {"agent_id": "global"}
                        ]}
                        docs = vector_store.similarity_search(query, k=3, metadata=metadata_filter)
                    else:
                        docs = vector_store.similarity_search(query, k=3)
                except Exception as inner_e:
                    print(f"Error with metadata filter approach: {inner_e}")
                    # Last resort, no filter
                    docs = vector_store.similarity_search(query, k=3)
            
            # Extract document contents and metadata
            retrieved_docs = []
            for doc in docs:
                source = doc.metadata.get("source", "Unknown")
                doc_name = doc.metadata.get("doc_name", "Unknown")
                content = f"[{doc_name}] {doc.page_content}"
                retrieved_docs.append(content)
            
            state["retrieved_docs"] = retrieved_docs
            print(f"Retrieved {len(retrieved_docs)} documents from vector store")
        else:
            # Fallback if vector store is not available
            state["retrieved_docs"] = ["No RAG documents available yet. Add documents with 'python main.py rag new'"]
            print("Vector store not available, using placeholder")
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        # Provide fallback content
        state["retrieved_docs"] = [f"Error retrieving documents: {str(e)}"]
        
        # Fallback to simple search without filters
        try:
            print("Trying fallback search without filters...")
            docs = vector_store.similarity_search(query, k=3)
            retrieved_docs = [f"[{doc.metadata.get('doc_name', 'Unknown')}] {doc.page_content}" for doc in docs]
            state["retrieved_docs"] = retrieved_docs
            print(f"Retrieved {len(retrieved_docs)} documents using fallback method")
        except Exception as fallback_e:
            print(f"Fallback search also failed: {fallback_e}")
    
    print(f"[RAG_MNG] state = {json.dumps(state)}")
    return state

# --- Node: Call LLM ---
llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
def call_llm(state: dict) -> dict:
    print(f"Calling LLM with state: {json.dumps(state)}")
    # Retrieve system prompt and RAG docs
    system_prompt = load_system_prompt(state["agent_id"])
    rag_docs = state.get("retrieved_docs", [])
    system_message = {
        "role": "system",
        "content": system_prompt + "\n\n# Documentos relevantes para a conversa:\n" + "\n".join(rag_docs)
    }

    # Load previous messages from state (should be a list of dicts with 'type' and 'content')
    history = state.get("messages", [])
    chat_messages = [system_message]
    for msg in history:
        if msg["type"] == "human":
            chat_messages.append({"role": "user", "content": msg["content"]})
        elif msg["type"] == "ai":
            chat_messages.append({"role": "assistant", "content": msg["content"]})

    # Add the new user message if not already in history
    if not history or history[-1].get("content") != state["input"]:
        chat_messages.append({"role": "user", "content": state["input"]})

    # Convert to LangChain message objects
    lc_messages = []
    for m in chat_messages:
        if m["role"] == "system":
            lc_messages.append(HumanMessage(content=m["content"]))  # LangChain doesn't have SystemMessage, so prepend as HumanMessage
        elif m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))

    response = llm.invoke(lc_messages)
    state["output"] = response.content

    # Update chat history: append the AI response
    updated_history = history + [{"type": "ai", "content": response.content}]
    state["messages"] = updated_history
    redis_client.set(state["thread_id"], json.dumps(updated_history), ex=3600)

    # Queue for long-term persistence
    threads_collection.update_one(
        {"thread_id": state["thread_id"]},
        {"$set": {"messages": updated_history}},
        upsert=True
    )
    return state

# --- Graph Setup ---
# Initialize the graph with a defined state type
graph = StateGraph(dict)

# Add nodes with explicit handling of the input state
graph.add_node("add_message", add_message)
graph.add_node("retrieve_docs", retrieve_docs)
graph.add_node("call_llm", call_llm)

# Set the entry point and define edges
graph.set_entry_point("add_message")
graph.add_edge("add_message", "retrieve_docs")
graph.add_edge("retrieve_docs", "call_llm")
graph.add_edge("call_llm", END)

# Use debug=True to help diagnose state passing issues
app = graph.compile()

# --- CLI Command Handlers ---
def create_new_agent(system_prompt):
    """Create a new agent with a unique ID and save the system prompt."""
    agent_id = f"ag_{uuid.uuid4()}"
    
    # Save system prompt to MongoDB
    system_prompts_collection.insert_one({
        "agent_id": agent_id,
        "prompt": system_prompt
    })
    
    print(f"Created new agent: {agent_id}")
    return agent_id

def generate_new_thread_id():
    """Generate a new thread ID."""
    return f"th_{uuid.uuid4()}"

def process_response(agent_id, thread_id, user_input):
    """Process a response using the specified agent and thread."""
    # If thread_id is "0", create a new thread
    if thread_id == "0":
        thread_id = generate_new_thread_id()
        print(f"Created new thread: {thread_id}")
    
    # Check if agent exists
    agent_doc = system_prompts_collection.find_one({"agent_id": agent_id})
    if not agent_doc:
        print(f"Error: Agent {agent_id} not found.")
        return
    
    # Create a GraphState instance instead of a plain dict
    state = {
        "thread_id": thread_id,
        "agent_id": agent_id,
        "input": user_input
    }
    
    print(f"Initial state: {json.dumps(state)}")  # Debug print

    try:
        final_state = app.invoke(state)
        print(f"Agent Response: {final_state['output']}")
        print(f"Thread ID: {thread_id}")
    except Exception as e:
        print(f"Error processing response: {e}")
        import traceback
        traceback.print_exc()  # Print full stack trace for debugging

def embed_file(filepath, agent_id=None, doc_name=None, chunk_size=1000, chunk_overlap=200):
    """
    Read a file, split it into chunks, and store embeddings in Qdrant.
    
    Args:
        filepath (str): Path to the file to embed
        agent_id (str, optional): Agent ID to associate with the document
        doc_name (str, optional): Name to give to the document
        chunk_size (int, optional): Size of text chunks
        chunk_overlap (int, optional): Overlap between chunks
    
    Returns:
        int: Number of chunks embedded
    """
    import os
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
    from langchain.docstore.document import Document
    
    # Set default document name if not provided
    if not doc_name:
        doc_name = os.path.basename(filepath)
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return 0
    
    try:
        # Load document based on file extension
        file_extension = os.path.splitext(filepath)[1].lower()
        
        if file_extension == ".pdf":
            loader = PDFMinerLoader(filepath)
            documents = loader.load()
        elif file_extension == ".csv":
            loader = CSVLoader(filepath)
            documents = loader.load()
        elif file_extension in [".txt", ".md", ".py", ".js", ".html", ".css", ".json"]:
            loader = TextLoader(filepath)
            documents = loader.load()
        else:
            # For unsupported file types, read as plain text
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
                documents = [Document(page_content=text, metadata={"source": filepath})]
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "source": filepath,
                "doc_name": doc_name,
                "chunk_id": i,
                "agent_id": agent_id if agent_id else "global"
            })
        
        print(f"Split {filepath} into {len(chunks)} chunks")
        
        # Store in vector database
        if vector_store is not None:
            vector_store.add_documents(chunks)
            print(f"Stored {len(chunks)} chunks in vector store")
            return len(chunks)
        else:
            print("Error: Vector store is not available")
            return 0
            
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return 0

# --- CLI Argument Parsing ---
def main():
    parser = argparse.ArgumentParser(description="Agent Connect CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # New agent command
    new_agent_parser = subparsers.add_parser("new", help="Create a new agent")
    new_agent_parser.add_argument("agent", help="Should be 'agent'")
    new_agent_parser.add_argument("system_prompt", help="System prompt for the new agent")
    
    # Response command
    response_parser = subparsers.add_parser("response", help="Generate a response")
    response_parser.add_argument("agent_id", help="Agent ID to use")
    response_parser.add_argument("thread_id", help="Thread ID to use (use '0' for new thread)")
    response_parser.add_argument("user_input", nargs="*", help="User input message")

    # List agents command
    list_agents_parser = subparsers.add_parser("list", help="List all agents")
    list_agents_parser.add_argument("--verbose", action="store_true", help="Show detailed agent information")

    # Show agent command
    show_agent_parser = subparsers.add_parser("show", help="Show agent details")
    show_agent_parser.add_argument("agent_id", help="Agent ID to show details for")

    # History command
    history_parser = subparsers.add_parser("history", help="Show thread history")
    history_parser.add_argument("thread_id", help="Thread ID to show history for")

    # RAG command
    rag_parser = subparsers.add_parser("rag", help="RAG manager")
    rag_subparsers = rag_parser.add_subparsers(dest="subcommand", help="RAG subcommands")

    # RAG new (add document)
    rag_new_parser = rag_subparsers.add_parser("new", help="Add a new RAG document")
    rag_new_parser.add_argument("agent_id", help="Agent ID to associate the document with")
    rag_new_parser.add_argument("doc_path", help="Path to the document file")
    rag_new_parser.add_argument("doc_name", help="Name to assign to the document")

    # RAG rm (remove document)
    rag_rm_parser = rag_subparsers.add_parser("rm", help="Remove a RAG document")
    rag_rm_parser.add_argument("agent_id", help="Agent ID associated with the document")
    rag_rm_parser.add_argument("doc_name", help="Name of the document to remove")

    # RAG list (list documents)
    rag_list_parser = rag_subparsers.add_parser("list", help="List RAG documents")
    rag_list_parser.add_argument("agent_id", help="Agent ID to list documents for")
    
    args = parser.parse_args()
    
    if args.command == "new":
        if args.agent != "agent":
            print("Error: Command should be 'new agent <system-prompt>'")
            return
        create_new_agent(args.system_prompt)
    
    elif args.command == "response":
        if not args.user_input:
            print("Error: Please provide user input for the response")
            return
        user_input = " ".join(args.user_input)
        process_response(args.agent_id, args.thread_id, user_input)
    
    elif args.command == "list":
        agents_cursor = system_prompts_collection.find()
        agents = list(agents_cursor)
        if len(agents) == 0:
            print("No agents found.")
            return
        if args.verbose:
            for agent in agents:
                print(f"Agent ID: {agent['agent_id']}, System Prompt: {agent['prompt']}")
        else:
            for agent in agents:
                print(f"Agent ID: {agent['agent_id']}")

    elif args.command == "show":
        agent_doc = system_prompts_collection.find_one({"agent_id": args.agent_id})
        if agent_doc:
            print(f"System Prompt: {agent_doc['prompt']}")
        else:
            print(f"Error: Agent {args.agent_id} not found.")

    elif args.command == "history":
        thread_id = args.thread_id
        history = redis_client.get(thread_id)
        if history:
            history = json.loads(history)
            print(f"History for thread {thread_id}:\n")
            for msg in history:
                print(f"{msg['type']}: {msg['content']}")
        else:
            print(f"No history found for thread {thread_id}.")

    elif args.command == "rag":
        if args.subcommand == "new":
            if not args.doc_path or not args.doc_name or not args.agent_id:
                print("Error: Please provide all required arguments for adding a RAG document.")
                return
            print(f"Adding new RAG document '{args.doc_name}' for agent {args.agent_id} from {args.doc_path}")
            chunks_added = embed_file(args.doc_path, args.agent_id, args.doc_name)
            if chunks_added > 0:
                print(f"Successfully added {chunks_added} chunks from '{args.doc_name}' to the RAG database")
            else:
                print(f"Failed to add document '{args.doc_name}' to the RAG database")
                
        elif args.subcommand == "rm":
            print(f"Removing RAG document '{args.doc_name}' for agent {args.agent_id}")
            if vector_store is not None:
                try:
                    # Delete embeddings by metadata filter
                    from qdrant_client.http import models as rest
                    filter_selector = rest.Filter(
                        must=[
                            rest.FieldCondition(
                                key="metadata.doc_name",
                                match=rest.MatchValue(value=args.doc_name)
                            ),
                            rest.FieldCondition(
                                key="metadata.agent_id",
                                match=rest.MatchValue(value=args.agent_id)
                            )
                        ]
                    )
                    # Perform delete operation
                    client.delete(
                        collection_name="rag",
                        points_selector=rest.FilterSelector(filter=filter_selector)
                    )
                    print(f"Document '{args.doc_name}' has been removed from the vector store")
                except Exception as e:
                    print(f"Error removing document: {e}")
            else:
                print("Vector store is not available")
                
        elif args.subcommand == "list":
            print(f"Listing RAG documents for agent {args.agent_id}")
            if vector_store is not None:
                try:
                    # Query to get distinct document names
                    from qdrant_client.http import models as rest
                    # Use scroll API to get points with the agent_id
                    scroll_result = client.scroll(
                        collection_name="rag",
                        scroll_filter=rest.Filter(
                            must=[
                                rest.FieldCondition(
                                    key="metadata.agent_id", 
                                    match=rest.MatchValue(value=args.agent_id)
                                )
                            ]
                        ),
                        limit=100,  # Adjust based on expected number of documents
                        with_payload=True,
                        with_vectors=False
                    )
                    
                    # Extract unique document names
                    doc_names = set()
                    for point in scroll_result[0]:
                        if point.payload and "metadata" in point.payload:
                            doc_name = point.payload["metadata"].get("doc_name")
                            if doc_name:
                                doc_names.add(doc_name)
                    
                    if doc_names:
                        print(f"Documents for agent {args.agent_id}:")
                        for doc_name in sorted(doc_names):
                            print(f"- {doc_name}")
                    else:
                        print(f"No documents found for agent {args.agent_id}")
                except Exception as e:
                    print(f"Error listing documents: {e}")
            else:
                print("Vector store is not available")

    else:
        parser.print_help()

# --- Example Run ---
if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        print("Example usage:")
        print("  python main.py new agent \"You are a helpful coding assistant specialized in Python.\"")
        print("  python main.py response <agent_id> 0 \"How do I use FastAPI?\"")
        print("  python main.py response <agent_id> <thread_id> \"Tell me more about async functions\"")
        print("  python main.py list --verbose")
        print("  python main.py show <agent_id>")
        print("  python main.py history <thread_id>")
        print("  python main.py rag new <agent_id> <doc_path> <doc_name> - embed and store a document")
        print("  python main.py rag rm <agent_id> <doc_name> - remove a document from RAG")
        print("  python main.py rag list <agent_id> - list RAG documents for an agent")

# Jeferson ag_4635b53d-301a-4bc8-a99e-c2dd61e2ac8b
# th_280cbe60-eaec-4ee6-8859-c883eabfc4f6