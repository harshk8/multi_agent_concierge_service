import asyncio
import torch

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import Tool

asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())


torch.mps.empty_cache()

torch.set_default_device("cpu")


hf_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=-1,   # ✅ explicitly force CPU
    max_length=256,       # ↓ shorten output
    truncation=True,      # ✅ discard overflow tokens safely
    temperature=0.2
)

# memory = ConversationSummaryBufferMemory(
#     llm=None,
#     max_token_limit=200   # summarize automatically when long
# )


# === Step 1: LLM ===
# hf_pipeline = pipeline(
#     "text2text-generation",
#     model="google/flan-t5-base",
#     max_length=512,
#     temperature=0.2
# )
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# === Step 2: Tools ===
def search_doctor(q: str):
    return f"Found a doctor for query: {q}"

def book_doctor(q: str):
    return f"Doctor booked successfully for: {q}"


tools = [
    Tool(
        name="search_doctor",
        func=search_doctor,
        description="Search for doctors by specialization or name"
    ),
    Tool(
        name="book_doctor",
        func=book_doctor,
        description="Book a doctor appointment"
    ),
]

# === Step 3: Memory ===
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=200   # summarize automatically when long
)

memory = ConversationBufferMemory(memory_key="chat_history")

if len(memory.buffer.split()) > 400:
    memory.clear()


# === Step 4: Create the Agent ===
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
)

# === Step 5: Run ===
# query = "Book me a cardiologist for tomorrow at 10 AM"
query = "Task: Schedule a doctor appointment.\nDetails: I need a cardiologist for tomorrow at 10 AM."

response = agent.invoke({"input": query})
print("💬 Response:", response["output"])





# # from langchain.agents import Tool, initialize_agent
# # from langchain.llms import OpenAI  # or HuggingFacePipeline
# # from langchain.llms import HuggingFacePipeline
# from db import get_providers, book_provider
# from langchain.memory import ConversationBufferMemory
# from dotenv import load_dotenv


# load_dotenv()

# # ---- Tools for domain agents ----
# def search_providers(domain, specialization: str = ""):
#     res = get_providers(domain, specialization)
#     if not res: return f"No providers found for {domain} with specialization '{specialization}'"
#     out = []
#     for r in res:
#         pid, name, spec, slots = r
#         out.append(f"{pid}: {name} ({spec}) — slots: {slots}")
#     return "\n".join(out)

# def schedule(domain, provider_id: int, user_name: str, time_iso: str):
#     ok, msg = book_provider(domain, provider_id, user_name, time_iso)
#     return msg if ok else f"Failed: {msg}"

# # Create Tool wrappers
# tools = [
#     Tool(name="search_doctors", func=lambda q: search_providers("doctor", q), description="Find doctors by specialization"),
#     Tool(name="book_doctor", func=lambda args: schedule("doctor", *args), description="Book a doctor; args: provider_id,user_name,time_iso"),
#     Tool(name="search_plumbers", func=lambda q: search_providers("plumber", q), description="Find plumbers"),
#     Tool(name="book_plumber", func=lambda args: schedule("plumber", *args), description="Book a plumber; args: provider_id,user_name,time_iso"),
#     Tool(name="search_househelp", func=lambda q: search_providers("househelp", q), description="Find house help"),
#     Tool(name="book_househelp", func=lambda args: schedule("househelp", *args), description="Book house help; args: provider_id,user_name,time_iso"),
#     # Flight tool would normally call external API; here we mock:
#     Tool(name="search_flights", func=lambda q: "Flight 101 at 2025-11-12T09:00 (mock)", description="Search flights"),
#     Tool(name="book_flight", func=lambda args: "Flight booked (mock)", description="Book a flight; args: flight_id,user_name")
# ]



# from transformers import pipeline
# # from langchain.llms import HuggingFacePipeline
# from langchain_community.llms import HuggingFacePipeline
# # Create a text generation pipeline
# hf_pipeline = pipeline(
#     "text2text-generation",
#     model="google/flan-t5-large",   # You can replace with "google/flan-t5-base" or "declare-lab/flan-alpaca-large"
#     max_length=512,
#     temperature=0.3
# )

# # Wrap it for LangChain
# llm = HuggingFacePipeline(pipeline=hf_pipeline)

# # ---- LLM & Router Agent ----
# # Use OpenAI or local HF model. For demo, you can set LLM to OpenAI(api_key=...) or use a small HF model.
# # llm = HuggingFacePipeline(temperature=0) #OpenAI(temperature=0)  # replace with HuggingFacePipeline if offline
# memory = ConversationBufferMemory(memory_key="chat_history")

# import asyncio
# from langchain_community.llms import HuggingFacePipeline

# # Ensure event loop exists
# try:
#     asyncio.get_running_loop()
# except RuntimeError:
#     asyncio.set_event_loop(asyncio.new_event_loop())

# # Now create your agent
# agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description", handle_parsing_errors=True)

# # agent = initialize_agent(
# #     tools,
# #     llm,
# #     agent_type="zero-shot-react-description",
# #     verbose=True,
# #     memory=memory,
# #     handle_parsing_errors=True   # 👈 this lets LangChain retry or skip bad parses
# # )

# #agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description", memory=memory, verbose=False)










# import asyncio
# asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# from transformers import pipeline
# from langchain_community.llms import HuggingFacePipeline
# from langgraph.prebuilt import create_react_agent, create_agent_executor
# from langchain_core.tools import Tool
# from langchain_core.memory import ConversationBufferMemory

# # Create LLM pipeline (Hugging Face)
# hf_pipeline = pipeline(
#     "text2text-generation",
#     model="google/flan-t5-base",
#     max_length=512,
#     temperature=0.2
# )
# llm = HuggingFacePipeline(pipeline=hf_pipeline)

# # Example tools
# def search_doctor(q: str):
#     return f"Found doctor for query: {q}"

# def book_doctor(q: str):
#     return f"Doctor booked for query: {q}"

# tools = [
#     Tool(name="search_doctor", func=search_doctor, description="Find doctors by specialization"),
#     Tool(name="book_doctor", func=book_doctor, description="Book a doctor appointment")
# ]

# memory = ConversationBufferMemory(memory_key="chat_history")

# # Create the REACT-style agent (replaces initialize_agent)
# agent = create_react_agent(llm, tools)
# executor = create_agent_executor(agent, tools)

# # agent = create_react_agent(llm, tools)
# # executor = AgentExecutor.from_agent_and_tools(
# #     agent=agent,
# #     tools=tools,
# #     memory=memory,
# #     handle_parsing_errors=True,
# #     verbose=True
# # )

# # Test run
# query = "Book me a cardiologist tomorrow morning"
# response = executor.invoke({"input": query})
# print(response["output"])











# import asyncio
# asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# from transformers import pipeline
# from langchain_community.llms import HuggingFacePipeline
# from langchain_core.tools import Tool
# # from langchain_core.memory import ConversationBufferMemory
# # from langchain_community.memory import ConversationBufferMemory

# from langgraph.prebuilt import create_react_agent
# from langgraph.graph import StateGraph

# # ---- Step 1: LLM ----
# hf_pipeline = pipeline(
#     "text2text-generation",
#     model="google/flan-t5-base",
#     max_length=512,
#     temperature=0.2,
# )
# llm = HuggingFacePipeline(pipeline=hf_pipeline)

# # ---- Step 2: Tools ----
# def search_doctor(q: str):
#     return f"Found doctor for query: {q}"

# def book_doctor(q: str):
#     return f"Doctor booked successfully for {q}"

# tools = [
#     Tool(name="search_doctor", func=search_doctor,
#          description="Find doctors by specialization or name"),
#     Tool(name="book_doctor", func=book_doctor,
#          description="Book a doctor appointment"),
# ]

# # ---- Step 3: Memory ----
# memory = ConversationBufferMemory(memory_key="chat_history")

# # ---- Step 4: Build the ReAct agent node ----
# agent_node = create_react_agent(llm, tools)

# # ---- Step 5: Create a graph and add the node ----
# graph = StateGraph(input_schema={"input": str})
# graph.add_node("agent", agent_node)
# graph.set_entry_point("agent")
# graph.set_finish_point("agent")

# # ---- Step 6: Compile the graph into an executable ----
# app = graph.compile()

# # ---- Step 7: Run ----
# query = "Book me a cardiologist tomorrow at 10 AM"
# result = app.invoke({"input": query})
# print(result)






# import asyncio
# asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# from transformers import pipeline
# from langchain_community.llms import HuggingFacePipeline
# from langchain_core.tools import Tool
# # from langchain_community.memory import ConversationBufferMemory
# from langchain.memory import ConversationBufferMemory

# # from langchain_experimental.agents import create_react_agent
# from langgraph.prebuilt import create_react_agent_executor

# from langgraph.graph import StateGraph

# # === Step 1: LLM ===
# hf_pipeline = pipeline(
#     "text2text-generation",
#     model="google/flan-t5-base",
#     max_length=512,
#     temperature=0.2
# )
# llm = HuggingFacePipeline(pipeline=hf_pipeline)

# # === Step 2: Tools ===
# def search_doctor(q: str):
#     return f"Found doctor for query: {q}"

# def book_doctor(q: str):
#     return f"Doctor booked successfully for {q}"

# tools = [
#     Tool(name="search_doctor", func=search_doctor, description="Find doctors by specialization"),
#     Tool(name="book_doctor", func=book_doctor, description="Book a doctor appointment")
# ]

# # === Step 3: Memory ===
# memory = ConversationBufferMemory(memory_key="chat_history")

# # === Step 4: Build Agent & Graph ===
# agent_node = create_react_agent(llm, tools)
# graph = StateGraph(input_schema={"input": str})
# graph.add_node("agent", agent_node)
# graph.set_entry_point("agent")
# graph.set_finish_point("agent")

# # === Step 5: Compile and Run ===
# app = graph.compile()
# response = app.invoke({"input": "Book me a cardiologist tomorrow at 10 AM"})
# print(response)



















