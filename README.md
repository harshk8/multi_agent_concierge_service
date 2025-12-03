Architecture of the Application:

                User (chat UI)
                      ↓
                Router Agent
         (interprets intent + routes)
            ↙       ↓        ↘
Doctor Agent  Plumber Agent  Flight Agent  HouseHelp Agent
   (tools)        (tools)       (tools)        (tools)
     ↓               ↓             ↓              ↓
Local DB (SQLite)  Local DB     Flight API (mock)  Local DB
 Vector DB / RAG for docs, bios, policies (optional)


Router Agent: decides which domain agent(s) to call (or calls multiple).

Domain Agents: expose tools for searching providers, checking availability, and booking.

Tools: small functions that interact with DB or external API.

Memory: conversation-level memory (user preferences).

Persistence: SQLite for bookings and provider schedules.

Front-end: Streamlit chat UI that calls the Router agent.


Folder Structure:

multi_agent_concierge/
├─ app.py                  # Streamlit UI
├─ agents.py               # LangChain router + agent setup + tools
├─ db.py                   # SQLite persistence helpers + seed data
├─ requirements.txt
└─ README.md

