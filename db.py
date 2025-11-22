import sqlite3
from datetime import datetime

DB = "concierge.db"

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    # providers table: domain (doctor/plumber/flight/househelp), name, specialization, slots (comma-separated)
    c.execute('''
    CREATE TABLE IF NOT EXISTS providers (
        id INTEGER PRIMARY KEY,
        domain TEXT,
        name TEXT,
        specialization TEXT,
        slots TEXT
    )
    ''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS bookings (
        id INTEGER PRIMARY KEY,
        domain TEXT,
        provider_id INTEGER,
        user_name TEXT,
        time TEXT,
        created_at TEXT
    )
    ''')
    conn.commit()
    conn.close()

def seed_data():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    # clear
    c.execute("DELETE FROM providers")
    # sample doctors
    c.executemany("INSERT INTO providers(domain,name,specialization,slots) VALUES (?,?,?,?)", [
        ("doctor","Dr. Priya Sharma","Cardiologist","2025-11-10T10:00,2025-11-10T11:00,2025-11-11T15:00"),
        ("doctor","Dr. Raj Mehta","Dermatologist","2025-11-10T09:00,2025-11-10T13:00"),
        ("plumber","Plumber Rahul","Plumber","2025-11-10T10:00,2025-11-10T12:00,2025-11-11T09:00"),
        ("househelp","Sita","HouseHelp","2025-11-10T08:00,2025-11-10T14:00")
    ])
    conn.commit()
    conn.close()

def get_providers(domain, specialization=None):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    if specialization:
        c.execute("SELECT id,name,specialization,slots FROM providers WHERE domain=? AND specialization LIKE ?", (domain, f"%{specialization}%"))
    else:
        c.execute("SELECT id,name,specialization,slots FROM providers WHERE domain=?", (domain,))
    res = c.fetchall()
    conn.close()
    return res

def book_provider(domain, provider_id, user_name, time_iso):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    # check slot availability
    c.execute("SELECT slots FROM providers WHERE id=? AND domain=?", (provider_id, domain))
    row = c.fetchone()
    if not row:
        conn.close()
        return False, "Provider not found"
    slots = row[0].split(",") if row[0] else []
    if time_iso not in slots:
        conn.close()
        return False, "Slot not available"
    # remove slot
    new_slots = ",".join([s for s in slots if s != time_iso])
    c.execute("UPDATE providers SET slots=? WHERE id=?", (new_slots, provider_id))
    c.execute("INSERT INTO bookings(domain,provider_id,user_name,time,created_at) VALUES (?,?,?,?,?)",
              (domain, provider_id, user_name, time_iso, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()
    return True, "Booked successfully"
