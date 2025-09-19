# yt_health_bootstrap.py
# Requisiti: pip install requests
import requests, sys

BASE_URL = ""
TOKEN = ""

HDRS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/json",
    "Content-Type": "application/json"
}

def req(method, path, **kwargs):
    r = requests.request(method, f"{BASE_URL}{path}", headers=HDRS, **kwargs)
    if r.status_code >= 300:
        print("ERRORE", r.status_code, r.text)
        sys.exit(1)
    return r.json() if r.text else {}

def get_me():
    return req("GET", "/users/me?fields=id,login")

def create_project(name, short_name, leader_id):
    # template=kanban per avere board e flusso base
    payload = {
        "name": name,
        "shortName": short_name,
        "leader": {"id": leader_id}
    }
    return req("POST", "/admin/projects?fields=id,shortName,name&template=kanban", json=payload)

def create_issue(project_id, summary, description=""):
    payload = {
        "project": {"id": project_id},
        "summary": summary,
        "description": description
    }
    return req("POST", "/issues?fields=id,idReadable,summary", json=payload)

if __name__ == "__main__":
    me = get_me()  # id utente corrente
    proj = create_project("Salute - Diario e Segnalazioni", "HLTH", me["id"])
    print("Creato progetto:", proj)

    # Ticket d’esempio (solo campi base; i custom arriveranno dopo)
    create_issue(proj["id"], "Cefalea improvvisa con nausea",
                 "Tipo: Episodio/Sintomo\nScala Dolore: 7\nData/ora esordio: 2025-09-17 22:10\nSoggetto: PZ-001\nNote: migliorata con riposo")
    create_issue(proj["id"], "Visita neurologica programmata",
                 "Tipo: Visita/Esame\nProssima azione al: 2025-09-25\nSoggetto: PZ-001")
    create_issue(proj["id"], "Terapia sintomatica",
                 "Tipo: Terapia\nFarmaco: paracetamolo 1g PRN\nSoggetto: PZ-001")

    print("Ticket iniziali creati.")
