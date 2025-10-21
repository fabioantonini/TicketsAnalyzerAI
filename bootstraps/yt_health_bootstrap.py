# yt_health_bootstrap.py
# Requisiti: pip install requests
import requests, sys, argparse, time

BASE_URL = "https://tuaistanza.youtrack.cloud/api"
TOKEN = "ybpt_xxx_copia_il_tuo_token_qui"

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

def find_project_by_short_name(short_name):
    # recupera tutti i progetti e filtra (poche decine: ok)
    projs = req("GET", "/admin/projects?fields=id,shortName,name")
    for p in projs:
        if p.get("shortName") == short_name:
            return p
    return None

def create_project(name, short_name, leader_id, dry_run=False):
    if dry_run:
        print(f"[DRY] Creerei progetto {name} ({short_name})")
        return {"id": "DRY-PROJ-ID", "name": name, "shortName": short_name}
    payload = {
        "name": name,
        "shortName": short_name,
        "leader": {"id": leader_id}
    }
    # template=kanban per board/flusso base
    return req("POST", "/admin/projects?fields=id,shortName,name&template=kanban", json=payload)

def create_issue(project_id, summary, description="", dry_run=False):
    if dry_run:
        print(f"[DRY] Creerei issue: {summary}")
        return {"id": "DRY-ISSUE-ID", "idReadable": "HLTH-XXX", "summary": summary}
    payload = {
        "project": {"id": project_id},
        "summary": summary,
        "description": description
    }
    return req("POST", "/issues?fields=id,idReadable,summary", json=payload)

def fetch_existing_summaries(project_short_name):
    # legge tutti i summary del progetto con paginazione ($skip/$top)
    summaries = set()
    skip, top = 0, 100
    while True:
        path = f"/issues?fields=summary&query=project:%20{project_short_name}&$skip={skip}&$top={top}"
        batch = req("GET", path)
        if not isinstance(batch, list) or not batch:
            break
        for it in batch:
            s = it.get("summary")
            if s:
                summaries.add(s.strip())
        if len(batch) < top:
            break
        skip += top
        # piccola pausa di cortesia
        time.sleep(0.05)
    return summaries

def seed_issues(project_id, project_short_name, idempotent=False, dry_run=False):
    issues_data = [
        # Episodi/Sintomi
        {"summary": "Cefalea improvvisa con nausea",
         "desc": "Tipo: Episodio/Sintomo\nScala Dolore: 7\nData/ora esordio: 2025-09-17 22:10\nSoggetto: PZ-001\nContesto: Riposo\nNote: migliorata con riposo"},
        {"summary": "Dolore toracico lieve sotto sforzo",
         "desc": "Tipo: Episodio/Sintomo\nScala Dolore: 4\nData/ora esordio: 2025-09-18 07:30\nSoggetto: PZ-002\nContesto: Sforzo\nNote: dura 5 minuti, si risolve con riposo"},
        {"summary": "Dispnea notturna ricorrente",
         "desc": "Tipo: Episodio/Sintomo\nScala Dolore: 0\nData/ora esordio: 2025-09-10 03:00\nSoggetto: PZ-003\nContesto: Sonno\nNote: 3 episodi settimanali"},
        {"summary": "Episodio di vertigini in posizione eretta",
         "desc": "Tipo: Episodio/Sintomo\nScala Dolore: 2\nData/ora esordio: 2025-09-14 11:20\nSoggetto: PZ-001\nContesto: Lavoro\nNote: risolto spontaneamente"},
        {"summary": "Rash cutaneo pruriginoso su avambracci",
         "desc": "Tipo: Episodio/Sintomo\nScala Dolore: 1\nData/ora esordio: 2025-09-12 09:00\nSoggetto: PZ-004\nContesto: Altro\nNote: peggiora al caldo"},
        {"summary": "Calo improvviso di pressione percepito",
         "desc": "Tipo: Episodio/Sintomo\nScala Dolore: 0\nData/ora esordio: 2025-09-16 16:40\nSoggetto: PZ-002\nContesto: Dopo pasto\nNote: lieve sudorazione"},
        {"summary": "Fitte addominali post-prandiali",
         "desc": "Tipo: Episodio/Sintomo\nScala Dolore: 5\nData/ora esordio: 2025-09-13 13:15\nSoggetto: PZ-003\nContesto: Alimentazione\nNote: sospetto intolleranza"},
        {"summary": "Formicolio mano destra transitorio",
         "desc": "Tipo: Episodio/Sintomo\nScala Dolore: 0\nData/ora esordio: 2025-09-11 18:50\nSoggetto: PZ-001\nContesto: Lavoro\nNote: durata 10 minuti"},
        {"summary": "Palpitazioni a riposo serali",
         "desc": "Tipo: Episodio/Sintomo\nScala Dolore: 0\nData/ora esordio: 2025-09-15 22:00\nSoggetto: PZ-004\nContesto: Riposo\nNote: frequenza percepita elevata"},
        {"summary": "Lombalgia dopo sollevamento pesi",
         "desc": "Tipo: Episodio/Sintomo\nScala Dolore: 6\nData/ora esordio: 2025-09-09 17:25\nSoggetto: PZ-002\nContesto: Sforzo\nNote: dolore irradiato gluteo sx"},
        # Visite/Esami
        {"summary": "Visita neurologica programmata",
         "desc": "Tipo: Visita/Esame\nProssima azione al: 2025-09-25\nSoggetto: PZ-001\nNote: portare diario cefalea"},
        {"summary": "ECG basale da eseguire",
         "desc": "Tipo: Visita/Esame\nProssima azione al: 2025-09-23\nSoggetto: PZ-004\nNote: valutare palpitazioni serali"},
        {"summary": "Holter pressorio 24h",
         "desc": "Tipo: Visita/Esame\nProssima azione al: 2025-10-02\nSoggetto: PZ-002\nNote: episodi ipotensione post-prandiale"},
        {"summary": "Esami ematici completi",
         "desc": "Tipo: Visita/Esame\nProssima azione al: 2025-09-26\nSoggetto: PZ-003\nNote: includere profilo metabolico"},
        {"summary": "RX rachide lombare",
         "desc": "Tipo: Visita/Esame\nProssima azione al: 2025-09-24\nSoggetto: PZ-002\nNote: valutazione lombalgia"},
        # Terapie
        {"summary": "Terapia sintomatica per cefalea",
         "desc": "Tipo: Terapia\nFarmaco: paracetamolo 1g PRN\nSoggetto: PZ-001\nNote: non più di 2/die"},
        {"summary": "Integrazione magnesio serale",
         "desc": "Tipo: Terapia\nFarmaco: magnesio 375 mg\nSoggetto: PZ-001\nNote: 20 giorni, poi rivalutare"},
        {"summary": "Applicazione crema cortisonica topica",
         "desc": "Tipo: Terapia\nFarmaco: crema 1 appl/sera\nSoggetto: PZ-004\nNote: rash avambracci; max 7 giorni"},
        {"summary": "Fisioterapia lombare 2x settimana",
         "desc": "Tipo: Terapia\nIntervento: 8 sedute programmate\nSoggetto: PZ-002\nNote: evitare sollevamenti"},
        # Follow-up
        {"summary": "Follow-up diario cefalea e trigger",
         "desc": "Tipo: Follow-up\nProssima azione al: 2025-10-05\nSoggetto: PZ-001\nNote: valutare frequenza e intensità"},
        {"summary": "Follow-up palpitazioni post-ECG",
         "desc": "Tipo: Follow-up\nProssima azione al: 2025-10-01\nSoggetto: PZ-004\nNote: verificare esito ECG"},
        {"summary": "Revisione dieta per dolori addominali",
         "desc": "Tipo: Follow-up\nProssima azione al: 2025-09-30\nSoggetto: PZ-003\nNote: diario alimentare"},
        {"summary": "Controllo dolore lombare dopo fisioterapia",
         "desc": "Tipo: Follow-up\nProssima azione al: 2025-10-12\nSoggetto: PZ-002\nNote: scala funzionale"},
        {"summary": "Valutazione rash dopo terapia topica",
         "desc": "Tipo: Follow-up\nProssima azione al: 2025-09-28\nSoggetto: PZ-004\nNote: foto prima/dopo"}
    ]

    existing = set()
    if idempotent:
        existing = fetch_existing_summaries(project_short_name)
        if existing:
            print(f"[IDEMPOTENT] Summary già presenti: {len(existing)}")

    created, skipped = 0, 0
    for item in issues_data:
        s = item["summary"].strip()
        if idempotent and s in existing:
            skipped += 1
            print(f"[SKIP] '{s}' esiste già")
            continue
        create_issue(project_id, s, item["desc"], dry_run=dry_run)
        created += 1

    print(f"Completato. Create: {created}, Skippate: {skipped}")

def main():
    ap = argparse.ArgumentParser(description="Bootstrap progetto YouTrack Salute")
    ap.add_argument("--short", default="HLTH", help="Short name progetto (default HLTH)")
    ap.add_argument("--name", default="Salute - Diario e Segnalazioni", help="Nome progetto")
    ap.add_argument("--idempotent", action="store_true", help="Evita duplicati se rilanci lo script")
    ap.add_argument("--dry-run", action="store_true", help="Non apporta modifiche, stampa cosa farebbe")
    args = ap.parse_args()

    me = get_me()
    proj = find_project_by_short_name(args.short)

    if proj:
        print(f"[INFO] Progetto esistente rilevato: {proj['name']} ({proj['shortName']}) -> id {proj['id']}")
    else:
        proj = create_project(args.name, args.short, me["id"], dry_run=args.dry_run)
        print(f"[INFO] Creato progetto: {proj['name']} ({proj['shortName']}) -> id {proj['id']}")

    seed_issues(proj["id"], args.short, idempotent=args.idempotent, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
