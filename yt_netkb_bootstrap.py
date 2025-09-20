# yt_netkb_bootstrap.py
# Requisiti: pip install requests
import requests, sys, argparse, time, re, json
from typing import Dict, List

BASE_URL = "https://fabioantonini.youtrack.cloud/api"
TOKEN = "perm-YWRtaW4=.NDQtMA==.QJNcVBhDks9awoACXR2127Up24kEem"

HDRS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/json",
    "Content-Type": "application/json"
}

# -------------------- HTTP helper --------------------
def req(method: str, path: str, **kwargs):
    r = requests.request(method, f"{BASE_URL}{path}", headers=HDRS, **kwargs)
    if r.status_code >= 300:
        print("ERRORE", r.status_code, r.text)
        sys.exit(1)
    return r.json() if r.text else {}

# -------------------- Project helpers --------------------
def get_me():
    return req("GET", "/users/me?fields=id,login")

def find_project_by_short_name(short_name: str):
    projs = req("GET", "/admin/projects?fields=id,shortName,name")
    for p in projs:
        if p.get("shortName") == short_name:
            return p
    return None

def create_project(name: str, short_name: str, leader_id: str, dry: bool):
    if dry:
        print(f"[DRY] Creerei progetto {name} ({short_name})")
        return {"id": "DRY-PROJ-ID", "name": name, "shortName": short_name}
    payload = {"name": name, "shortName": short_name, "leader": {"id": leader_id}}
    return req("POST", "/admin/projects?fields=id,shortName,name&template=kanban", json=payload)

# -------------------- Custom fields & bundles --------------------
def find_or_create_enum_bundle(name: str, values: List[str], dry: bool) -> Dict:
    # cerca bundle enum per nome; se non c'è, crea
    bundles = req("GET", "/admin/customFieldSettings/bundles/enum?fields=id,name,values(name)")
    for b in bundles:
        if b["name"] == name:
            # idempotente: assicura tutti i valori
            current = {v["name"] for v in b.get("values", [])}
            missing = [v for v in values if v not in current]
            if missing and not dry:
                body = {"values": [{"name": v} for v in values]}
                req("POST", f"/admin/customFieldSettings/bundles/enum/{b['id']}/values?fields=id,name", json={"name": None})  # NO-OP per creare endpoint
                # aggiunte una ad una per sicurezza
                for v in missing:
                    req("POST",
                        f"/admin/customFieldSettings/bundles/enum/{b['id']}/values?fields=id,name",
                        json={"name": v})
            return b
    if dry:
        print(f"[DRY] Creerei EnumBundle {name} con valori {values}")
        return {"id": f"DRY-BNDL-{name}", "name": name}
    b = req("POST", "/admin/customFieldSettings/bundles/enum?fields=id,name",
            json={"name": name, "values": [{"name": v} for v in values]})
    return b

def find_or_create_cf_enum(name: str, bundle_id: str, dry: bool) -> Dict:
    # cerca CF schema
    cfs = req("GET", "/admin/customFieldSettings/customFields?fields=id,name,fieldType(id)")
    for cf in cfs:
        if cf["name"] == name:
            return cf
    if dry:
        print(f"[DRY] Creerei CustomField enum {name} (bundle {bundle_id})")
        return {"id": f"DRY-CF-{name}", "name": name}
    body = {
        "name": name,
        "fieldType": {"id": "enum"},
        "bundle": {"id": bundle_id}
    }
    return req("POST", "/admin/customFieldSettings/customFields?fields=id,name", json=body)

def find_or_create_cf_text(name: str, cf_type: str, dry: bool) -> Dict:
    # cf_type: "string" o "text"
    cfs = req("GET", "/admin/customFieldSettings/customFields?fields=id,name,fieldType(id)")
    for cf in cfs:
        if cf["name"] == name:
            return cf
    if dry:
        print(f"[DRY] Creerei CustomField {cf_type} {name}")
        return {"id": f"DRY-CF-{name}", "name": name}
    body = {"name": name, "fieldType": {"id": cf_type}}
    return req("POST", "/admin/customFieldSettings/customFields?fields=id,name", json=body)

def ensure_project_cf(project_id: str, cf_id: str, required: bool, dry: bool) -> None:
    """
    Ensure that a custom field (cf_id) is associated with the given project (project_id).
    In dry-run mode (or when using a DRY-* fake project id), no API calls are performed.
    """
    # Skip real API calls in dry-run or with fake project id
    if dry or str(project_id).startswith("DRY-"):
        print(f"[DRY] Assocerei CF {cf_id} al progetto {project_id}")
        return
    # aggiunge il CF al progetto se assente
    # Check if the custom field is already linked to the project
    fields = req(
        "GET",
        f"/admin/projects/{project_id}/customFields?fields=id,field(id,name),canBeEmpty,emptyFieldText"
    )
    for f in fields:
        field = f.get("field", {})
        if field.get("id") == cf_id:
            return  # already present

    # Link the custom field to the project
    body = {
        "field": {"id": cf_id},
        "canBeEmpty": not required
    }
    req(
        "POST",
        f"/admin/projects/{project_id}/customFields?fields=id,field(name)",
        json=body
    )


# -------------------- Issues --------------------
def fetch_existing_summaries(project_short_name: str) -> set:
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
        time.sleep(0.05)
    return summaries

JSON_REQUIRED_KEYS = {"protocol", "category", "root_cause"}

def has_valid_json_block(description: str) -> bool:
    # blocco JSON è l'ultima riga (o in coda). Prende l'ultimo { ... } nel testo
    matches = list(re.finditer(r"\{[\s\S]*\}$", description.strip()))
    if not matches:
        # fallback: trova l'ultimo blocco { .. } nel testo
        matches = list(re.finditer(r"\{[\s\S]*\}", description))
        if not matches:
            return False
    block = matches[-1].group(0)
    try:
        data = json.loads(block)
    except Exception:
        return False
    return JSON_REQUIRED_KEYS.issubset(set(map(str, data.keys())))

def create_issue(project_id: str, summary: str, description: str, custom: Dict, dry: bool, prefix: str = ""):
    if prefix:
        summary = f"{prefix} {summary}".strip()
    if not has_valid_json_block(description):
        print(f"[VALIDATION] '{summary}' senza JSON valido (chiavi richieste: {JSON_REQUIRED_KEYS}). Skippo.")
        return None
    if dry:
        print(f"[DRY] Creerei issue: {summary}")
        return {"id": "DRY-ISSUE-ID", "idReadable": "NETKB-XXX", "summary": summary}

    # Mappa i CF (per semplicità setto per nome)
    cf_payload = []
    for fname, fval in custom.items():
        if fval is None:
            continue
        if isinstance(fval, dict) and fval.get("_type") == "enum":
            cf_payload.append({"name": fname, "value": {"name": fval["name"]}})
        else:
            # string/text
            cf_payload.append({"name": fname, "value": fval})

    payload = {
        "project": {"id": project_id},
        "summary": summary,
        "description": description,
        "customFields": cf_payload
    }
    return req("POST", "/issues?fields=id,idReadable,summary", json=payload)

# -------------------- Seed data --------------------
def case_block(
    titolo: str,
    problema: str,
    contesto: str,
    sintomi: List[str],
    evidenze: List[str],
    analisi: str,
    root: str,
    soluzione: List[str],
    comandi: List[str],
    verifica: List[str],
    lezioni: List[str],
    keywords: List[str],
    json_tail: Dict
) -> (str, str):
    desc = []
    desc.append("Problema\n" + problema.strip())
    desc.append("Contesto\n" + contesto.strip())
    if sintomi:
        desc.append("Sintomi\n" + "\n".join(f"- {s}" for s in sintomi))
    if evidenze:
        desc.append("Evidenze/Log\n" + "\n".join(f"- {e}" for e in evidenze))
    desc.append("Analisi\n" + analisi.strip())
    desc.append("Causa radice\n" + root.strip())
    if soluzione:
        desc.append("Soluzione adottata\n" + "\n".join(f"- {x}" for x in soluzione))
    if comandi:
        desc.append("Comandi/Config\n" + "\n".join(comandi))
    if verifica:
        desc.append("Verifica/Esito\n" + "\n".join(f"- {v}" for v in verifica))
    if lezioni:
        desc.append("Lezioni apprese\n" + "\n".join(f"- {l}" for l in lezioni))
    if keywords:
        desc.append("Parole chiave\n" + ", ".join(keywords))
    desc.append(json.dumps(json_tail, ensure_ascii=False))
    return titolo, "\n\n".join(desc)

def seed_cases() -> List[Dict]:
    cases = []

    # 1) OSPF MTU mismatch
    title, desc = case_block(
        "OSPF non va in FULL su link p2p: MTU mismatch (1500↔1400), workaround mtu-ignore",
        problema="Adiacenza OSPF bloccata in EXSTART/EXCHANGE dopo cambio su link p2p.",
        contesto="Area 0, A=Linux/FRR (eth1), B=Cisco IOS (Gi0/0); overlay GRE.",
        sintomi=["Neighbor oscillante EXSTART/EXCHANGE", "LSDB non sincronizzata"],
        evidenze=["FRR: 'MTU mismatch' su DBD", "Cisco: show ip ospf interface -> MTU 1400", "A ha MTU 1500", "Ping DF 1472 fallisce"],
        analisi="DBD rifiutati per MTU diversa; overlay riduce MTU effettiva.",
        root="MTU mismatch sul link p2p.",
        soluzione=["Allineato MTU a 1400 su entrambe", "Workaround temporaneo: ip ospf mtu-ignore su entrambe"],
        comandi=[
            "Linux/FRR: ip link set dev eth1 mtu 1400",
            "Cisco IOS: interface Gi0/0 ; mtu 1400 ; ip ospf mtu-ignore"
        ],
        verifica=["Neighbor FULL; DBD completi", "Ping DF size 1372 ok"],
        lezioni=["Standard MTU per link con overlay", "Evitare mtu-ignore in esercizio"],
        keywords=["ospf","exstart","exchange","mtu mismatch","mtu-ignore","dbd","gre","p2p"],
        json_tail={
            "protocol": "OSPF",
            "category": "Routing",
            "root_cause": "MTU_mismatch",
            "fix_type": "MTU_align",
            "checks": ["show ip ospf interface", "ping DF MTU-28"],
            "commands": ["ip link set dev eth1 mtu 1400", "ip ospf mtu-ignore"],
            "env": {"vendorA": "Linux/FRR", "vendorB": "Cisco IOS", "overlay": "GRE"},
            "keywords": ["ospf","exstart","exchange","mtu mismatch","mtu-ignore","dbd","gre","p2p"]
        }
    )
    cases.append({
        "summary": title,
        "description": desc,
        "custom": {
            "Categoria": {"_type":"enum","name":"Routing"},
            "Protocollo": {"_type":"enum","name":"OSPF"},
            "Severità": {"_type":"enum","name":"Media"},
            "Vendor": {"_type":"enum","name":"Cisco"},
            "Dispositivo": "Router A/B",
            "Versione/OS": "IOS 15.x / FRR 9.x"
        }
    })

    # 2) PPPoE MTU/MSS/PMTUD
    title, desc = case_block(
        "WAN PPPoE lenta e loss: MTU 1500 su PPPoE, PMTUD bloccato, MSS non clampato",
        "Packet loss 10–25% e stall TCP su PPPoE dopo migrazione.",
        "Edge Linux PPPoE VLAN 835; servizi IPsec/HTTPS/VoIP impattati.",
        ["Download bloccati >1MB", "VoIP jitter alto", "Ping DF 1472 fallisce"],
        ["MTU ppp0=1500", "Firewall blocca ICMP 3:4", "MSS eccessivo in SYN"],
        "PPPoe richiede MTU/MRU 1492; ICMP fragmentation-needed va permesso; MSS clamp.",
        "MTU errata + ICMP 3:4 bloccato + MSS non clampato.",
        ["Set MTU ppp0 1492", "Permesso ICMP 3:4", "TCP MSS clamp 1452"],
        [
            "ip link set dev ppp0 mtu 1492",
            "iptables -t mangle -A FORWARD -o ppp0 -p tcp --tcp-flags SYN,RST SYN -j TCPMSS --clamp-mss-to-pmtu"
        ],
        ["Ping DF 1472 ok", "Throughput ripristinato", "MOS>4"],
        ["Standard PPPoE: MTU 1492, MSS clamp default", "Non bloccare ICMP 3:4"],
        ["pppoe","mtu 1492","mss 1452","pmtud","icmp 3:4","vlan 835"],
        {
            "protocol":"IP",
            "category":"WAN",
            "root_cause":"MTU_MSS_PMUTD_combo",
            "fix_type":"MTU_1492_MSS_clamp",
            "checks":["ping DF 1472","show interface ppp0 mtu","firewall logs ICMP 3:4"],
            "commands":["ip link set ppp0 mtu 1492","iptables TCPMSS clamp-to-pmtu"],
            "env":{"access":"FTTC PPPoE","vlan":"835"},
            "keywords":["pppoe","mtu","mss","pmtud","icmp fragmentation-needed"]
        }
    )
    cases.append({
        "summary": title,
        "description": desc,
        "custom": {
            "Categoria": {"_type":"enum","name":"WAN"},
            "Protocollo": {"_type":"enum","name":"Altro"},
            "Severità": {"_type":"enum","name":"Alta"},
            "Vendor": {"_type":"enum","name":"Linux/FRR"},
            "Dispositivo": "edge-router",
            "Versione/OS": "Linux 5.x"
        }
    })

    # 3) SIP ALG one-way audio
    title, desc = case_block(
        "VoIP one-way audio con SBC esterno: SIP ALG riscrive SDP/RTP",
        "Audio monodirezionale su chiamate SIP dopo nuovo firewall.",
        "PBX on-prem + SBC cloud; firewall con SIP ALG attivo.",
        ["Chiamate con audio in una sola direzione", "Porte RTP errate nei trace"],
        ["SDP riscritti dall'ALG", "NAT traversal duplicato"],
        "ALG interferisce con SBC che già gestisce NAT traversal.",
        "SIP ALG attivo causa porta/indirizzo RTP errati.",
        ["Disabilitato SIP ALG", "Aperto range RTP necessario", "Pin-hole gestiti dallo SBC"],
        [
            "set firewall sip alg disable",
            "permit udp 10000-20000"
        ],
        ["Audio bidirezionale ripristinato", "Nessun rewrite SDP"],
        ["Evitare ALG quando esiste SBC", "Documentare range RTP"],
        ["sip alg","one-way audio","rtp","sdp","sbc","nat traversal"],
        {
            "protocol":"SIP",
            "category":"VoIP",
            "root_cause":"ALG_interference",
            "fix_type":"Disable_ALG_open_RTP",
            "checks":["pcap sdp","inspect firewall ALG"],
            "commands":["no sip alg","permit udp rtp-range"],
            "env":{"pbx":"on-prem","sbc":"cloud"},
            "keywords":["sip","alg","rtp","sdp","one-way"]
        }
    )
    cases.append({
        "summary": title,
        "description": desc,
        "custom": {
            "Categoria": {"_type":"enum","name":"VoIP"},
            "Protocollo": {"_type":"enum","name":"SIP"},
            "Severità": {"_type":"enum","name":"Alta"},
            "Vendor": {"_type":"enum","name":"Altro"},
            "Dispositivo": "firewall",
            "Versione/OS": "n/a"
        }
    })

    # (Aggiungo altri 21 casi sintetici ma completi)
    def add_case(
        summary: str,
        cat: str, proto: str, sev: str, vendor: str, dev: str, osver: str,
        problema: str, contesto: str,
        sintomi: list, evid: list, analisi: str, root: str,
        fix: list, cmds: list, verify: list, lessons: list, kws: list,
        js: dict = None
    ):
        # Se il blocco JSON non è passato, costruiamone uno minimale ma coerente
        def _slug(s: str) -> str:
            import re as _re
            s = s.strip() or "Unknown"
            s = s.replace(" ", "_")
            s = _re.sub(r"[^A-Za-z0-9_]+", "", s)
            return (s or "Unknown")[:48]

        if js is None:
            js = {
                "protocol": proto if proto and proto != "Altro" else "IP",
                "category": cat or "Routing",
                "root_cause": _slug(root),
                "fix_type": _slug(fix[0]) if fix else "Manual_Fix",
                "checks": sintomi if sintomi else (evid if evid else []),
                "commands": cmds or [],
                "env": {"vendor": vendor or "Altro", "device": dev or "n/a", "os": osver or "n/a"},
                "keywords": kws or []
            }

        t, d = case_block(
            summary, problema, contesto, sintomi, evid, analisi, root, fix, cmds, verify, lessons, kws, js
        )
        cases.append({
            "summary": t,
            "description": d,
            "custom": {
                "Categoria": {"_type": "enum", "name": cat},
                "Protocollo": {"_type": "enum", "name": proto},
                "Severità": {"_type": "enum", "name": sev},
                "Vendor": {"_type": "enum", "name": vendor},
                "Dispositivo": dev,
                "Versione/OS": osver
            }
        })


    add_case(
        "DHCP relay ignora richieste: manca Option 82 richiesta dal server",
        "DNS/DHCP","DHCP","Media","Cisco","Switch L3","IOS 15.x",
        "Client non ottengono IP su VLAN access.",
        "Relay centralizzato; server valida Circuit-ID/Remote-ID.",
        ["DISCOVER parte, nessun OFFER","Log server: richiede Option 82"],
        ["Sniffer conferma DISCOVER senza Option 82"],
        "Server scarta senza Option 82.",
        "Abilitata insert-option 82 e pass-through hop-count.",
        ["ip dhcp relay information option","ip dhcp relay information trust-all"],
        ["Lease assegnati, T1/T2 ok"],
        ["Standardizzare Option 82"],
        ["dhcp","option 82","relay"],
        {
            "protocol":"DHCP","category":"DNS/DHCP","root_cause":"Missing_Option82",
            "fix_type":"Enable_Option82","checks":["pcap dhcp","server logs"],
            "commands":["dhcp relay information option"],"env":{"topo":"L3 access"},
            "keywords":["dhcp","relay","option 82"]
        }
    )
    add_case(
        "OSPF su access: mancato passive-interface espone reti di accesso",
        "Routing","OSPF","Bassa","Cisco","Switch L3","IOS 15.x",
        "Adiacenze inattese su porte access.",
        "Area 10 su switch L3.",
        ["State BDR su porta utente"],
        ["show ip ospf interface"],
        "Hardening mancante.",
        "Impostato passive-interface default; no passive su uplink.",
        ["router ospf X","passive-interface default","no passive-interface Gi0/1"],
        ["Nessuna adiacenza su access"],
        ["Usare passive-interface default"],
        ["ospf","passive-interface","hardening"],
        {"protocol":"OSPF","category":"Routing","root_cause":"Hardening_missing","fix_type":"Passive_Interface_Default",
         "checks":["show ip ospf interface"],"commands":["passive-interface default"],"env":{},"keywords":["ospf","passive"]}
    )
    add_case(
        "BGP flapping: hold-time troppo basso e dampening aggressivo",
        "Routing","BGP","Alta","Cisco","Router edge","IOS XE 17.x",
        "Sessione eBGP instabile ogni 30s.",
        "Peer ISP con timers 1/3s.",
        ["ADJCHANGE Idle/Established frequente"],
        ["show ip bgp summary","timers keepalive 10 hold 30"],
        "Keepalive/hold inadeguati; dampening penalizza prefissi.",
        "Timers 10/30; disabilitato dampening su prefissi critici.",
        ["neighbor x timers 10 30","no bgp dampening su prefissi business"],
        ["Sessione stabile, flap <1/h"],
        ["Concordare timers standard con ISP"],
        ["bgp","timers","dampening"],
        {"protocol":"BGP","category":"Routing","root_cause":"Aggressive_Timers","fix_type":"Timers_Normalize",
         "checks":["bgp summary"],"commands":["neighbor timers 10 30"],"env":{},"keywords":["bgp","timers"]}
    )
    add_case(
        "STP loop: PortFast senza BPDU Guard su presa utente con mini-switch",
        "Switching","STP","Alta","Cisco","Access switch","IOS 15.x",
        "Tempeste broadcast e rete lenta.",
        "Mini-switch utente crea loop su due prese.",
        ["CPU alta, TCN incrementano"],
        ["show spanning-tree detail"],
        "PortFast abilitato ma senza protezioni.",
        "Abilitato bpduguard su access; portfast mantenuto.",
        ["spanning-tree portfast","spanning-tree bpduguard enable"],
        ["Nessun loop successivo"],
        ["Attivare bpduguard di default su access"],
        ["stp","portfast","bpduguard","loop"],
        {"protocol":"STP","category":"Switching","root_cause":"No_BPDU_Guard","fix_type":"Enable_BPDU_Guard",
         "checks":["stp detail"],"commands":["bpduguard enable"],"env":{},"keywords":["stp","bpduguard"]}
    )
    add_case(
        "VLAN Native mismatch su trunk 802.1Q causa perdita traffico",
        "Switching","Altro","Media","Cisco","Switch agg","IOS 15.x",
        "Pacchetti persi su VLAN utente.",
        "Trunk tra vendor diversi.",
        ["LLDP/CDP warning native mismatch"],
        ["show interfaces trunk"],
        "Native VLAN diversa per lato.",
        "Allineata native 10 su entrambi; tagging native.",
        ["switchport trunk native vlan 10","vlan dot1q tag native"],
        ["Traffico ristabilito"],
        ["Evitare VLAN1 come native"],
        ["vlan","trunk","native mismatch"],
        {"protocol":"802.1Q","category":"Switching","root_cause":"Native_Mismatch","fix_type":"Align_Native",
         "checks":["trunk status"],"commands":["native vlan 10"],"env":{},"keywords":["vlan","native"]}
    )
    add_case(
        "IPsec site-to-site dietro NAT: NAT-T disabilitato",
        "Security","IPsec","Alta","Altro","Firewall","n/a",
        "Tunnel non sale / instabile.",
        "Firewall NAT intermedio.",
        ["ESP bloccato; solo IKE/500 visibile"],
        ["pcap: no UDP 4500"],
        "Serve NAT-T e UDP 4500.",
        "Abilitato NAT-T e aperte porte 500/4500.",
        ["crypto isakmp nat-traversal 20","permit udp 500 4500"],
        ["SA stabili"],
        ["Verificare sempre NAT-T con NAT in path"],
        ["ipsec","nat-t","udp 4500"],
        {"protocol":"IPsec","category":"VPN","root_cause":"NAT_T_off","fix_type":"Enable_NAT_T",
         "checks":["ike logs"],"commands":["enable nat-t"],"env":{},"keywords":["ipsec","nat-t"]}
    )
    add_case(
        "QoS errata: shaper applicato in ingresso invece che in uscita",
        "QoS","Altro","Media","Altro","Edge router","Linux",
        "Jitter VoIP e upload saturato.",
        "Link 100/20 con code HFSC.",
        ["Coda tx vuota, rx drop alti"],
        ["tc -s qdisc show"],
        "Direzione policy invertita.",
        "Shaping su egress WAN + policing ingress.",
        ["tc qdisc add dev ppp0 root ...","class EF/AF"],
        ["MOS>4 stabile"],
        ["Controllare sempre direction"],
        ["qos","hfsc","egress"],
        {"protocol":"QoS","category":"QoS","root_cause":"Wrong_Direction","fix_type":"Fix_Egress",
         "checks":["qdisc stats"],"commands":["tc qdisc ..."],"env":{},"keywords":["qos","egress"]}
    )
    add_case(
        "NTP offset causa fallimenti Kerberos: UDP/123 bloccato",
        "Security","NTP","Media","Linux/FRR","Server","Linux",
        "Ticket TGT rifiutati, clock skew.",
        "Join AD; firewall restrittivo.",
        ["ntpq -p no-reach","kinit skew"],
        ["fw logs udp/123 drop"],
        "NTP non sincronizzato.",
        "Aperto UDP 123 verso pool; fallback interno.",
        ["ufw allow 123/udp","timesyncd iburst"],
        ["Offset < 50ms; Kerberos ok"],
        ["Monitorare NTP su host critici"],
        ["ntp","kerberos","skew"],
        {"protocol":"NTP","category":"Security","root_cause":"NTP_Blocked","fix_type":"Allow_UDP_123",
         "checks":["ntpq -p"],"commands":["allow 123/udp"],"env":{},"keywords":["ntp","kerberos"]}
    )
    add_case(
        "Port-channel instabile: LACP rate e hash non coerenti",
        "Switching","LACP","Media","Cisco","Core/Agg","IOS XE",
        "Flap e micro-outage.",
        "Bundle 2x10G.",
        ["LACP individual","hash diversi"],
        ["show etherchannel summary"],
        "Parametri non allineati.",
        "lacp rate fast su entrambi; hash src-dst-ip.",
        ["lacp rate fast","port-channel load-balance src-dst-ip"],
        ["Bundle stabile"],
        ["Standardizzare parametri LACP"],
        ["lacp","hashing","port-channel"],
        {"protocol":"LACP","category":"Switching","root_cause":"Mismatch_Params","fix_type":"Align_Params",
         "checks":["etherchannel sum"],"commands":["lacp rate fast"],"env":{},"keywords":["lacp","hash"]}
    )
    add_case(
        "VRRP flap con CPU alta: preempt troppo aggressivo",
        "Routing","Altro","Media","Cisco","Gateway","IOS",
        "Gateway master cambia spesso.",
        "HA VRRP.",
        ["VRRP transitions frequenti","CPU>90%"],
        ["show vrrp","show proc cpu"],
        "Preempt con skew basso.",
        "Delay 60s, priority gap 30, track interface.",
        ["vrrp preempt delay 60","priority 110 vs 80","track Gi0/1"],
        ["Stabilità ripristinata"],
        ["Hardening VRRP"],
        ["vrrp","preempt","priority"],
        {"protocol":"VRRP","category":"Routing","root_cause":"Aggressive_Preempt","fix_type":"Tune_Preempt",
         "checks":["vrrp logs"],"commands":["preempt delay"],"env":{},"keywords":["vrrp","preempt"]}
    )
    add_case(
        "LLDP disabilitato: telefoni non entrano in voice VLAN",
        "Switching","LLDP","Bassa","Cisco","Access switch","IOS",
        "Phone finiscono in VLAN dati.",
        "Auto-VLAN richiede LLDP-MED.",
        ["LLDP neighbors assenti"],
        ["show lldp neighbors"],
        "LLDP off.",
        "lldp run + med-tlv voice-vlan.",
        ["lldp run","lldp med-tlv-select network-policy voice"],
        ["Phone in VLAN voce"],
        ["Abilitare LLDP/LLDP-MED"],
        ["lldp","voice vlan","med"],
        {"protocol":"LLDP","category":"Switching","root_cause":"LLDP_Off","fix_type":"Enable_LLDP",
         "checks":["lldp neighbors"],"commands":["lldp run"],"env":{},"keywords":["lldp","voice"]}
    )
    add_case(
        "DNS ricorsivo lento: forwarder bloccato e cache TTL basso",
        "DNS/DHCP","DNS","Media","Linux/FRR","Resolver","bind9",
        "Lookups lenti / SERVFAIL sporadici.",
        "Firewall outbound restrittivo.",
        ["dig @8.8.8.8 timeout","named log SERVFAIL"],
        ["fw drop udp/tcp 53"],
        "Forwarding filtrato e TTL cache basso.",
        "Permessi 53 verso forwarder + max-cache-ttl 86400.",
        ["ufw allow 53","named.conf: forwarders {...}; max-cache-ttl 86400;"],
        ["Latency <50ms; cache hit >80%"],
        ["Aggiornare forwarder autorizzati"],
        ["dns","forwarder","ttl"],
        {"protocol":"DNS","category":"DNS/DHCP","root_cause":"FW_Block_53","fix_type":"Allow_53_and_Tune_TTL",
         "checks":["dig +trace"],"commands":["allow 53"],"env":{},"keywords":["dns","forwarder"]}
    )
    add_case(
        "RADIUS accounting intermittente: UDP/1813 non permesso",
        "Security","Altro","Media","Altro","WLC/RADIUS","n/a",
        "Sessioni non contabilizzate.",
        "WLAN enterprise con accounting.",
        ["fw drop udp 1813"],
        ["server logs missing acct"],
        "Regola outbound incompleta.",
        "Permesso 1812/1813 bidirezionale; retry 3.",
        ["permit udp 1812 1813","radius-server retransmit 3"],
        ["Accounting affidabile"],
        ["Validare sempre le porte AAA"],
        ["radius","accounting","1813"],
        {"protocol":"RADIUS","category":"Security","root_cause":"Port_1813_Blocked","fix_type":"Allow_1813",
         "checks":["server logs"],"commands":["permit 1813"],"env":{},"keywords":["radius","1813"]}
    )
    add_case(
        "Static route recursive fail: next-hop non risolvibile",
        "Routing","Altro","Media","Linux/FRR","Router","FRR 9.x",
        "Blackhole su rotte statiche.",
        "Next-hop fuori subnet.",
        ["ARP incomplete su next-hop"],
        ["ip route show","arp -an"],
        "Manca route verso subnet del next-hop.",
        "Aggiunta route intermedia o interfaccia d’uscita.",
        ["ip route add ... via ... dev X"],
        ["Connettività ripristinata"],
        ["Verificare sempre reachability next-hop"],
        ["static route","recursive","arp"],
        {"protocol":"IP","category":"Routing","root_cause":"NextHop_Unreachable","fix_type":"Add_Intermediate_Route",
         "checks":["arp","route show"],"commands":["ip route add"],"env":{},"keywords":["static","recursive"]}
    )
    add_case(
        "AnyConnect split-tunnel: MTU eccessiva, SaaS si blocca",
        "VPN","Altro","Media","Cisco","ASA/FTD","ASA 9.x",
        "Applicazioni SaaS lente/irraggiungibili.",
        "SSL VPN split-tunnel, PMTUD bloccato.",
        ["Stalli su payload grandi"],
        ["no ICMP 3:4 in path"],
        "MTU tunnel 1500 e MSS eccessivo.",
        "MTU 1350–1400 e MSS 1350.",
        ["sysopt connection tcpmss 1350","tunnel-group MTU 1400"],
        ["SaaS ok"],
        ["Definire MTU per SSL VPN"],
        ["ssl vpn","mtu","mss","pmtud"],
        {"protocol":"SSL","category":"VPN","root_cause":"Tunnel_MTU_Too_High","fix_type":"Lower_MTU_MSS",
         "checks":["ping df","path mtu"],"commands":["tcpmss 1350"],"env":{},"keywords":["ssl","mtu"]}
    )
    add_case(
        "WireGuard dietro CGNAT: assenza keepalive porta a drop",
        "VPN","WireGuard","Bassa","Linux/FRR","WG peer","Linux",
        "Tunnel si interrompe dopo inattività.",
        "Peer mobile CGNAT.",
        ["NAT entry scade"],
        ["no PersistentKeepalive"],
        "Manca keepalive periodico.",
        "PersistentKeepalive=25 su peer remoti.",
        ["PersistentKeepalive=25"],
        ["Tunnel stabile"],
        ["Impostare keepalive per NAT"],
        ["wireguard","keepalive","cgnat"],
        {"protocol":"WireGuard","category":"VPN","root_cause":"No_Keepalive","fix_type":"Enable_Keepalive",
         "checks":["wg show"],"commands":["PersistentKeepalive=25"],"env":{},"keywords":["wireguard","keepalive"]}
    )
    add_case(
        "GRE over IPsec: payload grandi in blackhole (PMTUD rotto)",
        "Routing","GRE","Media","Cisco","Router","IOS",
        "Ping piccoli ok, grandi no.",
        "GRE sopra IPsec.",
        ["ICMP 3:4 bloccato","overhead riduce MTU"],
        ["no clamp MSS"],
        "Serve clamp MSS e MTU tunnel.",
        "Clamp MSS 1360 e ip mtu 1400 su tunnel; permettere ICMP 3:4.",
        ["interface Tunnel0 ; ip mtu 1400","ip tcp adjust-mss 1360"],
        ["Traffico stabile"],
        ["Test PMTUD dopo change"],
        ["gre","ipsec","mtu","pmtud"],
        {"protocol":"GRE","category":"Routing","root_cause":"PMTUD_Blocked","fix_type":"Clamp_MSS_Set_Tunnel_MTU",
         "checks":["ping df"],"commands":["ip mtu 1400","mss 1360"],"env":{},"keywords":["gre","mtu"]}
    )
    add_case(
        "Asymmetric routing: firewall stateful droppa i ritorni",
        "Security","Altro","Media","Altro","Firewall","n/a",
        "TCP reset/timeout su alcune sessioni.",
        "Due uplink ECMP.",
        ["state miss sul ritorno"],
        ["session table miss"],
        "Ritorno esce su link diverso (no stickiness).",
        "PBR per stickiness o eccezione state per subnet note.",
        ["policy-based routing","disable state per subnet note"],
        ["Connessioni stabili"],
        ["Evitare asimmetria con stateful"],
        ["ecmp","stateful","pbr"],
        {"protocol":"IP","category":"Security","root_cause":"Asymmetric_Path","fix_type":"Sticky_PBR_or_State_Bypass",
         "checks":["conn table"],"commands":["pbr stickiness"],"env":{},"keywords":["ecmp","stateful"]}
    )
    add_case(
        "MPLS LDP non sale: TCP/646 filtrato nel core",
        "Routing","MPLS","Alta","Cisco","Core","IOS XR",
        "Niente label binding (solo IGP up).",
        "Core MPLS v4.",
        ["Discovery ok, sessione TCP fallisce"],
        ["fw interno blocca 646"],
        "Filtro inter-VLAN blocca 646.",
        "Consentito TCP 646; timers default.",
        ["permit tcp 646"],
        ["LDP up, label distribuite"],
        ["Validare porte LDP"],
        ["mpls","ldp","tcp 646"],
        {"protocol":"MPLS","category":"Routing","root_cause":"TCP646_Blocked","fix_type":"Allow_646",
         "checks":["ldp logs"],"commands":["allow 646"],"env":{},"keywords":["mpls","ldp"]}
    )
    add_case(
        "BFD non parte: min-tx/rx troppo bassi per la piattaforma",
        "Routing","Altro","Bassa","Cisco","WAN","IOS XE",
        "Sessione BFD down.",
        "Link WAN ad alta latenza.",
        ["diag: control packet too slow"],
        ["timers 50/50/3"],
        "Timer sotto capacità forwarding.",
        "Impostati 300ms/300ms mult=3.",
        ["bfd interval 300 min_rx 300 multiplier 3"],
        ["BFD up"],
        ["Timer in linea con piattaforma"],
        ["bfd","timers"],
        {"protocol":"BFD","category":"Routing","root_cause":"Timers_Too_Aggressive","fix_type":"Relax_Timers",
         "checks":["bfd logs"],"commands":["interval 300"],"env":{},"keywords":["bfd"]}
    )
    add_case(
        "IPv6 RA-guard blocca SLAAC legittimo dopo upgrade",
        "LAN","Altro","Media","Cisco","Access","IOS",
        "Client senza IPv6.",
        "RA-guard su access.",
        ["Drops su RA guard","MAC gw non whitelisted"],
        ["ra-guard profile obsoleto"],
        "Profile non aggiornato all’upgrade.",
        "Aggiornato bindings e permit su uplink.",
        ["ipv6 nd raguard policy ..."],
        ["SLAAC ok"],
        ["Rivedere RA-guard dopo upgrade"],
        ["ipv6","ra-guard","slaac"],
        {"protocol":"IPv6","category":"LAN","root_cause":"RA_Guard_Profile_Stale","fix_type":"Update_Profile",
         "checks":["ra guard counters"],"commands":["permit uplink"],"env":{},"keywords":["ipv6","raguard"]}
    )
    add_case(
        "PBR non applicata: route-map mancante sulla subinterface",
        "Routing","Altro","Bassa","Cisco","Router","IOS",
        "Traffico prende sempre default.",
        "Policy per link backup.",
        ["route-map counters zero"],
        ["policy su interfaccia errata"],
        "Route-map non legata alla subif corretta.",
        "ip policy route-map BACKUP su VLAN X.",
        ["int Gi0/0.X ; ip policy route-map BACKUP"],
        ["Traffico matcha policy"],
        ["Check policy per interfaccia corretta"],
        ["pbr","route-map","subinterface"],
        {"protocol":"IP","category":"Routing","root_cause":"Policy_On_Wrong_Interface","fix_type":"Bind_On_Correct_Subif",
         "checks":["route-map counters"],"commands":["ip policy"],"env":{},"keywords":["pbr","route-map"]}
    )
    add_case(
        "DHCP starvation da device anomalo: manca rate-limit/snooping",
        "LAN","Altro","Media","Cisco","Access","IOS",
        "Pool esaurito improvvisamente.",
        "Device anomalo genera DISCOVER burst.",
        ["DISCOVER pps elevato da singolo MAC"],
        ["no dhcp snooping"],
        "Assente protezione contro starvation.",
        "DHCP snooping + rate-limit 10pps + blocco MAC.",
        ["/ip dhcp-snooping enable","rate-limit 10pps","blacklist mac"],
        ["Pool stabile"],
        ["Abilitare snooping e rate-limit"],
        ["dhcp","snooping","starvation"],
        {"protocol":"DHCP","category":"LAN","root_cause":"No_Snooping_RateLimit","fix_type":"Enable_Snooping_RateLimit",
         "checks":["snooping stats"],"commands":["rate-limit"],"env":{},"keywords":["dhcp","snooping"]}
    )
    add_case(
        "802.11 roaming scarso: 2.4GHz troppo forte, 5GHz debole",
        "Wireless","Altro","Bassa","Altro","WLAN","n/a",
        "Client sticky, throughput basso.",
        "Rete enterprise.",
        ["RSSI alto su AP lontani","Preferenza 2.4GHz"],
        ["min-RSSI non impostato"],
        "Soglie/steering non configurati.",
        "Ridotto TX 2.4, min-RSSI -70, 802.11k/v/r attivi.",
        ["set tx-power 2.4 low","set min-RSSI -70","enable k/v/r"],
        ["Roaming fluido"],
        ["Bilanciare potenze e soglie"],
        ["wifi","band-steering","min-rssi","802.11kvr"],
        {"protocol":"WiFi","category":"Wireless","root_cause":"Power_Steering_Bad","fix_type":"Tune_Power_MinRSSI",
         "checks":["rssi hist"],"commands":["min-rssi -70"],"env":{},"keywords":["wifi","roaming"]}
    )
    add_case(
        "PBR di stickiness per evitare asimmetria su firewall",
        "Security","Altro","Bassa","Altro","Firewall","n/a",
        "Stati TCP si perdono su ritorno.",
        "Due link uplink ECMP.",
        ["Session miss in return path"],
        ["conntrack miss"],
        "Ritorno passa su link diverso.",
        "PBR con hash per stickiness; o eccezione state su subnet.",
        ["policy route hash src-dst","state bypass per subnet X"],
        ["Connessioni stabili"],
        ["Stickiness su stateful"],
        ["ecmp","pbr","state"],
        {"protocol":"IP","category":"Security","root_cause":"Asymmetry","fix_type":"PBR_Stickiness",
         "checks":["conn table"],"commands":["pbr hash"],"env":{},"keywords":["pbr","asymmetric"]}
    )
    return cases

# -------------------- Seeding engine --------------------
def seed_issues(project_id: str, project_short_name: str, idempotent: bool, dry: bool, project_exists: bool, prefix: str):
    data = seed_cases()

    existing = set()
    if idempotent:
        if dry and not project_exists:
            print("[DRY][IDEMPOTENT] Progetto inesistente (dry-run): salto fetch_existing_summaries().")
        else:
            existing = fetch_existing_summaries(project_short_name)
            if existing:
                print(f"[IDEMPOTENT] Summary già presenti: {len(existing)}")

    created, skipped = 0, 0
    for it in data:
        s = it["summary"].strip()
        if prefix:
            # il check idempotente considera anche il prefisso
            s_check = f"{prefix} {s}".strip()
        else:
            s_check = s
        if idempotent and s_check in existing:
            skipped += 1
            print(f"[SKIP] '{s_check}' esiste già")
            continue
        create_issue(project_id, s, it["description"], it["custom"], dry, prefix)
        created += 1
    print(f"Completato. Create: {created}, Skippate: {skipped}")

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(description="Bootstrap progetto YouTrack NETKB (RAG-friendly)")
    ap.add_argument("--short", default="NETKB", help="Short name progetto (default NETKB)")
    ap.add_argument("--name", default="Networking Knowledge Base", help="Nome progetto")
    ap.add_argument("--idempotent", action="store_true", help="Evita duplicati se rilanci lo script")
    ap.add_argument("--dry-run", action="store_true", help="Non apporta modifiche, stampa cosa farebbe")
    ap.add_argument("--prefix", default="[KB]", help="Prefisso per i summary (default [KB])")
    args = ap.parse_args()

    me = get_me()
    proj = find_project_by_short_name(args.short)
    project_existed = proj is not None

    if proj:
        print(f"[INFO] Progetto esistente: {proj['name']} ({proj['shortName']}) -> id {proj['id']}")
    else:
        proj = create_project(args.name, args.short, me["id"], dry=args.dry_run)
        print(f"[INFO] Creato progetto: {proj['name']} ({proj['shortName']}) -> id {proj['id']}")

    # Crea/associa campi custom (enum + text)
    # Enum bundles
    categoria_bundle = find_or_create_enum_bundle("NETKB_Categoria", ["WAN","LAN","Security","Wireless","Routing","Switching","VoIP","DNS/DHCP","VPN","QoS"], args.dry_run)
    protocollo_bundle = find_or_create_enum_bundle("NETKB_Protocollo", ["OSPF","BGP","STP","IPsec","SIP","DNS","DHCP","NTP","LLDP","LACP","MPLS","WireGuard","GRE","VRRP","BFD","WiFi","SSL","IPv6","802.1Q","Altro"], args.dry_run)
    severita_bundle = find_or_create_enum_bundle("NETKB_Severita", ["Bassa","Media","Alta"], args.dry_run)
    vendor_bundle = find_or_create_enum_bundle("NETKB_Vendor", ["Cisco","Juniper","MikroTik","Linux/FRR","Palo Alto","Fortinet","Ubiquiti","Altro"], args.dry_run)

    # Custom fields
    cf_categoria = find_or_create_cf_enum("Categoria", categoria_bundle["id"], args.dry_run)
    cf_protocollo = find_or_create_cf_enum("Protocollo", protocollo_bundle["id"], args.dry_run)
    cf_severita = find_or_create_cf_enum("Severità", severita_bundle["id"], args.dry_run)
    cf_vendor = find_or_create_cf_enum("Vendor", vendor_bundle["id"], args.dry_run)
    cf_device = find_or_create_cf_text("Dispositivo", "string", args.dry_run)
    cf_osver = find_or_create_cf_text("Versione/OS", "string", args.dry_run)

    # Associa CF al progetto
    for cf in [cf_categoria, cf_protocollo, cf_severita, cf_vendor, cf_device, cf_osver]:
        ensure_project_cf(proj["id"], cf["id"], required=False, dry=args.dry_run)

    # Seed issues
    seed_issues(proj["id"], args.short, args.idempotent, args.dry_run, project_existed, args.prefix)

if __name__ == "__main__":
    main()
