# yt_netkb_bootstrap.py
# Requisiti: pip install requests
import requests, sys, argparse, time, re, json
from typing import Dict, List, Optional

BASE_URL = "https://tuaistanza.youtrack.cloud/api"
TOKEN = "ybpt_xxx_copia_il_tuo_token_qui"

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
    bundles = req("GET", "/admin/customFieldSettings/bundles/enum?fields=id,name,values(name)")
    for b in bundles:
        if b["name"] == name:
            current = {v["name"] for v in b.get("values", [])}
            missing = [v for v in values if v not in current]
            if missing and not dry:
                for v in missing:
                    req("POST",
                        f"/admin/customFieldSettings/bundles/enum/{b['id']}/values?fields=id,name",
                        json={"name": v})
            elif missing and dry:
                print(f"[DRY] Aggiungerei al bundle {name} i valori mancanti: {missing}")
            return b
    if dry:
        print(f"[DRY] Creerei EnumBundle {name} con valori {values}")
        return {"id": f"DRY-BNDL-{name}", "name": name}
    b = req("POST", "/admin/customFieldSettings/bundles/enum?fields=id,name",
            json={"name": name, "values": [{"name": v} for v in values]})
    return b

def find_or_create_cf_enum(name: str, bundle_id: str, dry: bool) -> Dict:
    cfs = req("GET", "/admin/customFieldSettings/customFields?fields=id,name,fieldType(id)")
    for cf in cfs:
        if cf["name"] == name:
            return cf
    if dry:
        print(f"[DRY] Creerei CustomField enum {name} (bundle {bundle_id})")
        return {"id": f"DRY-CF-{name}", "name": name}
    body = {
        "name": name,
        "fieldType": {"id": "enum[1]"}  # tipo corretto per enum
    }
    return req("POST", "/admin/customFieldSettings/customFields?fields=id,name,fieldType(id)", json=body)

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

# ---- mapping & helpers per ProjectCustomField ----
def _proj_cf_type_from_fieldtype(fieldtype_id: str) -> str:
    mapping = {
        "enum[1]": "EnumProjectCustomField",
        "enum[*]": "EnumProjectCustomField",
        "integer": "SimpleProjectCustomField",
        "float": "SimpleProjectCustomField",
        "date": "SimpleProjectCustomField",
        "date and time": "SimpleProjectCustomField",
        "period": "PeriodProjectCustomField",
        "string": "SimpleProjectCustomField",
        "text": "TextProjectCustomField",
        "user[1]": "UserProjectCustomField",
        "user[*]": "UserProjectCustomField",
        "group[1]": "GroupProjectCustomField",
        "group[*]": "GroupProjectCustomField",
        "state[1]": "StateProjectCustomField",
        "version[1]": "VersionProjectCustomField",
        "version[*]": "VersionProjectCustomField",
        "ownedField[1]": "OwnedProjectCustomField",
        "ownedField[*]": "OwnedProjectCustomField",
        "build[1]": "BuildProjectCustomField",
        "build[*]": "BuildProjectCustomField",
    }
    return mapping.get(fieldtype_id, "SimpleProjectCustomField")

def get_customfield_type_id(cf_id: str) -> str:
    cf = req("GET", f"/admin/customFieldSettings/customFields/{cf_id}?fields=fieldType(id)")
    return (cf.get("fieldType") or {}).get("id", "string")

def ensure_project_cf(
    project_id: str,
    cf_id: str,
    required: bool,
    dry: bool,
    proj_cf_type: Optional[str] = None,
    bundle_id: Optional[str] = None
) -> None:
    """
    Collega un CustomField al progetto istanziando il ProjectCustomField corretto.
    Se proj_cf_type è None, viene dedotto dal fieldType del CustomField.
    Per i campi enum passare bundle_id.
    """
    if dry or str(project_id).startswith("DRY-"):
        msg = f"[DRY] Assocerei CF {cf_id} al progetto {project_id}"
        if proj_cf_type:
            msg += f" come {proj_cf_type}"
        if bundle_id:
            msg += f" (bundle {bundle_id})"
        print(msg)
        return

    fields = req(
        "GET",
        f"/admin/projects/{project_id}/customFields?fields=id,field(id,name),$type,bundle(id,name)"
    )
    if isinstance(fields, list):
        for f in fields:
            if f.get("field", {}).get("id") == cf_id:
                return  # già presente

    # deduci $type se non passato
    if not proj_cf_type:
        ft_id = get_customfield_type_id(cf_id)  # es: "enum[1]", "string"
        proj_cf_type = _proj_cf_type_from_fieldtype(ft_id)

    body = {
        "$type": proj_cf_type,
        "field": {"id": cf_id, "$type": "CustomField"},  # <-- importante
        "canBeEmpty": not required,
    }
    if bundle_id:
        body["bundle"] = {"id": bundle_id}

    req(
        "POST",
        f"/admin/projects/{project_id}/customFields?fields=id,$type,field(name),bundle(id,name)",
        json=body
    )

# -------------------- Issues helpers --------------------
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
    text = description.strip()
    m = re.search(r"\{[\s\S]*\}\s*$", text)
    if not m:
        return False
    try:
        data = json.loads(m.group(0))
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

    cf_payload = []
    for fname, fval in custom.items():
        if fval is None:
            continue
        if isinstance(fval, dict) and fval.get("_type") == "enum":
            cf_payload.append({"name": fname, "value": {"name": fval["name"]}})
        else:
            cf_payload.append({"name": fname, "value": fval})

    payload = {
        "project": {"id": project_id},
        "summary": summary,
        "description": description,
        "customFields": cf_payload
    }
    return req("POST", "/issues?fields=id,idReadable,summary", json=payload)

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

def _slug(s: str) -> str:
    s = (s or "Unknown").strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_]+", "", s)[:48] or "Unknown"

def add_case_to(
    cases: List[Dict],
    summary: str,
    cat: str, proto: str, sev: str, vendor: str, dev: str, osver: str,
    problema: str, contesto: str,
    sintomi: List[str], evid: List[str], analisi: str, root: str,
    fix: List[str], cmds: List[str], verify: List[str], lessons: List[str], kws: List[str],
    js: Optional[Dict] = None
):
    if js is None:
        js = {
            "protocol": proto if proto and proto != "Altro" else "IP",
            "category": cat or "Routing",
            "root_cause": _slug(root),
            "fix_type": _slug(fix[0]) if fix else "Manual_Fix",
            "checks": (sintomi or evid or []),
            "commands": cmds or [],
            "env": {"vendor": vendor or "Altro", "device": dev or "n/a", "os": osver or "n/a"},
            "keywords": kws or []
        }
    t, d = case_block(summary, problema, contesto, sintomi, evid, analisi, root, fix, cmds, verify, lessons, kws, js)
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

# -------------------- Seed data (28 casi) --------------------
def seed_cases() -> List[Dict]:
    cases: List[Dict] = []

    add_case_to(cases, "OSPF non va in FULL su link p2p: MTU mismatch (1500↔1400), workaround mtu-ignore",
        "Routing","OSPF","Media","Cisco","Router A/B","IOS 15.x / FRR 9.x",
        "Adiacenza OSPF bloccata in EXSTART/EXCHANGE dopo cambio su link p2p.",
        "Area 0, A=Linux/FRR (eth1), B=Cisco IOS (Gi0/0); overlay GRE.",
        ["Neighbor EXSTART/EXCHANGE","LSDB non sincronizzata"],
        ["FRR: 'MTU mismatch'","Cisco: MTU 1400, lato A 1500","Ping DF 1472 fallisce"],
        "DBD rifiutati per MTU diversa; overlay riduce MTU effettiva.",
        "MTU mismatch sul link p2p.",
        ["Allineato MTU 1400 su entrambi","Workaround: ip ospf mtu-ignore su entrambe"],
        ["ip link set dev eth1 mtu 1400","interface Gi0/0 ; mtu 1400 ; ip ospf mtu-ignore"],
        ["Neighbor FULL","Ping DF size 1372 ok"],
        ["Standard MTU per link con overlay","Evitare mtu-ignore in esercizio"],
        ["ospf","exstart","exchange","mtu mismatch","mtu-ignore","dbd","gre","p2p"],
        {
            "protocol":"OSPF","category":"Routing","root_cause":"MTU_mismatch","fix_type":"MTU_align",
            "checks":["show ip ospf interface","ping DF 1372"],
            "commands":["mtu 1400","ip ospf mtu-ignore"],
            "env":{"vendorA":"Linux/FRR","vendorB":"Cisco IOS","overlay":"GRE"},
            "keywords":["ospf","exstart","exchange","mtu mismatch","mtu-ignore","dbd","gre","p2p"]
        }
    )

    add_case_to(cases, "WAN PPPoE lenta e loss: MTU 1500 su PPPoE, PMTUD bloccato, MSS non clampato",
        "WAN","Altro","Alta","Linux/FRR","edge-router","Linux 5.x",
        "Loss 10–25% e stall TCP su PPPoE dopo migrazione.",
        "PPPoE VLAN 835; IPsec/HTTPS/VoIP impattati.",
        ["Download >1MB si bloccano","VoIP jitter alto","Ping DF 1472 fallisce"],
        ["MTU ppp0=1500","ICMP 3:4 bloccato","MSS eccessivo"],
        "PPPoe richiede MTU/MRU 1492; permettere ICMP fragmentation-needed; MSS clamp.",
        "MTU errata + ICMP 3:4 bloccato + MSS non clampato.",
        ["Set MTU 1492","Permesso ICMP 3:4","MSS clamp 1452"],
        ["ip link set ppp0 mtu 1492","iptables -t mangle ... TCPMSS --clamp-mss-to-pmtu"],
        ["Ping DF 1472 ok","Throughput ripristinato"],
        ["Standard PPPoE: MTU 1492 + MSS clamp","Non bloccare ICMP 3:4"],
        ["pppoe","mtu 1492","mss 1452","pmtud","icmp 3:4","vlan 835"], None)

    add_case_to(cases, "VoIP one-way audio con SBC esterno: SIP ALG riscrive SDP/RTP",
        "VoIP","SIP","Alta","Altro","firewall","n/a",
        "Audio monodirezionale dopo nuovo firewall.",
        "PBX on-prem + SBC cloud; SIP ALG attivo.",
        ["Audio in una sola direzione","Porte RTP errate"],
        ["SDP riscritti dall'ALG","NAT traversal duplicato"],
        "ALG interferisce con SBC.",
        "SIP ALG attivo causa porta/indirizzo RTP errati.",
        ["Disabilitato SIP ALG","Aperto range RTP necessario"],
        ["no sip alg","permit udp 10000-20000"],
        ["Audio bidirezionale ripristinato"],
        ["Evitare ALG con SBC presente","Documentare range RTP"],
        ["sip alg","one-way audio","rtp","sdp","sbc","nat traversal"], None)

    add_case_to(cases, "DHCP relay ignora richieste: manca Option 82 richiesta dal server",
        "DNS/DHCP","DHCP","Media","Cisco","Switch L3","IOS 15.x",
        "Client non ottengono IP su VLAN access.",
        "Relay centralizzato; server valida Circuit-ID/Remote-ID.",
        ["DISCOVER senza OFFER","Server richiede Option 82"],
        ["Sniffer conferma DISCOVER senza Option 82"],
        "Server scarta senza Option 82.",
        "Abilitata insert-option 82 e pass-through hop-count.",
        ["ip dhcp relay information option","ip dhcp relay information trust-all"],
        ["Lease assegnati"],
        ["Standardizzare Option 82"],
        ["dhcp","option 82","relay"], None)

    add_case_to(cases, "OSPF su access: mancato passive-interface espone reti di accesso",
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
        ["ospf","passive-interface","hardening"], None)

    add_case_to(cases, "BGP flapping: hold-time troppo basso e dampening aggressivo",
        "Routing","BGP","Alta","Cisco","Router edge","IOS XE 17.x",
        "Sessione eBGP instabile ogni 30s.",
        "Peer ISP con timers 1/3s.",
        ["ADJCHANGE Idle/Established frequente"],
        ["bgp summary, timers 1/3"],
        "Keepalive/hold inadeguati; dampening penalizza prefissi.",
        "Timers 10/30 e no dampening su prefissi critici.",
        ["neighbor x timers 10 30","no bgp dampening su prefissi business"],
        ["Sessione stabile"],
        ["Concordare timers standard con ISP"],
        ["bgp","timers","dampening"], None)

    add_case_to(cases, "STP loop: PortFast senza BPDU Guard su presa utente con mini-switch",
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
        ["stp","portfast","bpduguard","loop"], None)

    add_case_to(cases, "VLAN Native mismatch su trunk 802.1Q causa perdita traffico",
        "Switching","802.1Q","Media","Cisco","Switch agg","IOS 15.x",
        "Pacchetti persi su VLAN utente.",
        "Trunk tra vendor diversi.",
        ["LLDP/CDP warning native mismatch"],
        ["show interfaces trunk"],
        "Native VLAN diversa per lato.",
        "Allineata native 10 su entrambi; tagging native.",
        ["switchport trunk native vlan 10","vlan dot1q tag native"],
        ["Traffico ristabilito"],
        ["Evitare VLAN1 come native"],
        ["vlan","trunk","native mismatch"], None)

    add_case_to(cases, "IPsec site-to-site dietro NAT: NAT-T disabilitato",
        "VPN","IPsec","Alta","Altro","Firewall","n/a",
        "Tunnel non sale / instabile.",
        "Firewall NAT intermedio.",
        ["ESP bloccato; solo IKE/500 visibile"],
        ["pcap: no UDP 4500"],
        "Serve NAT-T e UDP 4500.",
        "Abilitato NAT-T e aperte porte 500/4500.",
        ["crypto isakmp nat-traversal 20","permit udp 500 4500"],
        ["SA stabili"],
        ["Verificare sempre NAT-T con NAT in path"],
        ["ipsec","nat-t","udp 4500"], None)

    add_case_to(cases, "QoS errata: shaper applicato in ingresso invece che in uscita",
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
        ["qos","hfsc","egress"], None)

    add_case_to(cases, "NTP offset causa fallimenti Kerberos: UDP/123 bloccato",
        "Security","NTP","Media","Linux/FRR","Server","Linux",
        "Ticket TGT rifiutati, clock skew.",
        "Join AD; firewall restrittivo.",
        ["ntpq -p no-reach","kinit skew"],
        ["fw drop udp/123"],
        "NTP non sincronizzato.",
        "Aperto UDP 123 verso pool; fallback interno.",
        ["ufw allow 123/udp","timesyncd iburst"],
        ["Offset < 50ms; Kerberos ok"],
        ["Monitorare NTP su host critici"],
        ["ntp","kerberos","skew"], None)

    add_case_to(cases, "Port-channel instabile: LACP rate e hash non coerenti",
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
        ["lacp","hashing","port-channel"], None)

    add_case_to(cases, "VRRP flap con CPU alta: preempt troppo aggressivo",
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
        ["vrrp","preempt","priority"], None)

    add_case_to(cases, "LLDP disabilitato: telefoni non entrano in voice VLAN",
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
        ["lldp","voice vlan","med"], None)

    add_case_to(cases, "DNS ricorsivo lento: forwarder bloccato e cache TTL basso",
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
        ["dns","forwarder","ttl"], None)

    add_case_to(cases, "RADIUS accounting intermittente: UDP/1813 non permesso",
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
        ["radius","accounting","1813"], None)

    add_case_to(cases, "Static route recursive fail: next-hop non risolvibile",
        "Routing","Altro","Media","Linux/FRR","Router","FRR 9.x",
        "Blackhole su rotte statiche.",
        "Next-hop fuori subnet.",
        ["ARP incomplete su next-hop"],
        ["ip route show","arp -an"],
        "Manca route verso subnet del next-hop.",
        "Aggiunta route intermedia o interfaccia d’uscita.",
        ["ip route add ... via ... dev X"],
        ["Connettività ripristinata"],
        ["Verificare reachability next-hop"],
        ["static route","recursive","arp"], None)

    add_case_to(cases, "AnyConnect split-tunnel: MTU eccessiva, SaaS si blocca",
        "VPN","SSL","Media","Cisco","ASA/FTD","ASA 9.x",
        "Applicazioni SaaS lente/irraggiungibili.",
        "SSL VPN split-tunnel, PMTUD bloccato.",
        ["Stalli su payload grandi"],
        ["no ICMP 3:4 in path"],
        "MTU tunnel 1500 e MSS eccessivo.",
        "MTU 1350–1400 e MSS 1350.",
        ["sysopt connection tcpmss 1350","tunnel-group MTU 1400"],
        ["SaaS ok"],
        ["Definire MTU per SSL VPN"],
        ["ssl vpn","mtu","mss","pmtud"], None)

    add_case_to(cases, "WireGuard dietro CGNAT: assenza keepalive porta a drop",
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
        ["wireguard","keepalive","cgnat"], None)

    add_case_to(cases, "GRE over IPsec: payload grandi in blackhole (PMTUD rotto)",
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
        ["gre","ipsec","mtu","pmtud"], None)

    add_case_to(cases, "Asymmetric routing: firewall stateful droppa i ritorni",
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
        ["ecmp","stateful","pbr"], None)

    add_case_to(cases, "MPLS LDP non sale: TCP/646 filtrato nel core",
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
        ["mpls","ldp","tcp 646"], None)

    add_case_to(cases, "BFD non parte: min-tx/rx troppo bassi per la piattaforma",
        "Routing","BFD","Bassa","Cisco","WAN","IOS XE",
        "Sessione BFD down.",
        "Link WAN ad alta latenza.",
        ["diag: control packet too slow"],
        ["timers 50/50/3"],
        "Timer sotto capacità forwarding.",
        "Impostati 300ms/300ms mult=3.",
        ["bfd interval 300 min_rx 300 multiplier 3"],
        ["BFD up"],
        ["Timer in linea con piattaforma"],
        ["bfd","timers"], None)

    add_case_to(cases, "IPv6 RA-guard blocca SLAAC legittimo dopo upgrade",
        "LAN","IPv6","Media","Cisco","Access","IOS",
        "Client senza IPv6.",
        "RA-guard su access.",
        ["Drops su RA guard","MAC gw non whitelisted"],
        ["ra-guard profile obsoleto"],
        "Profile non aggiornato all’upgrade.",
        "Aggiornato bindings e permit su uplink.",
        ["ipv6 nd raguard policy ..."],
        ["SLAAC ok"],
        ["Rivedere RA-guard dopo upgrade"],
        ["ipv6","ra-guard","slaac"], None)

    add_case_to(cases, "PBR non applicata: route-map mancante sulla subinterface",
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
        ["pbr","route-map","subinterface"], None)

    add_case_to(cases, "DHCP starvation da device anomalo: manca rate-limit/snooping",
        "LAN","DHCP","Media","Cisco","Access","IOS",
        "Pool esaurito improvvisamente.",
        "Device anomalo genera DISCOVER burst.",
        ["DISCOVER pps elevato da singolo MAC"],
        ["no dhcp snooping"],
        "Assente protezione contro starvation.",
        "DHCP snooping + rate-limit 10pps + blocco MAC.",
        ["/ip dhcp-snooping enable","rate-limit 10pps","blacklist mac"],
        ["Pool stabile"],
        ["Abilitare snooping e rate-limit"],
        ["dhcp","snooping","starvation"], None)

    add_case_to(cases, "802.11 roaming scarso: 2.4GHz troppo forte, 5GHz debole",
        "Wireless","WiFi","Bassa","Altro","WLAN","n/a",
        "Client sticky, throughput basso.",
        "Rete enterprise.",
        ["RSSI alto su AP lontani","Preferenza 2.4GHz"],
        ["min-RSSI non impostato"],
        "Soglie/steering non configurati.",
        "Ridotto TX 2.4, min-RSSI -70, 802.11k/v/r attivi.",
        ["set tx-power 2.4 low","set min-RSSI -70","enable k/v/r"],
        ["Roaming fluido"],
        ["Bilanciare potenze e soglie"],
        ["wifi","band-steering","min-rssi","802.11kvr"], None)

    add_case_to(cases, "PBR di stickiness per evitare asimmetria su firewall",
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
        ["ecmp","pbr","state"], None)

    # (aggiunte varianti/casi ulteriori per arrivare a ~28)
    add_case_to(cases, "RADIUS CoA non funziona: UDP/3799 non aperto",
        "Security","Altro","Media","Altro","WLC/RADIUS","n/a",
        "Cambio VLAN dinamico non applicato.",
        "RADIUS CoA/DM tra NAS e server.",
        ["CoA Request senza Reply","timeout sul 3799"],
        ["fw blocca udp/3799"],
        "Porta CoA non consentita.",
        "Aperto UDP 3799 bidirezionale.",
        ["permit udp 3799"],
        ["CoA applicato correttamente"],
        ["Validare porte AAA dinamiche"],
        ["radius","coa","3799"], None)

    add_case_to(cases, "DNSSEC fallisce per skew orario",
        "DNS/DHCP","DNS","Bassa","Linux/FRR","Resolver","unbound",
        "SERVFAIL su domini con DNSSEC.",
        "Host con orologio fuori sync.",
        ["val-bogus: signature expired"],
        ["ntp offset > 300s"],
        "Time skew rompe validazione.",
        "Sincronizzato NTP; makestep iniziale.",
        ["timedatectl set-ntp true"],
        ["Validation OK"],
        ["Abilitare monitor NTP"],
        ["dnssec","ntp","skew"], None)

    add_case_to(cases, "Port mirroring saturato: SPAN in uscita verso uplink",
        "Switching","Altro","Bassa","Cisco","Agg switch","IOS",
        "Perdita traffico durante capture.",
        "SPAN inviato sull’uplink di produzione.",
        ["drop intermittenti su uplink"],
        ["show int counters"],
        "SPAN causa congestione.",
        "Porta dedicata per SPAN o rate-limit.",
        ["monitor session 1 destination Gi1/0/48"],
        ["No drop in produzione"],
        ["Isolare SPAN dal traffico utente"],
        ["span","mirror","congestion"], None)

    add_case_to(cases, "ARP cache poisoning evitato con DAI",
        "LAN","Altro","Bassa","Cisco","Access","IOS",
        "Allarmi spoof ARP sporadici.",
        "Client non affidabili su access.",
        ["gratuitous sospetti","MAC/IP mismatch"],
        ["no dhcp snooping bindings"],
        "Assenza DAI + bindings.",
        "Abilitato DHCP snooping & DAI.",
        ["ip dhcp snooping","ip arp inspection vlan X"],
        ["Nessun falso positivo"],
        ["Sicurezza L2 di base"],
        ["dai","dhcp snooping","arp"], None)

    add_case_to(cases, "NAT hairpin mancante per accesso a VIP interno",
        "Security","Altro","Bassa","Altro","Firewall","n/a",
        "Client interni non accedono a VIP pubblico.",
        "VIP/DNAT su firewall per servizio interno.",
        ["interni verso IP pubblico falliscono"],
        ["no hairpin nat"],
        "Manca NAT loopback/hairpin.",
        "Aggiunte regole hairpin e policy corrispondente.",
        ["nat hairpin enable","dnat+snat rules"],
        ["Accesso interno funziona"],
        ["Documentare hairpin per VIP interni"],
        ["nat","hairpin","vip"], None)

    add_case_to(cases, "Storm-control assente: broadcast/multicast saturano accesso",
        "Switching","Altro","Media","Cisco","Access","IOS",
        "Rallentamenti intermittenti su piano L2.",
        "Host rumorosi in VLAN ampia.",
        ["broadcast > soglia"],
        ["no storm-control"],
        "Assenza limiti di tempesta.",
        "storm-control broadcast/multicast 1-2%.",
        ["storm-control broadcast level 1.00"],
        ["Rete stabile"],
        ["Applicare storm-control di default"],
        ["storm-control","broadcast","multicast"], None)

    add_case_to(cases, "DHCP snooping trust non impostato su uplink",
        "LAN","DHCP","Bassa","Cisco","Access","IOS",
        "DISCOVER/OFFER bloccati sull’access.",
        "Server DHCP fuori VLAN, uplink verso L3.",
        ["packets dropped by dhcp snooping"],
        ["uplink non trust"],
        "Porta uplink va trustata.",
        "ip dhcp snooping trust su uplink.",
        ["ip dhcp snooping trust"],
        ["Lease regolari"],
        ["Checklist: trust uplink"],
        ["dhcp","snooping","trust"], None)

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
        s_check = f"{prefix} {s}".strip() if prefix else s
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

    # Enum bundles
    categoria_bundle  = find_or_create_enum_bundle("NETKB_Categoria",
        ["WAN","LAN","Security","Wireless","Routing","Switching","VoIP","DNS/DHCP","VPN","QoS"], args.dry_run)
    protocollo_bundle = find_or_create_enum_bundle("NETKB_Protocollo",
        ["OSPF","BGP","STP","IPsec","SIP","DNS","DHCP","NTP","LLDP","LACP","MPLS","WireGuard","GRE","VRRP","BFD","WiFi","SSL","IPv6","802.1Q","Altro"], args.dry_run)
    severita_bundle   = find_or_create_enum_bundle("NETKB_Severita", ["Bassa","Media","Alta"], args.dry_run)
    vendor_bundle     = find_or_create_enum_bundle("NETKB_Vendor",
        ["Cisco","Juniper","MikroTik","Linux/FRR","Palo Alto","Fortinet","Ubiquiti","Altro"], args.dry_run)

    # Custom fields (schema)
    cf_categoria  = find_or_create_cf_enum("Categoria",  categoria_bundle["id"],  args.dry_run)
    cf_protocollo = find_or_create_cf_enum("Protocollo", protocollo_bundle["id"], args.dry_run)
    cf_severita   = find_or_create_cf_enum("Severità",   severita_bundle["id"],   args.dry_run)
    cf_vendor     = find_or_create_cf_enum("Vendor",     vendor_bundle["id"],     args.dry_run)
    cf_device     = find_or_create_cf_text("Dispositivo", "string", args.dry_run)
    cf_osver      = find_or_create_cf_text("Versione/OS", "string", args.dry_run)

    # Associa CF al progetto (bundle obbligatorio per gli enum; $type dedotto auto)
    ensure_project_cf(proj["id"], cf_categoria["id"],  required=False, dry=args.dry_run, bundle_id=categoria_bundle["id"])
    ensure_project_cf(proj["id"], cf_protocollo["id"], required=False, dry=args.dry_run, bundle_id=protocollo_bundle["id"])
    ensure_project_cf(proj["id"], cf_severita["id"],   required=False, dry=args.dry_run, bundle_id=severita_bundle["id"])
    ensure_project_cf(proj["id"], cf_vendor["id"],     required=False, dry=args.dry_run, bundle_id=vendor_bundle["id"])
    ensure_project_cf(proj["id"], cf_device["id"],     required=False, dry=args.dry_run)
    ensure_project_cf(proj["id"], cf_osver["id"],      required=False, dry=args.dry_run)

    # Seed issues
    seed_issues(proj["id"], args.short, args.idempotent, args.dry_run, project_existed, args.prefix)

if __name__ == "__main__":
    main()
