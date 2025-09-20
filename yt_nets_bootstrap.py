# yt_nets_bootstrap.py
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
    r = requests.request(method, f"{BASE_URL}{path}", headers=HDRRS if (HDRRS:=HDRS) else HDRS, **kwargs)
    if r.status_code >= 300:
        print("ERRORE", r.status_code, r.text)
        sys.exit(1)
    return r.json() if r.text else {}

def get_me():
    return req("GET", "/users/me?fields=id,login")

def find_project_by_short_name(short_name):
    projs = req("GET", "/admin/projects?fields=id,shortName,name")
    for p in projs:
        if p.get("shortName") == short_name:
            return p
    return None

def create_project(name, short_name, leader_id, dry_run=False):
    if dry_run:
        print(f"[DRY] Creerei progetto {name} ({short_name})")
        return {"id": "DRY-PROJ-ID", "name": name, "shortName": short_name}
    payload = {"name": name, "shortName": short_name, "leader": {"id": leader_id}}
    return req("POST", "/admin/projects?fields=id,shortName,name&template=kanban", json=payload)

def create_issue(project_id, summary, description="", dry_run=False):
    if dry_run:
        print(f"[DRY] Creerei issue: {summary}")
        return {"id": "DRY-ISSUE-ID", "idReadable": "NETS-XXX", "summary": summary}
    payload = {"project": {"id": project_id}, "summary": summary, "description": description}
    return req("POST", "/issues?fields=id,idReadable,summary", json=payload)

def fetch_existing_summaries(project_short_name):
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

def seed_issues(project_id, project_short_name, idempotent=False, dry_run=False):
    issues_data = [
        {
            "summary": "PPPoE WAN lenta e packet loss dopo migrazione VLAN 835: MTU 1500, PMTUD bloccato, MSS non clampato",
            "desc":
"""Contesto
Sede Torino, edge Linux, PPPoE VLAN 835
Sintomi
Loss 10-25%, stall TCP, VoIP jitter
Evidenze/Log
ping -M do -s 1472 fallisce DF; MTU ppp0=1500; firewall blocca ICMP 3:4
Analisi
PPPoE richiede MTU/MRU 1492; PMTUD rotto; MSS eccessivo
Causa radice
MTU 1500 su ppp0, niente MSS clamp, ICMP 3:4 bloccato
Soluzione adottata
ip link set dev ppp0 mtu 1492
iptables -t mangle -A FORWARD -o ppp0 -p tcp --tcp-flags SYN,RST SYN -j TCPMSS --clamp-mss-to-pmtu
iptables -A INPUT -p icmp --icmp-type fragmentation-needed -j ACCEPT
Verifica/Esito
ping 1472 ok, throughput ripristinato, VoIP stabile
Parole chiave
PPPoE MTU 1492 MSS 1452 PMTUD ICMP 3:4"""
        },
        {
            "summary": "DNS ricorsivo lento: forwarder pubblico bloccato da firewall; cache TTL assente",
            "desc":
"""Contesto
Resolver interno bind9, firewall perimetrale restrittivo
Sintomi
Lookup lenti, timeouts sporadici
Evidenze/Log
dig +trace rapido; dig @8.8.8.8 timeouts; named log SERVFAIL
Analisi
Forwarding verso 8.8.8.8 filtrato; no caching efficace
Causa radice
Policy firewall outbound DNS bloccata; max-cache-ttl default basso
Soluzione adottata
Consentito UDP/TCP 53 verso forwarder autorizzati
named.conf: forwarders { 1.1.1.1; 9.9.9.9; }; max-cache-ttl 86400;
Verifica/Esito
latency < 50ms, hit cache > 80%
Parole chiave
DNS forwarder firewall cache TTL"""
        },
        {
            "summary": "DHCP relay non inoltra: Option 82 richiesta dal server ma assente",
            "desc":
"""Contesto
Switch L3 come relay, server DHCP centralizzato
Sintomi
Client senza IP in VLAN access
Evidenze/Log
Sniffer: DISCOVER esce ma server ignora; log server richiede option 82
Analisi
Server ha policy che valida Circuit-ID/Remote-ID
Causa radice
Relay configurato senza insert-option
Soluzione adottata
ip dhcp relay information option; policy pass-through su hop-count
Verifica/Esito
Lease assegnati, T1/T2 ok
Parole chiave
DHCP relay Option 82 Circuit-ID Remote-ID"""
        },
        {
            "summary": "OSPF adiacenza stuck in EXSTART: mismatch MTU su link punto-punto",
            "desc":
"""Contesto
Router A/B backbone area 0
Sintomi
Full adjacency mai raggiunta
Evidenze/Log
OSPF logs: EXSTART/EXCHANGE loop; ifconfig MTU A=1500, B=1400
Analisi
MSS/DBD size incongruenti per MTU mismatch
Causa radice
MTU interfacce non allineate
Soluzione adottata
Uniformato MTU 1500 su entrambi; ip ospf mtu-ignore su vendor che lo supporta
Verifica/Esito
Neighbor FULL, LSDB sincronizzata
Parole chiave
OSPF EXSTART MTU mismatch mtu-ignore"""
        },
        {
            "summary": "OSPF rotte di accesso esposte: interfacce access non passive",
            "desc":
"""Contesto
Switch L3 con OSPF area 10
Sintomi
Rete access annunciata come transit; instabilità
Evidenze/Log
show ip ospf interface: state BDR su interfacce access
Analisi
Mancata passive-interface default
Causa radice
Errore di hardening OSPF
Soluzione adottata
router ospf; passive-interface default; no passive su uplink
Verifica/Esito
Nessuna adiacenza su access, LSDB pulita
Parole chiave
OSPF passive-interface hardening"""
        },
        {
            "summary": "BGP flapping ogni 30s: hold-time 3s su peer remoto e route-dampening aggressivo",
            "desc":
"""Contesto
Peering eBGP con ISP
Sintomi
Sessione attiva/inattiva di continuo, rotte instabili
Evidenze/Log
%BGP-5-ADJCHANGE Idle/Established; timers 1/3 secondi
Analisi
Keepalive troppo stretto; dampening penalizza prefissi
Causa radice
Timings non standard e policy ISP
Soluzione adottata
Timers 10/30, disabilitato dampening su prefissi critici, richiesta adeguamento ISP
Verifica/Esito
Sessione stabile, flap < 1/h
Parole chiave
BGP timers hold-time dampening"""
        },
        {
            "summary": "ARP gratuitous genera allarme spoofing: failover HA non riconosciuto",
            "desc":
"""Contesto
Coppia firewall HA active/passive
Sintomi
IDS segnala ARP spoof, ma è failover lecito
Evidenze/Log
Gratuitous ARP all'evento di preemption
Analisi
IDS non whitelista MAC virtuale
Causa radice
Mancata eccezione per VRRP/HSRP/HA MAC
Soluzione adottata
Whitelist MAC virtuale e subnet VIP, tuning soglie IDS
Verifica/Esito
Nessun falso positivo al prossimo failover
Parole chiave
ARP gratuitous VRRP HSRP HA spoofing"""
        },
        {
            "summary": "STP loop accidentale su access: PortFast senza BPDU guard",
            "desc":
"""Contesto
Access layer con PC e mini-switch utente
Sintomi
Tempeste broadcast, rete lenta
Evidenze/Log
CPU switch alta, incrementi TCN
Analisi
Loop creato da mini-switch; PortFast abilitato senza protezioni
Causa radice
Assenza bpduguard/bpdufilter/loopguard
Soluzione adottata
spanning-tree portfast; spanning-tree bpduguard enable su access
Verifica/Esito
Stabilizzazione in pochi secondi, nessun loop
Parole chiave
STP PortFast BPDU guard loop"""
        },
        {
            "summary": "VLAN Native mismatch su trunk: tag inconsistente causa perdita traffico",
            "desc":
"""Contesto
Trunk 802.1Q tra switch diversi
Sintomi
Disallineamento VLAN utente, pacchetti persi
Evidenze/Log
CDP/LLDP avvisa native mismatch
Analisi
Una estremità con native VLAN 1, l'altra 10
Causa radice
Configurazioni trunk asimmetriche
Soluzione adottata
Allineata native VLAN 10 su entrambi; preferito tag native
Verifica/Esito
Connettività ripristinata
Parole chiave
VLAN native mismatch trunk 802.1Q"""
        },
        {
            "summary": "IPsec site-to-site non si stabilisce dietro NAT: NAT-T disabilitato",
            "desc":
"""Contesto
Tunnel verso partner, firewall NAT in mezzo
Sintomi
Phase1 fallisce o Phase2 instabile
Evidenze/Log
ESP blocked, solo IKE/UDP 500 visto
Analisi
Necessario UDP 4500 e NAT-T
Causa radice
nat-traversal off
Soluzione adottata
Abilitato NAT-T; aperto UDP 500/4500; keepalive 20s
Verifica/Esito
SA stabili, traffico cifrato ok
Parole chiave
IPsec NAT-T UDP 4500 IKE"""
        },
        {
            "summary": "SIP one-way audio: ALG interferisce con SDP in presenza di SBC esterno",
            "desc":
"""Contesto
PBX interno, SBC cloud, firewall con SIP ALG
Sintomi
Audio monodirezionale
Evidenze/Log
SDP riscritti, porte RTP errate
Analisi
ALG duplicava NAT traversal già gestito da SBC
Causa radice
SIP ALG attivo
Soluzione adottata
Disabilitato SIP ALG; pin-hole RTP statici o via SBC
Verifica/Esito
Audio bidirezionale ripristinato
Parole chiave
SIP ALG RTP SBC one-way audio"""
        },
        {
            "summary": "QoS shaping errato su WAN: profilo applicato in ingresso invece che in uscita",
            "desc":
"""Contesto
Link 100/20 Mbps con coda HFSC
Sintomi
VoIP jitter, upload saturato
Evidenze/Log
Coda tx quasi vuota, rx drop elevati
Analisi
Shaper applicato su interfaccia sbagliata
Causa radice
Policy direction invertita
Soluzione adottata
Shaping su egress WAN; classi per EF/AF; policing in ingresso
Verifica/Esito
MOS>4, latenze stabili
Parole chiave
QoS shaping egress EF AF"""
        },
        {
            "summary": "NTP offset elevato causa fallimenti Kerberos/AD: sorgenti non raggiungibili",
            "desc":
"""Contesto
Server Linux join a dominio
Sintomi
Ticket TGT rifiutati, clock skew
Evidenze/Log
ntpq -p no reach; kinit clock skew
Analisi
Firewall blocca UDP 123
Causa radice
NTP non sincronizzato
Soluzione adottata
Aperto UDP 123 verso pool; fallback interno; iburst
Verifica/Esito
Offset < 50ms, Kerberos ok
Parole chiave
NTP offset Kerberos clock skew"""
        },
        {
            "summary": "RADIUS accounting intermittente: pacchetti dalla NAS droppati",
            "desc":
"""Contesto
WLC e server RADIUS centralizzato
Sintomi
Sessioni non contabilizzate
Evidenze/Log
Firewall log drop UDP 1813
Analisi
Policy outbound non include 1813
Causa radice
Regola incompleta
Soluzione adottata
Permesso UDP 1812/1813 bidirezionale; retry 3
Verifica/Esito
Accounting affidabile
Parole chiave
RADIUS accounting UDP 1813"""
        },
        {
            "summary": "Port-channel instabile: LACP rate mismatch e hashing non coerente",
            "desc":
"""Contesto
Aggregazione 2x10G tra core e agg
Sintomi
Flapping e micro-outage
Evidenze/Log
LACP moved to individual; different LACP rate; hash src-dst-ip vs src-mac
Analisi
Parametri non allineati
Causa radice
LACP fast vs normal e per-hash diverso
Soluzione adottata
lacp rate fast su entrambi; port-channel load-balance src-dst-ip
Verifica/Esito
Bundle stabile
Parole chiave
LACP port-channel hashing"""
        },
        {
            "summary": "VRRP flap durante picchi CPU: preempt aggressivo",
            "desc":
"""Contesto
Gateway ridondato VRRP
Sintomi
Default gateway cambia master spesso
Evidenze/Log
VRRP state transitions, CPU>90%
Analisi
Preempt con skew basso
Causa radice
Priorità e preempt non conservative
Soluzione adottata
vrrp preempt delay 60; priority gap 30; track interface
Verifica/Esito
Stabilità ripristinata
Parole chiave
VRRP preempt priority flap"""
        },
        {
            "summary": "LLDP disabilitato impedisce voice VLAN auto-assignment",
            "desc":
"""Contesto
IP phone su switch access
Sintomi
Telefoni finiscono in VLAN dati
Evidenze/Log
LLDP neighbors assenti; CDP non standard
Analisi
Auto-VLAN richiede LLDP-MED
Causa radice
LLDP off
Soluzione adottata
lldp run; lldp med-tlv voice-vlan; policy voice
Verifica/Esito
Phone in VLAN voce
Parole chiave
LLDP LLDP-MED voice VLAN"""
        },
        {
            "summary": "DNSSEC validation fail per skew orario di 5 minuti",
            "desc":
"""Contesto
Unbound come resolver
Sintomi
SERVFAIL per domini con DNSSEC
Evidenze/Log
val-bogus: signature expired; clock fuori sync
Analisi
Time skew rompe validazione
Causa radice
NTP non in sync
Soluzione adottata
Systemd-timesyncd verso pool; makestep
Verifica/Esito
Validation OK
Parole chiave
DNSSEC time skew NTP"""
        },
        {
            "summary": "IPv6 RA guard blocca prefissi legittimi dopo upgrade",
            "desc":
"""Contesto
LAN dual-stack con RA-guard sugli access
Sintomi
Client senza IPv6
Evidenze/Log
Drops su RA guard; MAC gw non in whitelist
Analisi
Firmware cambia signature RA
Causa radice
RA guard profile obsoleto
Soluzione adottata
Aggiornato RA guard bindings; permit su porta uplink
Verifica/Esito
SLAAC ok
Parole chiave
IPv6 RA guard SLAAC"""
        },
        {
            "summary": "PBR non applicata su sottorete specifica: route-map mancante su subinterface",
            "desc":
"""Contesto
Policy routing per uscire verso link backup
Sintomi
Traffico prende sempre default
Evidenze/Log
show route-map counters zero
Analisi
Interfaccia sbagliata legata alla policy
Causa radice
route-map non applicata sulla subif corretta
Soluzione adottata
ip policy route-map BACKUP su subif VLAN X
Verifica/Esito
Traffico match policy
Parole chiave
PBR route-map subinterface"""
        },
        {
            "summary": "Static route recursive failure: next-hop non risolvibile per ARP",
            "desc":
"""Contesto
Statiche verso next-hop fuori subnet
Sintomi
Blackhole
Evidenze/Log
ARP incomplete; route recursive unresolved
Analisi
Manca route verso next-hop
Causa radice
Statiche configurate senza via intermedia
Soluzione adottata
Aggiunta route verso subnet next-hop o interfaccia uscita
Verifica/Esito
Connettività ripristinata
Parole chiave
Static route recursive ARP"""
        },
        {
            "summary": "AnyConnect MTU eccessiva in split-tunnel: frammentazione e stall su SaaS",
            "desc":
"""Contesto
VPN SSL split-tunnel
Sintomi
Siti SaaS lenti/irraggiungibili
Evidenze/Log
PMTUD ICMP bloccato
Analisi
MTU tunnel 1500, MSS eccessivo
Causa radice
Client profile non imposta MTU
Soluzione adottata
Set MTU 1350-1400 su tunnel; sysopt connection tcpmss 1350
Verifica/Esito
SaaS ok
Parole chiave
SSL VPN MTU MSS PMTUD"""
        },
        {
            "summary": "WireGuard dietro NAT con drop periodici: keepalive assente",
            "desc":
"""Contesto
Peer mobile dietro CGNAT
Sintomi
Tunnel si interrompe dopo inattività
Evidenze/Log
No traffic; NAT entry scade
Analisi
Manca keepalive
Causa radice
PersistentKeepalive non configurato
Soluzione adottata
PersistentKeepalive=25 su peer remoti; UDP consentito
Verifica/Esito
Tunnel stabile
Parole chiave
WireGuard NAT keepalive"""
        },
        {
            "summary": "GRE tra sedi blackhole su payload grandi: PMTUD rotto",
            "desc":
"""Contesto
GRE over IPsec
Sintomi
Ping piccoli ok, grandi no
Evidenze/Log
ICMP 3:4 bloccato in path
Analisi
Overhead GRE+IPsec riduce MTU effettiva
Causa radice
Assenza clamp MSS e PMTUD
Soluzione adottata
Clamp MSS 1360; ip mtu 1400 su tunnel; permettere ICMP 3:4
Verifica/Esito
Traffico stabile
Parole chiave
GRE IPsec MTU PMTUD"""
        },
        {
            "summary": "Asymmetric routing causa drop su firewall stateful",
            "desc":
"""Contesto
Due uplink con equal-cost
Sintomi
Connessioni TCP resettate
Evidenze/Log
State miss su ritorno
Analisi
Ritorno esce da link diverso
Causa radice
ECMP senza session stickiness
Soluzione adottata
Policy based routing per stickiness o disable state su subnet note
Verifica/Esito
Connessioni stabili
Parole chiave
Asymmetric routing stateful ECMP"""
        },
        {
            "summary": "MPLS LDP non forma sessione: filtro TCP 646 in core",
            "desc":
"""Contesto
Core MPLS LDPv4
Sintomi
No label binding, solo IGP up
Evidenze/Log
LDP Discovery ok, TCP session fail
Analisi
Firewall inter-vlan blocca 646
Causa radice
Regola mancante
Soluzione adottata
Permesso TCP 646; hello holdtime default
Verifica/Esito
LDP sessione up, label distribuite
Parole chiave
MPLS LDP TCP 646"""
        },
        {
            "summary": "BFD non parte: min-tx/rx troppo bassi per piattaforma",
            "desc":
"""Contesto
BFD su link WAN
Sintomi
Sessione down
Evidenze/Log
BFD diag control packet too slow
Analisi
Timer sotto capacità forwarding
Causa radice
Parametri aggressivi
Soluzione adottata
min-tx/rx 300ms; multiplier 3
Verifica/Esito
BFD up
Parole chiave
BFD timers min-tx rx"""
        },
        {
            "summary": "802.11 roaming scarso: 2.4GHz troppo forte, band-steering inefficace",
            "desc":
"""Contesto
WLAN enterprise
Sintomi
Sticky clients, throughput basso
Evidenze/Log
RSSI alto su AP lontani, 2.4 preferito
Analisi
Soglie e min-RSSI non impostate
Causa radice
Power 2.4 alto, 5GHz debole
Soluzione adottata
Ridotto TX 2.4; min-RSSI -70; band-steering aggressivo; 802.11k/v/r
Verifica/Esito
Roaming fluido
Parole chiave
WiFi band-steering min-RSSI 802.11kvr"""
        },
        {
            "summary": "DHCP starvation da device anomalo: soglia rate-limit assente",
            "desc":
"""Contesto
LAN ufficio
Sintomi
Pool esaurito
Evidenze/Log
DISCOVER burst da MAC singolo
Analisi
Client malevolo o bug
Causa radice
Manca snooping/rate-limit
Soluzione adottata
DHCP snooping; rate-limit 10pps; blocco MAC
Verifica/Esito
Pool stabile
Parole chiave
DHCP snooping starvation rate-limit"""
        },
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
    ap = argparse.ArgumentParser(description="Bootstrap progetto YouTrack Networking")
    ap.add_argument("--short", default="NETS", help="Short name progetto (default NETS)")
    ap.add_argument("--name", default="Networking - Knowledge Cases", help="Nome progetto")
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
