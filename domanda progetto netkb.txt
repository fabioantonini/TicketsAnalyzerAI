### 1) GRE over IPsec: PMTUD rotto / MTU

**breve query:** ping grandi in blackhole su GRE over IPsec
**descrizione:** su un tunnel GRE incapsulato in IPsec i ping piccoli passano, mentre payload >1400 si perdono. Navigazione lenta, alcuni servizi HTTPS si bloccano. Sospetto PMTUD bloccato / MSS non clampato.
**contesto:** router Cisco, tunnel `Tunnel0` sopra IPsec site-to-site; firewall intermedi probabilmente filtrano ICMP type 3 code 4.
**evidenze:**

* ping con DF e size 1472 fallisce; size 1200 ok
* nessun ICMP “fragmentation needed” visto in capture
* tcpdump mostra MSS troppo alto verso il tunnel
  **keywords:** gre, ipsec, mtu, pmtud, df, mss, clamp

---

### 2) IPsec dietro NAT: NAT-T mancante

**breve query:** ipsec s2s non sale dietro NAT (solo IKE/500 visibile)
**descrizione:** tunnel site-to-site non negozia o flappa; vedo traffico su UDP/500 ma niente su UDP/4500. L’ISP dice che c’è NAT a monte.
**contesto:** firewall “A” on-prem ↔ firewall “B” cloud; politica restrittiva sulle porte in uscita.
**evidenze:**

* log: “no proposal chosen / NAT detected”
* capture: ESP non passa, assente UDP/4500
* quando forzo NAT-T lato “B”, lo stato rimane “MM\_SA\_SETUP”
  **keywords:** ipsec, nat-t, udp 4500, ike, esp, s2s, firewall

---

### 3) Trunk 802.1Q: Native VLAN mismatch

**breve query:** perdita traffico su trunk multivendor, warning native mismatch
**descrizione:** su un trunk tra switch di vendor diversi alcuni client in VLAN utenti perdono pacchetti. CDP/LLDP segnala “native VLAN mismatch”.
**contesto:** uplink access→aggregation; su uno dei due lati la native è 1, sull’altro 10.
**evidenze:**

* `show interfaces trunk` indica native diversa per lato
* LLDP/CDP warning “native mismatch”
* ping layer-3 intermittente solo per VLAN utenti
  **keywords:** vlan, trunk, 802.1q, native mismatch, lldp, cdp, multivendor

Se il retriever è a posto, dovresti vedere match forti con i tuoi ticket:

* **GRE over IPsec / PMTUD** → caso “GRE over IPsec: payload grandi in blackhole (PMTUD rotto)”
* **IPsec NAT-T** → caso “IPsec site-to-site dietro NAT: NAT-T disabilitato”
* **Native VLAN** → caso “VLAN Native mismatch su trunk 802.1Q causa perdita traffico”


4) SIP ALG one-way audio

breve query: chiamate VoIP con audio monodirezionale dopo inserimento nuovo firewall
descrizione: dopo il cambio firewall le chiamate SIP passano ma spesso l’audio è monodirezionale (solo in uscita). Sospetto che l’ALG SIP stia riscrivendo SDP/RTP rompendo il NAT traversal gestito dallo SBC.
contesto: PBX on-prem, SBC in cloud; firewall appena introdotto tra LAN voce e Internet.
evidenze:

media solo in una direzione; signaling 200 OK regolare

SDP nei pacchetti INVITE/200OK mostra IP/porte RTP diversi lato WAN

disattivando temporaneamente SIP ALG su firewall, una chiamata di prova funziona
keywords: sip, alg, sdp, rtp, one-way audio, sbc, nat traversal

5) DHCP Option 82 (relay)

breve query: client non ottengono IP via DHCP relay; server richiede Option 82
descrizione: su alcune VLAN di accesso i client non ricevono lease. Il server DHCP centrale applica policy basate su Circuit-ID/Remote-ID e sembra scartare richieste senza Option 82.
contesto: switch L3 fa da relay verso server DHCP centrale; uplink L3 verso core.
evidenze:

DISCOVER escono dalla VLAN ma non tornano OFFER

sniff su SVI mostra giunte richieste senza option 82

log server: “missing option 82 – request ignored”
keywords: dhcp, relay, option 82, circuit-id, remote-id, svi, access

6) LACP incoerente (port-channel instabile)

breve query: bundle LACP con flapping e throughput altalenante
descrizione: port-channel a 2×10G tra access-agg/core con continui eventi “individual” sui membri; hash di bilanciamento e rate LACP sembrano diversi sui due lati.
contesto: switch Cisco su entrambi i lati; traffico east-west sensibile.
evidenze:

show etherchannel summary: membri passano up/individual

contatori errore/timeout LACP aumentano durante picchi

lato A lacp rate fast, lato B default; hash src-mac vs src-dst-ip
keywords: lacp, port-channel, hashing, rate fast, individual, etherchannel



Risposta gpt-oss:20b alla domanda 4

Soluzione rapida

Disabilita l’ALG SIP sul firewall

# esempio per un firewall Cisco ASA
no policy-map type inspect sip SIP-ALG
no inspect sip
# per FortiGate
config system settings
set sip-inspection disable
end

Apri il range di porte RTP che l’SBC utilizza

Se l’SBC espone un range statico (es. 10000‑20000), apri quelle porte sul firewall.
Se l’SBC usa porte dinamiche, crea una policy di “pin‑hole” per le porte RTP in uscita (es. inspect sip con sip-udp-port).
Verifica la corretta propagazione degli SDP

Usa Wireshark o tcpdump sul PBX e sull’SBC per controllare che l’IP/porta RTP nei pacchetti INVITE/200 OK non vengano riscritti dal firewall.
Se ancora riscritto, controlla eventuali “NAT‑helper” o “NAT‑policy” che potrebbero sovrascrivere l’ALG.
Test di chiamata

Effettua una chiamata di prova e verifica che l’audio sia bidirezionale.
Se persiste, prova a disabilitare temporaneamente l’SBC (o a usare un altro SBC) per confermare che il problema sia effettivamente l’ALG.
Ticket di riferimento

[NETS-11] SIP one‑way audio: ALG interferisce con SDP in presenza di SBC esterno
[NETKB-3] VoIP one‑way audio con SBC esterno: SIP ALG riscrive SDP/RTP
Entrambi i ticket indicano che la causa radice è l’attivazione dell’ALG SIP, che riscrive l’SDP già gestito dallo SBC, causando porte RTP errate e audio monodirezionale. La soluzione adottata è la disabilitazione dell’ALG e l’apertura del range RTP necessario.

Domande di chiarimento

Quale modello e firmware del firewall è in uso?
L’SBC utilizza porte RTP statiche o dinamiche?
Hai già verificato che l’ALG non sia attivo in modalità “inspect” o “helper” su altri servizi (es. SIP‑UDP 5060)?


7) sto avendo problemi di roaming su protocollo 802.11. cosa posso fare?
Per risolvere i problemi di roaming su protocollo 802.11, ti consiglio di seguire questi passaggi, basati sul ticket simile [NETKB-26]:

Controlla il livello di segnale (RSSI): Assicurati che il segnale degli AP non sia troppo forte per la banda 2.4GHz e che la banda 5GHz non sia troppo debole.

Imposta il min-RSSI: Configura il valore di min-RSSI a -70 dBm per garantire che i client non si connettano a AP lontani con segnale debole.

Regola la potenza di trasmissione: Imposta la potenza di trasmissione per la banda 2.4GHz su un valore basso per bilanciare il segnale.

Abilita 802.11k/v/r: Assicurati che le funzionalità di roaming assistito siano attive per migliorare l'esperienza di roaming.

Verifica le soglie di roaming: Controlla che le soglie di steering siano configurate correttamente.

Dopo aver applicato queste modifiche, verifica se il roaming è migliorato. Se hai bisogno di ulteriori dettagli o assistenza, fammi sapere!
