domanda 1:
WAN PPPoE lenta dopo cambio VLAN
breve query: dopo migrazione a pppoe su vlan 835 ho perdite e download che si bloccano
descrizione: dopo passaggio a pppoe su vlan 835 noto packet loss 10–20%, https lento, ping con df e payload grandi falliscono. possibile problema mtu/mss o pmtud?
keywords: pppoe, mtu 1492, mss clamp, pmtud, icmp fragmentation-needed

Risposta openai

Per il problema di WAN PPPoE lenta dopo il cambio a VLAN 835, sembra che tu stia affrontando un problema simile a quello descritto nel ticket [NETS-1].

Analisi e Soluzione Proposta:
MTU: Verifica che l'MTU sulla tua interfaccia PPPoE sia impostato a 1492. Attualmente sembra essere 1500, il che è inappropriato per PPPoE.

Comando da eseguire:
ip link set dev ppp0 mtu 1492

MSS Clamp: Assicurati che il MSS sia clampato per evitare frammentazione. Puoi utilizzare il seguente comando:

Comando da eseguire:
iptables -t mangle -A FORWARD -p tcp -o ppp0 -j TCPMSS --clamp-mss-to-pmtu

ICMP: Controlla che il traffico ICMP non sia bloccato, in particolare i messaggi di "fragmentation needed". Questo è cruciale per il PMTUD.

Comando da eseguire:
iptables -A INPUT -p icmp --icmp-type fragmentation-needed -j ACCEPT

Verifica:
Dopo aver applicato queste modifiche, esegui un test di ping con pacchetti di dimensioni diverse per confermare che il problema di packet loss e stall TCP sia risolto.

Se i problemi persistono, ti consiglio di monitorare ulteriormente il traffico e fornire log dettagliati per ulteriori analisi.

Per ulteriori riferimenti, puoi consultare il ticket [NETS-1] che tratta un problema simile.



domanda 2:
adiacenza ospf non arriva a full
breve query: ospf stuck in exstart/exchange tra due router dopo intervento
descrizione: due router in area 0 non raggiungono stato full, i log oscillano exstart/exchange. su una porta vedo mtu 1500 e sull’altra 1400. che verifiche/config consigli?
keywords: ospf, exstart, exchange, mtu mismatch, mtu-ignore

risposta openai:
Per risolvere il problema di OSPF che rimane in stato EXSTART/EXCHANGE tra i due router, ti consiglio di verificare e uniformare il valore MTU delle interfacce coinvolte. Attualmente hai un MTU di 1500 su una porta e 1400 sull'altra, il che causa un mismatch.

Passi da seguire:

Uniforma il MTU: Imposta il MTU a 1500 su entrambe le interfacce.

Esempio di comando: interface <nome_interfaccia>; mtu 1500.
Configura OSPF per ignorare il MTU (se il vendor lo supporta):

Esempio di comando: ip ospf mtu-ignore.
Verifica lo stato dell'adiacenza: Controlla i log OSPF per confermare che l'adiacenza sia passata a stato FULL.

Questa soluzione è simile a quella adottata nel ticket [NETS-4], dove un mismatch di MTU ha causato un problema analogo. Dopo aver applicato queste modifiche, dovresti vedere un miglioramento nella sincronizzazione della LSDB.


domanda 3:
audio monodirezionale su chiamate voip
breve query: one-way audio con sbc cloud, forse sip alg
descrizione: pbx on-prem collegato a sbc esterno; da quando abbiamo messo nuovo firewall alcune chiamate hanno audio in una sola direzione. vedo sdp riscritti e porte rtp strane. possibile che il sip alg interferisca?
keywords: sip alg, rtp, sdp rewrite, sbc, nat traversal

risposta openai:
Il problema di audio monodirezionale che stai riscontrando potrebbe effettivamente essere causato dal SIP ALG attivo sul nuovo firewall, come evidenziato nel ticket simile [NETS-11]. In quel caso, l'ALG interferiva con la riscrittura del SDP e duplicava la gestione del NAT traversal già effettuata dallo SBC.

Ti consiglio di disabilitare il SIP ALG sul firewall e, se necessario, configurare pin-hole RTP statici o tramite lo SBC. Questo dovrebbe ripristinare l'audio bidirezionale.

