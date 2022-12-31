netsh interface portproxy add v4tov4 listenport=7077 listenaddress=0.0.0.0 connectport=7077 connectaddress=ip_vm_wsl
netsh interface portproxy add v4tov4 listenport=20002 listenaddress=0.0.0.0 connectport=20002 connectaddress=ip_vm_wsl
netsh interface portproxy add v4tov4 listenport=6060 listenaddress=0.0.0.0 connectport=6060 connectaddress=ip_vm_wsl

Rem netsh interface portproxy reset
Rem netsh interface portproxy show v4tov4