@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul


REM ============================================================
REM  Aria - Configuration acces LAN au conteneur
REM  ----------------------------------------------------------
REM  - Detecte automatiquement l'IP LAN (adaptateur actif, hors WSL)
REM  - (Re)configure netsh portproxy : IP_LAN:PORT -> 127.0.0.1:PORT
REM  - (Re)cree la regle de pare-feu Windows pour le port
REM  - Verifie iphlpsvc, proxy systeme ; rappelle antivirus / Wi-Fi / micro
REM
REM  Le conteneur (podman) publie deja 7860 sur le localhost
REM  Windows ; portproxy expose ce port sur l'IP LAN.
REM
REM  A relancer si l'IP change (DHCP). Le script est idempotent.
REM ============================================================


set "PORT_WEB=7860"
set "CONNECT=127.0.0.1"
set "RULE_NAME=Aria LAN %PORT_WEB%"


REM --- Elevation administrateur (portproxy + pare-feu l'exigent) ---
net session >nul 2>&1
if %errorlevel% neq 0 (
  echo [i] Elevation des privileges ^(administrateur^)...
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)


echo ============================================================
echo   Configuration acces LAN - Aria
echo ============================================================
echo.


REM --- 1. Detection de l'IP LAN (adaptateur UP avec passerelle, hors WSL/vEthernet) ---
set "HOSTIP="
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "$c=Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway -and $_.NetAdapter.Status -eq 'Up' -and $_.InterfaceAlias -notlike '*WSL*' -and $_.InterfaceAlias -notlike '*vEthernet*' -and $_.InterfaceAlias -notlike '*Loopback*' } | Select-Object -First 1; if($c){$c.IPv4Address.IPAddress}"`) do set "HOSTIP=%%i"


if not defined HOSTIP (
  echo [!] Impossible de detecter l'IP LAN automatiquement.
  echo     Verifiez la connexion reseau ^(Wi-Fi / Ethernet^).
  echo.
  pause
  exit /b 1
)


echo [i] IP LAN detectee : %HOSTIP%
echo.


REM --- 2. Nettoyage des anciennes redirections portproxy pour ce port (evite IP perimee) ---
echo [i] Nettoyage des anciennes redirections portproxy (%PORT_WEB%)...
powershell -NoProfile -Command "netsh interface portproxy show v4tov4 | Select-String '^\s*\d' | ForEach-Object { $p = ($_ -split '\s+') | Where-Object { $_ }; if ($p.Count -ge 2 -and $p[1] -eq '%PORT_WEB%') { netsh interface portproxy delete v4tov4 listenport=$($p[1]) listenaddress=$($p[0]) | Out-Null } }"


REM --- 3. Ajout de la redirection IP_LAN:PORT -> 127.0.0.1:PORT ---
echo [i] portproxy %HOSTIP%:%PORT_WEB% -^> %CONNECT%:%PORT_WEB%
netsh interface portproxy add v4tov4 listenport=%PORT_WEB% listenaddress=%HOSTIP% connectport=%PORT_WEB% connectaddress=%CONNECT%
echo.


REM --- 4. Regle pare-feu Windows Defender (idempotent : delete puis add) ---
echo [i] Regle pare-feu Windows Defender...
netsh advfirewall firewall delete rule name="%RULE_NAME%" >nul 2>&1
netsh advfirewall firewall add rule name="%RULE_NAME%" dir=in action=allow protocol=TCP localport=%PORT_WEB% >nul
echo [i] Port TCP %PORT_WEB% autorise en entree.
echo.


REM --- 5. Verifications informatives (non bloquantes) ---
echo ============================================================
echo   Redirections portproxy actives :
echo ============================================================
netsh interface portproxy show v4tov4
echo.


echo [i] Service IP Helper (iphlpsvc, requis par portproxy) :
sc query iphlpsvc | findstr /i "STATE"
echo.


echo [i] Proxy systeme WinHTTP (pour info) :
netsh winhttp show proxy
echo.


echo [!] Antivirus tiers : ce script ne configure QUE le pare-feu Windows.
echo     Si un antivirus/pare-feu tiers est installe, autorisez-y manuellement
echo     le port TCP %PORT_WEB% en entree.
echo.
echo [!] Wi-Fi : si un PC distant ne joint pas le serveur, desactivez
echo     l'isolation des clients (AP isolation / mode invite) sur la box,
echo     et verifiez que les 2 PC sont sur le meme sous-reseau.
echo.
echo [!] Micro : un navigateur n'autorise le micro que sur une page securisee
echo     (HTTPS ou localhost). Pour la voix depuis un autre PC ou un smartphone :
echo       1. dans configs/default.json, section "https" :
echo            "enabled": true, "hosts": ["%HOSTIP%"]
echo       2. redemarrer Aria ^(le certificat est cree au demarrage^)
echo       3. ouvrir https://%HOSTIP%:%PORT_WEB% et accepter l'avertissement
echo          du navigateur ^(certificat auto-signe^).
echo     Sans HTTPS, l'interface et le texte saisi fonctionnent, pas le micro.
echo.


echo ============================================================
echo   PRET. Depuis un autre PC du reseau local, ouvrez :
echo.
echo        http://%HOSTIP%:%PORT_WEB%    ^(texte seulement^)
echo        https://%HOSTIP%:%PORT_WEB%   ^(micro, si "https" est active^)
echo.
echo ============================================================
echo.
pause
endlocal
