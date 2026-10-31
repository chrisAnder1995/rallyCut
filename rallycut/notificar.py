"""Notificação do Windows (canto da tela) quando um vídeo fica pronto.

Usa o próprio PowerShell do Windows 10/11 — não precisa instalar nada. Clicar na notificação abre o
RallyCut no navegador, já no vídeo certo.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

RAIZ = Path(__file__).resolve().parent.parent
CONFIG = RAIZ / "dados" / "config.json"


def ler_config() -> dict:
    try:
        return {"notificar_windows": True, **json.loads(CONFIG.read_text(encoding="utf-8"))}
    except Exception:
        return {"notificar_windows": True}


def salvar_config(cfg: dict) -> dict:
    atual = ler_config()
    atual.update({k: v for k, v in cfg.items() if k in ("notificar_windows",)})
    CONFIG.parent.mkdir(exist_ok=True)
    CONFIG.write_text(json.dumps(atual), encoding="utf-8")
    return atual


def montar_xml(titulo: str, texto: str, link: str | None = None) -> str:
    abrir = f' activationType="protocol" launch="{escape(link, {chr(34): "&quot;"})}"' if link else ""
    return (f'<toast{abrir}><visual><binding template="ToastGeneric">'
            f'<text>{escape(titulo)}</text><text>{escape(texto)}</text>'
            f'</binding></visual><audio src="ms-winsoundevent:Notification.Default"/></toast>')


def notificar(titulo: str, texto: str, link: str | None = None) -> bool:
    """Mostra a notificação no Windows. Em outros sistemas (ou se desligado) não faz nada."""
    if sys.platform != "win32" or not ler_config().get("notificar_windows", True):
        return False
    xml = montar_xml(titulo, texto, link).replace("'", "''")   # aspas simples escapadas para o PowerShell
    script = (
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null;"
        "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null;"
        "$x = New-Object Windows.Data.Xml.Dom.XmlDocument;"
        f"$x.LoadXml('{xml}');"
        # identificador do próprio PowerShell: já registrado no Windows, então a notificação aparece sem instalar nada
        "$app = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\\WindowsPowerShell\\v1.0\\powershell.exe';"
        "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app).Show("
        "[Windows.UI.Notifications.ToastNotification]::new($x))"
    )
    try:
        subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-Command", script],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return True
    except Exception:
        return False
