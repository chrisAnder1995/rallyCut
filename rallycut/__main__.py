"""Inicia o RallyCut: python -m rallycut"""
import threading
import webbrowser

import uvicorn

PORTA = 8765


def main() -> None:
    url = f"http://127.0.0.1:{PORTA}"
    print(f"\n  RallyCut rodando em {url}\n  (feche esta janela para encerrar)\n")
    threading.Timer(2.0, lambda: webbrowser.open(url)).start()
    uvicorn.run("rallycut.app:app", host="127.0.0.1", port=PORTA, log_level="warning")


if __name__ == "__main__":
    main()
