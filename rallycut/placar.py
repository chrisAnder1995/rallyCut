"""Placar automático.

Regra usada: no vôlei de praia quem ganha o ponto saca o próximo rally. Então basta descobrir
de que lado da rede saiu cada saque (perto ou longe da câmera). A partir disso a página monta o
placar com a regra oficial (sets de 21, troca de lado a cada 7 pontos; set 3 até 15, troca a cada 5).

Lado do saque: a bola sacada do lado de lá vem em direção à câmera e CRESCE na imagem;
a sacada do lado de cá se afasta e ENCOLHE. Quando o movimento é ambíguo, usamos o tamanho.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .detector import Config

LARGURA_PERTO = 0.02   # bola mais larga que 2% da tela no saque => provavelmente lado de cá


def _bolas_moveis(frames: list[dict], cfg: Config) -> list[tuple[float, float, float, float]]:
    """(t, x, y, largura) das bolas em movimento (descarta bolas paradas e detecções sem largura)."""
    ts = np.array([f["t"] for f in frames], dtype=float)
    bolas = [[b for b in f["b"] if b[0] >= cfg.conf_bola and len(b) >= 4] for f in frames]
    saida = []
    for i, bs in enumerate(bolas):
        for b in bs:
            _, x, y, w = b[:4]

            def perto(lo: float, hi: float) -> bool | None:
                js = np.where((ts >= ts[i] + lo) & (ts <= ts[i] + hi))[0]
                if len(js) == 0:
                    return None
                return any(math.hypot(x - o[1], y - o[2]) < cfg.raio_parado for j in js for o in bolas[j])

            antes, depois = perto(-1.2, -0.4), perto(0.4, 1.2)
            if not ((antes is not False) and (depois is not False)):
                saida.append((float(ts[i]), x, y, w))
    return saida


def _inclinacao(ts: list[float], vs: list[float]) -> float:
    """Inclinação robusta (Theil-Sen): mediana das inclinações entre todos os pares."""
    inc = [(vs[j] - vs[i]) / (ts[j] - ts[i]) for i in range(len(ts)) for j in range(i + 1, len(ts))
           if ts[j] - ts[i] > 0.05]
    return float(np.median(inc)) if inc else 0.0


def lado_do_saque(moveis: list[tuple[float, float, float, float]], ini: float, janela: float = 3.5) -> dict:
    """Retorna {"lado": "perto"|"longe"|None, "certeza": "alta"|"baixa"}."""
    pts = [(t, w) for t, x, y, w in moveis if ini <= t <= ini + janela and y < 0.8 and w > 0]
    if not pts:
        return {"lado": None, "certeza": "baixa"}
    pts = pts[:10]
    ts = [p[0] for p in pts]
    lw = [math.log(p[1]) for p in pts]
    if len(pts) >= 3:
        k = _inclinacao(ts, lw)  # variação do log da largura por segundo
        if k > 0.2:
            return {"lado": "longe", "certeza": "alta" if k > 0.4 else "baixa"}
        if k < -0.2:
            return {"lado": "perto", "certeza": "alta" if k < -0.4 else "baixa"}
    w0 = float(np.median([p[1] for p in pts[:3]]))
    return {"lado": "perto" if w0 > LARGURA_PERTO else "longe", "certeza": "baixa"}


def saques(frames: list[dict], segmentos: list[dict], cfg: Config | None = None) -> list[dict]:
    cfg = cfg or Config()
    if not frames or not any(len(b) >= 4 for f in frames for b in f["b"]):
        # análise antiga (sem o tamanho da bola): é preciso analisar o vídeo de novo
        return [{"lado": None, "certeza": "sem_dados"} for _ in segmentos]
    moveis = _bolas_moveis(frames, cfg)
    return [lado_do_saque(moveis, float(s["ini"])) for s in segmentos]


# ---------------------------------------------------------------- arte do placar no vídeo
def _fonte(tamanho: int):
    from PIL import ImageFont
    for nome in ("arialbd.ttf", "Arial Bold.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(nome, tamanho)
        except OSError:
            continue
    return ImageFont.load_default(size=tamanho)


def desenhar_placar(p: dict, altura_video: int, destino: Path) -> Path:
    """Gera um PNG transparente com o placar, no estilo de transmissão, proporcional à altura do vídeo."""
    from PIL import Image, ImageDraw

    e = altura_video / 1080
    fn, fp, fs = _fonte(int(34 * e)), _fonte(int(40 * e)), _fonte(int(22 * e))
    nomes = [str(p.get("nome_a") or "Dupla A")[:18], str(p.get("nome_b") or "Dupla B")[:18]]
    pontos = [str(p.get("a", 0)), str(p.get("b", 0))]
    sacador = p.get("saque")  # "A" | "B" | None
    destaque = p.get("destaque")  # dupla que acabou de pontuar ("A" | "B" | None)

    tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    larg_nome = max(tmp.textlength(n.upper(), font=fn) for n in nomes)
    linha, pad, larg_pt, bola = int(56 * e), int(16 * e), int(64 * e), int(14 * e)
    W = int(pad + bola + pad + larg_nome + pad * 2 + larg_pt)
    H = int(linha * 2 + 36 * e)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = int(12 * e)
    d.rounded_rectangle([0, 0, W - 1, linha * 2], radius=r, fill=(16, 32, 44, 225))
    d.rectangle([W - larg_pt, 0, W - 1, linha * 2], fill=(15, 110, 140, 240))
    d.rounded_rectangle([W - larg_pt, 0, W - 1, linha * 2], radius=r, fill=(15, 110, 140, 240))
    d.rectangle([W - larg_pt, 0, W - larg_pt + r, linha * 2], fill=(15, 110, 140, 240))
    d.line([0, linha, W - 1, linha], fill=(255, 255, 255, 60), width=max(1, int(2 * e)))
    for i in range(2):
        cy = linha * i + linha // 2
        if sacador == "AB"[i]:
            d.ellipse([pad, cy - bola // 2, pad + bola, cy + bola // 2], fill=(232, 176, 75, 255))
        d.text((pad * 2 + bola, cy), nomes[i].upper(), font=fn, fill=(255, 255, 255, 255), anchor="lm")
        if destaque == "AB"[i]:
            d.rectangle([W - larg_pt, linha * i + max(1, int(3 * e)), W - 1 - max(1, int(3 * e)),
                         linha * (i + 1) - max(1, int(3 * e))], fill=(232, 176, 75, 255))
        d.text((W - larg_pt // 2, cy), pontos[i], font=fp,
               fill=(16, 32, 44, 255) if destaque == "AB"[i] else (255, 255, 255, 255), anchor="mm")
    info = f"SET {p.get('set', 1)}"
    if p.get("sets_a") or p.get("sets_b"):
        info += f"   ·   SETS {p.get('sets_a', 0)}–{p.get('sets_b', 0)}"
    d.text((pad, linha * 2 + int(8 * e)), info, font=fs, fill=(255, 255, 255, 230), anchor="lt",
           stroke_width=max(1, int(2 * e)), stroke_fill=(0, 0, 0, 160))
    destino.parent.mkdir(parents=True, exist_ok=True)
    img.save(destino)
    return destino
