"""Detecção exaustiva: 2ª passagem (revisão dos intervalos) e relatório.

Prioridade: NÃO PERDER PONTOS. Um evento só é descartado depois de investigado; o que não dá para
confirmar fica marcado para revisão manual em vez de sumir.

Sinais usados além da bola:
  * MOVIMENTO DOS JOGADORES — a velocidade das pessoas na quadra (em alturas de corpo por segundo).
    Num ponto todo mundo corre, salta e se desloca; entre os pontos andam devagar ou ficam parados.
    Isso acha jogadas mesmo quando a bola não aparece (fora do quadro, contra o sol, muito longe).
"""
from __future__ import annotations

import numpy as np

JANELA_SUAVE = 5           # quadros (a 5 q/s = 1 s) para suavizar a movimentação
MIN_DUR_CANDIDATO = 2.5    # s de movimentação forte para virar um possível ponto
FOLGA_PONTO = 2.0          # s em volta de um ponto já detectado que não contam como intervalo
INTERVALO_LONGO = 45.0     # s: intervalo maior que isso entre pontos é revisado com atenção
PONTO_LONGO = 45.0         # s: clipe maior que isso pode ter dois pontos juntos


def movimento_jogadores(frames: list[dict]) -> list[float]:
    """Velocidade média dos jogadores mais rápidos em cada quadro (alturas de corpo por segundo).

    Liga cada pessoa à mais próxima no quadro anterior e mede o deslocamento dividido pela altura dela
    (assim quem está longe da câmera pesa igual a quem está perto). Pega a média das 4 maiores
    velocidades (os 4 jogadores); espectadores parados não contam.
    """
    mov = [0.0] * len(frames)
    for i in range(1, len(frames)):
        ant, at = frames[i - 1].get("_p") or [], frames[i].get("_p") or []
        dt = max(1e-3, frames[i]["t"] - frames[i - 1]["t"])
        if not ant or not at:
            continue
        vs = []
        for x, y, h in at:
            if h < 0.03 or h > 0.6 or y > 0.98:
                continue                         # muito longe, ou alguém bem na frente da câmera / cortado
            d = min(((x - a) ** 2 + (y - b) ** 2) ** 0.5 for a, b, _ in ant)
            if d < 0.8 * h:                      # mesma pessoa (não pulou meio corpo num quadro)
                vs.append(d / h / dt)
        if vs:
            vs.sort(reverse=True)
            mov[i] = float(np.mean(vs[:4]))
    if len(mov) >= JANELA_SUAVE:
        k = np.ones(JANELA_SUAVE) / JANELA_SUAVE
        mov = np.convolve(np.array(mov), k, mode="same").tolist()
    return mov


def limiar_movimento(ts: np.ndarray, mov: np.ndarray, pontos: list[tuple[float, float]]) -> float | None:
    """Calibra no próprio vídeo: meio caminho entre o movimento DURANTE os pontos e FORA deles."""
    if not len(mov) or not pontos:
        return None
    dentro = np.zeros(len(ts), dtype=bool)
    for a, b in pontos:
        dentro |= (ts >= a) & (ts <= b)
    if dentro.sum() < 10 or (~dentro).sum() < 10:
        return None
    m_in, m_out = float(np.median(mov[dentro])), float(np.median(mov[~dentro]))
    if m_in < m_out * 1.25:          # o sinal não separa jogo de pausa neste vídeo
        return None
    return (m_in + m_out) / 2


def candidatos(frames: list[dict], mov: list[float], segs: list[dict], duracao: float) -> list[dict]:
    """Trechos FORA dos pontos já detectados com sinais de jogo (jogadores correndo e/ou bola voando).

    Retorna [{"a", "b", "mov": fração do tempo com movimento forte, "bola": nº de bolas em movimento}].
    """
    from .detector import Config, bola_em_movimento
    ts = np.array([f["t"] for f in frames], dtype=float)
    mv = np.array(mov, dtype=float)
    ativos = [(s["ini"], s["fim"]) for s in segs if s.get("ativo")]
    thr = limiar_movimento(ts, mv, ativos)
    coberto = np.zeros(len(ts), dtype=bool)
    for a, b in ativos:
        coberto |= (ts >= a - FOLGA_PONTO) & (ts <= b + FOLGA_PONTO)
    bola = bola_em_movimento(frames, Config()) > 0
    forte = (mv > thr) if thr is not None else np.zeros(len(ts), dtype=bool)
    sinal = (forte | bola) & ~coberto
    out: list[dict] = []
    i = 0
    while i < len(ts):
        if not sinal[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(ts) and (sinal[j + 1] or (j + 3 < len(ts) and sinal[j + 2:j + 4].any())):
            j += 1                                    # tolera falhas curtas (até ~0,6 s)
        a, b = float(ts[i]), float(ts[j])
        sel = slice(i, j + 1)
        n_bola = int(bola[sel].sum())
        f_mov = float(forte[sel].mean()) if thr is not None else 0.0
        if b - a >= MIN_DUR_CANDIDATO and (f_mov >= 0.5 or n_bola >= 3):
            out.append({"a": round(a, 2), "b": round(b, 2), "mov": round(f_mov, 2), "bola": n_bola})
        i = j + 1
    return out


def relatorio(segs: list[dict], mov: list[float], frames: list[dict], duracao: float) -> dict:
    """Resumo pedido: totais, início inferido, baixa confiança, possíveis perdidos e intervalos a revisar."""
    ativos = sorted([s for s in segs if s.get("ativo")], key=lambda s: s["ini"])
    perdidos = [s for s in segs if not s.get("ativo") and s.get("revisar")]
    intervalos = []
    for a, b in zip(ativos, ativos[1:]):
        gap = b["ini"] - a["fim"]
        if gap >= INTERVALO_LONGO:
            intervalos.append({"de": a["fim"], "ate": b["ini"], "motivo": f"intervalo longo ({gap:.0f} s) sem ponto detectado"})
    for s in ativos:
        if s["fim"] - s["ini"] >= PONTO_LONGO:
            intervalos.append({"de": s["ini"], "ate": s["fim"], "motivo": "clipe longo: verificar se há dois pontos juntos"})
    return {
        "total": len(ativos),
        "inicio_inferido": sum(1 for s in ativos if s.get("inicio_inferido")),
        "baixa_confianca": sum(1 for s in ativos if s.get("confianca") == "baixa"),
        "fim_a_confirmar": [{"ini": s["ini"], "fim": s["fim"], "motivo": s.get("fim_motivo", "")}
                            for s in ativos if s.get("fim_status") == "a_confirmar"],
        "possiveis_perdidos": [{"ini": s["ini"], "fim": s["fim"], "motivo": s.get("revisar")} for s in perdidos],
        "intervalos": sorted(intervalos, key=lambda x: x["de"]),
    }
