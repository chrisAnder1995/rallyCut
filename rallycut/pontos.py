"""Detecção de PONTOS (início no saque, fim quando a bola cai) — algoritmo v3.

Regras (pedidas pelo usuário):
  INÍCIO: o primeiro quadro do movimento de saque (o lançamento da bola), com uma folga curta.
          Nunca a bola cruzando a rede; sem a espera/posicionamento antes.
  FIM:    quando a bola toca o chão / sai / a jogada é interrompida — o momento em que o encerramento
          fica evidente. Nunca só porque a bola está perto do chão: numa defesa a bola desce e VOLTA A SUBIR.
  - acompanha a bola em bloqueios, ataques, recepções e contra-ataques;
  - não mistura dois pontos consecutivos (cada saque abre um ponto novo);
  - não descarta pontos curtos (ace / erro de saque).

Como funciona:
  1. A análise rápida (5 quadros/s) marca as JANELAS de jogo (bola em movimento, tolerando sumiços de até 3 s).
  2. Cada janela é reanalisada a 10 quadros/s.
  3. Na janela, procura um saque; a partir dele procura o fim; depois procura o próximo saque, e assim por diante.
  4. Janela sem saque, mas com bola no ar -> ponto começando 0,6 s antes da recepção (ou da 1ª bola no ar).
"""
from __future__ import annotations

import numpy as np

from .detector import Config, _quadros
from .saque import (RECEPCAO_VOLTA, _prever, _trilha, achar_recepcao, achar_saque, bola_no_ar,
                    calibrar_rede)

FPS_FINO = 10.0
FOLGA_INICIO = 0.6      # antes do lançamento do saque: preserva o movimento completo, sem a espera
FOLGA_FIM = 0.7         # depois da bola cair: o encerramento fica evidente
SOBE_DE_NOVO = 0.12     # a bola voltou a subir de verdade (fração da altura da tela) -> foi defesa, o ponto continua
                        # (bola quicando/rolando na areia sobe bem menos que isso)
HORIZONTE = 0.40        # acima dessa altura da tela a bola está no ar, nunca no chão
JANELA_DEFESA = 1.5     # numa defesa a bola volta a subir em até esse tempo (s)
SUMIU_DE_VEZ = 3.0      # depois de cair, a bola não sobe mais por esse tempo -> ponto encerrado (s)
PAUSA_ENTRE_PONTOS = 4.0  # entre a bola cair e o próximo lançamento de saque passam pelo menos ~4 s


def janelas_de_jogo(frames: list[dict], duracao: float, cfg: Config, gap: float = 3.0) -> list[tuple[float, float]]:
    from .detector import bola_em_movimento
    ts = np.array([f["t"] for f in frames], dtype=float)
    mov = bola_em_movimento(frames, cfg)
    js: list[list[float]] = []
    for t in ts[mov > 0].tolist():
        if js and t - js[-1][1] <= gap:
            js[-1][1] = t
        else:
            js.append([t, t])
    return [(a, b) for a, b in js if b - a >= 1.0]


def calibrar_chao(frames: list[dict], cfg: Config) -> float:
    """Altura (fração da tela) a partir da qual a bola pode estar no chão.

    Bolas PARADAS na areia (reservas, bola morta) mostram onde fica o chão na imagem. Usa a parte mais
    alta dessas posições, com uma margem. Sem bolas paradas suficientes, usa HORIZONTE.
    """
    from .detector import bola_em_movimento
    mov = bola_em_movimento(frames, cfg)
    ys = [b[2] for f, m in zip(frames, mov) if m == 0 for b in f["b"] if b[0] >= cfg.conf_bola]
    if len(ys) < 8:
        return HORIZONTE
    return float(np.clip(np.percentile(ys, 10) - 0.08, HORIZONTE, 0.75))


def achar_fim(tr: list[tuple], desde: float, ate: float, chao: float = HORIZONTE) -> float | None:
    """Momento em que a bola cai e o ponto acaba.

    Candidato: a bola vinha DESCENDO. Só vira fim se, nos SUMIU_DE_VEZ segundos seguintes, a bola não
    voltar a subir (nenhuma aparição mais alta que o ponto de queda). Defesa, levantamento, ataque ou bola
    alta que some por cima do quadro fazem a bola subir de novo -> o ponto continua.
    """
    pts = [p for p in tr if desde <= p[0] <= ate]
    for i, (t, x, y, w) in enumerate(pts):
        antes = [q for q in pts[:i] if t - 0.7 <= q[0] < t]
        if not antes or y - min(q[2] for q in antes) < 0.04:
            continue                                   # não estava descendo
        if y < chao:
            continue                                   # bola acima da altura do chão: está no ar
        if any(q[2] > y + 0.05 for q in pts[i + 1:] if q[0] - t <= 0.6):
            continue                                   # ainda está caindo: o fim é o ponto mais baixo da queda
        # defesa/levantamento: a bola sobe logo (em até JANELA_DEFESA s). Bola recolhida e devolvida demora mais.
        logo = [q for q in pts[i + 1:] if q[0] <= t + JANELA_DEFESA]
        if any(q[2] < y - SOBE_DE_NOVO for q in logo):
            continue                                   # voltou a subir: o ponto continua
        if not logo:
            # sumiu: se reaparece vindo de cima logo em seguida, era bola alta fora do quadro (o jogo continua)
            volta = [q for q in pts[i + 1:] if q[0] <= t + SUMIU_DE_VEZ]
            if volta and volta[0][2] < 0.25:
                continue
        return t
    return None


def _ultima_bola(tr: list[tuple], desde: float, ate: float) -> float | None:
    pts = [p[0] for p in tr if desde <= p[0] <= ate]
    return pts[-1] if pts else None


def _saque_fora_do_quadro(tr: list[tuple], desde: float) -> dict | None:
    """Sacador fora do quadro: a primeira bola vista já está voando com cara de saque.

    Grande e encolhendo (saque do lado de cá) ou pequena e crescendo (lado de lá), com certeza alta.
    """
    from .placar import lado_do_saque
    pts = [p for p in tr if p[0] >= desde]
    if len(pts) < 3:
        return None
    r = lado_do_saque(pts, pts[0][0] - 0.01, janela=2.0)
    if r["certeza"] != "alta":
        return None
    return {"t": pts[0][0], "lado": r["lado"]}


def _lado_pela_bola(tr: list[tuple], desde: float) -> str | None:
    """Saque com o sacador fora do quadro: o lado sai do tamanho da bola nas primeiras aparições.

    Sacada do lado de cá, a bola aparece grande e ENCOLHE; sacada do lado de lá, aparece pequena e CRESCE.
    """
    from .placar import lado_do_saque
    pts = [p for p in tr if p[0] >= desde]
    if len(pts) < 2:
        return None
    r = lado_do_saque(pts, pts[0][0] - 0.01, janela=2.0)
    return r["lado"] if r["certeza"] == "alta" or len(pts) >= 3 else None


def pontos_da_janela(fino: list[dict], a: float, b: float, w_rede: float, cfg: Config,
                     frames_rapidos: list[dict], chao: float = HORIZONTE) -> list[dict]:
    tr = _trilha(fino)
    pontos: list[dict] = []
    t0 = a - 3.5
    while True:
        sq = achar_saque([f for f in fino if f["t"] >= t0], w_rede)
        if not sq or sq["voo"] > b + 1.0:
            break
        if pontos and sq["lancamento"] - pontos[-1]["queda"] < PAUSA_ENTRE_PONTOS:
            # "saque" logo depois do fim de um ponto: é a bola devolvida ao sacador ou o ponto ainda em jogo
            t0 = max(t0 + 0.5, sq["voo"] + 0.5)   # sempre avança (evita laço infinito)
            continue
        fim = achar_fim(tr, sq["voo"] + 0.6, b + 2.5, chao)
        if fim is None:
            fim = _ultima_bola(tr, sq["voo"], b + 2.5) or sq["voo"] + 1.0
        rc = achar_recepcao(tr, w_rede, desde=sq["voo"])
        pontos.append({"lanc": sq["lancamento"], "saque_t": sq["voo"], "saque_lado": sq["lado"],
                       "saque_ref": sq.get("ref"), "queda": round(fim, 2),
                       # ace/erro de saque: sem recepção E o ponto acaba logo depois do saque
                       "obs": None if (rc or fim - sq["voo"] > 3.5) else "saque sem recepção (ace ou erro de saque)"})
        t0 = max(t0 + 0.5, fim + 1.0)
    if not pontos:
        no_ar = bola_no_ar(frames_rapidos, a, b, w_rede, cfg)
        if no_ar is not None:
            rc = achar_recepcao(tr, w_rede)
            fq = None if rc else _saque_fora_do_quadro(tr, a - 0.5)
            if fq:
                # saque com o sacador fora do quadro: a bola entra voando -> o saque foi ~0,6 s antes
                fim = achar_fim(tr, fq["t"] + 0.6, b + 2.5, chao) or _ultima_bola(tr, fq["t"], b + 2.5) or b
                pontos.append({"lanc": None, "saque_t": fq["t"], "inicio_ref": fq["t"] - 0.6 - FOLGA_INICIO,
                               "saque_ref": "fora", "saque_lado": fq["lado"], "queda": round(fim, 2), "obs": None})
                return pontos
            ref_t = rc["t"] if rc else no_ar
            fim = achar_fim(tr, ref_t + 0.3, b + 2.5, chao) or _ultima_bola(tr, ref_t, b + 2.5) or b
            pontos.append({"lanc": None, "saque_t": ref_t, "inicio_ref": ref_t - RECEPCAO_VOLTA,
                           "saque_ref": "recepcao" if rc else "no_ar",
                           "saque_lado": ("longe" if rc["lado"] == "perto" else "perto") if rc else _lado_pela_bola(tr, a - 0.5),
                           "queda": round(fim, 2), "obs": None})
    return pontos


def analisar_pontos(video, frames_rapidos: list[dict], duracao: float, cfg: Config, w: int, h: int,
                    antes_do_saque: float = FOLGA_INICIO, progresso=None) -> list[dict]:
    from ultralytics import YOLO
    from .detector import CLASSE_BOLA, _caminho_modelo

    modelo = YOLO(_caminho_modelo(cfg.modelo))
    w_rede = calibrar_rede(frames_rapidos)
    chao = calibrar_chao(frames_rapidos, cfg)
    janelas = janelas_de_jogo(frames_rapidos, duracao, cfg)
    segs: list[dict] = []
    for k, (a, b) in enumerate(janelas):
        j_ini, j_fim = max(0.0, a - 3.5), min(duracao, b + 2.5)
        fino, lote, tempos = [], [], []
        for t, img in _quadros(video, FPS_FINO, cfg.largura, w, h, inicio=j_ini, duracao=j_fim - j_ini):
            lote.append(img)
            tempos.append(t + j_ini)
            if len(lote) == 8:
                fino += _prever(modelo, lote, tempos, cfg, CLASSE_BOLA)
                lote, tempos = [], []
        if lote:
            fino += _prever(modelo, lote, tempos, cfg, CLASSE_BOLA)
        pts = pontos_da_janela(fino, a, b, w_rede, cfg, frames_rapidos, chao)
        if not pts:
            segs.append({"ini": round(max(0.0, a - 1.0), 2), "fim": round(min(duracao, b + 1.0), 2), "ativo": False,
                         "motivo": "sem saque e sem bola no ar (bola sendo passada ou buscada)"})
        for p in pts:
            ini = (p["lanc"] - antes_do_saque) if p.get("lanc") is not None else p["inicio_ref"]
            seg = {"ini": round(max(0.0, ini), 2), "fim": round(min(duracao, p["queda"] + FOLGA_FIM), 2),
                   "ativo": True, "voo_fim": p["queda"], "saque_t": p["saque_t"], "lanc": p.get("lanc"),
                   "saque_lado": p.get("saque_lado"), "saque_ref": p.get("saque_ref")}
            if p.get("obs"):
                seg["obs"] = p["obs"]
            segs.append(seg)
        if progresso:
            progresso((k + 1) / len(janelas))
    return _limpar(_juntar_pedacos(segs))


JUNTA_SE = 1.5   # pedaço que começa até 1,5 s depois do fim do anterior é o MESMO rally (não é saque novo)
SAIU_DO_QUADRO = 8.0   # pedaço SEM saque até 8 s depois do rally anterior: a bola saiu do quadro e o ponto continuou
SERVE_COMO_SAQUE = ("lancamento", "rede", "recepcao", "fora")


def _juntar_pedacos(segs: list[dict]) -> list[dict]:
    """Um rally partido em dois não pode virar dois pontos.

    Se um ponto começa colado no anterior (até JUNTA_SE s) ou o "saque" dele vem menos de PAUSA_ENTRE_PONTOS s
    depois da bola cair no anterior, é continuação da mesma jogada: junta os dois e mantém o saque do primeiro.
    """
    segs = sorted(segs, key=lambda s: s["ini"])
    saida: list[dict] = []
    for s in segs:
        ant = next((x for x in reversed(saida) if x.get("ativo")), None)
        if s.get("ativo") and s.get("saque_ref") not in SERVE_COMO_SAQUE:
            # sem saque detectado: um rally novo só começa no saque
            if ant and s["ini"] - ant["fim"] <= SAIU_DO_QUADRO:
                ant["fim"] = max(ant["fim"], s["fim"])     # a bola saiu do quadro e voltou: o ponto continua
                ant["voo_fim"] = max(ant.get("voo_fim") or 0, s.get("voo_fim") or 0)
                ant.pop("obs", None)
                continue
            s = {**s, "ativo": False, "motivo": "sem saque detectado (o rally só começa no saque)"}
        if ant and ant.get("ativo") and s.get("ativo"):
            colado = s["ini"] - ant["fim"] <= JUNTA_SE
            t_saque = s.get("lanc") if s.get("lanc") is not None else s.get("saque_t")
            cedo = t_saque is not None and ant.get("voo_fim") is not None and t_saque - ant["voo_fim"] < PAUSA_ENTRE_PONTOS
            if colado or cedo:
                ant["fim"] = max(ant["fim"], s["fim"])
                ant["voo_fim"] = max(ant.get("voo_fim") or 0, s.get("voo_fim") or 0)
                ant.pop("obs", None)        # teve continuação: não foi ace
                continue
        saida.append(s)
    return saida


def _limpar(segs: list[dict]) -> list[dict]:
    """Ordena e garante que nenhum ponto invada o seguinte (sem misturar pontos consecutivos)."""
    segs = sorted(segs, key=lambda s: s["ini"])
    for a, b in zip(segs, segs[1:]):
        if a["fim"] > b["ini"]:
            if a.get("ativo") and b.get("ativo"):
                a["fim"] = round(max(a["ini"] + 0.5, b["ini"]), 2)
            elif not b.get("ativo"):
                b["ini"] = a["fim"]
            else:
                a["fim"] = b["ini"]
    return [s for s in segs if s["fim"] - s["ini"] >= 0.3]
