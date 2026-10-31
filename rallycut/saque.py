"""Refino do início dos rallys: começar o corte pouco antes do movimento de saque.

A detecção principal roda a 5 quadros/s e às vezes pega o rally um pouco tarde (o lançamento do
saque é rápido e a bola sai do quadro por cima) ou cedo demais (gente batendo bola, buscando a bola
e passando para o sacador antes do saque). Aqui reanalisamos a ~15 quadros/s alguns segundos em volta
do começo de cada rally e procuramos:

  * o VOO do saque: a primeira sequência longa em que a bola anda e muda de tamanho sempre no mesmo
    sentido (cresce = veio do lado de lá; encolhe = saiu do lado de cá);
  * o LANÇAMENTO: a bola vista logo antes desse voo (até 1,5 s antes).

O corte passa a começar `antes_do_saque` segundos antes do lançamento.
"""
from __future__ import annotations

import math

import numpy as np

from .detector import Config, _quadros


LARGURA_REDE_PADRAO = 0.0175


def calibrar_rede(frames: list[dict]) -> float:
    """Largura aparente (fração da tela) de uma bola passando pelo plano da rede.

    Com a câmera parada, a distância até a rede é ~constante, então toda bola cruzando a rede aparece
    com o mesmo tamanho. As bolas no ar formam dois grupos (quadra de lá: pequenas; de cá: grandes);
    o limite entre eles, na escala logarítmica, é o tamanho da bola na rede.
    """
    ws = [b[3] for f in frames for b in f["b"] if len(b) >= 4 and b[0] >= 0.4 and b[2] < 0.62 and b[3] > 0]
    if len(ws) < 40:
        return LARGURA_REDE_PADRAO
    lw = np.log(np.array(ws))
    c1, c2 = np.percentile(lw, 25), np.percentile(lw, 85)
    for _ in range(30):  # k-means com 2 grupos
        g = np.abs(lw - c1) <= np.abs(lw - c2)
        if g.all() or (~g).all():
            break
        c1, c2 = lw[g].mean(), lw[~g].mean()
    if c2 - c1 < 0.35:   # grupos mal separados: usa o padrão
        return LARGURA_REDE_PADRAO
    return float(np.clip(np.exp((c1 + c2) / 2), 0.010, 0.030))


def travessia_da_rede(tr: list[tuple], w_rede: float, desde: float = -1e9) -> dict | None:
    """Primeira vez em que a bola, voando, cruza o plano da rede (o tamanho passa pelo da rede)."""
    for i in range(len(tr) - 1):
        t0, x0, y0, w0 = tr[i]
        t1, x1, y1, w1 = tr[i + 1]
        if t0 < desde or t1 - t0 > 0.8 or y0 > 0.62 or y1 > 0.62:
            continue
        if (w0 - w_rede) * (w1 - w_rede) > 0 or w0 == w1:
            continue
        if math.hypot(x1 - x0, y1 - y0) / max(t1 - t0, 0.05) < 0.25:   # tem que estar voando
            continue
        # confirma: depois de cruzar, a bola segue do outro lado da rede por um tempo
        depois = [p for p in tr[i + 1:i + 6] if p[0] - t1 <= 0.8]
        lado = np.sign(w1 - w_rede)
        if len(depois) >= 2 and sum(np.sign(p[3] - w_rede) == lado for p in depois) < len(depois) * 0.6:
            continue
        f = (math.log(w_rede) - math.log(w0)) / (math.log(w1) - math.log(w0))
        return {"t": round(t0 + f * (t1 - t0), 2), "lado": "perto" if w1 < w0 else "longe"}
    return None


def achar_recepcao(tr: list[tuple], w_rede: float, desde: float = -1e9) -> dict | None:
    """Primeira recepção: a bola vem descendo e, num toque, passa a subir (vira para cima na tela).

    Retorna {"t": momento do toque, "lado": lado de quem RECEBEU ("perto"/"longe")} ou None.
    """
    pts = [p for p in tr if p[0] >= desde]
    for i in range(1, len(pts) - 1):
        t, x, y, w = pts[i]
        if y > 0.85:                       # bola quicando na areia bem perto da câmera
            continue
        antes = [p for p in pts[:i] if t - 0.6 <= p[0] < t]
        depois = [p for p in pts[i + 1:] if t < p[0] <= t + 0.6]
        if len(antes) < 2 or len(depois) < 2:
            continue
        # continuidade (sem saltos de mais de ~1,3 tela/s entre pontos vizinhos)
        seq = antes + [pts[i]] + depois
        if any(math.hypot(b[1] - a[1], b[2] - a[2]) > 1.3 * (b[0] - a[0]) + 0.04 or b[0] - a[0] > 0.35
               for a, b in zip(seq, seq[1:])):
            continue
        desce = y - min(p[2] for p in antes)      # quanto desceu na tela antes do toque
        sobe = y - min(p[2] for p in depois)      # quanto subiu depois
        if not (desce >= 0.05 and sobe >= 0.05 and y >= max(p[2] for p in antes) - 0.01
                and y >= max(p[2] for p in depois) - 0.01):
            continue
        # a bola tem que estar VINDO de lá para quem recebe: crescendo (recepção do lado de cá)
        # ou encolhendo (do lado de lá). Quicar a bola na mão não muda o tamanho de forma consistente.
        lado = "perto" if w > w_rede else "longe"
        vinda = [p for p in pts[:i + 1] if t - 1.0 <= p[0] <= t]
        k = float(np.polyfit([p[0] for p in vinda], [math.log(p[3]) for p in vinda], 1)[0]) if len(vinda) >= 3 else 0.0
        vindo = (lado == "perto" and k >= 0.15) or (lado == "longe" and k <= -0.15)
        cruzou = any((p[3] > w_rede) != (w > w_rede) for p in pts[:i] if t - 1.5 <= p[0] < t and p[2] < 0.62)
        if not (vindo or cruzou):
            continue
        return {"t": round(t, 2), "lado": lado}
    return None


def bola_no_ar(frames: list[dict], ini: float, fim: float, w_rede: float, cfg: Config) -> float | None:
    """'Bola no ar' = bola voando de verdade durante o trecho: cruzou o plano da rede pelo menos uma vez
    ou foi vista alta na tela (acima de ~1/3 da altura) pelo menos 2 vezes. Bola passada, chutada,
    quicada ou recolhida não passa nesse teste. Retorna o momento em que a bola foi vista no ar, ou None."""
    from .detector import bola_em_movimento
    ts = np.array([f["t"] for f in frames], dtype=float)
    mov = bola_em_movimento(frames, cfg)
    pts = []
    for i in np.where((ts >= ini) & (ts <= fim) & (mov > 0))[0]:
        bs = [b for b in frames[i]["b"] if b[0] >= cfg.conf_bola and len(b) >= 4]
        if bs:
            b = max(bs, key=lambda b: b[0])
            pts.append((float(ts[i]), b[1], b[2], b[3]))
    altos = [p for p in pts if p[2] < 0.35]
    cruzamentos = [q for p, q in zip(pts, pts[1:])
                   if p[2] < 0.62 and q[2] < 0.62 and (p[3] > w_rede) != (q[3] > w_rede)]
    if cruzamentos or len(altos) >= 2:
        return min([p[0] for p in altos[:1]] + [q[0] for q in cruzamentos[:1]])
    return None


def _trilha(frames: list[dict], conf_min: float = 0.3, raio_parado: float = 0.015) -> list[tuple]:
    """Uma bola por quadro (a mais confiável que não esteja parada): lista de (t, x, y, w)."""
    ts = np.array([f["t"] for f in frames], dtype=float)
    cand = [[b for b in f["b"] if b[0] >= conf_min and len(b) >= 4] for f in frames]
    trilha = []
    for i, bs in enumerate(cand):
        moveis = []
        for b in bs:
            _, x, y, w = b[:4]
            js = np.where((np.abs(ts - ts[i]) >= 0.6) & (np.abs(ts - ts[i]) <= 1.5))[0]
            parada = sum(1 for j in js for o in cand[j] if math.hypot(x - o[1], y - o[2]) < raio_parado) >= 3
            if not parada:
                moveis.append(b)
        if moveis:
            b = max(moveis, key=lambda b: b[0])
            trilha.append((float(ts[i]), b[1], b[2], b[3]))
    return trilha


# tempos típicos medidos nos vídeos: o toque do saque vem ~0,6 s depois do lançamento e
# a bola cruza a rede ~0,75 s depois do toque
TOQUE_ATE_REDE = 0.75
RECEPCAO_VOLTA = 0.6    # sem saque visível: o rally começa esse tanto antes da recepção
LANCAMENTO_ATE_TOQUE = 0.6


def achar_saque(frames: list[dict], w_rede: float | None = None) -> dict | None:
    """Procura o saque. Retorna {"lancamento", "voo", "lado", "ref"} ou None.

    ref = "lancamento": viu a bola subir e o voo do saque.
    ref = "rede": o sacador estava fora do quadro; usou a passagem da bola pela rede como referência.
    """
    tr = _trilha(frames)
    voo = _primeiro_voo(tr)
    rede = travessia_da_rede(tr, w_rede) if w_rede else None
    # a primeira passagem pela rede vem bem antes do voo achado? então o saque foi essa passagem
    # (o voo achado já era a recepção/ataque) — típico quando o sacador está fora do quadro
    if rede and (voo is None or rede["t"] < voo["voo"] - 0.3):
        toque = rede["t"] - TOQUE_ATE_REDE
        return {"lancamento": round(toque - LANCAMENTO_ATE_TOQUE, 2), "voo": round(toque, 2),
                "lado": rede["lado"], "ref": "rede"}
    return voo


def _primeiro_voo(tr: list[tuple]) -> dict | None:
    """Primeira sequência em que a bola anda rápido e muda de tamanho num só sentido (voo longo)."""
    n = len(tr)
    for i in range(n):
        if tr[i][2] > 0.62:          # voo de saque começa no alto (bola rolando na areia não conta)
            continue
        run = [tr[i]]
        for j in range(i + 1, n):
            t, x, y, w = tr[j]
            pt, px, py, pw = run[-1]
            dt = t - pt
            if dt > 0.45:
                break
            if math.hypot(x - px, y - py) > 1.3 * dt + 0.03:   # bola não anda mais que ~1,3 tela/s
                break
            run.append(tr[j])
        run = [r for r in run if r[0] - run[0][0] <= 1.0]   # só o começo do voo
        if len(run) < 3 or run[-1][0] - run[0][0] < 0.25:
            continue
        dur = run[-1][0] - run[0][0]
        lw = [math.log(r[3]) for r in run]
        k = float(np.polyfit([r[0] for r in run], lw, 1)[0])   # variação do tamanho (log) por segundo
        mudanca = lw[-1] - lw[0]
        velocidade = math.hypot(run[-1][1] - run[0][1], run[-1][2] - run[0][2]) / dur
        # voo: anda rápido e muda de tamanho num só sentido (bola parada na mão ou carregada não)
        if velocidade >= 0.3 and abs(k) >= 0.4 and abs(mudanca) >= 0.2 and np.sign(k) == np.sign(mudanca):
            voo = run[0][0]
            antes = [p for p in tr if voo - 1.5 <= p[0] < voo - 0.05]
            lanc = antes[0][0] if antes else voo - LANCAMENTO_ATE_TOQUE
            return {"lancamento": round(lanc, 2), "voo": round(voo, 2),
                    "lado": "longe" if k > 0 else "perto", "ref": "lancamento"}
    return None


def refinar_inicios(video, segmentos: list[dict], duracao: float, cfg: Config, w: int, h: int,
                    antes_do_saque: float = 1.5, progresso=None, frames_rapidos: list[dict] | None = None) -> list[dict]:
    """Ajusta o início de cada segmento para `antes_do_saque` s antes do lançamento do saque."""
    from ultralytics import YOLO
    from .detector import _caminho_modelo, CLASSE_BOLA

    modelo = YOLO(_caminho_modelo(cfg.modelo))
    fps = 15.0
    w_rede = calibrar_rede(frames_rapidos or [])
    for k, s in enumerate(segmentos):
        primeira = s["ini"] + cfg.folga_antes            # primeira bola em movimento da análise rápida
        j_ini = max(0.0, primeira - 3.5)
        j_fim = min(duracao, primeira + 10.0, s["fim"])
        if j_fim - j_ini < 1.0:
            continue
        frames = []
        lote, tempos = [], []
        for t, img in _quadros(video, fps, cfg.largura, w, h, inicio=j_ini, duracao=j_fim - j_ini):
            lote.append(img)
            tempos.append(t + j_ini)
            if len(lote) == 8:
                frames += _prever(modelo, lote, tempos, cfg, CLASSE_BOLA)
                lote, tempos = [], []
        if lote:
            frames += _prever(modelo, lote, tempos, cfg, CLASSE_BOLA)
        # Rally válido = tem SAQUE ou tem BOLA NO AR. O começo:
        #   1) viu o saque (lançamento, voo ou bola passando na rede) -> começa pelo saque;
        #   2) sem saque, mas com a bola no ar -> começa 0,6 s antes da recepção (ou da 1ª bola no ar)
        #      e conta como um saque do lado oposto ao de quem recebeu;
        #   3) nem saque nem bola no ar -> não é rally (bola sendo passada/buscada): fica desmarcado.
        anterior = segmentos[k - 1]["fim"] if k > 0 else 0.0
        sq = achar_saque(frames, w_rede)
        if sq:
            # no máximo 3 s antes do que a análise rápida achou (evita engolir o fim do rally anterior)
            novo = max(anterior, 0.0, s["ini"] - 3.0, sq["lancamento"] - antes_do_saque)
            if novo < s["fim"] - 2.0:
                s["ini_detectado"] = s["ini"]
                s["ini"] = round(novo, 2)
            s.update(saque_t=sq["voo"], lanc=sq["lancamento"], saque_lado=sq["lado"], saque_ref=sq.get("ref"))
            # saque sem recepção (ace ou erro de saque) continua valendo: o ponto vai para quem sacar o próximo
            if not achar_recepcao(_trilha(frames), w_rede, desde=sq["voo"]):
                s["obs"] = "saque sem recepção (ace ou erro de saque)"
        else:
            no_ar = bola_no_ar(frames_rapidos or frames, s["ini"], s["fim"], w_rede, cfg)
            if no_ar is None:
                s.update(ativo=False, motivo="sem saque e sem bola no ar (bola sendo passada ou buscada)")
            else:
                rc = achar_recepcao(_trilha(frames), w_rede)
                ref_t = rc["t"] if rc else no_ar
                novo = max(anterior, 0.0, ref_t - RECEPCAO_VOLTA)
                if novo < s["fim"] - 2.0:
                    s["ini_detectado"] = s["ini"]
                    s["ini"] = round(novo, 2)
                s.update(saque_t=ref_t, lanc=None, saque_ref="recepcao" if rc else "no_ar",
                         saque_lado=("longe" if rc["lado"] == "perto" else "perto") if rc else None)
        if progresso:
            progresso((k + 1) / len(segmentos))
    return juntar_continuacoes(segmentos, frames_rapidos, cfg)


PAUSA_ANTES_DO_SAQUE = 0.8   # o sacador segura/quica a bola antes de lançar: a bola fica parada esse tanto


def juntar_continuacoes(segmentos: list[dict], frames: list[dict] | None = None, cfg: Config | None = None) -> list[dict]:
    """Um 'saque' que vem com a bola ainda em jogo (sem nenhuma pausa antes do lançamento) é, na verdade,
    o mesmo rally (um ataque confundido com saque). Junta os dois para não contar um ponto a mais.

    Importante: o tempo desde o fim do rally anterior NÃO decide — a bola devolvida ao sacador também se
    move e "estica" o rally anterior até perto do saque. O que decide é se a bola PAROU antes do lançamento.
    """
    segmentos = sorted(segmentos, key=lambda s: s["ini"])
    voando = None
    if frames:
        from .detector import bola_em_movimento
        ts = np.array([f["t"] for f in frames], dtype=float)
        voando = ts[bola_em_movimento(frames, cfg or Config()) > 0]
    saida: list[dict] = []
    for s in segmentos:
        ant = next((p for p in reversed(saida) if p.get("ativo", True)), None)
        if ant is not None and s.get("ativo", True) and s.get("saque_t") is not None and voando is not None:
            lanc = s.get("lanc") if s.get("lanc") is not None else s["saque_t"] - 0.6
            antes = voando[(voando < lanc - 0.1) & (voando >= ant["ini"])]
            sem_pausa = len(antes) > 0 and lanc - float(antes[-1]) < PAUSA_ANTES_DO_SAQUE
            if sem_pausa and s["saque_t"] - ant.get("voo_fim", ant["fim"]) < 3.0:
                ant["fim"] = max(ant["fim"], s["fim"])
                if s.get("voo_fim") is not None:
                    ant["voo_fim"] = max(ant.get("voo_fim", 0), s["voo_fim"])
                continue
        saida.append(s)
    return sem_sobreposicao(saida)


def sem_sobreposicao(segmentos: list[dict]) -> list[dict]:
    """Garante que nenhum trecho invada o próximo (trecho sobreposto acabava com duração zero e sumia)."""
    segmentos = sorted(segmentos, key=lambda s: s["ini"])
    for a, b in zip(segmentos, segmentos[1:]):
        if a["fim"] > b["ini"]:
            a["fim"] = round(max(a["ini"] + 0.5, b["ini"]), 2)
            if a.get("voo_fim") is not None:
                a["voo_fim"] = min(a["voo_fim"], a["fim"])
    return [s for s in segmentos if s["fim"] - s["ini"] >= 0.3]


def _prever(modelo, lote, tempos, cfg, classe):
    res = modelo.predict(lote, imgsz=cfg.largura, classes=[classe], conf=0.15, verbose=False)
    out = []
    for t, r in zip(tempos, res):
        H, W = r.orig_shape
        bolas = [[round(c, 3), round((b[0] + b[2]) / 2 / W, 4), round((b[1] + b[3]) / 2 / H, 4), round((b[2] - b[0]) / W, 4)]
                 for c, b in zip(r.boxes.conf.tolist(), r.boxes.xyxy.cpu().numpy().tolist())]
        out.append({"t": round(t, 3), "b": bolas})
    return out
