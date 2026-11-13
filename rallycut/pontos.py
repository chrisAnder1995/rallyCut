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
# CONFIRMAÇÃO DO FIM (antes de cortar): a queda da bola só vira fim se o resto da cena concordar
REACAO_DE, REACAO_ATE = 1.0, 3.0  # janela depois da queda em que os jogadores já reagem ao fim da jogada (s)
VOLTA_AO_JOGO = 3.0     # bola voando alto de novo até esse tempo depois da "queda", com jogadores correndo -> não acabou
OUTRA_QUEDA = 6.0       # fim duvidoso: procura uma queda confirmada até esse tempo depois (s)
ESPERA_MAX_FIM = 3.0    # fim não confirmado: o corte espera os jogadores pararem, no máximo esse tempo (s)


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


def _candidatos_fim(tr: list[tuple], desde: float, ate: float, chao: float = HORIZONTE):
    """Momentos em que a bola parece cair e o ponto acabar (em ordem): gera (t, x, y).

    Candidato: a bola vinha DESCENDO. Só vira candidato se, nos segundos seguintes, a bola não
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
        yield t, x, y


def achar_fim(tr: list[tuple], desde: float, ate: float, chao: float = HORIZONTE) -> float | None:
    """Primeiro candidato a fim (sem confirmação). Mantido para quem só precisa da queda da bola."""
    c = next(_candidatos_fim(tr, desde, ate, chao), None)
    return c[0] if c else None


# ---------------------------------------------------------------- confirmação do fim da jogada
def contexto_movimento(frames_rapidos: list[dict], mov: list[float] | None,
                       janelas: list[tuple[float, float]]) -> dict | None:
    """Movimento dos jogadores (corpos/s) por quadro e o limiar jogo/pausa calibrado no próprio vídeo."""
    if not mov or len(mov) != len(frames_rapidos):
        return None
    from .revisao import limiar_movimento
    ts = np.array([f["t"] for f in frames_rapidos], dtype=float)
    mv = np.array(mov, dtype=float)
    return {"ts": ts, "mov": mv, "thr": limiar_movimento(ts, mv, janelas)}


def _mov_medio(ctx: dict | None, a: float, b: float) -> float | None:
    if not ctx or ctx.get("thr") is None:
        return None
    sel = (ctx["ts"] >= a) & (ctx["ts"] <= b)
    return float(ctx["mov"][sel].mean()) if sel.any() else None


def _bola_voa_de_novo(tr: list[tuple], t: float, chao: float, ate: float, ate_rel: float) -> bool:
    """Depois da queda, a bola volta a VOAR alto (vários quadros seguidos se deslocando acima da areia)?"""
    altos = [p for p in tr if t + 0.3 < p[0] <= min(ate, t + ate_rel) and p[2] < chao - 0.12]
    voando = 0
    for p, q in zip(altos, altos[1:]):
        dt = q[0] - p[0]
        if 0 < dt <= 0.4 and ((q[1] - p[1]) ** 2 + (q[2] - p[2]) ** 2) ** 0.5 / dt > 0.3:
            voando += 1
    return voando >= 2


def _bola_parada_no_chao(fino: list[dict], t: float, x: float, y: float, chao: float) -> bool:
    """A bola aparece parada (ou rolando devagar) perto de onde caiu: bola morta na areia."""
    perto = [b for f in fino if t + 0.2 <= f["t"] <= t + 2.5 for b in f["b"]
             if b[0] >= 0.3 and b[2] >= chao - 0.05 and ((b[1] - x) ** 2 + (b[2] - y) ** 2) ** 0.5 < 0.08]
    if len(perto) < 4:
        return False
    xs, ys = [b[1] for b in perto], [b[2] for b in perto]
    return (max(xs) - min(xs)) + (max(ys) - min(ys)) < 0.08


def confirmar_fim(t: float, x: float, y: float, tr: list[tuple], fino: list[dict], chao: float,
                  ctx: dict | None, ate: float) -> dict:
    """Confere a queda da bola com o resto da cena antes de cortar.

    Retorna {"ok": True | False | None, "por": motivo}:
      True  -> fim confirmado (jogadores pararam / bola morta na areia / bola não voltou ao jogo)
      False -> NÃO é o fim: a bola voltou a voar com os jogadores em disputa (defesa não vista)
      None  -> duvidoso: a bola não voltou, mas os jogadores continuam correndo
    """
    jog = _mov_medio(ctx, t + REACAO_DE, t + REACAO_ATE)
    thr = ctx.get("thr") if ctx else None
    correndo = jog is not None and jog >= thr
    parados = jog is not None and jog < thr
    volta = _bola_voa_de_novo(tr, t, chao, ate, VOLTA_AO_JOGO if jog is not None else 2.0)
    parada = _bola_parada_no_chao(fino, t, x, y, chao)
    if volta and not parados and not parada:
        return {"ok": False, "por": "a bola voltou a voar com os jogadores em disputa"}
    if parados and parada:
        return {"ok": True, "por": "bola parada na areia e jogadores pararam"}
    if parados:
        return {"ok": True, "por": "jogadores pararam depois da queda"}
    if parada:
        return {"ok": True, "por": "bola parada na areia"}
    if not correndo:
        return {"ok": True, "por": "a bola não voltou ao jogo"}
    return {"ok": None, "por": "a bola caiu, mas os jogadores continuaram correndo"}


def _jogadores_param(ctx: dict | None, t: float) -> float | None:
    """Primeiro momento depois de t em que a movimentação dos jogadores cai abaixo do limiar (até ESPERA_MAX_FIM)."""
    if not ctx or ctx.get("thr") is None:
        return None
    sel = np.where((ctx["ts"] > t + 0.5) & (ctx["ts"] <= t + ESPERA_MAX_FIM))[0]
    for i in sel:
        if ctx["mov"][i] < ctx["thr"]:
            return float(ctx["ts"][i])
    return t + ESPERA_MAX_FIM if len(sel) else None


def fim_confirmado(tr: list[tuple], fino: list[dict], desde: float, ate: float, chao: float,
                   ctx: dict | None) -> tuple[float | None, dict]:
    """Fim da jogada CONFIRMADO antes do corte.

    Percorre as quedas candidatas: descarta as que o jogo desmente (bola voltou a voar com disputa) e fica
    com a primeira confirmada. Se a primeira plausível for duvidosa, ainda procura uma queda confirmada até
    OUTRA_QUEDA s depois; não achando, usa a duvidosa e estende o corte até os jogadores pararem, marcando
    o ponto para revisão ("fim a confirmar").
    Retorna (t_queda, info) com info = {"status": "confirmado" | "a_confirmar", "por", "estende_ate"?}.
    """
    duvida = None
    for t, x, y in _candidatos_fim(tr, desde, ate, chao):
        if duvida and t - duvida[0] > OUTRA_QUEDA:
            break
        r = confirmar_fim(t, x, y, tr, fino, chao, ctx, ate)
        if r["ok"] is True:
            return t, {"status": "confirmado", "por": r["por"]}
        if r["ok"] is None and duvida is None:
            duvida = (t, r)
    if duvida:
        t, r = duvida
        info = {"status": "a_confirmar", "por": r["por"]}
        parou = _jogadores_param(ctx, t)
        if parou is not None and parou > t:
            info["estende_ate"] = round(parou, 2)
        return t, info
    return None, {"status": "a_confirmar", "por": "a bola sumiu sem ser vista caindo"}


def _fim_ou_ultima(tr, fino, desde, ate, chao, ctx, padrao):
    t, info = fim_confirmado(tr, fino, desde, ate, chao, ctx)
    if t is None:
        t = _ultima_bola(tr, desde, ate) or padrao
    return t, info


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
                     frames_rapidos: list[dict], chao: float = HORIZONTE, ctx: dict | None = None) -> list[dict]:
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
        fim, fi = _fim_ou_ultima(tr, fino, sq["voo"] + 0.6, b + 2.5, chao, ctx, sq["voo"] + 1.0)
        rc = achar_recepcao(tr, w_rede, desde=sq["voo"])
        pontos.append({"lanc": sq["lancamento"], "saque_t": sq["voo"], "saque_lado": sq["lado"],
                       "saque_ref": sq.get("ref"), "queda": round(fim, 2), "fim_info": fi,
                       # ace/erro de saque: sem recepção E o ponto acaba logo depois do saque
                       "obs": None if (rc or fim - sq["voo"] > 3.5) else "saque sem recepção (ace ou erro de saque)"})
        t0 = max(t0 + 0.5, fim + 1.0, fi.get("estende_ate", 0) + 0.5)
    if not pontos:
        no_ar = bola_no_ar(frames_rapidos, a, b, w_rede, cfg)
        if no_ar is not None:
            rc = achar_recepcao(tr, w_rede)
            fq = None if rc else _saque_fora_do_quadro(tr, a - 0.5)
            if fq:
                # saque com o sacador fora do quadro: a bola entra voando -> o saque foi ~0,6 s antes
                fim, fi = _fim_ou_ultima(tr, fino, fq["t"] + 0.6, b + 2.5, chao, ctx, b)
                pontos.append({"lanc": None, "saque_t": fq["t"], "inicio_ref": fq["t"] - 0.6 - FOLGA_INICIO,
                               "saque_ref": "fora", "saque_lado": fq["lado"], "queda": round(fim, 2), "fim_info": fi,
                               "obs": None})
                return pontos
            ref_t = rc["t"] if rc else no_ar
            fim, fi = _fim_ou_ultima(tr, fino, ref_t + 0.3, b + 2.5, chao, ctx, b)
            pontos.append({"lanc": None, "saque_t": ref_t, "inicio_ref": ref_t - RECEPCAO_VOLTA, "fim_info": fi,
                           "saque_ref": "recepcao" if rc else "no_ar",
                           "saque_lado": ("longe" if rc["lado"] == "perto" else "perto") if rc else _lado_pela_bola(tr, a - 0.5),
                           "queda": round(fim, 2), "obs": None})
    return pontos


CONFIANCA = {"lancamento": "alta", "rede": "media", "fora": "media", "recepcao": "media", "no_ar": "baixa",
             "inferido": "baixa"}


def _fino(modelo, video, a: float, b: float, cfg: Config, w: int, h: int, duracao: float) -> list[dict]:
    """Reanalisa um trecho a FPS_FINO quadros/s procurando a bola."""
    from .detector import CLASSE_BOLA
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
    return fino


def _seg_do_ponto(p: dict, antes_do_saque: float, duracao: float, origem: str = "passagem 1") -> dict:
    ini = (p["lanc"] - antes_do_saque) if p.get("lanc") is not None else p["inicio_ref"]
    fi = p.get("fim_info") or {}
    # fim não confirmado: o corte espera os jogadores pararem (não corta uma jogada que pode estar viva)
    fim_jogada = max(p["queda"], (fi.get("estende_ate") or 0) - FOLGA_FIM / 2)
    seg = {"ini": round(max(0.0, ini), 2), "fim": round(min(duracao, fim_jogada + FOLGA_FIM), 2),
           "ativo": True, "voo_fim": round(fim_jogada, 2), "queda_bola": p["queda"], "saque_t": p["saque_t"], "lanc": p.get("lanc"),
           "saque_lado": p.get("saque_lado"), "saque_ref": p.get("saque_ref"),
           "confianca": CONFIANCA.get(p.get("saque_ref"), "baixa"), "origem": origem}
    if p.get("lanc") is None or p.get("saque_ref") != "lancamento":
        seg["inicio_inferido"] = True     # START_INFERIDO: o começo do saque não foi visto (estimado)
    if p.get("obs"):
        seg["obs"] = p["obs"]
    _marcar_fim(seg, fi)
    return seg


def _marcar_fim(seg: dict, fi: dict | None) -> None:
    fi = fi or {}
    seg["fim_status"] = fi.get("status", "a_confirmar")
    seg["fim_motivo"] = fi.get("por", "")
    if seg["fim_status"] != "confirmado":
        seg["revisar_fim"] = f"fim a confirmar: {seg['fim_motivo']}"
    else:
        seg.pop("revisar_fim", None)


def _bola_voando(tr: list[tuple], chao: float) -> bool:
    """Evidência de disputa: a bola VOANDO (alta e se deslocando), não só na mão de alguém andando."""
    altos = [p for p in tr if p[2] < min(0.35, chao - 0.1)]
    if len(altos) < 3:
        return False
    xs, ys = [p[1] for p in altos], [p[2] for p in altos]
    return (max(xs) - min(xs)) + (max(ys) - min(ys)) > 0.15


def analisar_pontos(video, frames_rapidos: list[dict], duracao: float, cfg: Config, w: int, h: int,
                    antes_do_saque: float = FOLGA_INICIO, progresso=None, mov: list[float] | None = None) -> list[dict]:
    """PASSAGEM 1: pontos a partir do saque em cada janela com bola em jogo.
    PASSAGEM 2: revisa tudo o que ficou FORA dos pontos (jogadores correndo, bola voando) atrás de pontos perdidos.
    """
    from ultralytics import YOLO
    from .detector import _caminho_modelo
    from .revisao import candidatos

    modelo = YOLO(_caminho_modelo(cfg.modelo))
    w_rede = calibrar_rede(frames_rapidos)
    chao = calibrar_chao(frames_rapidos, cfg)
    janelas = janelas_de_jogo(frames_rapidos, duracao, cfg)
    ctx = contexto_movimento(frames_rapidos, mov, janelas)
    segs: list[dict] = []
    # ---------- passagem 1
    for k, (a, b) in enumerate(janelas):
        fino = _fino(modelo, video, a, b, cfg, w, h, duracao)
        pts = pontos_da_janela(fino, a, b, w_rede, cfg, frames_rapidos, chao, ctx)
        if not pts:
            segs.append({"ini": round(max(0.0, a - 1.0), 2), "fim": round(min(duracao, b + 1.0), 2), "ativo": False,
                         "motivo": "sem saque e sem bola no ar (bola sendo passada ou buscada)"})
        segs += [_seg_do_ponto(p, antes_do_saque, duracao) for p in pts]
        if progresso:
            progresso(0.8 * (k + 1) / max(1, len(janelas)))
    if mov:
        _marcar_saques_suspeitos(segs, frames_rapidos, mov)
    segs = _limpar(_juntar_pedacos(segs))
    # ---------- passagem 2: revisão dos intervalos
    if mov:
        cands = candidatos(frames_rapidos, mov, segs, duracao)
        novos: list[dict] = []
        for k, c in enumerate(cands):
            a, b = c["a"], c["b"]
            fino = _fino(modelo, video, a, b, cfg, w, h, duracao)
            pts = pontos_da_janela(fino, a, b, w_rede, cfg, frames_rapidos, chao, ctx)
            tr = [p for p in _trilha(fino) if a - 1.0 <= p[0] <= b + 2.5]
            if pts:
                for p in pts:
                    sg = _seg_do_ponto(p, antes_do_saque, duracao, "revisão dos intervalos")
                    sg["confianca"] = "media" if sg["confianca"] == "alta" else "baixa"
                    novos.append(sg)
            elif _bola_voando(tr, chao):
                # houve disputa (bola voando + jogadores) mas o saque não foi visto: START_INFERIDO
                t0 = tr[0][0] if tr else a
                fim, fi = _fim_ou_ultima(tr, fino, t0 + 0.3, b + 2.5, chao, ctx, tr[-1][0] if tr else b)
                fim = max(fim, (fi.get("estende_ate") or 0) - FOLGA_FIM / 2)
                novos.append({"ini": round(max(0.0, min(a, t0) - 1.5), 2), "fim": round(min(duracao, fim + FOLGA_FIM), 2),
                              "ativo": True, "voo_fim": round(fim, 2), "saque_t": round(t0, 2), "lanc": None,
                              "saque_lado": _lado_pela_bola(tr, t0 - 0.1), "saque_ref": "inferido",
                              "inicio_inferido": True, "confianca": "baixa", "origem": "revisão dos intervalos"})
                _marcar_fim(novos[-1], fi)
            elif c["mov"] >= 0.6:
                novos.append({"ini": round(max(0.0, a - 1.5), 2), "fim": round(min(duracao, b + 1.0), 2), "ativo": False,
                              "motivo": "jogadores em movimento, bola não vista",
                              "revisar": "possível ponto perdido: jogadores se movimentando como num rally, mas a bola não apareceu"})
            if progresso:
                progresso(0.8 + 0.2 * (k + 1) / max(1, len(cands)))
        if novos:
            # os trechos revisados substituem os "sem saque" que eles cobrem
            def cobre(x, y):
                return x["ini"] < y["fim"] and y["ini"] < x["fim"]
            segs = [s for s in segs if s.get("ativo") or not any(cobre(s, n) for n in novos)] + novos
            _marcar_saques_suspeitos(segs, frames_rapidos, mov)
            segs = _limpar(_juntar_pedacos(segs))
    if progresso:
        progresso(1.0)
    return segs


JUNTA_SE = 1.5   # pedaço que começa até 1,5 s depois do fim do anterior é o MESMO rally (não é saque novo)
SAIU_DO_QUADRO = 8.0   # pedaço SEM saque até 8 s depois do rally anterior: a bola saiu do quadro e o ponto continuou
SERVE_COMO_SAQUE = ("lancamento", "rede", "recepcao", "fora")


def _marcar_saques_suspeitos(segs: list[dict], frames: list[dict], mov: list[float]) -> None:
    """Saque de verdade: antes dele os jogadores estão PARADOS esperando. Se no "saque" eles vinham correndo
    (movimento de rally), é um ataque/defesa no meio da jogada confundido com saque."""
    ts = np.array([f["t"] for f in frames], dtype=float)
    mv = np.array(mov, dtype=float)
    if not len(mv):
        return
    rally = [mv[(ts >= s["saque_t"] + 1) & (ts <= (s.get("voo_fim") or s["fim"]))] for s in segs
             if s.get("ativo") and s.get("saque_t") is not None]
    rally = np.concatenate([r for r in rally if len(r)]) if any(len(r) for r in rally) else np.array([])
    if len(rally) < 10:
        return
    limite = max(0.25, float(np.median(rally)))   # tão intenso quanto um rally (andar na troca de lado não conta)
    for s in segs:
        t = s.get("lanc") if s.get("lanc") is not None else s.get("saque_t")
        if not s.get("ativo") or t is None:
            continue
        antes = mv[(ts >= t - 2.5) & (ts <= t - 0.3)]
        if len(antes) >= 5 and float(np.median(antes)) > limite:
            s["saque_suspeito"] = True


def _juntar_pedacos(segs: list[dict]) -> list[dict]:
    """Um rally partido em dois não pode virar dois pontos.

    Se um ponto começa colado no anterior (até JUNTA_SE s) ou o "saque" dele vem menos de PAUSA_ENTRE_PONTOS s
    depois da bola cair no anterior, é continuação da mesma jogada: junta os dois e mantém o saque do primeiro.
    """
    segs = sorted(segs, key=lambda s: s["ini"])
    saida: list[dict] = []
    for s in segs:
        ant = next((x for x in reversed(saida) if x.get("ativo")), None)
        if s.get("ativo") and (s.get("saque_ref") not in SERVE_COMO_SAQUE or s.get("saque_suspeito")):
            # sem saque detectado (ou "saque" com os jogadores correndo antes): um rally novo só começa no saque
            if ant and s["ini"] - ant["fim"] <= SAIU_DO_QUADRO:
                ant["fim"] = max(ant["fim"], s["fim"])     # a bola saiu do quadro e voltou: o ponto continua
                ant["voo_fim"] = max(ant.get("voo_fim") or 0, s.get("voo_fim") or 0)
                ant.pop("obs", None)
                _herdar_fim(ant, s)
                continue
            if s.get("saque_suspeito") and s.get("saque_ref") in SERVE_COMO_SAQUE:
                s = {**s, "confianca": "baixa"}   # longe de outro ponto: fica, mas para conferir
            elif s.get("saque_ref") in ("no_ar", "inferido"):
                # houve disputa, mas o saque não foi visto: fica como ponto com START_INFERIDO (não descarta)
                s = {**s, "inicio_inferido": True, "confianca": "baixa"}
            else:
                s = {**s, "ativo": False, "motivo": s.get("motivo") or "sem saque detectado"}
        if ant and ant.get("ativo") and s.get("ativo"):
            colado = s["ini"] - ant["fim"] <= JUNTA_SE
            t_saque = s.get("lanc") if s.get("lanc") is not None else s.get("saque_t")
            cedo = t_saque is not None and ant.get("voo_fim") is not None and t_saque - ant["voo_fim"] < PAUSA_ENTRE_PONTOS
            if colado or cedo:
                ant["fim"] = max(ant["fim"], s["fim"])
                ant["voo_fim"] = max(ant.get("voo_fim") or 0, s.get("voo_fim") or 0)
                ant.pop("obs", None)        # teve continuação: não foi ace
                _herdar_fim(ant, s)
                continue
        saida.append(s)
    return saida


def _herdar_fim(ant: dict, s: dict) -> None:
    """Dois pedaços viraram um rally só: o fim (e a confirmação dele) é o do pedaço de depois."""
    if "fim_status" in s:
        _marcar_fim(ant, {"status": s["fim_status"], "por": s.get("fim_motivo", "")})
    if s.get("queda_bola") is not None:
        ant["queda_bola"] = s["queda_bola"]


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
