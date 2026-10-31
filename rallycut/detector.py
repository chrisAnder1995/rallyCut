"""Detecção de rallys de vôlei de praia.

Ideia: durante um rally a bola está VOANDO. Entre um ponto e outro ela fica
na mão de alguém ou parada na areia. Então:

1. Lemos o vídeo em baixa taxa (5 quadros/s) com o ffmpeg.
2. Um detector YOLO encontra a bola ("sports ball") em cada quadro.
3. Descartamos bolas paradas (as reservas na areia, a bola na mão do sacador).
4. Juntamos os momentos com bola em movimento em trechos contínuos, tolerando
   pequenas falhas de detecção, e damos uma folga antes (saque) e depois.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

CLASSE_BOLA = 32  # "sports ball" no COCO


@dataclass
class Config:
    fps_analise: float = 5.0      # quadros analisados por segundo
    largura: int = 1280           # largura usada na análise (bola é pequena: 1280 detecta bem mais que 960)
    modelo: str = "yolo11n.pt"    # yolo11s.pt é mais preciso e ~3x mais lento
    conf_bola: float = 0.35       # confiança mínima da detecção da bola
    raio_parado: float = 0.02     # bola que anda menos que isso (fração da tela) em ~1s é "parada"
    gap_max: float = 2.5          # buracos menores que isso dentro de um rally são ignorados (s)
    dur_min: float = 3.0          # rallys mais curtos que isso são descartados (s)
    folga_antes: float = 1.5      # pega o saque (s)
    folga_depois: float = 1.5     # pega o fim do lance (s)
    refinar_saque: bool = True    # reanalisa o começo de cada rally a 15 quadros/s para achar o saque
    antes_do_saque: float = 0.5   # o corte começa esse tanto antes do lançamento do saque (s)
    pausa_bola_morta: float = 1.0 # bola vista parada por esse tempo depois de voar = fim do rally (s)
    dur_min_fragmento: float = 1.0  # trechos de bola voando mais curtos que isso são ignorados (s)


def _probe(video: Path) -> dict:
    exe = shutil.which("ffprobe")
    if not exe:
        raise RuntimeError("ffprobe não encontrado no PATH. Instale o FFmpeg.")
    out = subprocess.run(
        [exe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height:format=duration", "-of", "json", str(video)],
        capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    s = j["streams"][0]
    return {"duracao": float(j["format"]["duration"]), "w": int(s["width"]), "h": int(s["height"])}


def _quadros(video: Path, fps: float, largura: int, w: int, h: int, inicio: float = 0.0, duracao: float | None = None):
    """Gera (tempo, imagem BGR) já reduzidos pelo ffmpeg — bem mais rápido que decodificar tudo em Python.
    A rotação do celular já vem aplicada pelo ffmpeg."""
    altura = int(round(h * largura / w / 2)) * 2
    trecho = (["-ss", f"{inicio:.3f}"] if inicio > 0 else [])
    limite = (["-t", f"{duracao:.3f}"] if duracao else [])
    cmd = [shutil.which("ffmpeg"), "-v", "error", "-hwaccel", "auto", *trecho, "-i", str(video), *limite,
           "-an", "-vf", f"fps={fps},scale={largura}:{altura}:flags=area",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10 ** 8)
    tam = largura * altura * 3
    i = 0
    try:
        while True:
            buf = proc.stdout.read(tam)
            if len(buf) < tam:
                break
            yield i / fps, np.frombuffer(buf, np.uint8).reshape(altura, largura, 3)
            i += 1
    finally:
        proc.stdout.close()
        proc.kill()


def _dimensoes_exibidas(video: Path, info: dict) -> tuple[int, int]:
    """Considera vídeos gravados com o celular em pé (rotação nos metadados)."""
    exe = shutil.which("ffprobe")
    out = subprocess.run([exe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream_side_data=rotation:stream_tags=rotate", "-of", "json", str(video)],
                         capture_output=True, text=True).stdout
    rot = 0
    try:
        j = json.loads(out)["streams"][0]
        rot = int(float(j.get("tags", {}).get("rotate", 0) or 0))
        for sd in j.get("side_data_list", []):
            rot = int(float(sd.get("rotation", rot)))
    except Exception:
        pass
    if abs(rot) % 180 == 90:
        return info["h"], info["w"]
    return info["w"], info["h"]


def detectar_bolas(video: Path, cfg: Config, progresso=None) -> tuple[list[dict], float]:
    from ultralytics import YOLO  # import tardio: demora alguns segundos

    info = _probe(video)
    w, h = _dimensoes_exibidas(video, info)
    modelo = YOLO(_caminho_modelo(cfg.modelo))
    total = max(1, int(info["duracao"] * cfg.fps_analise))
    frames = []
    lote, tempos = [], []

    def processar():
        resultados = modelo.predict(lote, imgsz=cfg.largura, classes=[CLASSE_BOLA], conf=0.15, verbose=False)
        for t, r in zip(tempos, resultados):
            H, W = r.orig_shape
            bolas = []
            for c, b in zip(r.boxes.conf.tolist(), r.boxes.xyxy.cpu().numpy().tolist()):
                bolas.append([round(c, 3), round((b[0] + b[2]) / 2 / W, 4), round((b[1] + b[3]) / 2 / H, 4),
                              round((b[2] - b[0]) / W, 4)])  # conf, x, y (centro) e largura, em fração da tela
            frames.append({"t": round(t, 3), "b": bolas})
        lote.clear()
        tempos.clear()

    for t, img in _quadros(video, cfg.fps_analise, cfg.largura, w, h):
        lote.append(img)
        tempos.append(t)
        if len(lote) == 8:
            processar()
            if progresso:
                progresso(min(0.99, len(frames) / total))
    if lote:
        processar()
    return frames, info["duracao"]


def _fps(video: Path) -> float:
    """Quadros por segundo do vídeo original (para as marcações em número de frame)."""
    try:
        out = subprocess.run([shutil.which("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=avg_frame_rate,r_frame_rate", "-of", "json", str(video)],
                             capture_output=True, text=True).stdout
        st = json.loads(out)["streams"][0]
        for k in ("avg_frame_rate", "r_frame_rate"):
            n, d = st.get(k, "0/0").split("/")
            if float(d) > 0 and float(n) > 0:
                return round(float(n) / float(d), 3)
    except Exception:
        pass
    return 30.0


def dispositivo() -> str:
    """Onde a IA vai rodar: nome da placa de vídeo NVIDIA ou "CPU"."""
    try:
        import torch
        if torch.cuda.is_available():
            return "GPU " + torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "CPU (sem placa de vídeo compatível — mais lento)"


def _caminho_modelo(nome: str) -> str:
    local = Path(__file__).resolve().parent.parent / "modelos" / nome
    return str(local) if local.exists() else nome


def bola_em_movimento(frames: list[dict], cfg: Config) -> np.ndarray:
    """Para cada quadro, a confiança da melhor bola que NÃO está parada (0 se nenhuma)."""
    ts = np.array([f["t"] for f in frames])
    bolas = [[b for b in f["b"] if b[0] >= cfg.conf_bola] for f in frames]
    mov = np.zeros(len(frames))
    for i, bs in enumerate(bolas):
        if not bs:
            continue

        def perto(lo: float, hi: float, x: float, y: float) -> bool | None:
            js = np.where((ts >= ts[i] + lo) & (ts <= ts[i] + hi))[0]
            if len(js) == 0:
                return None
            return any(np.hypot(x - b[1], y - b[2]) < cfg.raio_parado for j in js for b in bolas[j])

        for c, x, y, *_ in bs:
            antes, depois = perto(-1.2, -0.4, x, y), perto(0.4, 1.2, x, y)
            parada = (antes is not False) and (depois is not False)  # parada nos dois lados da janela
            if not parada:
                mov[i] = max(mov[i], c)
    return mov


def segmentar(frames: list[dict], duracao: float, cfg: Config) -> tuple[list[dict], list[float]]:
    if not frames:
        return [], []
    ts = np.array([f["t"] for f in frames], dtype=float)
    mov = bola_em_movimento(frames, cfg)
    trechos: list[list[float]] = []
    for t in ts[mov > 0].tolist():
        if trechos and t - trechos[-1][1] <= cfg.gap_max:
            trechos[-1][1] = t
        else:
            trechos.append([t, t])
    segs = []
    for a, b in trechos:
        if b - a < cfg.dur_min:
            continue
        ini, fim = max(0.0, float(a) - cfg.folga_antes), min(float(duracao), float(b) + cfg.folga_depois)
        if segs and ini <= segs[-1]["fim"]:
            segs[-1]["fim"] = round(fim, 2)
        else:
            segs.append({"ini": round(ini, 2), "fim": round(fim, 2), "ativo": True})
    # curva de "atividade" para a linha do tempo (média móvel de ~2 s)
    k = max(1, int(2 * cfg.fps_analise))
    score = np.convolve((mov > 0).astype(float), np.ones(k) / k, mode="same")
    return segs, [round(float(v), 3) for v in score]


def _bola_morta(frames: list[dict], ts: np.ndarray, mov: np.ndarray, t0: float, t1: float,
                ref: tuple[float, float] | None, cfg: Config) -> bool:
    """Entre t0 (última bola voando) e t1 (próxima bola voando) a bola foi vista PARADA perto de onde
    estava voando? (caiu na areia, foi segurada, parou) -> o rally acabou."""
    if ref is None:
        return False
    # bolas que já estavam paradas ali antes do lance (bolas reservas na areia) não contam
    reservas = [(b[1], b[2]) for j in np.where((ts >= t0 - 8) & (ts <= t0 - 1.5))[0]
                for b in frames[j]["b"] if b[0] >= cfg.conf_bola]
    for i in np.where((ts > t0) & (ts < t1) & (mov == 0))[0]:
        for b in frames[i]["b"]:
            if b[0] < cfg.conf_bola or math.hypot(b[1] - ref[0], b[2] - ref[1]) > 0.15:
                continue
            if sum(1 for r in reservas if math.hypot(b[1] - r[0], b[2] - r[1]) <= 0.03) >= 2:
                continue
            return True
    return False


def agrupar_rallys(frames: list[dict], duracao: float, cfg: Config) -> list[dict]:
    """Junta os momentos de bola voando em rallys. Um rally termina quando a bola some por mais de
    `gap_max` s OU quando, depois de voar, ela aparece parada (bola morta) por pelo menos `pausa_bola_morta` s."""
    ts = np.array([f["t"] for f in frames], dtype=float)
    mov = bola_em_movimento(frames, cfg)
    idx = np.where(mov > 0)[0]
    grupos: list[dict] = []
    for i in idx:
        t = float(ts[i])
        # posição da bola voando neste quadro (a mais confiável que não está parada)
        bs = [b for b in frames[i]["b"] if b[0] >= cfg.conf_bola]
        pos = (bs[0][1], bs[0][2]) if bs else None
        if grupos:
            g = grupos[-1]
            pausa = t - g["b"]
            continua = pausa <= cfg.gap_max and not (
                pausa >= cfg.pausa_bola_morta and _bola_morta(frames, ts, mov, g["b"], t, g["pos"], cfg))
            if continua:
                g["b"] = t
                g["pos"] = pos or g["pos"]
                continue
            g["fim_por"] = "bola_morta" if pausa <= cfg.gap_max else "pausa"
        grupos.append({"a": t, "b": t, "pos": pos, "fim_por": None})
    segs = []
    for g in grupos:
        if g["b"] - g["a"] < cfg.dur_min_fragmento:
            continue
        ini = max(0.0, g["a"] - cfg.folga_antes, segs[-1]["fim"] if segs else 0.0)
        fim = min(float(duracao), g["b"] + cfg.folga_depois)
        segs.append({"ini": round(ini, 2), "fim": round(fim, 2), "ativo": True, "voo_fim": round(g["b"], 2)})
    # não deixa um rally invadir o próximo
    for a, b in zip(segs, segs[1:]):
        if a["fim"] > b["ini"]:
            a["fim"] = round(max(a["voo_fim"] + 0.3, b["ini"]), 2)
            b["ini"] = max(b["ini"], a["fim"])
    return segs


def analisar(video: Path, cfg: Config | None = None, progresso=None) -> dict:
    """progresso(p, etapa) recebe a fração concluída (0..1) e o nome da etapa atual."""
    cfg = cfg or Config()
    n = 2 if cfg.refinar_saque else 1
    e1 = f"Procurando a bola (etapa 1 de {n})"
    peso1 = 0.8 if cfg.refinar_saque else 1.0
    if progresso:
        progresso(0.0, e1)
    frames, duracao = detectar_bolas(video, cfg, (lambda p: progresso(p * peso1, e1)) if progresso else None)
    segs, score = segmentar(frames, duracao, cfg)
    if cfg.refinar_saque:
        # algoritmo v3: cada ponto começa no lançamento do saque e termina quando a bola cai
        from .pontos import analisar_pontos
        e2 = "Marcando o saque e o fim de cada ponto (etapa 2 de 2)"
        if progresso:
            progresso(peso1, e2)
        info = _probe(video)
        w, h = _dimensoes_exibidas(video, info)
        segs = analisar_pontos(video, frames, duracao, cfg, w, h, cfg.antes_do_saque,
                               progresso=(lambda p: progresso(peso1 + (1 - peso1) * p, e2)) if progresso else None)
    if progresso:
        progresso(1.0, "Concluído")
    return {"duracao": duracao, "fps": _fps(video), "passo": 1 / cfg.fps_analise, "segmentos": segs, "score": score,
            "config": asdict(cfg), "frames": frames}
