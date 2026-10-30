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


def _quadros(video: Path, fps: float, largura: int, w: int, h: int):
    """Gera (tempo, imagem BGR) já reduzidos pelo ffmpeg — bem mais rápido que decodificar tudo em Python.
    A rotação do celular já vem aplicada pelo ffmpeg."""
    altura = int(round(h * largura / w / 2)) * 2
    cmd = [shutil.which("ffmpeg"), "-v", "error", "-hwaccel", "auto", "-i", str(video),
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
                bolas.append([round(c, 3), round((b[0] + b[2]) / 2 / W, 4), round((b[1] + b[3]) / 2 / H, 4)])
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

        for c, x, y in bs:
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


def analisar(video: Path, cfg: Config | None = None, progresso=None) -> dict:
    cfg = cfg or Config()
    frames, duracao = detectar_bolas(video, cfg, progresso)
    segs, score = segmentar(frames, duracao, cfg)
    if progresso:
        progresso(1.0)
    return {"duracao": duracao, "passo": 1 / cfg.fps_analise, "segmentos": segs, "score": score,
            "config": asdict(cfg), "frames": frames}
