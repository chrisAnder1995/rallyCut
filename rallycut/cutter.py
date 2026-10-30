"""Corte e montagem do vídeo final com ffmpeg."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg não encontrado no PATH. Instale com: winget install Gyan.FFmpeg")
    return exe


@lru_cache(maxsize=1)
def detectar_encoder() -> tuple[str, ...]:
    """Usa a placa de vídeo para codificar quando existir (muito mais rápido)."""
    try:
        out = subprocess.run([_ffmpeg(), "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    except Exception:
        out = ""
    for enc, args in (
        ("h264_nvenc", ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "21"]),
        ("h264_qsv", ["-c:v", "h264_qsv", "-global_quality", "21"]),
        ("h264_amf", ["-c:v", "h264_amf", "-quality", "balanced", "-rc", "cqp", "-qp_i", "21", "-qp_p", "21"]),
    ):
        if enc in out and _encoder_funciona(args):
            return tuple(args)
    return ("-c:v", "libx264", "-preset", "veryfast", "-crf", "20")


def _encoder_funciona(args: list[str]) -> bool:
    cmd = [_ffmpeg(), "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2", *args, "-f", "null", "-"]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=30).returncode == 0
    except Exception:
        return False


def filtro_audio(volume: float = 1.0, reduzir_vento: bool = False) -> list[str]:
    """Monta o filtro de áudio do ffmpeg.

    volume: multiplicador (1.0 = original, 2.0 = dobro).
    reduzir_vento: corta os graves abaixo de ~120 Hz, onde fica a maior parte do ruído de vento.
    Com volume acima de 1, um limitador evita que os sons fortes (gritos, batida na bola) estourem.
    """
    filtros = []
    if reduzir_vento:
        filtros.append("highpass=f=120:poles=2")
    if abs(volume - 1.0) > 0.01:
        filtros.append(f"volume={volume:.2f}")
        if volume > 1.0:
            filtros.append("alimiter=limit=0.95:attack=5:release=50:level=disabled")
    return ["-af", ",".join(filtros)] if filtros else []


PROPORCOES = {"16:9": (1920, 1080), "4:3": (1440, 1080)}


def filtro_video(proporcao: str = "original", modo: str = "cortar", pos: float = 0.5) -> list[str]:
    """Muda o formato do vídeo para redes sociais.

    proporcao: "original", "16:9" (1920x1080) ou "4:3" (1440x1080).
    modo: "cortar" preenche o quadro cortando as laterais; "desfocar" mostra a imagem inteira
          com um fundo desfocado da própria cena no lugar das barras pretas.
    pos: posição horizontal do corte, de 0 (esquerda) a 1 (direita).
    """
    if proporcao not in PROPORCOES:
        return []
    w, h = PROPORCOES[proporcao]
    ar = w / h
    p = min(1.0, max(0.0, pos))
    if modo == "desfocar":
        vf = (f"split[a][b];"
              f"[a]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=25:2,eq=brightness=-0.08[bg];"
              f"[b]scale={w}:{h}:force_original_aspect_ratio=decrease[fg];"
              f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1")
    else:
        vf = (f"crop=w='min(iw,ih*{ar:.6f})':h='min(ih,iw/{ar:.6f})':x='(iw-ow)*{p:.3f}':y='(ih-oh)/2',"
              f"scale={w}:{h},setsar=1")
    return ["-vf", vf]


def cortar(entrada: Path, segmentos: list[tuple[float, float]], saida: Path, progresso=None,
           volume: float = 1.0, reduzir_vento: bool = False,
           proporcao: str = "original", modo: str = "cortar", pos: float = 0.5) -> Path:
    """Recorta cada segmento (com precisão de quadro) e junta tudo num único MP4."""
    if not segmentos:
        raise ValueError("Nenhum segmento selecionado.")
    enc = detectar_encoder()
    af = filtro_audio(volume, reduzir_vento)
    vf = filtro_video(proporcao, modo, pos)
    saida.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=saida.parent) as tmp:
        tmpd = Path(tmp)
        partes = []
        for i, (ini, fim) in enumerate(segmentos):
            parte = tmpd / f"parte_{i:04d}.mp4"
            cmd = [
                _ffmpeg(), "-v", "error", "-y",
                "-ss", f"{ini:.3f}", "-i", str(entrada), "-t", f"{fim - ini:.3f}",
                "-map", "0:v:0", "-map", "0:a:0?",
                *vf, *enc, "-pix_fmt", "yuv420p", *af,
                "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
                "-avoid_negative_ts", "make_zero", str(parte),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"ffmpeg falhou no trecho {i + 1}: {r.stderr[-800:]}")
            partes.append(parte)
            if progresso:
                progresso((i + 1) / len(segmentos))
        lista = tmpd / "lista.txt"
        lista.write_text("".join(f"file '{p.name}'\n" for p in partes), encoding="utf-8")
        cmd = [_ffmpeg(), "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lista),
               "-c", "copy", "-movflags", "+faststart", str(saida)]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=tmpd)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg falhou ao juntar: {r.stderr[-800:]}")
    return saida
