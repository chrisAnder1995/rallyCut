"""Servidor web local do RallyCut."""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from . import cutter, detector

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "dados"
STATIC = Path(__file__).resolve().parent / "static"
DADOS.mkdir(exist_ok=True)

app = FastAPI(title="RallyCut")
_lock = threading.Lock()
_fila: list[str] = []
_evento = threading.Event()


# ---------------------------------------------------------------- persistência
def _dir(job_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{12}", job_id):
        raise HTTPException(404, "Trabalho não encontrado")
    d = DADOS / job_id
    if not d.exists():
        raise HTTPException(404, "Trabalho não encontrado")
    return d


def _ler(job_id: str) -> dict:
    return json.loads((_dir(job_id) / "job.json").read_text(encoding="utf-8"))


def _salvar(job: dict) -> None:
    p = DADOS / job["id"] / "job.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def _atualizar(job_id: str, **campos) -> dict:
    with _lock:
        job = _ler(job_id)
        job.update(campos)
        _salvar(job)
        return job


def _novo_job(nome: str, video: Path, mover: bool = False) -> dict:
    job_id = uuid.uuid4().hex[:12]
    (DADOS / job_id).mkdir()
    if mover:  # upload: o vídeo passa a morar na pasta do trabalho
        video = video.replace(DADOS / job_id / nome)
    job = {"id": job_id, "nome": nome, "video": str(video), "criado": time.time(),
           "status": "na_fila", "etapa": "Aguardando", "progresso": 0.0, "erro": None}
    _salvar(job)
    _fila.append(job_id)
    _evento.set()
    return job


# ---------------------------------------------------------------- worker
def _worker() -> None:
    while True:
        _evento.wait()
        while _fila:
            job_id = _fila.pop(0)
            job = _ler(job_id)
            try:
                if job.get("acao") == "exportar":
                    _exportar(job)
                else:
                    _analisar(job)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                _atualizar(job_id, status="erro", erro=str(e))
        _evento.clear()


def _analisar(job: dict) -> None:
    jid = job["id"]
    _atualizar(jid, status="analisando", etapa="Analisando o vídeo", progresso=0.0)
    ultimo = [0.0]

    def prog(p: float) -> None:
        if p - ultimo[0] >= 0.01 or p >= 1:
            ultimo[0] = p
            _atualizar(jid, progresso=round(p, 3))

    res = detector.analisar(Path(job["video"]), progresso=prog)
    (DADOS / jid / "analise.json").write_text(json.dumps(res), encoding="utf-8")
    _atualizar(jid, status="revisao", etapa="Pronto para revisar", progresso=1.0,
               duracao=res["duracao"], segmentos=res["segmentos"])


def _exportar(job: dict) -> None:
    jid = job["id"]
    segs = [(s["ini"], s["fim"]) for s in job["segmentos"] if s.get("ativo", True)]
    _atualizar(jid, status="exportando", etapa="Cortando e juntando os rallys", progresso=0.0)
    fmt = (job.get("formato") or {}).get("proporcao", "original")
    sufixo = {"16:9": "_16x9", "4:3": "_4x3"}.get(fmt, "")
    nome = Path(job["nome"]).stem + f"_rallys{sufixo}.mp4"
    saida = DADOS / jid / nome
    audio = job.get("audio") or {}
    cutter.cortar(Path(job["video"]), segs, saida, progresso=lambda p: _atualizar(jid, progresso=round(p, 3)),
                  volume=float(audio.get("volume", 1.0)), reduzir_vento=bool(audio.get("vento", False)),
                  **(job.get("formato") or {}))
    _atualizar(jid, status="concluido", etapa="Vídeo pronto", progresso=1.0, saida=str(saida), acao=None)


threading.Thread(target=_worker, daemon=True).start()
for _d in DADOS.iterdir():  # retoma trabalhos interrompidos
    if _d.name.startswith("_upload_"):
        _d.unlink(missing_ok=True)
        continue
    _j = _d / "job.json"
    if _j.exists():
        _job = json.loads(_j.read_text(encoding="utf-8"))
        if _job["status"] in ("na_fila", "analisando", "exportando"):
            _fila.append(_job["id"])
            _evento.set()


# ---------------------------------------------------------------- rotas
@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.post("/api/upload")
async def upload(request: Request):
    from urllib.parse import unquote
    nome = Path(unquote(request.headers.get("x-filename", "video.mp4"))).name
    nome = re.sub(r"[^\w.\- ]", "_", nome) or "video.mp4"
    destino = DADOS / f"_upload_{uuid.uuid4().hex}"
    try:
        with open(destino, "wb") as f:
            async for bloco in request.stream():
                f.write(bloco)
    except BaseException:
        destino.unlink(missing_ok=True)
        raise
    return _novo_job(nome, destino, mover=True)


class CaminhoIn(BaseModel):
    caminho: str


@app.post("/api/caminho")
def por_caminho(dados: CaminhoIn):
    p = Path(dados.caminho.strip().strip('"'))
    if not p.is_file():
        raise HTTPException(400, f"Arquivo não encontrado: {p}")
    return _novo_job(p.name, p)


@app.get("/api/jobs")
def listar():
    jobs = []
    for d in DADOS.iterdir():
        if d.is_dir() and (d / "job.json").exists():
            jobs.append(json.loads((d / "job.json").read_text(encoding="utf-8")))
    return sorted(jobs, key=lambda j: j["criado"], reverse=True)


@app.get("/api/jobs/{job_id}")
def obter(job_id: str):
    return _ler(job_id)


@app.get("/api/jobs/{job_id}/analise")
def analise(job_id: str):
    p = _dir(job_id) / "analise.json"
    if not p.exists():
        raise HTTPException(404, "Análise ainda não terminou")
    return FileResponse(p, media_type="application/json")


class AudioIn(BaseModel):
    volume: float = 1.0      # 1.0 = original, 2.0 = dobro
    vento: bool = False      # filtro de ruído de vento


class FormatoIn(BaseModel):
    proporcao: str = "original"  # original | 16:9 | 4:3
    modo: str = "cortar"         # cortar | desfocar
    pos: float = 0.5             # posição horizontal do corte (0 = esquerda, 1 = direita)


class SegmentosIn(BaseModel):
    segmentos: list[dict]
    audio: AudioIn = AudioIn()
    formato: FormatoIn = FormatoIn()


@app.post("/api/jobs/{job_id}/exportar")
def exportar(job_id: str, dados: SegmentosIn):
    job = _ler(job_id)
    if job["status"] not in ("revisao", "concluido", "erro") or "duracao" not in job:
        raise HTTPException(409, "Trabalho ainda não está pronto para exportar")
    segs = []
    for s in dados.segmentos:
        ini, fim = max(0.0, float(s["ini"])), min(job["duracao"], float(s["fim"]))
        if fim - ini >= 0.3:
            segs.append({"ini": round(ini, 2), "fim": round(fim, 2), "ativo": bool(s.get("ativo", True))})
    segs.sort(key=lambda s: s["ini"])
    if not any(s["ativo"] for s in segs):
        raise HTTPException(400, "Selecione pelo menos um rally")
    audio = {"volume": round(min(4.0, max(0.5, dados.audio.volume)), 2), "vento": dados.audio.vento}
    f = dados.formato
    formato = {"proporcao": f.proporcao if f.proporcao in ("16:9", "4:3") else "original",
               "modo": "desfocar" if f.modo == "desfocar" else "cortar",
               "pos": round(min(1.0, max(0.0, f.pos)), 3)}
    job = _atualizar(job_id, segmentos=segs, audio=audio, formato=formato, acao="exportar", status="na_fila", etapa="Na fila para exportar",
                     progresso=0.0, erro=None)
    _fila.append(job_id)
    _evento.set()
    return job


class ClipeIn(BaseModel):
    ini: float
    fim: float
    numero: int = 1
    audio: AudioIn = AudioIn()
    formato: FormatoIn = FormatoIn()


_lock_clipe = threading.Lock()


@app.post("/api/jobs/{job_id}/clipe")
def clipe(job_id: str, dados: ClipeIn):
    """Gera (ou reaproveita) o vídeo de um único rally com o formato e o áudio escolhidos."""
    job = _ler(job_id)
    if "duracao" not in job:
        raise HTTPException(409, "A análise ainda não terminou")
    ini, fim = max(0.0, dados.ini), min(job["duracao"], dados.fim)
    if fim - ini < 0.3:
        raise HTTPException(400, "Trecho muito curto")
    f, a = dados.formato, dados.audio
    prop = f.proporcao if f.proporcao in ("16:9", "4:3") else "original"
    modo = "desfocar" if f.modo == "desfocar" else "cortar"
    pos = round(min(1.0, max(0.0, f.pos)), 3)
    vol = round(min(4.0, max(0.5, a.volume)), 2)
    chave = hashlib.sha1(f"{ini:.2f}|{fim:.2f}|{prop}|{modo}|{pos}|{vol}|{a.vento}".encode()).hexdigest()[:10]
    sufixo = {"16:9": "_16x9", "4:3": "_4x3"}.get(prop, "")
    nome = f"{Path(job['nome']).stem}_rally{dados.numero:02d}{sufixo}.mp4"
    pasta = _dir(job_id) / "clipes" / chave
    saida = pasta / nome
    with _lock_clipe:  # um clipe por vez, para não disputar a placa de vídeo
        if not saida.exists():
            cutter.cortar(Path(job["video"]), [(ini, fim)], saida, volume=vol, reduzir_vento=a.vento,
                          proporcao=prop, modo=modo, pos=pos)
    return FileResponse(saida, media_type="video/mp4", filename=nome)


@app.get("/api/jobs/{job_id}/video")
def video(job_id: str):
    return FileResponse(_ler(job_id)["video"], media_type="video/mp4")


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str):
    job = _ler(job_id)
    if not job.get("saida") or not Path(job["saida"]).exists():
        raise HTTPException(404, "Vídeo final ainda não existe")
    return FileResponse(job["saida"], media_type="video/mp4", filename=Path(job["saida"]).name)


@app.delete("/api/jobs/{job_id}")
def apagar(job_id: str):
    import shutil
    d = _dir(job_id)
    if _ler(job_id)["status"] in ("analisando", "exportando"):
        raise HTTPException(409, "Aguarde o processamento terminar")
    shutil.rmtree(d)
    return {"ok": True}
