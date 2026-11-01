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

from . import cutter, detector, placar
from .notificar import ler_config, notificar, salvar_config

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "dados"
STATIC = Path(__file__).resolve().parent / "static"
DADOS.mkdir(exist_ok=True)

app = FastAPI(title="RallyCut")
URL_APP = "http://127.0.0.1:8765"


def _tempo(seg: float) -> str:
    m, s = divmod(int(seg), 60)
    return f"{m}min{s:02d}s" if m else f"{s}s"
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
                notificar("⚠️ RallyCut: algo deu errado", f"{job.get('nome', '')}: {str(e)[:120]}", f"{URL_APP}/#job={job_id}")
        _evento.clear()


def _analisar(job: dict) -> None:
    jid = job["id"]
    t0 = time.time()
    try:
        dur_video = detector._probe(Path(job["video"]))["duracao"]
    except Exception:
        dur_video = None
    _atualizar(jid, status="analisando", etapa="Preparando a análise", progresso=0.0, inicio=t0,
               duracao_video=dur_video, dispositivo="verificando…", eta=None, tempo_analise=None)
    _atualizar(jid, dispositivo=detector.dispositivo())
    estado = {"p": -1.0, "t": 0.0, "etapa": None, "ini_etapa": t0, "p_etapa": 0.0}

    def prog(p: float, etapa: str | None = None) -> None:
        agora = time.time()
        mudou = etapa and etapa != estado["etapa"]
        if mudou:
            estado.update(etapa=etapa, ini_etapa=agora, p_etapa=p)
        if not mudou and p < 1 and p - estado["p"] < 0.005 and agora - estado["t"] < 1.0:
            return
        estado.update(p=p, t=agora)
        # estimativa do tempo restante pelo ritmo atual (ignora os primeiros segundos, quando o modelo carrega)
        decorrido = agora - t0
        eta = None
        if 0.02 < p < 1 and decorrido > 5:
            ritmo = (p - estado["p_etapa"]) / max(1e-6, agora - estado["ini_etapa"]) if agora - estado["ini_etapa"] > 5 else p / decorrido
            if ritmo > 0:
                eta = round((1 - p) / ritmo)
        campos = {"progresso": round(p, 4), "eta": eta, "eta_em": round(agora, 1)}
        if etapa:
            campos["etapa"] = etapa
        _atualizar(jid, **campos)

    res = detector.analisar(Path(job["video"]), progresso=prog)
    (DADOS / jid / "analise.json").write_text(json.dumps(res), encoding="utf-8")
    _atualizar(jid, status="revisao", etapa="Pronto para revisar", progresso=1.0, eta=None,
               tempo_analise=round(time.time() - t0, 1), duracao=res["duracao"], segmentos=res["segmentos"], removidos=[])
    n = sum(1 for s in res["segmentos"] if s.get("ativo", True))
    notificar("🏐 Pronto para revisar", f"{job['nome']}: {n} rallys encontrados em {_tempo(time.time() - t0)}.",
              f"{URL_APP}/#job={jid}")


def _exportar(job: dict) -> None:
    jid = job["id"]
    ativos = [s for s in job["segmentos"] if s.get("ativo", True)]
    segs = [(s["ini"], s["fim"]) for s in ativos]
    usar_placar = bool((job.get("placar_cfg") or {}).get("ativo"))
    placares = [s.get("placar") if usar_placar else None for s in ativos]
    _atualizar(jid, status="exportando", etapa="Cortando e juntando os rallys", progresso=0.0)
    fmt = (job.get("formato") or {}).get("proporcao", "original")
    sufixo = {"16:9": "_16x9", "4:3": "_4x3"}.get(fmt, "")
    nome = Path(job["nome"]).stem + f"_rallys{sufixo}.mp4"
    saida = DADOS / jid / nome
    audio = job.get("audio") or {}
    cutter.cortar(Path(job["video"]), segs, saida, progresso=lambda p: _atualizar(jid, progresso=round(p, 3)),
                  volume=float(audio.get("volume", 1.0)), reduzir_vento=bool(audio.get("vento", False)),
                  placares=placares, **(job.get("formato") or {}))
    _atualizar(jid, status="concluido", etapa="Vídeo pronto", progresso=1.0, saida=str(saida), acao=None)
    notificar("✅ Vídeo pronto", f"{saida.name} — clique para abrir e baixar.", f"{URL_APP}/#job={jid}")


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


@app.get("/api/config")
def config_ler():
    return ler_config()


@app.post("/api/config")
def config_salvar(dados: dict):
    return salvar_config(dados)


@app.post("/api/notificar-teste")
def notificar_teste():
    ok = notificar("🏐 RallyCut", "As notificações estão funcionando!", URL_APP)
    return {"ok": ok}


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
    placar_cfg: dict = {}
    removidos: list[dict] = []


def _limpar_placar(p) -> dict | None:
    if not isinstance(p, dict):
        return None
    num = lambda k: max(0, min(99, int(p.get(k) or 0)))  # noqa: E731
    lado = lambda k: p.get(k) if p.get(k) in ("A", "B") else None  # noqa: E731
    r = {"a": num("a"), "b": num("b"), "set": max(1, min(5, int(p.get("set") or 1))),
         "sets_a": num("sets_a"), "sets_b": num("sets_b"), "saque": lado("saque"), "destaque": lado("destaque"),
         "nome_a": str(p.get("nome_a") or "Dupla A")[:24], "nome_b": str(p.get("nome_b") or "Dupla B")[:24],
         "destaque_ate": max(0.0, min(30.0, float(p.get("destaque_ate") or 0)))}
    f = p.get("final")
    if isinstance(f, dict) and p.get("final_em") is not None:
        r["final"] = {"a": max(0, min(99, int(f.get("a") or 0))), "b": max(0, min(99, int(f.get("b") or 0))),
                      "destaque": f.get("destaque") if f.get("destaque") in ("A", "B") else None}
        r["final_em"] = max(0.0, float(p["final_em"]))
    return r


def _seg_valido(job: dict, s: dict) -> dict | None:
    if True:
        ini, fim = max(0.0, float(s["ini"])), min(job["duracao"], float(s["fim"]))
        if fim - ini < 0.3:   # nunca descarta um trecho em silêncio: corrige o fim
            fim = min(job["duracao"], ini + 1.0)
        if fim - ini >= 0.3:
            seg = {"ini": round(ini, 2), "fim": round(fim, 2), "ativo": bool(s.get("ativo", True)),
                   "sacador": s.get("sacador") if s.get("sacador") in ("A", "B") else None,
                   "ponto_final": s.get("ponto_final") if s.get("ponto_final") in ("A", "B") else None,
                   "placar": _limpar_placar(s.get("placar")), "ini_manual": bool(s.get("ini_manual")),
                   "fim_manual": bool(s.get("fim_manual"))}
            pm = s.get("placar_manual")
            if isinstance(pm, dict):
                try:
                    seg["placar_manual"] = {"a": max(0, min(99, int(pm["a"]))), "b": max(0, min(99, int(pm["b"])))}
                except (KeyError, TypeError, ValueError):
                    pass
            for k in ("lanc", "saque_t", "voo_fim"):
                if isinstance(s.get(k), (int, float)):
                    seg[k] = round(float(s[k]), 2)
            if s.get("saque_lado") in ("perto", "longe"):
                seg["saque_lado"] = s["saque_lado"]
            if s.get("motivo"):
                seg["motivo"] = str(s["motivo"])[:120]
            if s.get("obs"):
                seg["obs"] = str(s["obs"])[:120]
            if s.get("saque_ref") in ("lancamento", "rede", "recepcao", "no_ar"):
                seg["saque_ref"] = s["saque_ref"]
            return seg
    return None


def _validar(job: dict, dados: "SegmentosIn") -> dict:
    segs = [x for x in (_seg_valido(job, s) for s in dados.segmentos) if x]
    segs.sort(key=lambda s: s["ini"])
    # jogadas excluídas ficam guardadas à parte, para poder trazer de volta
    removidos = [x for x in (_seg_valido(job, s) for s in (dados.removidos or [])[:500]) if x]
    audio = {"volume": round(min(4.0, max(0.5, dados.audio.volume)), 2), "vento": dados.audio.vento}
    f = dados.formato
    formato = {"proporcao": f.proporcao if f.proporcao in ("16:9", "4:3") else "original",
               "modo": "desfocar" if f.modo == "desfocar" else "cortar",
               "pos": round(min(1.0, max(0.0, f.pos)), 3)}
    c = dados.placar_cfg or {}
    antes = round(max(0.0, min(5.0, float(c.get("antes_saque", 0.6) or 0))), 2)
    sets_cfg = {}
    for n, v in (c.get("sets_cfg") or {}).items():
        if str(n) in ("2", "3") and isinstance(v, dict):
            r = v.get("rally")
            sets_cfg[str(n)] = {"perto": v.get("perto") if v.get("perto") in ("A", "B") else "auto",
                                "rally": int(r) if isinstance(r, (int, float)) and r >= 1 else None}
    placar_cfg = {"sets_cfg": sets_cfg, "antes_saque": antes, "fim_modo": "queda" if c.get("fim_modo") == "queda" else "saque", "ativo": bool(c.get("ativo")), "nome_a": str(c.get("nome_a") or "Dupla A")[:24],
                  "nome_b": str(c.get("nome_b") or "Dupla B")[:24], "perto": "B" if c.get("perto") == "B" else "A",
                  "ini_a": max(0, min(99, int(c.get("ini_a") or 0))), "ini_b": max(0, min(99, int(c.get("ini_b") or 0))),
                  "set": max(1, min(3, int(c.get("set") or 1))),
                  "sets_a": max(0, min(2, int(c.get("sets_a") or 0))), "sets_b": max(0, min(2, int(c.get("sets_b") or 0)))}
    return {"segmentos": segs, "removidos": removidos, "audio": audio, "formato": formato, "placar_cfg": placar_cfg}


@app.post("/api/jobs/{job_id}/salvar")
def salvar(job_id: str, dados: SegmentosIn):
    """Salva as edições da revisão (sem exportar)."""
    job = _ler(job_id)
    if "duracao" not in job:
        raise HTTPException(409, "A análise ainda não terminou")
    if job["status"] in ("na_fila", "exportando"):
        return {"ok": False}
    _atualizar(job_id, **_validar(job, dados))
    return {"ok": True}


@app.post("/api/jobs/{job_id}/exportar")
def exportar(job_id: str, dados: SegmentosIn):
    job = _ler(job_id)
    if job["status"] not in ("revisao", "concluido", "erro") or "duracao" not in job:
        raise HTTPException(409, "Trabalho ainda não está pronto para exportar")
    v = _validar(job, dados)
    if not any(s["ativo"] for s in v["segmentos"]):
        raise HTTPException(400, "Selecione pelo menos um rally")
    job = _atualizar(job_id, **v, acao="exportar", status="na_fila", etapa="Na fila para exportar",
                     progresso=0.0, erro=None)
    _fila.append(job_id)
    _evento.set()
    return job


@app.post("/api/jobs/{job_id}/reanalisar")
def reanalisar(job_id: str):
    job = _ler(job_id)
    if job["status"] in ("na_fila", "analisando", "exportando"):
        raise HTTPException(409, "Aguarde o processamento atual terminar")
    if not Path(job["video"]).exists():
        raise HTTPException(400, f"O vídeo original não está mais em {job['video']}")
    job = _atualizar(job_id, status="na_fila", etapa="Na fila para analisar", progresso=0.0, erro=None, acao=None)
    _fila.append(job_id)
    _evento.set()
    return job


class SaquesIn(BaseModel):
    inicios: list[float]


@app.post("/api/jobs/{job_id}/saques")
def saques(job_id: str, dados: SaquesIn):
    """Lado (perto/longe da câmera) de onde saiu o saque de cada rally."""
    p = _dir(job_id) / "analise.json"
    if not p.exists():
        raise HTTPException(409, "A análise ainda não terminou")
    frames = json.loads(p.read_text(encoding="utf-8")).get("frames", [])
    return placar.saques(frames, [{"ini": t} for t in dados.inicios])


class ClipeIn(BaseModel):
    ini: float
    fim: float
    numero: int = 1
    audio: AudioIn = AudioIn()
    formato: FormatoIn = FormatoIn()
    placar: dict | None = None


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
    pl = _limpar_placar(dados.placar)
    chave = hashlib.sha1(f"{ini:.2f}|{fim:.2f}|{prop}|{modo}|{pos}|{vol}|{a.vento}|{json.dumps(pl, sort_keys=True)}"
                         .encode()).hexdigest()[:10]
    sufixo = {"16:9": "_16x9", "4:3": "_4x3"}.get(prop, "")
    nome = f"{Path(job['nome']).stem}_rally{dados.numero:02d}{sufixo}.mp4"
    pasta = _dir(job_id) / "clipes" / chave
    saida = pasta / nome
    with _lock_clipe:  # um clipe por vez, para não disputar a placa de vídeo
        if not saida.exists():
            cutter.cortar(Path(job["video"]), [(ini, fim)], saida, volume=vol, reduzir_vento=a.vento,
                          proporcao=prop, modo=modo, pos=pos, placares=[pl])
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
