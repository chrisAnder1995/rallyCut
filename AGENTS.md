# RallyCut — guia para agentes de IA

App local (Windows) que recebe vídeos de treino de vôlei de praia e gera um vídeo só com os rallys.
O usuário revisa os cortes numa página web local e exporta (com placar desenhado, formato e áudio ajustáveis).
O README.md descreve o uso e o algoritmo em detalhe: consulte-o só quando a tarefa precisar.

## Como rodar

- Primeira vez: `instalar.bat` (Python 3.12, FFmpeg via winget, `.venv`, PyTorch com GPU se houver NVIDIA).
- Rodar: `iniciar.bat` ou `.venv\Scripts\python -m rallycut` → http://127.0.0.1:8765
- Dependências Python: `requirements.txt` (ultralytics, fastapi, uvicorn, numpy). FFmpeg é externo, chamado por subprocess.

## Stack

Python 3.12 · FastAPI + uvicorn · YOLO11 (ultralytics, `modelos/yolo11n.pt`, classe *sports ball*) · numpy · FFmpeg.
Frontend: um único arquivo `rallycut/static/index.html` (HTML/CSS/JS puro, ~1400 linhas, sem build).

## Mapa do código (`rallycut/`)

| Arquivo | Responsabilidade |
|---|---|
| `__main__.py` | Sobe o uvicorn na porta 8765 e abre o navegador |
| `app.py` | API FastAPI, fila de jobs e lotes, análise e exportação em thread (`_worker`, `_analisar`, `_exportar`) |
| `detector.py` | `Config` (todos os parâmetros), leitura de quadros via FFmpeg, detecção da bola com YOLO, segmentação inicial |
| `pontos.py` | Núcleo da lógica: janelas de jogo, início e fim de cada ponto, confirmação do fim, 2ª passagem |
| `saque.py` | Calibra a rede, acha saque/travessia da rede/recepção, refina inícios, junta continuações |
| `placar.py` | Lado do saque (pelo tamanho da bola), sequência de saques, desenha o placar (imagem) |
| `revisao.py` | Movimento dos jogadores, candidatos a ponto perdido, relatório |
| `cutter.py` | Corte e exportação com FFmpeg (encoder de GPU quando existe, filtros de vídeo e áudio) |
| `notificar.py` | Notificação do Windows ao fim da análise e config do usuário |
| `static/index.html` | Interface: upload/caminho, lista de vídeos, revisão na linha do tempo, exportação |

**ATENÇÃO:** `rallycut/rallycut/` é uma cópia antiga (13/11) de `app.py`, `pontos.py`, `revisao.py` e `static/`.
Não leia nem edite essa pasta; o código ativo é o de `rallycut/`.

## Pipeline (resumo)

1. FFmpeg lê o vídeo a 5 fps em 1280 px; YOLO acha a bola; bolas paradas são ignoradas → janelas com jogo.
2. Cada janela é reanalisada a 10 fps.
3. Início do ponto = saque detectado (lançamento, bola cruzando a rede, ou bola entrando no quadro com cara de saque quando o sacador está fora da tela; a 1ª ação vista sendo um passe/recepção também conta como início), com 0,6 s de folga.
4. Fim = bola cai na areia e não volta a subir em 1,5 s (+0,7 s), confirmado pela cena (jogadores param, bola parada). Sem confirmação → selo `fim?` (`FIM_A_CONFIRMAR`).
5. 2ª passagem procura pontos perdidos nos intervalos (movimento dos jogadores + bola voando).
6. Exportação com precisão de quadro.

## Regras do vôlei (domínio)

- Set padrão de 21 pontos (opções 12/15/18/21); troca de lado a cada 1/3 do set (4/5/6/7 pontos).
- Tie-break de 15 pontos, troca a cada 5.
- O ponto é sempre de quem saca o próximo rally. O placar muda entre o fim da jogada e o próximo saque.

## Dados

- `dados/<job_id>/job.json` (estado do job) e `analise.json` (resultado da análise); `dados/_lotes/<id>.json` (lotes).
- `dados/`, `*.mp4`, `*.mov` e `.venv/` ficam fora do git. Nunca versione vídeos.
- Vídeos de treino do usuário: `Videos\peladas\<data>` (1 a 5 GB, 1080p, celular em tripé).

## Convenções

- Nomes de funções, variáveis e mensagens em português; type hints em tudo.
- Funções internas do módulo começam com `_`.
- Parâmetros ajustáveis ficam em `Config` (`detector.py`), não espalhados como números mágicos.
- Mudou comportamento visível ao usuário → atualize o README.md.

## Validação

Não há testes automatizados ainda. Valide rodando o app e analisando o vídeo curto de teste `teste.mp4`
(41 s): o esperado são 5 saques e placar final 3x2. Não rode análise de vídeo longo sem pedir: leva minutos.

## Regras para agentes (economia de tokens)

- Siga o `PLANO.md` quando existir: faça só a sua tarefa, nos arquivos indicados.
- Leia só os arquivos da tarefa. Para achar algo, use busca (grep) antes de abrir arquivos inteiros.
- Não leia `static/index.html` inteiro: busque a função ou o id do elemento e leia só o trecho.
- Não abra `dados/*.json`, `modelos/` nem `.venv/`.
- Mudanças pequenas e focadas; não refatore o que a tarefa não pediu.
- Especificação ambígua → pare e devolva a dúvida, em vez de adivinhar.
- Ao terminar, resuma em poucas linhas: arquivos alterados, o que mudou e como validou.
