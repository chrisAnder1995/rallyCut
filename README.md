# RallyCut 🏐

Aplicação local que recebe o vídeo do treino de vôlei de praia e gera um novo vídeo **só com os rallys**, sem os momentos parados.

## Como usar (Windows)

1. Dê dois cliques em **`instalar.bat`** (só na primeira vez). Ele instala Python 3.12, FFmpeg e as bibliotecas. Se o PC tiver placa NVIDIA, já instala a versão com GPU.
2. Dê dois cliques em **`iniciar.bat`**. O navegador abre em `http://127.0.0.1:8765`.
3. Arraste o vídeo para a página **ou** cole o caminho do arquivo (melhor para vídeos grandes, porque não faz cópia).
4. Quando a análise terminar, clique em **Revisar**:
   - a linha do tempo mostra em azul os trechos que vão para o vídeo final;
   - desmarque, apague, ajuste início/fim ou adicione um rally no ponto atual do player;
   - com "Pular os trechos parados" marcado, o player já toca só os rallys (prévia);
   - **Formato**: Original, 16:9 (1920×1080) ou 4:3 (1440×1080), cortando as laterais (com ajuste de posição) ou mostrando a imagem inteira com fundo desfocado;
   - **Áudio**: volume de 50% a 400% (com limitador para não estourar) e filtro de ruído de vento;
   - **⬇ em cada rally**: baixa só aquele rally, com o formato e o áudio escolhidos;
   - **✕ em cada rally**: manda a jogada para **🗑 Jogadas excluídas**, de onde ela pode ser restaurada;
   - **Placar**: placar automático desenhado no vídeo (regra oficial: sets de 21 com troca de lado a cada 7 pontos, tie-break de 15 com troca a cada 5). O placar de cada rally é editável e a correção ajusta os rallys seguintes;
   - **📋 Marcações dos pontos**: baixa um .txt com `PONTO n / START FRAME / END FRAME / RESULTADO`.
5. Clique em **Gerar vídeo só com rallys** e depois em **Baixar**. Uma notificação do Windows avisa quando a análise termina.

Os arquivos ficam na pasta `dados/` (um subdiretório por vídeo). Apagar um trabalho pela página não apaga o vídeo original quando ele foi aberto pelo caminho.

## Como funciona a detecção

Cada **ponto** vai do lançamento da bola no saque até a bola cair (`rallycut/pontos.py`).

1. O FFmpeg lê o vídeo a 5 quadros/s em 1280 px e o modelo YOLO11 (classe *sports ball*) localiza a bola. Bolas paradas (reservas na areia, bola na mão) são ignoradas. Isso marca as janelas com jogo.
2. Cada janela é reanalisada a 10 quadros/s.
3. **Início**: o lançamento da bola no saque, com 0,6 s de folga (sem a espera antes). Se o sacador estiver fora do quadro, vale a bola cruzando a rede ou 0,6 s antes da recepção.
4. **Fim**: a bola desce até a altura da areia (calibrada pelas bolas paradas no chão) e não volta a subir em 1,5 s. Defesas, bloqueios e contra-ataques fazem a bola subir de novo e o ponto continua. Na página dá para escolher onde o rally termina: 1 s antes do próximo saque (padrão, até 20 s depois da bola cair) ou quando a bola cai (+0,7 s).
5. **Placar**: o ponto é sempre de quem saca o próximo rally (o lado do saque é visto pelo tamanho da bola: a que vem do lado de lá cresce na imagem). O placar muda entre o fim da jogada e o próximo saque.
6. O corte é feito com precisão de quadro e usa a placa de vídeo para codificar quando existir (NVENC/QuickSync/AMF).

Os parâmetros ficam em `rallycut/detector.py` (`Config`). Os principais:

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `modelo` | `yolo11n.pt` | `yolo11s.pt` acha a bola mais vezes, ~3x mais lento |
| `largura` | 1280 | maior = acha bola menor/mais longe, mais lento |
| `gap_max` | 2.5 s | aumente se um rally estiver sendo quebrado em dois |
| `dur_min` | 3.0 s | aumente se aparecerem trechos curtos de bola sendo devolvida |
| `antes_do_saque` | 0.6 s | margem antes do lançamento do saque (também ajustável na página) |

## Dicas de filmagem

- Tripé fixo, quadra inteira no quadro e com bastante céu acima da rede (a bola contra o céu é muito fácil de detectar).
- Evite deixar a câmera onde as pessoas passam na frente.
