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
   - **Placar**: placar automático desenhado no vídeo (sets de 12, 15, 18 ou 21 pontos — padrão 21 — com troca de lado a cada 1/3 do set: 4, 5, 6 ou 7 pontos; tie-break de 15 com troca a cada 5). O placar de cada rally é editável e a correção ajusta os rallys seguintes;
   - **📋 Marcações dos pontos**: baixa um .txt com `PONTO n / START FRAME / END FRAME / RESULTADO`.
5. Clique em **Gerar vídeo só com rallys** e depois em **Baixar**. Uma notificação do Windows avisa quando a análise termina.

Os arquivos ficam na pasta `dados/` (um subdiretório por vídeo). Apagar um trabalho pela página não apaga o vídeo original quando ele foi aberto pelo caminho.

## Como funciona a detecção

Cada **ponto** vai do lançamento da bola no saque até a bola cair (`rallycut/pontos.py`).

1. O FFmpeg lê o vídeo a 5 quadros/s em 1280 px e o modelo YOLO11 (classe *sports ball*) localiza a bola. Bolas paradas (reservas na areia, bola na mão) são ignoradas. Isso marca as janelas com jogo.
2. Cada janela é reanalisada a 10 quadros/s.
3. **Início**: um rally só começa quando há um **saque detectado** — o lançamento da bola, a bola cruzando a rede ou a bola entrando no quadro com cara de saque (sacador fora do quadro) — com 0,6 s de folga antes.
4. **Fim**: quando a bola cai (+0,7 s): ela desce até a altura da areia e não volta a subir em 1,5 s. Se a bola sai do quadro no meio do ponto, o rally **não termina**: o que vier até o próximo saque é continuação da mesma jogada. Na página dá para trocar para "1 s antes do próximo saque".
5. **Placar**: o ponto é sempre de quem saca o próximo rally (o lado do saque é visto pelo tamanho da bola: a que vem do lado de lá cresce na imagem). O placar muda entre o fim da jogada e o próximo saque.
6. **2ª passagem (revisão dos intervalos)**: tudo o que ficou fora dos pontos é revisado procurando pontos perdidos — jogadores se movimentando como num rally (velocidade das pessoas na quadra, calibrada no próprio vídeo) e bola voando. Disputa sem saque visível vira ponto com **início inferido** e **baixa confiança**; movimento sem bola fica desmarcado como **possível ponto perdido** para você conferir.
7. **Relatório**: total de pontos, com início inferido, com baixa confiança, possíveis pontos perdidos e intervalos que precisam de revisão (também no .txt das marcações).
8. O corte é feito com precisão de quadro e usa a placa de vídeo para codificar quando existir (NVENC/QuickSync/AMF).

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
