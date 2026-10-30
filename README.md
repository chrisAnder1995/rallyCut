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
   - **⬇ em cada rally**: baixa só aquele rally, com o formato e o áudio escolhidos.
5. Clique em **Gerar vídeo só com rallys** e depois em **Baixar**.

Os arquivos ficam na pasta `dados/` (um subdiretório por vídeo). Apagar um trabalho pela página não apaga o vídeo original quando ele foi aberto pelo caminho.

## Como funciona a detecção

Durante um rally, a bola está **voando**; entre os pontos ela fica na mão do sacador ou parada na areia.

1. O FFmpeg lê o vídeo a 5 quadros/s em 1280 px de largura.
2. O modelo YOLO11 (classe *sports ball*) localiza a bola em cada quadro.
3. Bolas que ficam no mesmo lugar por ~1 s são ignoradas (bolas reservas na areia, bola na mão).
4. Momentos com bola em movimento são unidos em trechos (tolerando falhas de até 2,5 s), trechos com menos de 3 s são descartados e cada rally ganha 1,5 s de folga antes (saque) e depois.
5. O corte é feito com precisão de quadro e usa a placa de vídeo para codificar quando existir (NVENC/QuickSync/AMF).

Os parâmetros ficam em `rallycut/detector.py` (`Config`). Os principais:

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `modelo` | `yolo11n.pt` | `yolo11s.pt` acha a bola mais vezes, ~3x mais lento |
| `largura` | 1280 | maior = acha bola menor/mais longe, mais lento |
| `gap_max` | 2.5 s | aumente se um rally estiver sendo quebrado em dois |
| `dur_min` | 3.0 s | aumente se aparecerem trechos curtos de bola sendo devolvida |
| `folga_antes` / `folga_depois` | 1.5 s | margem antes do saque / depois do fim |

## Dicas de filmagem

- Tripé fixo, quadra inteira no quadro e com bastante céu acima da rede (a bola contra o céu é muito fácil de detectar).
- Evite deixar a câmera onde as pessoas passam na frente.
