@AGENTS.md

## Específico do Claude Code

- O contexto do projeto está todo no AGENTS.md (importado acima), compartilhado com o Codex e outros agentes.
- Tarefas com passo a passo claro (scripts, README, ajustes mecânicos) vão para o subagente `executor`
  (`.claude/agents/executor.md`, modelo Haiku). Lógica de detecção (`pontos.py`, `saque.py`) fica com o modelo principal.
