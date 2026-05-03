# Guia de Contribuição — ML Telco Churn

Obrigado por contribuir! Este documento descreve o fluxo de trabalho e os padrões
obrigatórios do repositório.

---

## Fluxo de Contribuição

1. Abra uma **Issue** descrevendo o bug ou a feature antes de começar.
2. Crie uma branch a partir de `main` seguindo o padrão de nomenclatura abaixo.
3. Implemente as mudanças com testes correspondentes.
4. Execute a suíte completa de qualidade antes de abrir o PR:

```bash
uv run ruff check src/ tests/   # lint
uv run ruff format src/ tests/  # formatação
make test                       # testes
```

5. Abra um **Pull Request** com descrição clara e o checklist abaixo preenchido.

---

## Padrão de Branches

```
main                    ← branch principal (protegida)
feat/<descricao>        ← novas funcionalidades
fix/<descricao>         ← correções de bugs
chore/<descricao>       ← manutenção, deps, configs
docs/<descricao>        ← documentação
experiment/<descricao>  ← experimentos de modelo/features
```

---

## Conventional Commits em pt-BR

**Regra obrigatória:** todas as mensagens de commit devem ser em **Português (pt-BR)**.

```
<tipo>: <descrição curta em pt-BR>

[corpo opcional]
[footer: Closes #123]
```

| Tipo | Quando usar |
|---|---|
| `feat:` | Nova funcionalidade |
| `fix:` | Correção de bug |
| `chore:` | Atualização de deps, configs sem mudança funcional |
| `docs:` | Documentação |
| `test:` | Adição ou correção de testes |
| `refactor:` | Refatoração sem mudança de comportamento |
| `experiment:` | Novo experimento de ML |

**Exemplos:**

```bash
git commit -m "feat: adicionar endpoint de feedback de predição"
git commit -m "fix: corrigir cálculo de is_high_spender usando q75 do treino"
git commit -m "docs: criar ADR-010 sobre estratégia de rollback"
git commit -m "experiment: testar ResNet com embeddings de features categóricas"
```

---

## Checklist de PR

```markdown
## Checklist

- [ ] `ruff check src/ tests/` passa sem erros
- [ ] Todos os testes existentes continuam passando (`make test`)
- [ ] Novos comportamentos têm testes correspondentes
- [ ] Caminhos e configs referenciados no PR existem no repo
- [ ] Se o modelo foi retreinado: métricas documentadas no MLflow e comparadas com a versão anterior
- [ ] Se feature de ML adicionada: sem data leakage (fit apenas em X_train)
- [ ] Se mudança arquitetural: ADR criado em `docs/specs/adrs/`
- [ ] Se mudança de deploy: `docs/DEPLOYMENT_ARCHITECTURE.md` atualizado
- [ ] Mensagem de commit em pt-BR seguindo Conventional Commits
```

---

## Configuração do Ambiente de Desenvolvimento

```bash
# 1. Clonar e instalar todas as dependências
git clone <url-do-repositorio>
cd ML_TELCO_CHURN
uv sync

# 2. Inicializar o banco MLflow
make db-upgrade

# 3. Verificar que os testes passam
make test
```

Consulte o [README.md](README.md) para instruções detalhadas de treinamento e execução da API.
