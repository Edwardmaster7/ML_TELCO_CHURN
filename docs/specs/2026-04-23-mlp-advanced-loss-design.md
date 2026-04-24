# Especificação de Design: Fase 1 - Perda Avançada e Schedulers (MLP PyTorch)

**Data:** 23 de Abril de 2026
**Tópico:** Implementação de Focal Loss e OneCycleLR (Phase 1 do ADR-006)
**Notebook Alvo:** `notebooks/06_mlp_advanced_loss.ipynb`

## 1. Visão Geral
Esta especificação detalha a primeira fase da implementação das estratégias avançadas para quebrar o teto linear de PR-AUC (0.662) estabelecido pela Regressão Logística. Conforme decidido via brainstorming, adotaremos uma abordagem incremental. Nesta fase, manteremos a arquitetura de rede e o pipeline de dados originais (One-Hot Encoding), focando exclusivamente na alteração da topologia de otimização através de uma nova função de perda (Focal Loss) e regimes de taxa de aprendizado dinâmicos (OneCycleLR + AdamW).

## 2. Arquitetura e Componentes

### 2.1 Módulo Customizado: Focal Loss
Implementaremos uma classe `FocalLoss` herdando de `nn.Module`.
*   **Fórmula Base:** $\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$
*   **Parâmetros:**
    *   $\gamma$ (Gamma): Controla o foco em exemplos difíceis (reduz o peso de exemplos em que o modelo já tem alta confiança).
    *   $\alpha$ (Alpha): Fator de balanceamento de classes (substitui o antigo `pos_weight` rígido do BCE).
*   **Comportamento:** A loss calculará a Entropia Cruzada Binária (BCE) com logits, extrairá as probabilidades via sigmoid e aplicará os fatores moduladores antes da redução pela média.

### 2.2 Otimizador e Scheduler
*   **Otimizador:** Substituiremos o `torch.optim.Adam` pelo `torch.optim.AdamW`. O AdamW desacopla o *weight decay* do cálculo do gradiente, sendo fundamental para o sucesso das estratégias do "Regularization Cocktail" em dados tabulares.
*   **Scheduler:** Implementaremos `torch.optim.lr_scheduler.OneCycleLR`.
    *   *Mecânica:* O scheduler causará um pico na taxa de aprendizado (até `max_lr`) no primeiro terço do treinamento e depois decairá, ajudando o modelo a escapar de mínimos locais precoces causados pelo desbalanceamento.
    *   *Integração:* Diferente de schedulers tradicionais, o `scheduler.step()` deve ser chamado *após cada batch* no loop interno de treinamento, e não após cada época.

## 3. Fluxo de Dados e Integração MLflow

*   **Fonte de Dados:** Reutilizaremos o arquivo `data/processed/churn_processed_advanced.csv` gerado no notebook 05. Isso garante isolamento de variáveis: se houver melhora, saberemos que foi puramente devido à otimização matemática e não aos dados.
*   **MLflow:**
    *   Novo experimento: `telco_churn_advanced_loss`.
    *   Rastreamento estendido de parâmetros: Logar `focal_gamma`, `focal_alpha`, `max_lr` e `optimizer_type`.
    *   Model Registry: O modelo final será registrado sob o nome `MLP_Focal_OneCycle`.

## 4. Estratégia de Tuning (Optuna)

A função objetivo do Optuna será atualizada para buscar hiperparâmetros neste novo espaço topológico:

*   **Métrica Alvo:** Maximizar a Validação PR-AUC (`val_pr_auc`).
*   **Espaço de Busca (Search Space):**
    *   `gamma`: float uniforme entre `[0.0, 5.0]`
    *   `alpha`: float uniforme entre `[0.1, 0.9]`
    *   `max_lr`: float log-uniforme entre `[1e-4, 1e-1]`
    *   *(Manter a busca por arquitetura oculta e weight decay definida anteriormente).*

## 5. Critérios de Sucesso
O experimento será considerado um sucesso se o MLP otimizado através deste novo regime conseguir ultrapassar o benchmark de PR-AUC de **0.662** de forma consistente no conjunto de validação, mitigando o comportamento de "máquina de recall" (Recall extremo vs Precision baixa) documentado no experimento 05. Caso contrário, a implementação seguirá para a Fase 2 (Micro-arquitetura ResNet).