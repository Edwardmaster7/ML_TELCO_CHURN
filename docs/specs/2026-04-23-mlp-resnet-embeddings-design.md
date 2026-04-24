# Especificação de Design: Fase 2 e 3 - Entity Embeddings e ResNet MLP

**Data:** 23 de Abril de 2026
**Tópico:** Implementação do Cocktail de Regularização Final (ADR-006)
**Notebook Alvo:** `notebooks/07_mlp_resnet_embeddings.ipynb`

## 1. Visão Geral
Esta especificação descreve a arquitetura final de Deep Learning para dados tabulares do projeto. Tendo comprovado que a topologia Linear + One-Hot Encoding gera um "teto linear" intransponível perante a regressão logística (Iteração 4 / Notebook 06), esta implementação migra a representação de dados para **Entity Embeddings** (espaços latentes densos para variáveis categóricas) e atualiza a macro-arquitetura para conter **Skip Connections (ResNet Blocks)**. O objetivo é fornecer à rede neural o fluxo de gradiente e a representação semântica necessários para superar a Regressão Logística.

## 2. Preparação de Dados e Pre-processamento

Para não quebrar a compatibilidade com a Regressão Logística (que depende vitalmente do One-Hot Encoding), o pipeline de pré-processamento deste experimento será isolado.

*   **Entrada:** Dados crus (apenas com as features avançadas já imputadas via engenharia do notebook 04).
*   **Pipeline Numérico:** `SimpleImputer(median)` -> `StandardScaler()`.
*   **Pipeline Categórico:** `SimpleImputer(most_frequent)` -> `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)`.
*   **Dataset PyTorch Customizado:** Criação da classe `ChurnEmbeddingDataset` que herda de `torch.utils.data.Dataset`. No método `__getitem__`, ela deve retornar uma tupla contendo o tensor numérico (`float32`), o tensor categórico de índices (`long`) e a target (`float32`).

## 3. Micro-Arquitetura: O Bloco ResNet Tabular

Em vez da sequência linear padrão, a rede será formada por uma sequência de blocos residuais para dados tabulares, estabilizados por Layer Normalization.

```python
class ResNetBlock(nn.Module):
    # Entrada: Tensor [Batch, Dim]
    # Passos:
    # 1. LayerNorm(Dim)
    # 2. Linear(Dim, Dim)
    # 3. GELU()
    # 4. Dropout(P)
    # 5. Linear(Dim, Dim)
    # 6. Dropout(P)
    # Saída: Passo 6 + Entrada (Skip Connection)
```
*Observação: A dimensionalidade de entrada e saída é constante, evitando a necessidade de projeções lineares na skip connection.*

## 4. Macro-Arquitetura: `AdvancedChurnMLP`

A rede global absorverá os componentes e criará as representações densas.

*   **Camadas de Embedding:** O modelo receberá uma lista com a cardinalidade (número de classes únicas) de cada variável categórica. Para cada uma, criará uma camada `nn.Embedding(num_classes + 1, dim_emb)`.
    *   *Heurística de Dimensionamento:* O `dim_emb` não será hiperparâmetro, mas sim calculado fixamente pela regra de ouro: `min(50, (num_classes // 2) + 1)`.
    *   O `+ 1` no `num_classes` existe para acomodar valores não vistos mapeados pelo OrdinalEncoder.
*   **Camada de Projeção:** Após passar os índices pelas embeddings, todos os vetores densos resultantes são concatenados com o vetor numérico original. Essa mega-matriz passa por uma `nn.Linear(dim_total, hidden_dim)` para unificar e projetar para o tamanho do ResNet Block.
*   **Corpo da Rede:** Um `nn.Sequential` contendo $N$ cópias do `ResNetBlock` (onde $N$ e $hidden\_dim$ são otimizados pelo Optuna).
*   **Head:** Uma `nn.Linear(hidden_dim, 1)` final sem ativação (os logits).

## 5. Treinamento e Otimização (Optuna)

Mantemos a estratégia bem-sucedida de otimização matemática da Fase 1 (Notebook 06).
*   **Função de Perda:** Custom `FocalLoss` ($\alpha$, $\gamma$).
*   **Otimizador:** `AdamW` desacoplado.
*   **Scheduler:** `OneCycleLR` com step por batch.
*   **Espaço de Busca (Optuna):**
    *   `hidden_dim`: Categórico `[64, 128, 256]` (Tamanho constante dos blocos).
    *   `num_blocks`: Int `[2, 3, 4]` (Profundidade da rede).
    *   `dropout_rate`: Float uniforme `[0.1, 0.4]`.
    *   `focal_gamma`: Float `[0.0, 5.0]`.
    *   `focal_alpha`: Float `[0.1, 0.9]`.
    *   `max_lr`: Log-uniforme `[1e-4, 1e-2]`.

## 6. Governança e Rastreabilidade
*   **Métrica Alvo:** A validação e *Early Stopping* devem focar puramente em `PR-AUC`.
*   **MLflow:** O notebook registrará o melhor modelo da rodada sob o *registered_model_name* de `"MLP_ResNet_Embeddings"`.
*   **Prevenção de Warnings:** Garantir o uso seguro do `infer_signature` com o modelo realocado para a CPU antes do salvamento (`model.cpu()`), conforme debugado na Iteração 4.