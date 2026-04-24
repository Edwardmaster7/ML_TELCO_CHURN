# Quebrando o Teto de PR-AUC: Estratégias Avançadas para MLP PyTorch em Churn Tabular

## Visão Geral do Problema

O problema descrito — um MLP PyTorch estagnado em PR-AUC 0.649 enquanto uma Regressão Logística atinge 0.662 no IBM Telco Churn — é um fenômeno bem documentado na literatura de Deep Learning tabular. A dificuldade central não é falta de capacidade expressiva do MLP, mas sim um **problema conjunto de representação, otimização e regime de treinamento inadequados** para dados tabulares de escala moderada.[^1][^2]

O comportamento diagnosticado (Recall ~82%, Precision ~49%) indica que o MLP está aprendendo um mapeamento sub-ótimo onde a função de perda BCEWithLogitsLoss, sem ajustes para desbalanceamento, força a rede a calibrar mal sua fronteira de decisão no espaço de probabilidade, não no espaço de ranqueamento que define a PR-AUC.[^3][^4]

Este guia apresenta as quatro intervenções técnicas mais impactantes, baseadas em evidências da literatura recente, com análise matemática e recomendações de implementação específicas para PyTorch MLP puro.

***

## 1. Tratamento de Entradas: Entity Embeddings vs. One-Hot Encoding

### O Problema com One-Hot Encoding em MLPs

One-Hot Encoding trata todas as categorias como equidistantes entre si — a representação vetorial de "Month-to-Month" é ortogonal à de "Two year" e à de "One year". Para uma Linear layer do PyTorch, isso significa que as relações ordinais ou de similaridade semântica entre categorias **não podem ser inferidas sem espaço extra de parâmetros**. Em datasets com ~7k amostras, o modelo não tem exemplos suficientes para aprender essas relações implicitamente através de camadas densas.[^5]

### Entity Embeddings: A Solução Comprovada

Entity Embeddings, introduzidos por Guo & Berkhahn (2016) para dados tabulares, mapeiam cada categoria a um vetor denso contínuo aprendido durante o treinamento padrão com backpropagation. A ideia central é que `nn.Embedding(num_categorias, dim_embedding)` cria um lookup table onde o gradiente flui e posiciona categorias similares próximas no espaço embedding — análogo ao word2vec para texto.[^5]

Evidências empíricas demonstram que entity embeddings superam One-Hot Encoding em redes neurais para dados tabulares, especialmente quando há features categóricas de alta cardinalidade. Em um benchmark de detecção de fraude aduaneira, entity embeddings elevaram AUC-ROC e F1-Score em Logistic Regression, SVM e Neural Networks comparados com outros métodos de encoding.[^6][^7]

### Dimensionamento da Embedding

A heurística mais usada (validada empiricamente, incluindo Guo & Berkhahn) é:

\[
\text{dim\_emb}(c) = \min\left(50, \left\lfloor\frac{\text{num\_classes}(c)}{2}\right\rfloor + 1\right)
\]

Para o IBM Telco Churn, onde `Contract` tem 3 classes, `PaymentMethod` tem 4 e a maioria das categóricas binárias tem 2, as embeddings serão pequenas (dim 2–5), o que é adequado e não introduz overfitting significativo.

### Implementação PyTorch

```python
class ChurnMLP(nn.Module):
    def __init__(self, cat_dims, num_cont, emb_dims, hidden_sizes, dropout):
        super().__init__()
        # Embeddings para variáveis categóricas
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cat, emb_dim) for num_cat, emb_dim in zip(cat_dims, emb_dims)
        ])
        self.emb_dropout = nn.Dropout(emb_dropout_p)
        
        total_input = sum(emb_dims) + num_cont
        # Resto da arquitetura...
```

**Veredito:** Para o IBM Telco Churn (que tem ~10 variáveis categóricas binárias e algumas de cardinalidade baixa), o ganho de Entity Embeddings sobre One-Hot será **moderado, mas real** — tipicamente 1-3 pontos de AUC em datasets similares. A principal vantagem é que o MLP aprende uma geometria de entrada mais rica do que vetores esparsos permitem.[^7][^8]

***

## 2. Função de Perda: A Chave para Melhorar PR-AUC

Esta é a **intervenção de maior impacto isolado**. O problema fundamental é que BCEWithLogitsLoss trata cada exemplo com peso igual, fazendo com que os exemplos fáceis da classe majoritária dominem os gradientes e calibrem a rede para prever a maioria.

### Focal Loss: Matemática e Intuição

A Focal Loss, introduzida para detecção de objetos mas aplicável a qualquer classificação binária desbalanceada, adiciona um fator de modulação que reduz a contribuição de exemplos fáceis:[^4][^3]

\[
\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
\]

Onde:
- \( p_t \) é a probabilidade estimada para a classe correta
- \( (1 - p_t)^\gamma \) é o **fator de ponderação**: para amostras bem classificadas (\( p_t \to 1 \)), este fator tende a 0 e suprime o gradiente; para amostras difíceis (\( p_t \to 0 \)), o fator é próximo de 1 e mantém o gradiente.
- \( \alpha_t \) é o peso de balanceamento de classe (inverso da frequência da classe positiva)
- \( \gamma \geq 0 \) controla a força do down-weighting (tipicamente \( \gamma \in [0.5, 5] \))

A Focal Loss **não apenas resolve o desbalanceamento** (papel do \( \alpha \)), mas também força o modelo a se concentrar nos exemplos de churn que são "difíceis de classificar" — exatamente o regime que maximiza PR-AUC, pois eleva o ranqueamento relativo de positivos borderline.[^9][^4]

### Focal Loss vs. BCE em Dados Tabulares

Um estudo empírico recente (SSRN, 2025) encontrou que Focal Loss **não ofereceu benefício sobre BCE padrão em dados tabulares** em seu experimento específico. Isso é importante: Focal Loss funciona melhor quando o desbalanceamento é severo (>10:1) e o gradiente da maioria é o bottleneck. Para o IBM Telco Churn (~26% churn), o desbalanceamento é moderado (≈3:1), então **o parâmetro \( \alpha \) (weighted BCE) pode ser suficiente**.[^10]

A recomendação é: **comece com `BCEWithLogitsLoss(pos_weight=tensor([n_neg/n_pos]))`** e depois teste Focal Loss com Optuna buscando os hiperparâmetros \( \alpha \) e \( \gamma \) ótimos. A combinação de \( \alpha = 0.75 \) (peso para a classe positiva) e \( \gamma \in [1.0, 2.5] \) é um bom ponto de partida.[^11]

```python
# Implementação mínima de Focal Loss em PyTorch
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # probabilidade da classe correta
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce_loss).mean()
```

### Dice Loss: Alternativa para PR-AUC

A Dice Loss é matematicamente equivalente a otimizar diretamente o F1-Score:[^12]

\[
\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \cdot \text{TP}}{2\cdot\text{TP} + \text{FP} + \text{FN}}
\]

Pesquisa em NLP (ACL 2020) demonstrou que Dice Loss supera cross-entropy em tarefas de NLP desbalanceadas, pois penaliza FP e FN igualmente, tornando-a mais imune ao desbalanceamento. No entanto, sua principal limitação em contexto tabular com desbalanceamento moderado é que ela pode sobre-otimizar o limiar de decisão padrão (0.5), ao invés do ranqueamento de probabilidades que define a PR-AUC. **Focal Loss com \( \alpha \) tunado tende a ser mais robusta para a métrica PR-AUC neste cenário.**[^12]

### Tabela Comparativa de Funções de Perda

| Função de Perda | Endereça Desbalanceamento | Foca em Exemplos Difíceis | Maximiza PR-AUC | Recomendação |
|---|---|---|---|---|
| BCEWithLogitsLoss | ❌ | ❌ | Indiretamente | Baseline fraco |
| BCE + `pos_weight` | ✅ | ❌ | Moderado | **Primeiro teste** |
| Focal Loss (α, γ tunados) | ✅ | ✅ | **Forte** | **Candidato principal** |
| Dice Loss | ✅ (implícito) | Parcial | Moderado | Teste secundário |

***

## 3. Micro-Arquitetura: Skip Connections, Normalização e Ativações

Esta seção aborda as modificações estruturais que têm o maior impacto documentado em MLPs tabulares na literatura recente.

### 3.1 Skip Connections: ResNet-like MLP

O paper "Revisiting Deep Learning Models for Tabular Data" (Gorishniy et al., NeurIPS 2021) estabeleceu empiricamente que uma arquitetura ResNet para dados tabulares **supera o MLP vanilla em 7 de 11 datasets benchmark**, com ranking médio de 3.3 vs. 4.8 do MLP, sem ser consistentemente pior em nenhum. O bloco ResNet para dados tabulares proposto é:[^13]

\[
\text{ResNetBlock}(x) = x + \text{Dropout}(\text{Linear}(\text{Dropout}(\text{ReLU}(\text{Linear}(\text{BatchNorm}(x))))))
\]

A razão matemática é que skip connections garantem que **gradientes fluam diretamente para camadas iniciais** durante backpropagation, pois o gradiente da soma \( y = x + F(x, W) \) em relação a \( x \) inclui sempre o termo identidade \( 1 \). Isso é especialmente crítico para MLPs tabulares mais profundos (4+ camadas), onde o gradiente de camadas iniciais fica atenuado sem shortcuts.[^14][^15]

**A implementação correta** requer que a dimensão de entrada e saída do bloco coincidam, ou que se use uma projeção linear para alinhar dimensões:

```python
class ResNetBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return x + self.block(x)  # Skip connection
```

### 3.2 BatchNorm vs. LayerNorm para MLPs Tabulares

Esta é uma das questões mais debatidas e a resposta é **context-dependent**.[^16][^17]

**BatchNorm** calcula média e variância **através do batch** para cada feature. Benefícios: suaviza a superfície de loss, acelera convergência, funciona bem quando o batch size é suficientemente grande (≥32). O paper de Gorishniy (ResNet tabular) usa BatchNorm e a obtém excelentes resultados.[^18][^13]

**LayerNorm** calcula média e variância **através das features** para cada sample individualmente. Benefícios: comportamento **idêntico em treino e inferência**, não depende de estatísticas do batch, mais estável com batch sizes pequenos. Um estudo de normalização em DNN para dados tabulares encontrou que LayerNorm mantém performance "relativamente alta em todos os datasets testados", enquanto BatchNorm "apresenta variação maior dependendo do dataset específico".[^19][^20][^21]

**Recomendação prática para este cenário:**
- Se batch size ≥ 64: **BatchNorm** é boa escolha (como no paper de Gorishniy)
- Se batch size < 32 ou se há problemas de instabilidade: migrar para **LayerNorm**
- A combinação **skip connection + LayerNorm** é especialmente robusta, conforme demonstrado na literatura de Transformers — LayerNorm mitiga problemas de gradiente que skip connections com escala errada podem introduzir.[^22]

### 3.3 Ativações Modernas: GELU e Mish

ReLU é estável e eficiente, mas GELU e Mish têm mostrado vantagens em MLPs ao introduzirem **non-linearidade suave** que evita o "dying ReLU problem" e permite gradientes mais consistentes.[^23]

Um benchmark recente de funções de ativação em MLPs e CNNs concluiu que GELU é a "função de ativação universal superior", alcançando maior validation accuracy em MLPs, enquanto Mish oferece "máxima estabilidade em modelos mais profundos". As funções são definidas como:[^23]

\[
\text{GELU}(x) = x \cdot \Phi(x) \approx x \cdot \sigma(1.702 \cdot x)
\]

\[
\text{Mish}(x) = x \cdot \tanh(\text{softplus}(x)) = x \cdot \tanh(\ln(1 + e^x))
\]

Ambas têm valores negativos suaves (ao contrário do ReLU que trunca em 0), o que **preserva informação de gradiente** em regiões sub-zero. Para MLPs tabulares com ReLU atualmente falhando, a troca para GELU é de baixo risco e alto potencial.

```python
# Bloco modernizado
class ModernBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()  # ou nn.Mish()
        self.linear2 = nn.Linear(dim * 4, dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        return x + self.dropout(self.linear2(self.act(self.linear1(self.norm(x)))))
```

***

## 4. Regime de Treinamento: Schedulers de Learning Rate

### O Problema de Mínimos Locais Estruturais

O comportamento observado — MLP "estacionado" em uma solução com Recall muito alto e Precision baixa — é um sintoma de **convergência prematura para um mínimo local estrutural**. Com BCEWithLogitsLoss não ponderada, esse mínimo é "atraente" porque prever a classe positiva liberalmente minimiza a loss nos dados desbalanceados.[^24]

Schedulers agressivos como OneCycleLR e CosineAnnealingWarmRestarts atacam este problema aumentando temporariamente o learning rate para forçar o parâmetro a "escapar" do poço de energia atual.[^25][^26]

### OneCycleLR

OneCycleLR implementa um ciclo único de LR em 3 fases: crescimento gradual até `max_lr`, decrescimento de volta à LR inicial, e decrescimento adicional para `min_lr` (geralmente `max_lr / 1e4`).[^27]

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-2,        # Pico alto para escape de mínimos
    total_steps=n_epochs * steps_per_epoch,
    pct_start=0.3,      # 30% do tempo na fase de crescimento
    anneal_strategy='cos'
)
# Chamar scheduler.step() após cada batch, não após cada época
```

### CosineAnnealingWarmRestarts

Este scheduler aplica um decaimento cosseno e **reinicia periodicamente** o LR para o valor inicial, permitindo múltiplos ciclos de exploração:[^26][^25]

\[
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{T_{\text{cur}}}{T_i}\pi\right)\right)
\]

Cada reinício pode "revelar" um novo mínimo de menor energia. O parâmetro `T_mult > 1` aumenta progressivamente o período entre reinícios, uma estratégia que combina exploração inicial intensa com refinamento gradual.

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=20,       # Primeiro reinício após 20 épocas
    T_mult=2,     # Dobrar período a cada reinício
    eta_min=1e-6
)
```

**Comparação de Schedulers:**

| Scheduler | Estratégia de Escape | Melhor Para | Risco |
|---|---|---|---|
| Constante | Nenhuma | Baseline apenas | Alto (convergência prematura) |
| Step/Cosine | Decaimento suave | Datasets estáveis | Médio |
| OneCycleLR | Spike único de LR | Treinamento curto/rápido | Baixo |
| CosineWarmRestarts | Múltiplos spikes | Treinamento longo, múltiplas tentativas | Baixo-médio |

***

## 5. O "Regularization Cocktail": A Intervenção de Maior Impacto Comprovado

A descoberta mais importante da literatura recente para este problema específico vem do paper **"Well-tuned Simple Nets Excel on Tabular Datasets"** (NeurIPS 2021, Müller et al.), que demonstrou que MLPs simples com a combinação certa de regularização **superam consistentemente XGBoost e arquiteturas especializadas (TabNet, etc.) em 40 datasets tabulares**.[^28][^2][^1]

O conceito de **"regularization cocktail"** é que nenhuma técnica isolada é suficiente — a combinação otimizada (buscada via HPO como Optuna) das 13 técnicas testadas é o que move a agulha.[^29]

As regularizações de maior impacto documentado para MLPs tabulares incluem:

1. **Weight Decay (L2)**: Tunado no espaço [1e-6, 1e-2], com AdamW (que aplica weight decay corretamente, de forma desacoplada da adaptação do gradiente)
2. **Dropout variável por camada**: Rates diferentes por profundidade, tunáveis via Optuna
3. **Noise Injection nas entradas (Gaussian Noise)**: Adicionar ruído gaussiano \( \mathcal{N}(0, \sigma^2) \) às features contínuas durante treino melhora generalização em datasets tabulares de forma consistente
4. **Early Stopping com paciência longa** (20-50 épocas) monitorando PR-AUC no validation set, **não a loss**
5. **SMOTE + Undersampling combinados** aplicados **apenas no training set** antes de treinar, calibrando a distribuição de classes para ~50:50, permitindo que o modelo aprenda fronteiras de decisão mais equilibradas.[^30]

O paper RealMLP (NeurIPS 2024), que define o estado da arte em MLPs tabulares, incorpora adicionalmente **per-feature normalization** (normalizar cada feature individualmente com robust scaler antes de entrar na rede) e embeddings para features numéricas via piece-wise linear encoding — evidências que preprocessing cuidadoso é tão importante quanto a arquitetura.[^31][^32]

***

## 6. Stack Integrada: Implementação Prioritizada

A seguir, a ordem de intervenções por **impacto esperado vs. esforço de implementação**, com base nas evidências compiladas:

### Prioridade 1 — Loss Function (Alto Impacto, Baixo Esforço)
Substituir `BCEWithLogitsLoss` por `BCEWithLogitsLoss(pos_weight=pos_weight_tensor)` onde `pos_weight = n_neg / n_pos`. Adicionar ao espaço de busca do Optuna `alpha` e `gamma` da Focal Loss. Esta mudança isolada é a mais provável de mover PR-AUC significativamente, pois ataca diretamente o trade-off Recall/Precision observado.[^3][^4]

### Prioridade 2 — Arquitetura ResNet-like (Alto Impacto, Médio Esforço)
Converter o MLP atual para blocos ResNet com skip connections. A literatura mostra ganho médio de ~1 ponto de ranking versus MLP vanilla em 11 datasets, com vantagem em tarefas onde representações mais profundas ajudam. Para o IBM Telco Churn com ~36 features, 3-4 blocos ResNet com dimensão 256-512 é um bom ponto de partida.[^13]

### Prioridade 3 — Entity Embeddings (Médio Impacto, Médio Esforço)
Substituir One-Hot Encoding das variáveis categóricas por `nn.Embedding`. Para o IBM Telco Churn, as categóricas mais impactantes são `Contract`, `PaymentMethod`, `InternetService` e `TechSupport`.[^7][^5]

### Prioridade 4 — GELU + LayerNorm (Médio Impacto, Baixo Esforço)
Trocar ReLU por GELU e BatchNorm por LayerNorm. O custo computacional é similar e o ganho em estabilidade de gradiente e comportamento em test time é consistente.[^21][^23]

### Prioridade 5 — OneCycleLR + AdamW (Médio Impacto, Baixo Esforço)
Substituir optimizer atual por AdamW e adicionar OneCycleLR com max_lr buscado pelo Optuna no range [1e-3, 1e-1]. Usar early stopping monitorando diretamente PR-AUC (via `sklearn.metrics.average_precision_score`) no validation set.[^25]

### Prioridade 6 — SMOTE no Training Set (Médio Impacto, Baixo Esforço)
Aplicar SMOTE + RandomUnderSampler apenas no training set antes do treino. Em experimentos com datasets similares, a combinação de oversampling+undersampling melhora ROC-AUC de ~0.76 para ~0.83. **Atenção: nunca aplicar SMOTE no validation/test set.**[^30]

***

## 7. Diagnóstico do Comportamento Atual e Causa Raiz

O comportamento super-agressivo (Recall ~82%, Precision ~49%) é uma assinatura diagnóstica clara. O MLP está aprendendo um **limiar de decisão implicitamente baixo** porque:

1. A BCEWithLogitsLoss sem `pos_weight` força o modelo a prever probabilidades baixas para churn (classe minoritária), mas o threshold default de 0.5 aplicado durante avaliação captura muitos falsos positivos
2. O Optuna, ao **maximizar PR-AUC durante o tuning**, pode estar induzindo hiperparâmetros que maximizam o recall ao custo da precisão, já que PR-AUC integra sobre todos os thresholds mas é desproporcionalmente influenciada por recall alto

A solução não é calibrar a saída (como a Isotonic Regression fez, revertendo ao comportamento da Regressão Logística), mas sim **reformular o sinal de treinamento via a função de perda** para que o modelo aprenda intrinsecamente uma fronteira de decisão mais equilibrada.

***

## Conclusão e Roteiro de Ação

A combinação com maior probabilidade de quebrar o baseline de PR-AUC 0.662 é:

1. **Focal Loss com α e γ tunados** como função de perda principal
2. **Arquitetura ResNet-like** (skip connections + 3-4 blocos) com **LayerNorm + GELU**
3. **Entity Embeddings** para variáveis categóricas de cardinalidade ≥ 2
4. **AdamW + OneCycleLR ou CosineWarmRestarts** no regime de otimização
5. **SMOTE** aplicado somente no training set para balancear a distribuição de treino

Esta stack representa a essência do "regularization cocktail" para MLPs tabulares documentado na literatura mais recente. A ordem de implementação recomendada permite isolar o impacto de cada mudança e identificar quais contribuem de fato para o problema específico do IBM Telco Churn.[^2][^1][^29]

O ponto mais importante: o teto da Regressão Logística (0.662) é um **teto linear**, e um MLP adequadamente configurado — especialmente com skip connections e função de perda adequada ao desbalanceamento — tem capacidade expressiva superior para capturar interações não lineares entre features como `tenure × Contract type` ou `MonthlyCharges × InternetService`. O problema atual não é capacidade do modelo, mas sim sinal de treinamento e regime de otimização inadequados.

---

## References

1. [Well-tuned Simple Nets Excel on Tabular Datasets](https://arxiv.org/pdf/2106.11189.pdf) - ...
combination/cocktail of 13 regularization techniques for each dataset using a
joint optimization...

2. [Well-tuned Simple Nets Excel on Tabular Datasets - arXiv](https://arxiv.org/abs/2106.11189) - We propose regularizing plain Multilayer Perceptron (MLP) networks by searching for the optimal comb...

3. [Focal Loss vs. Binary Cross Entropy Loss - Daily Dose of Data Science](https://blog.dailydoseofds.com/p/focal-loss-vs-binary-cross-entropy) - The model trained with BCE loss (left) always predicts the majority class. The model trained with fo...

4. [Focal Loss : A better alternative for Cross-Entropy](https://towardsdatascience.com/focal-loss-a-better-alternative-for-cross-entropy-1d073d92d075/) - Focal loss focuses on the examples that the model gets wrong rather than the ones that it can confid...

5. [Entity Embeddings of Categorical Variables](https://www.semanticscholar.org/paper/f9408afe434ab6ea5e852d92d5454063815a8685) - We map categorical variables in a function approximation problem into Euclidean spaces, which are th...

6. [Feature Encoding For High Cardinality Categorical Variables Using Entity Embeddings: A Case Study in Customs Fraud Detection](https://ieeexplore.ieee.org/document/9995764/) - Customs authorities nowadays are pressurized by the increasing levels of international trade and ins...

7. [Predictive Business Process Monitoring – Remaining Time Prediction using Deep Neural Network with Entity Embedding](https://linkinghub.elsevier.com/retrieve/pii/S1877050919319301) - Abstract Most process mining study focuses on analysis of past data. This differs from predictive pr...

8. [Categorical Embeddings for Tabular Data using PyTorch](https://www.semanticscholar.org/paper/Categorical-Embeddings-for-Tabular-Data-using-Khedkar-Lambor/be1ced921f6c6e31a75a9f7160e18de206e3935f) - This research paper applies a feedforward neural network model in PyTorch to a multiclass classifica...

9. [A Loss Function Suitable for Class Imbalanced Data: "Focal Loss"](https://towardsdatascience.com/a-loss-function-suitable-for-class-imbalanced-data-focal-loss-af1702d75d75/) - Here in this post we discuss Focal Loss and how it can improve classification task when the data is ...

10. [[PDF] Deep Learning for Imbalanced Classification: When Do Specialized ...](https://papers.ssrn.com/sol3/Delivery.cfm/5915582.pdf?abstractid=5915582&mirid=1) - 2) Focal Loss ineffectiveness: Despite revolutionizing ob- ject detection, Focal Loss provided no be...

11. [Use Focal Loss To Train Model Using Imbalanced Dataset - Lei Mao](https://leimao.github.io/blog/Focal-Loss-Explained/) - Focal loss is very useful for training imbalanced dataset, especially in object detection tasks. How...

12. [[PDF] Dice Loss for Data-imbalanced NLP Tasks - ACL Anthology](https://aclanthology.org/2020.acl-main.45.pdf) - Taking the binary classification task as an example, at test time, an example will be classified as ...

13. [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/pdf/2106.11959.pdf) - ...researchers and practitioners
what models perform best. Additionally, the field still lacks effec...

14. [[D] Why and how do residual/skip connections work? Looking for ...](https://www.reddit.com/r/MachineLearning/comments/rmxri9/d_why_and_how_do_residualskip_connections_work/) - My handwavy explanation is that skip connections initialize the network as a deep ensemble of neural...

15. [What are skip connections or residual connections? - Milvus](https://milvus.io/ai-quick-reference/what-are-skip-connections-or-residual-connections) - Skip connections, also called residual connections, are a neural network design technique that helps...

16. [When should BatchNorm be used and when should LayerNorm be ...](https://www.reddit.com/r/deeplearning/comments/1ozps3j/when_should_batchnorm_be_used_and_when_should/) - BatchNorm can be easily fused into a Conv, making it faster for edge devices. Many backends do it au...

17. [LayerNorm vs BatchNorm: The Choice That Quietly Decides ...](https://pub.aimind.so/layernorm-vs-batchnorm-the-choice-that-quietly-decides-whether-your-model-scales-be7e3fbda4ac) - BatchNorm is not “bad.” It's just context-dependent. LayerNorm isn't “better.” It's just aligned wit...

18. [Impact of Batch Normalization on Convolutional Network Representations](https://arxiv.org/pdf/2501.14441.pdf) - Batch normalization (BatchNorm) is a popular layer normalization technique
used when training deep n...

19. [Layer Normalization vs. Batch Normalization: What's the Difference?](https://www.coursera.org/articles/layer-normalization-vs-batch-normalization) - While batch normalization computes and adjusts the mean and variance over each mini-batch, layer nor...

20. [Difference between Batch Normalization and Layer Normalization](https://aiml.com/what-is-the-difference-between-batch-and-layer-normalization/) - However, as mentioned above, BatchNorm takes mean and variance across samples while LayerNorm takes ...

21. [[PDF] Understanding the Effect of Normalization On Deep Neural Network ...](https://arxiv.org/pdf/2006.12753.pdf) - BatchNorm and GroupNorm slightly underperform LayerNorm based approaches on Avazu dataset. (2) Compa...

22. [[PDF] Rethinking Skip Connection with Layer Normalization - ACL Anthology](https://aclanthology.org/2020.coling-main.320.pdf) - Skip Connection bypasses the gradient exploding or vanishing problem and tries to solve the model op...

23. [[PDF] International Journal of Electrical and Computer Engineering (IJECE)](https://ijece.iaescore.com/index.php/IJECE/article/download/40374/18698) - Thus, the. GELU and Mish were highly effective for MLP processing, thus providing effective generali...

24. [How to optimize two sets of parameters and jump out local minima?](https://discuss.pytorch.org/t/how-to-optimize-two-sets-of-parameters-and-jump-out-local-minima/117629) - I tried to use torch.optim.lr_scheduler. CosineAnnealingLR and a large learning rate to jump out of ...

25. [Advanced Learning Rate Schedules in PyTorch](https://apxml.com/courses/advanced-pytorch/chapter-3-optimization-training-strategies/advanced-lr-scheduling) - Implement learning rate strategies like cosine annealing with restarts, linear/polynomial decay, and...

26. [CosineAnnealingWarmRestarts — PyTorch 2.11 documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html) - Set the learning rate of each parameter group using a cosine annealing schedule. ... When T c u r = ...

27. [OneCycleLR does not follow the algorithm described by the original ...](https://github.com/pytorch/pytorch/issues/40362) - This implies a learning rate schedule with 3 phases: Increase from initial learning rate to max lear...

28. [Well-tuned Simple Nets Excel on Tabular Datasets](https://www.semanticscholar.org/paper/d2196723bfc17837337f75aede2fb35a025349b9) - Tabular datasets are the last"unconquered castle"for deep learning, with traditional ML methods like...

29. [Regularization Cocktails for Tabular Datasets | OpenReview](https://openreview.net/forum?id=2d34y5bRWxB) - We perform a large-scale empirical study on 40 tabular datasets, concluding that, firstly, regulariz...

30. [SMOTE for Imbalanced Classification with Python](https://www.machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/) - In this case, we can see a modest improvement in performance from a ROC AUC of about 0.76 to about 0...

31. [Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data](https://arxiv.org/abs/2407.04491) - For classification and regression on tabular data, the dominance of gradient-boosted decision trees ...

32. [Strong pre-tuned MLPs and boosted trees on tabular data](https://neurips.cc/virtual/2024/poster/96765) - RealMLP and RealTabR perform strongly among NNs. On most benchmarks, RealMLP-TD and RealTabR-D bring...

