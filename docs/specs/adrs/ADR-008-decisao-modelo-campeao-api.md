# ADR-008: Decisão sobre o Modelo Campeão para a API

## Status
Aceito

## Data
25 de Abril de 2026

## Contexto
Durante a fase de modelagem do projeto de predição de Churn da Telco, realizamos extensas experimentações avaliando tanto modelos lineares tradicionais quanto arquiteturas de Deep Learning baseadas em PyTorch (conforme restrição do desafio).

Nossas descobertas evidenciaram que a Regressão Logística (Logistic Regression) superou matematicamente todas as abordagens de Deep Learning testadas neste dataset tabular, alcançando um PR-AUC de 0.6624 contra 0.6539 da melhor Rede Neural. 

No entanto, o projeto (um "Tech Challenge" para a FIAP) exige estritamente a implantação de uma Rede Neural Multi-Layer Perceptron (MLP) construída com PyTorch na API final. Executamos múltiplas arquiteturas, incluindo um MLP Vanilla e uma ResNet com Entity Embeddings. A arquitetura avançada de ResNet com embeddings apresentou underperformance e sofreu com o "Generalization Gap", enquanto a arquitetura `MLP_Focal_KFold` (utilizando Focal Loss para tratar o desbalanceamento e validada com K-Fold cross-validation) obteve o melhor score entre as redes neurais que generalizaram bem (PR-AUC 0.6539).

## Decisão
A arquitetura `MLP_Focal_KFold` (registrada no MLflow) será o modelo formalmente escolhido e implantado em produção via a API FastAPI.

## Consequências

### Positivas
- Cumprimos os requisitos estritos do Tech Challenge, entregando um modelo baseado em PyTorch/MLP na API.
- Demonstramos a capacidade de aplicar soluções matemáticas avançadas (Focal Loss, validação K-Fold robusta) para empurrar um MLP tabular ao seu limite funcional.

### Negativas (Trade-offs)
- Estamos conscientemente implantando um modelo que é levemente sub-ótimo em comparação com um modelo linear mais simples (queda de aproximadamente ~1% na métrica PR-AUC).
- Sacrificamos um pouco de precisão em prol da conformidade acadêmica, assumindo a complexidade extra (manutenção e tracking) de um modelo Deep Learning sobre um modelo linear base.
