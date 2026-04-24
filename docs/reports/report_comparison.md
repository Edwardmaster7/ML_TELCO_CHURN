## Comparativo: MLP PyTorch vs. Baselines

Dados coletados diretamente do MLflow:**127.0.0**+3

| Modelo                | ROC-AUC          | PR-AUC           | F1               | Precision        | Recall           |
| --------------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| **mlp_pytorch** | **0.8449** | **0.6510** | **0.6230** | 0.5050           | **0.8128** |
| log_reg_baseline      | 0.8460           | 0.6524           | 0.5952           | **0.6711** | 0.5348           |
| gradient_boosting     | 0.8365           | 0.6284           | 0.5664           | 0.6316           | 0.5134           |
| random_forest         | 0.8241           | 0.6137           | 0.5476           | 0.6052           | 0.5000           |

---

## Análise Métrica por Métrica

## ROC-AUC — MLP: 0.8449 ⚠️ Quase empatado

O MLP ficou praticamente igual à Regressão Logística (0.846 vs. 0.8449), e ambos superam Gradient Boosting e Random Forest. Na prática, a diferença de 0.001 é estatisticamente irrelevante — o MLP **não ganhou capacidade discriminatória geral** em relação ao baseline mais simples. Isso é um sinal de alerta: uma rede neural deve, em tese, capturar relações mais complexas. Se não está superando nem uma Logística no ROC-AUC, pode haver underfitting, falta de tuning, ou os dados simplesmente são bem lineares nessa dimensão.

## PR-AUC — MLP: 0.651 ⚠️ Quase empatado

Mesma história: a Logística tem 0.6524 e o MLP tem 0.6510. A diferença é de apenas 0.0014. O MLP não trouxe ganho real na curva Precision-Recall, que é  **a métrica mais importante para churn com classes desbalanceadas** . Gradient Boosting (0.628) e Random Forest (0.614) ficam para trás, mas a margem também não é enorme.

## F1-Score — MLP: 0.623 ✅ Melhor do grupo

Aqui o MLP **vence todos os outros modelos** com 0.623 vs. 0.595 da Logística. Isso acontece porque o MLP encontrou um equilíbrio melhor entre precision e recall — não o melhor em precision, mas compensa com recall muito superior.

## Recall — MLP: 0.8128 ✅ Grande destaque

Este é o  **diferencial real do MLP** : recall de 81.3% contra apenas 53.5% da Logística. Ou seja, o MLP identifica muito mais churners corretamente. Para um problema de churn onde **falsos negativos são o maior custo de negócio** (clientes que saem sem ser detectados = receita perdida), essa diferença é enorme e muito relevante.

## Precision — MLP: 0.505 ❌ Ponto fraco

A contra-partida do alto recall é uma precision baixa: 0.505. Isso significa que metade dos clientes que o MLP sinaliza como churners  **não vão churnar de fato** . Ações de retenção (cupons, ligações, ofertas) serão desperdiçadas em ~50% dos casos. A Logística (0.671) é bem superior nesse aspecto.

---

## Diagnóstico: O que o MLP está fazendo?

O MLP claramente **ajustou o threshold implicitamente** para um ponto que maximiza recall em detrimento da precision. O modelo está sendo mais "agressivo" — chuta churn com mais frequência, acerta mais churners reais, mas também gera muitos falsos positivos. Isso explica o F1 mais alto (recall alto levanta o F1) mas precision baixa.

---

## Ranking Geral para Churn

| #   | Modelo                      | Por quê?                                                         |
| --- | --------------------------- | ----------------------------------------------------------------- |
| 🥇  | **MLP PyTorch**       | Melhor recall (0.81) e F1 (0.62) — detecta mais churners reais   |
| 🥈  | **log_reg_baseline**  | Melhor precision (0.67), ROC-AUC e PR-AUC ligeiramente superiores |
| 🥉  | **gradient_boosting** | Equilibrado, mas abaixo dos dois anteriores em tudo               |
| 4º | **random_forest**     | Pior em todas as métricas nesse experimento                      |

---

## Recomendações

* **Se o custo de perder um churner >> custo de acionar um não-churner** : o MLP já é a melhor opção — priorize recall
* **Se o custo de retenção é alto** (ex: desconto generoso): vale ajustar o threshold do MLP para 0.55–0.65 e ganhar precision sem perder tanto recall
* **Próximo passo técnico** : tunar o MLP (learning rate, camadas, dropout, batch size) e testar `class_weight` — o ganho de recall pode vir simplesmente do modelo aprendendo um threshold mais baixo, e com tuning adequado é possível subir tanto recall quanto precision
* **Avaliar calibração** : com precision tão baixa, vale checar se as probabilidades do MLP estão bem calibradas (Platt scaling ou isotonic regression podem ajudar)
