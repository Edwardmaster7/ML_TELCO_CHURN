## Comparativo Completo: MLP Tuned vs. Todos os Modelos

Dados coletados diretamente do MLflow:**127.0.0**+4

| Modelo                             | ROC-AUC          | PR-AUC           | F1               | Precision        | Recall           |
| ---------------------------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| **MLP Baseline**             | **0.8465** | **0.6591** | **0.6180** | 0.4976           | **0.8155** |
| **MLP Tuned (Uncalibrated)** | 0.8470           | 0.6548           | 0.6163           | 0.4833           | **0.8503** |
| **MLP Tuned (Calibrated)**   | 0.8370           | 0.6199           | 0.5701           | **0.6727** | 0.4947           |
| log_reg_baseline                   | 0.8460           | 0.6524           | 0.5952           | 0.6711           | 0.5348           |
| gradient_boosting                  | 0.8365           | 0.6284           | 0.5664           | 0.6316           | 0.5134           |
| random_forest                      | 0.8241           | 0.6137           | 0.5476           | 0.6052           | 0.5000           |

---

## Análise por Dimensão

## ROC-AUC — MLP Tuned lidera (0.847), mas por margem mínima

Todos os MLPs e a Logística estão praticamente empatados na faixa 0.846–0.847. A diferença de 0.001 é ruído estatístico — não dá pra afirmar superioridade real aqui. O que importa é que o conjunto MLP + LogReg claramente bate GB (0.836) e RF (0.824).

## PR-AUC — MLP Baseline é o melhor (0.6591)

Esse é o dado mais importante para churn com desbalanceamento. O **MLP Baseline supera o Tuned** (0.6591 vs. 0.6548) — o tuning não melhorou a curva precision-recall global. A calibração então derruba bastante para 0.6199, ficando abaixo até do `log_reg_baseline`. Isso é um sinal importante.

## F1 — MLP Baseline na frente (0.618)

Os dois MLPs não calibrados dominam o F1 (0.618 e 0.616), seguidos da Logística (0.595). A calibração volta o MLP Tuned para 0.570 — pior que o baseline de regressão logística. Random Forest continua no fundo (0.548).

## Recall — MLP Tuned Uncalibrated é o campeão absoluto (0.850)

O modelo tuned sem calibração captura **85% dos churners reais** — o maior recall de todos. O MLP Baseline fica em 0.815, também muito forte. Ambos superam amplamente a Logística (0.535). Isso confirma que a rede neural aprendeu a ser mais agressiva na detecção de churn.

## Precision — Calibrado e Logística lideram (~0.671–0.673)

O MLP Tuned Calibrado tem a melhor precision do experimento (0.6727), empatado com a Logística (0.6711). Mas o custo foi brutal: recall caiu para 0.495, ou seja, a calibração simplesmente inverteu o comportamento do modelo — de agressivo para conservador demais.

---

## O Problema da Calibração

A calibração via Platt Scaling/Isotonic produziu um  **trade-off desfavorável** :

* PR-AUC caiu de 0.6548 → 0.6199 (−0.035)
* Recall caiu de 0.850 → 0.495 (−0.355)
* Precision subiu de 0.483 → 0.673 (+0.190)

O modelo calibrado ficou comportamentalmente semelhante à Regressão Logística — o que questiona o valor da calibração nesse contexto. O problema provavelmente é que **a calibração foi feita no validation set, mas o threshold de 0.5 já não é o ideal** para nenhum dos dois modelos. Calibração corrige as probabilidades, não o ponto de corte — e você ainda precisa ajustar o threshold separadamente após calibrar.

---

## Ranking Final para Churn

| #   | Modelo                             | Justificativa                                                                |
| --- | ---------------------------------- | ---------------------------------------------------------------------------- |
| 🥇  | **MLP Baseline**             | Melhor PR-AUC (0.6591), F1 alto (0.618), recall forte (0.815)                |
| 🥈  | **MLP Tuned (Uncalibrated)** | Maior recall (0.850), ROC-AUC levemente melhor, mas PR-AUC um pouco inferior |
| 🥉  | **log_reg_baseline**         | Melhor precision, bom ROC-AUC, mas recall muito inferior (0.535)             |
| 4º | **MLP Tuned (Calibrated)**   | Precision boa, mas PR-AUC e recall decepcionantes — perde valor             |
| 5º | **gradient_boosting**        | Mediano em tudo                                                              |
| 6º | **random_forest**            | Pior em todas as métricas                                                   |

---

## Recomendações

* **O tuning não trouxe ganho real no PR-AUC** — o MLP Baseline ficou com 0.6591 contra 0.6548 do Tuned. Revise a arquitetura, learning rate scheduler ou o número de épocas do modelo tuned
* **Abandone o threshold fixo de 0.5** para os MLPs: com recall de 0.85 e precision de 0.48, o ponto de operação ideal está provavelmente entre 0.35 e 0.45. Plote a curva precision-recall e escolha o threshold que maximiza F1 ou que atende a uma restrição de negócio (ex: "quero pelo menos 70% de precision")
* **A calibração piorou o modelo** — se a intenção era melhorar as probabilidades para scoring, use uma amostra de calibração separada e avalie com Brier Score ou ECE (Expected Calibration Error), não só com as métricas de classificação
* **Próximo passo natural** : otimizar o threshold dos dois MLPs não calibrados com base na curva PR e no custo de negócio real (custo de ação de retenção vs. receita perdida por churn)
