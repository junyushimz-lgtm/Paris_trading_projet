# Pairs Trading Project

Projet de data science appliqué à la finance.

## Objectif

Construire un pipeline automatisé permettant de générer une recommandation d'investissement à partir de plusieurs signaux :

- clustering d'entreprises similaires ;
- sélection de paires corrélées ;
- test de cointégration ;
- stratégie de pair trading ;
- rolling beta ;
- classification ML de convergence du spread ;
- régression du rendement à J+1 ;
- sentiment analysis simple sur news financières ;
- agrégation finale des signaux.

## Structure

```text
src/
  config.py
  data.py
  clustering.py
  pairs_trading.py
  ml_models.py
  sentiment.py
  aggregation.py
  reporting.py

main.py
outputs/
notebooks/
reports/