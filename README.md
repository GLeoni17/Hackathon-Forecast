# Hackaton Forecast - Gabriel Leoni e Luan Bonizi

## 🎯 Objetivo
Você deverá desenvolver um modelo de previsão de vendas (forecast) para apoiar o varejo na reposição de produtos. A tarefa é prever a quantidade semanal de vendas por PDV (Ponto de Venda) /SKU Stock Keeping Unit (ou Unidade de Manutenção de Estoque) para as cinco semanas de janeiro/2023, utilizando como base o histórico de vendas de 2022.

## Submissão
CSV no formato semana|pdv|produto|quantidade

## Descrição do problema
O problema é da categoria de previsão de vendas, também chamado de forecast. Deve ser realizada uma previsão da quantidade de produtos a serem vendidos por PDV, para as cinco primeiras semanas de janeiro/2023. A métrica utilizada para avaliação é WMAPE, que é um erro ponderado, ou seja, aquele que se aproxima mais do valor, apresenta WMAPE menor.

## Modelos já utilizados
Avaliação realizada em cima da previsão das semanas do mês de novembro e dezembro/2022, dos dados de treino
- lightbgm
- xgboost
- lstm

## Etapas Necessárias
- Criação dos .parquet utilizados no preprocessing.py