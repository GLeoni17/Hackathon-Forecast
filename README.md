# Hackaton Forecast Big Data - Gabriel Leoni e Luan Bonizi

## 🎯 Objetivo
O objetivo do Hackathon foi desenvolver um modelo de previsão de vendas (forecast) para apoiar o varejo na reposição de produtos da empresa Big Data. A tarefa é prever a quantidade semanal de vendas por PDV (Ponto de Venda) /SKU Stock Keeping Unit (ou Unidade de Manutenção de Estoque) para as cinco semanas de janeiro/2023, utilizando como base o histórico de vendas de 2022.

## Submissão
O arquivo de submissão precisa ser um csv ou parquet com a seguinte estrutura de colunas: [semana| pdv | produto | quantidade], onde:
  - 'semana' é um atributo que pode conter os valores inteiro de 1 a 5, representando uma semana específica de janeiro/2023;
  - 'pdv' é o código (ID) da loja que está vendendo um determinado produto;
  - 'produto' é o ID do produto em si;
  - 'quantidade' é a quantidade de vendas do produto representado pela tripla (semana, pdv, produto)

## Descrição do problema
O problema é da categoria de previsão de vendas, também chamado de forecast. Deve ser realizada uma previsão da quantidade de produtos a serem vendidos por PDV, para as cinco primeiras semanas de janeiro/2023. A métrica utilizada para avaliação é WMAPE, que é um erro ponderado, ou seja, aquele que se aproxima mais do valor, apresenta WMAPE menor. o WMAPE é a versão ponderada do MAPE, uma métrica que calcula a média dos erros percentuais absolutos entre as previsões e os valores reais.

## Modelos já utilizados
Avaliação realizada em cima da previsão das semanas do mês de novembro e dezembro/2022, dos dados de treino
- lightbgm
- xgboost
- lstm

## Etapas Necessárias
- Criação dos .parquet utilizados no preprocessing.py
