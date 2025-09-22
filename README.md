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

## Modelos utilizados
Para a resolução do problema, foram testados os seguintes modelos:
- lightbgm
- xgboost
- catboost
- ridge
- lstm

Para validação, foram utilizadas as 5 últimas semanas de 2022. O modelo com a melhor validação foi o Catboost, alcançando 0.217 de WMAPE. Porém, o modelo que alcançou o melhor resultado final em relação aos dados de teste reais foi o XGBoost, cujos parâmetros foram os mesmos do códido em 'train_model.py', alcançando um WMAPE de 0.701.

## Como rodar o código e realizar as predições
<ol>
  <li>Primeiro, é necessário baixar os dados de treino do Hackathon, os quais são 3 arquivos .parquet com os dados dos produtos, lojas e transações de 2022;</li>
  <li>Depois de baixar os arquivos, é necessário movê-los para a pasta <strong>data/</strong>, onde eles serão lidos e pré-processados; </li>
  <li>Para pré-processar os dados, basta rodar, primeiramente, o arquivo <strong>preprocessing.py</strong> através do comando <code>python3 preprocessing.py </code>. O outuput deste código é o arquivo weekly.parquet, uma tabela que agrega as transações com as outras informações dos produtos e lojas. O arquivo será salvo na pasta <strong>data/</strong>;</li>
  <li>Depois de pré-processar, é necessário rodar o arquivo <strong>create_weekly_future.py</strong> usando o código <code>python3 create_weekly_future.py</code>. Com isso, será criado o arquivo weekly_future.parquet, o qual é o esqueleto dos dados de teste, contendo as 5 semanas de janeiro/2023 com os produtos que aparecem no arquivo weekly.parquet. Este arquivo também é salvo na pasta <strong>data/</strong>;</li>
  <li>Com os arquivos criados, agora é necessário rodar o código <code>python3 feature_extraction.py</code> para extrair as características de treinamento e teste. Este código produz os dados de treino e teste weekly_train.parquet e weekly_teste.parquet, respectivamente. Esses arquivos também serão salvos na pasta <strong>data/</strong>;</li>
  <li>Por último, basta rodar o código <code>python3 train_model.py</code> para treinar um modelo XGBoost e, logo em seguida, realizar as predições para as 5 semanas de janeiro/2023. As predições serão salvas na pasta <strong>predicts/</strong> com o nome de predicoes.parquet.</li>
</ol>
