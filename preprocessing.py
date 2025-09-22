import pandas as pd
from pathlib import Path

absolute_path = Path.cwd()

def create_weekly():
    
    print('Carregando arquivos de treino...')
    #Leitura dos arquivos de treino
    pdv = pd.read_parquet(Path.joinpath(absolute_path,"data/part-00000-tid-2779033056155408584-f6316110-4c9a-4061-ae48-69b77c7c8c36-4-1-c000.snappy.parquet"))
    transacao = pd.read_parquet(Path.joinpath(absolute_path,"data/part-00000-tid-5196563791502273604-c90d3a24-52f2-4955-b4ec-fb143aae74d8-4-1-c000.snappy.parquet"))
    produto = pd.read_parquet(Path.joinpath(absolute_path,"data/part-00000-tid-7173294866425216458-eae53fbf-d19e-4130-ba74-78f96b9675f1-4-1-c000.snappy.parquet"))
    
    print('Arquivos carregados!')
    
    #Converter data das transações para o tipo DateTime
    transacao['transaction_date'] = pd.to_datetime(transacao['transaction_date'])
    
    #Renomeação das colunas para ficar mais organizado
    transacao = transacao.rename(columns={
    'internal_store_id':'pdv_id',
    'internal_product_id':'produto_id'
    })
    
    pdv = pdv.rename(columns={'pdv':'pdv_id'})
    produto = produto.rename(columns={'produto':'produto_id'})
    
    print('Criando weekly.parquet com as transações e suas quantidades...')
    
    #Realiza um Merge na tabela 'transacao' com as outras tabelas, a fim de agregar as colunas por pdv
    # e id do produto
    transacao = transacao.merge(pdv, left_on='pdv_id', right_on='pdv_id', how='left')
    transacao = transacao.merge(produto, left_on='produto_id', right_on='produto_id', how='left')
    
    #Substitui os dados NaN por 'unknowm' a fim de evitar erros
    for c in ['categoria_pdv','premise','zipcode']:
        if c in transacao.columns:
            transacao[c] = transacao[c].fillna('unknown')
    
    #Transforma as datas de transação para o começo da semana, com isso é possível agregar as vendas por semana
    # com mais facilidade
    transacao['week_start'] = transacao['transaction_date'] - pd.to_timedelta(transacao['transaction_date'].dt.weekday, unit='d')
    
    #Agrega as vendas pela tripla (week_start, pdv_id, produto_id), somando os valores dos lucros e quantidades
    # de venda para ter as informações de vendas das semanas
    weekly = (transacao.groupby(['week_start','pdv_id','produto_id'], as_index=False)
                .agg(quantity=('quantity','sum'),
                    gross_value=('gross_value','sum'),
                    net_value=('net_value','sum'),
                    transactions=('quantity','size')))
    
    #Transforma a tabela no tipo DataFrame e salva 
    weekly = pd.DataFrame(weekly)
    weekly.to_parquet(Path.joinpath(absolute_path,"data/weekly.parquet"), index=False)
    
    print('weekly.parquet criado e salvo em: ', Path.joinpath(absolute_path,"data/weekly.parquet"))
    
    #O parquet 'weekly' contém as vendas dos produtos considerando todas as semanas que apareceram no
    # dataframe 'transação'
    
if __name__=='__main__':
    #Cria o df weekly
    create_weekly()
