# Projeto_2

# Projeto

Projeto para a segunda fase do programa de Trainee Wise sendo um complemento do projeto 1 transformando ele em uma Api consumivel.

# Como utilizar

#### 1 Passo - Execute o script ele vai na port 5000

#### 2 Passo - Se preferir utilizando o postman teste se esta funcionando com http://127.0.0.1:5000/

#### 3 Passo - Popule o banco com http://127.0.0.1:5000/popular/

#### 4 Passo - se quiser adicione os modelos ao banco com http://127.0.0.1:5000/modelos/

#### 5 Passo - se quiser calcule a probabilidade dos clientes do banco com http://127.0.0.1:5000/probs/<num_para_vc_add>

#### 6 Passo - Add o seu cliente com o http://127.0.0.1:5000/adicionar/ mas tem que enviar um objeto do tipo json como :

#### {
        "CODIGO_FORMA_PGTO_MENSALIDADE" : "B",
        "DIAS_ATE_REALIZAR_ALTO_CUSTO" : "31-60",
        "ESTADO_CIVIL" : "solteiro",
        "IDADE" : 40.0,
       "NUM_BENEFICIARIOS_FAMILIA" : 2.0,
       "PLANO" : "408875991",
        "QTDE_ATO_COBERTO_EXECUTADO" : 20.0,
       "QTDE_DIAS_ATIVO" : 654.0,
       "REALIZOU_PROCEDIMEN_ALTO_CUSTO" : "SIM",
       "SEXO" : "M"
 ####   }
