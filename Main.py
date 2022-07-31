from multiprocessing.dummy import DummyProcess
from multiprocessing.sharedctypes import Value
from optparse import Values
from pydoc import cli
import pandas as pd
from bson import ObjectId
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from pymongo import MongoClient
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from flask import Flask,request,jsonify
import warnings

from visions import Object

app = Flask(__name__)

warnings.filterwarnings('ignore')
np.random.seed = 3050
data_set = pd.read_csv('dataset_customer_churn.csv', sep='^')
data_set = data_set.dropna()
client = MongoClient("localhost", 27017)
clientes = client.test.get_collection('CLIENTES')
model = client.test.get_collection('MODELOS')
data_set = data_set.drop(['CD_ASSOCIADO','CODIGO_BENEFICIARIO','REALIZOU_EXODONTIA_COBERTA','REALIZOU_ENDODONTIA_COBERTA', 'A006_REGISTRO_ANS','A006_NM_PLANO','CD_USUARIO','CLIENTE','FORMA_PGTO_MENSALIDADE','QTDE_ATO_N_COBERTO_EXECUTADO','QTDE_ATENDIMENTOS'], axis=1)
colunas = ['CODIGO_FORMA_PGTO_MENSALIDADE','DIAS_ATE_REALIZAR_ALTO_CUSTO','ESTADO_CIVIL','IDADE','NUM_BENEFICIARIOS_FAMILIA','PLANO','QTDE_ATO_COBERTO_EXECUTADO','QTDE_DIAS_ATIVO','REALIZOU_PROCEDIMEN_ALTO_CUSTO','SEXO']

#Pegando o Y

def y():
    y = data_set['SITUACAO']
    dici_trad = {
        'DESATIVADO' : 0,
        'ATIVO':1
    }
    y =  y.replace(dici_trad)
    return pd.Series(y)
y = y()

#Pegando o X

def x():
    dummies = pd.get_dummies(data_set[['SEXO','ESTADO_CIVIL','REALIZOU_PROCEDIMEN_ALTO_CUSTO','DIAS_ATE_REALIZAR_ALTO_CUSTO','PLANO','CODIGO_FORMA_PGTO_MENSALIDADE']])
    num = data_set.drop(['SEXO','ESTADO_CIVIL','REALIZOU_PROCEDIMEN_ALTO_CUSTO','DIAS_ATE_REALIZAR_ALTO_CUSTO','PLANO','CODIGO_FORMA_PGTO_MENSALIDADE','SITUACAO'], axis=1)
    return pd.concat([dummies, num], axis=1)
x = x()

def separando_x(data_set):
    dummy = pd.get_dummies(data_set[['SEXO','ESTADO_CIVIL','REALIZOU_PROCEDIMEN_ALTO_CUSTO','DIAS_ATE_REALIZAR_ALTO_CUSTO','PLANO','CODIGO_FORMA_PGTO_MENSALIDADE']])
    numeros = data_set.drop(['SEXO','ESTADO_CIVIL','REALIZOU_PROCEDIMEN_ALTO_CUSTO','DIAS_ATE_REALIZAR_ALTO_CUSTO','PLANO','CODIGO_FORMA_PGTO_MENSALIDADE'], axis=1)
    return pd.concat([dummy, numeros], axis=1)

#Inserindo no mongoDB
def populando_banco():
    data_set_banco = data_set.drop('SITUACAO', axis=1)
    data_set_banco = data_set_banco.to_dict(orient='records')
    clientes.insert_many(data_set_banco)
    inserindo_situacao()
    
    
def inserindo_situacao():
    ids_lista = clientes.distinct('_id')
    
    clientes.update_many({}, {
        '$set' : {'PREVISOES' : []}
    })
    
    for (i,j) in zip(y.to_list(), ids_lista):
        clientes.update_one({'_id' : j}, {
            '$push' : {'PREVISOES' : i}
        })
    

### TREINO E TESTE DOS MODELOS ###

x_treino,x_teste,y_treino,y_teste = train_test_split(x, y, test_size=0.3, stratify=y)

def view_score(y_teste, p, mod) :
    baseline = np.ones(p.shape)

    ac = accuracy_score(y_teste,p)
    pc = precision_score(y_teste,p)
    f1 = f1_score(y_teste,p)

    print("Pelo Accuracy: ",ac * 100,"%")
    print("Pelo Precision: ",pc * 100,"%")
    print("Pelo F1: ",f1 * 100,"%")
    print()
    print("BASELINE")

    ac_b = accuracy_score(y_teste,baseline)
    pc_b = precision_score(y_teste,baseline)
    f1_b = f1_score(y_teste,baseline)

    print("Pelo Accuracy: ",ac_b * 100,"%")
    print("Pelo Precision: ",pc_b * 100,"%")
    print("Pelo F1: ",f1_b * 100,"%")
    print()
    print("Pelo cross validation")
    cross_validation(x, y,modelo=mod)
    
    
def cross_validation(x, y, modelo):
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    results = cross_validate(modelo, x, y,cv = cv)
    results = pd.DataFrame(results)
    media = results.test_score.mean() * 100
    desvio = results.test_score.std()
    print("Média Teste: %.2f" %(media),"%")
    print("Entre : [%.2f, %.2f]" %((media - 2 * desvio), (media + 2 * desvio)),"%")
    
    
def modelos_banco(mod) :
    model.insert_one({
        'MODELO' : str(mod.__class__)  
        })

def rf():
    print("Random Forest Classifier")
    rf = RandomForestClassifier(max_depth=3, n_estimators=10)
    rf.fit(x_treino, y_treino)
    return rf
    
    
def lr():
    print("Logistic Regression")
    lr = LogisticRegression(max_iter=50)
    lr.fit(x_treino,y_treino)
    return lr
    
    
def mlp():
    print("MLP Classifier")
    mlp = MLPClassifier(max_iter=3)
    mlp.fit(x_treino,y_treino)
    return mlp


def add_modelos_ao_banco():
    modelos_banco(rf())
    modelos_banco(lr())
    modelos_banco(mlp())


def clustering():
    print()
    print("Clustering dos dados pelo KMeans")
    print()
    data_final = pd.concat([x,y], axis=1)
    data_final = data_final.dropna()
    values = Normalizer().fit_transform(data_final.values)
    km = KMeans(n_clusters=5,n_init=10,max_iter=300)
    km_pred = km.fit_predict(values)
    clus = data_final
    clus['CLUSTER'] = km.labels_
    descp = clus.groupby('CLUSTER')['QTDE_DIAS_ATIVO','QTDE_ATO_COBERTO_EXECUTADO']
    descp = pd.DataFrame(descp.mean())
    descp['CLIENTES'] = clus['CLUSTER'].value_counts()
    print(descp)
    

def main():
    print("Iniciando...")
    populando_banco()
    print("Concluido com sucesso!")


@app.route('/busca/', methods=['POST'])
def busca():
    req = request.get_json()
    dado = req["_id"]
    try :
        pessoa = clientes.find_one({"_id" : ObjectId(dado)})
    except :
        return jsonify(nao_consegui_achar = str(dado))
    pessoa["_id"] = str(pessoa["_id"])
    return jsonify(previsoes = pessoa["PREVISOES"])

@app.route('/popular/')
def popular():
    main()
    return "MongoDB populado"

@app.route('/')
def home() :
    return "Api para projeto 2"


@app.route('/adicionar/', methods=['POST'])
def adicionar_pred():
    print("Adicionando...")
    mod1 = rf()
    mod2 = lr()
    mod3 = mlp()
    req = request.get_json()
    input = [req[col] for col in colunas]
    pessoa = clientes.insert_one({
        'CODIGO_FORMA_PGTO_MENSALIDADE' : input[0],
        'DIAS_ATE_REALIZAR_ALTO_CUSTO' : input[1],
        'ESTADO_CIVIL' : input[2],
        'IDADE' : input[3],
        'NUM_BENEFICIARIOS_FAMILIA' : input[4],
        'PLANO' : input[5],
        'PREVISOES' : [],
        'QTDE_ATO_COBERTO_EXECUTADO' : input[6],
        'QTDE_DIAS_ATIVO' : input[7],
        'REALIZOU_PROCEDIMEN_ALTO_CUSTO' : input[8],
        'SEXO' : input[9]
    })
    
    pessoa = clientes.find_one({
        'CODIGO_FORMA_PGTO_MENSALIDADE' : input[0],
        'DIAS_ATE_REALIZAR_ALTO_CUSTO' : input[1],
        'ESTADO_CIVIL' : input[2],
        'IDADE' : input[3],
        'NUM_BENEFICIARIOS_FAMILIA' : input[4],
        'PLANO' : input[5],
        'PREVISOES' : [],
        'QTDE_ATO_COBERTO_EXECUTADO' : input[6],
        'QTDE_DIAS_ATIVO' : input[7],
        'REALIZOU_PROCEDIMEN_ALTO_CUSTO' : input[8],
        'SEXO' : input[9]
    })
    
    data = pd.DataFrame(list(clientes.find()))
    data = data.drop(["_id","PREVISOES"], axis=1)
    data = list(separando_x(data).iloc[-1])
    print(data)
    pred1 = mod1.predict([data])
    pred2 = mod2.predict([data])
    pred3 = mod3.predict([data])
    resultados = [int(pred1[0]),int(pred2[0]),int(pred3[0])]
    print(resultados)
    clientes.find_one_and_update({"_id" : ObjectId(pessoa['_id'])}, {
       '$push' : {
          'PREVISOES' : {'$each' : resultados }
       }
    })
    
    return jsonify(pessoa_adicionada = str(pessoa["_id"]),
                    previsoes_dela = resultados
                   )

app.run(debug=True)

