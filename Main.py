from logging import exception
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

app = Flask(__name__)

warnings.filterwarnings('ignore')
np.random.seed = 3050
data_set = pd.read_csv('dataset_customer_churn.csv', sep='^')
data_set = data_set.dropna()
client = MongoClient("localhost", 27017)
clientes = client.test.get_collection('CLIENTES')
model = client.test.get_collection('MODELOS')
data_set = data_set.drop(['CD_ASSOCIADO','CODIGO_BENEFICIARIO','REALIZOU_EXODONTIA_COBERTA','REALIZOU_ENDODONTIA_COBERTA', 'A006_REGISTRO_ANS','A006_NM_PLANO','CD_USUARIO','CLIENTE','FORMA_PGTO_MENSALIDADE','QTDE_ATO_N_COBERTO_EXECUTADO','QTDE_ATENDIMENTOS'], axis=1)

#Pegando o Y

y = data_set['SITUACAO']
dici_trad = {
    'DESATIVADO' : 0,
    'ATIVO':1
}
y =  y.replace(dici_trad)
y = pd.Series(y)

#Pegando o X

dummies = pd.get_dummies(data_set[['SEXO','ESTADO_CIVIL','REALIZOU_PROCEDIMEN_ALTO_CUSTO','DIAS_ATE_REALIZAR_ALTO_CUSTO','PLANO','CODIGO_FORMA_PGTO_MENSALIDADE']])
num = data_set.drop(['SEXO','ESTADO_CIVIL','REALIZOU_PROCEDIMEN_ALTO_CUSTO','DIAS_ATE_REALIZAR_ALTO_CUSTO','PLANO','CODIGO_FORMA_PGTO_MENSALIDADE','SITUACAO'], axis=1)
x = pd.concat([dummies, num], axis=1)

#Inserindo no mongoDB

data_set = data_set.drop('SITUACAO', axis=1)
data_set = data_set.to_dict(orient='records')
clientes.insert_many(data_set)
data = pd.DataFrame(clientes.find())

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
    print("Concluido com sucesso!")

main()

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

@app.route('/')
def home() :
    return "Api para projeto 2"

app.run(debug=True)
