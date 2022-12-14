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
import pickle
import warnings
from random import randint

app = Flask(__name__)

warnings.filterwarnings('ignore')
np.random.seed = 3050
data_set = pd.read_csv('dataset_customer_churn.csv', sep='^')
data_set = data_set.dropna()
client = MongoClient("localhost", 27017)
clientes = client.test.get_collection('CLIENTES')
model = client.test.get_collection('MODELOS')
data_set = data_set.drop(['CD_ASSOCIADO','REALIZOU_EXODONTIA_COBERTA','REALIZOU_ENDODONTIA_COBERTA', 'A006_REGISTRO_ANS','A006_NM_PLANO','CD_USUARIO','FORMA_PGTO_MENSALIDADE','QTDE_ATO_N_COBERTO_EXECUTADO','QTDE_ATENDIMENTOS','CODIGO_BENEFICIARIO'], axis=1)
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
    num = data_set.drop(['SEXO','ESTADO_CIVIL','REALIZOU_PROCEDIMEN_ALTO_CUSTO','DIAS_ATE_REALIZAR_ALTO_CUSTO','PLANO','CODIGO_FORMA_PGTO_MENSALIDADE','SITUACAO','CLIENTE'], axis=1)
    return pd.concat([dummies, num], axis=1)
x = x()


def separando_x(data_set):
    dummy = pd.get_dummies(data_set[['SEXO','ESTADO_CIVIL','REALIZOU_PROCEDIMEN_ALTO_CUSTO','DIAS_ATE_REALIZAR_ALTO_CUSTO','PLANO','CODIGO_FORMA_PGTO_MENSALIDADE']])
    numeros = data_set.drop(['SEXO','ESTADO_CIVIL','REALIZOU_PROCEDIMEN_ALTO_CUSTO','DIAS_ATE_REALIZAR_ALTO_CUSTO','PLANO','CODIGO_FORMA_PGTO_MENSALIDADE','CLIENTE'], axis=1)
    return pd.concat([dummy, numeros], axis=1)


#Inserindo no mongoDB
def populando_banco():
    data_set_banco = data_set.drop('SITUACAO', axis=1)
    data_set_banco = data_set_banco.to_dict(orient='records')
    clientes.insert_many(data_set_banco)
    inserindo_situacao()
    
    
def inserindo_situacao():
    ids_lista = clientes.distinct('_id')
    resp = y
    
    clientes.update_many({}, {
        '$set' : {'PREVISOES' : []}
    })
    
    for (i,j) in zip(resp.to_list(), ids_lista):
        clientes.update_one({'_id' : j}, {
            '$push' : {'PREVISOES' : i}
        })
    
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
    print("M??dia Teste: %.2f" %(media),"%")
    print("Entre : [%.2f, %.2f]" %((media - 2 * desvio), (media + 2 * desvio)),"%")
    return results
    
    
def modelos_banco(mod) :
    cross_v = cross_validation(x, y, mod)
    
    model.insert_one({
        'MODELO' : str(mod.__class__),
        'CROSS_V_ACERT' : cross_v.test_score.mean() * 100,
        'DESVIO' : cross_v.test_score.std()
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
    
    
def mod_download():
    pickle.dump(rf(), open('rf.sav', 'wb'))
    pickle.dump(lr(), open('lr.sav', 'wb'))
    pickle.dump(mlp(), open('mlp.sav', 'wb'))


def ler_rf():
    return pickle.load(open('rf.sav', 'rb'))


def ler_lr():
    return pickle.load(open('lr.sav', 'rb'))


def ler_mlp():
    return pickle.load(open('mlp.sav', 'rb'))


@app.route('/probs/<num_registers>')
def prob_lista(num_registers):
    predi_prob1 = ler_rf()
    predi_prob2 = ler_lr()
    predi_prob3 = ler_mlp()
    
    prim_faixa = []
    seg_faixa = []
    terc_faixa = []
    quar_faixa = []
    
    registros = pd.DataFrame(clientes.find())
    registros = registros.drop(["_id","PREVISOES"], axis=1)
    registros = separando_x(registros)
        
    probabilidades1 = predi_prob1.predict_proba(registros[:int(num_registers)])
    probabilidades2 = predi_prob2.predict_proba(registros[:int(num_registers)])
    probabilidades3 = predi_prob3.predict_proba(registros[:int(num_registers)])
    
    respostas = np.concatenate((probabilidades1,probabilidades2,probabilidades3))
    
    respostas = list(respostas)
    
    for i in respostas:
        if (i.mean() > 0 and i.mean() <= 0.25):
            prim_faixa.append(i.mean())
        elif (i.mean() > 0.25 and i.mean() <= 0.5):
            seg_faixa.append(i.mean())
        elif (i.mean() > 0.5 and i.mean() <= 0.75):
            terc_faixa.append(max(i))  
        elif (i.mean() > 0.75 and i.mean() <= 1.0):
            seg_faixa.append(i.mean())
    
    return jsonify(
        de_0_a_25 = len(prim_faixa),
        de_25_a_50 = len(seg_faixa),
        de_50_a_75 =  len(terc_faixa),
        de_75_a_100 = len(quar_faixa)
    )
    

@app.route('/busca/', methods=['POST'])
def busca():
    req = request.get_json()
    dado = req['CLIENTE']
    try :
        pessoa = clientes.find_one({'CLIENTE' : int(dado)})
    except :
        return jsonify(nao_consegui_achar = int(dado))
    return jsonify(previsoes = pessoa["PREVISOES"])


@app.route('/popular/')
def popular():
    populando_banco()
    return "MongoDB populado"


@app.route('/')
def home() :
    return "Api para projeto 2"


@app.route('/adicionar/', methods=['POST'])
def adicionar_pred():
    print("Adicionando...")
    mod1 = ler_rf()
    mod2 = ler_lr()
    mod3 = ler_mlp()
    req = request.get_json()
    input = [req[col] for col in colunas]
    cod_cliente = clientes.distinct('CLIENTE')
    cod_cliente = max(cod_cliente) + 1
    pessoa = clientes.insert_one({
        'CLIENTE' : cod_cliente,
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
        'CLIENTE' : cod_cliente,
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
    
    return jsonify(pessoa_adicionada = str(pessoa["CLIENTE"]),
                    previsoes_dela = resultados
                   )


@app.route('/modelos/')
def add_modelos_ao_banco():
    modelos_banco(ler_lr())
    modelos_banco(ler_rf())
    modelos_banco(ler_mlp())
    
    return jsonify(
        resposta = "Adicionado com sucesso os modelos ao banco!"
    )
    
    
app.run(debug=True)
