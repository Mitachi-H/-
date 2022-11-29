import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#データ読み込み
data = pd.read_csv("./data.csv")
x1List=np.array(data["x1"],dtype=float)
x2List=np.array(data["x2"],dtype=float)
classList=np.array(data["クラス"],dtype=str)

data2 = pd.read_csv("./data2.csv")
x1List2=np.array(data["x1"],dtype=float)
x2List2=np.array(data["x2"],dtype=float)

#最近傍決定則
#prot=proto(使いたいテストデータ)=>classificateNNDBで結果得られる
def proto(x1List,x2List,classList):
    answer=[[],[]]
    for i in range(0,3):
        answer[0].append(np.nanmean(np.where(classList=="C"+str(i+1),x1List,np.nan)))
        answer[1].append(np.nanmean(np.where(classList=="C"+str(i+1),x2List,np.nan)))
    return answer

def NNDB(x1List,x2List):
    index=np.argmin([(x1List-prot[0][0])**2+(x2List-prot[1][0])**2,
    (x1List-prot[0][1])**2+(x2List-prot[1][1])**2,
    (x1List-prot[0][2])**2+(x2List-prot[1][2])**2])
    return "C"+str(index+1)

classificateNNDB = np.frompyfunc(NNDB, 2, 1)

#パーセプトロン
#w=perceptron(使いたいデータ)=>classficatePerceptronで結果取得
def perceptron(x1List,x2List,classList):
    w=[np.array([1,0,0],dtype=float),np.array([2.0,0,0],dtype=float),np.array([1,0,1],dtype=float)]
    p=1
    while(True):
        changed=False
        for x1,x2,_class in zip(x1List,x2List,classList):
            realClass=int(_class[-1])-1
            X=np.array([1,x1,x2])
            judgeClass=np.argmax([np.dot(w[i],X) for i in range(0,3)])
            if(realClass!=judgeClass):
                changed=True
                w[realClass]+=p*X
                w[judgeClass]-=p*X
        if(not changed):
            break
    return w

def classificatePerceptron(x1List,x2List,w):
    answer=[]
    for x1,x2 in zip(x1List,x2List):
        X=np.array([1,x1,x2])
        judgeClass=np.argmax([np.dot(w[i],X) for i in range(0,3)])
        answer.append("C"+str(judgeClass+1))
    return answer

#区分的線形識別法
#testListを更新=>classficatePLAMで結果取得
def PLAM(x1List,x2List):
    arr=[]
    for studyx1,studyx2 in zip(studyx1List,studyx2List):
        arr.append((x1List-studyx1)**2+(x2List-studyx2)**2)
    return studyclassList[np.argmin(arr)]
classificatePLAM = np.frompyfunc(PLAM, 2, 1)

#正誤判定
def checkClassification(judge,real):
    l= len(judge)
    answer=0
    for j,r in zip(judge,real):
        if j==r:
            answer+=1/l
    return answer

#性能評価
#5個を学習データ、5個をテストデータに分け正解率を計算。学習データとテストデータを逆にした場合も計算して平均を出す。

def checkresult():
    print(f"最近傍決定則:正解率={checkClassification(classificateNNDB(testx1List,testx2List),testclassList)}")
    print(f"パーセプトロン:正解率={checkClassification(classificatePerceptron(testx1List,testx2List,perceptron(studyx1List,studyx2List,studyclassList)),testclassList)}")
    print(f"区分的線形識別法:正解率={checkClassification(classificatePLAM(testx1List,testx2List),testclassList)}\n")
    pass

studyx1List=np.concatenate([x1List[0:5],x1List[10:15],x1List[20:25]])
studyx2List=np.concatenate([x2List[0:5],x2List[10:15],x2List[20:25]])
studyclassList=np.concatenate([classList[0:5],classList[10:15],classList[20:25]])
testx1List=np.concatenate([x1List[5:10],x1List[15:20],x1List[25:30]])
testx2List=np.concatenate([x2List[5:10],x2List[15:20],x2List[25:30]])
testclassList=np.concatenate([classList[5:10],classList[15:20],classList[25:30]])

prot=proto(studyx1List,studyx2List,studyclassList)
checkresult()

studyx1List=np.concatenate([x1List[5:10],x1List[15:20],x1List[25:30]])
studyx2List=np.concatenate([x2List[5:10],x2List[15:20],x2List[25:30]])
studyclassList=np.concatenate([classList[5:10],classList[15:20],classList[25:30]])
testx1List=np.concatenate([x1List[0:5],x1List[10:15],x1List[20:25]])
testx2List=np.concatenate([x2List[0:5],x2List[10:15],x2List[20:25]])
testclassList=np.concatenate([classList[0:5],classList[10:15],classList[20:25]])

prot=proto(studyx1List,studyx2List,studyclassList)
checkresult()

studyx1List=x1List
studyx2List=x2List
studyclassList=classList
prot=proto(studyx1List,studyx2List,studyclassList)
withNNDB=classificateNNDB(x1List2,x2List2)
withPerceptron=classificatePerceptron(x1List2,x2List2,perceptron(studyx1List,studyx2List,studyclassList))
withPLAM=classificatePLAM(x1List2,x2List2)
print(withNNDB)
print(withPerceptron)
print(withPLAM)
print(f"最近傍決定則とパーセプトロンの一致度={checkClassification(withNNDB,withPerceptron)}")
print(f"パーセプトロンと区分的線形識別法の一致度={checkClassification(withPerceptron,withPLAM)}")
print(f"区分的線形識別法と最近傍決定則の一致度={checkClassification(withPLAM,withNNDB)}")



#クラス未知のものの分類

plt.scatter(x1List,x2List,c=classList)
plt.scatter(prot[0],prot[1],marker='^')
plt.show()