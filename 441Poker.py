import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


data="poker-hand-training-true.data"
df=pd.read_csv(data)


sameRanks=df.loc[(df.Y == 1) | (df.Y == 2) | (df.Y == 3) | (df.Y == 6) | (df.Y == 7)]
sequentialRanks=df.loc[(df.Y == 0) | (df.Y == 4) | (df.Y == 5) | (df.Y == 8) | (df.Y == 9)]


features=['Type','Num','Type2','Num2','Type3','Num3','Type4','Num4','Type5','Num5']
X_train_same=sameRanks[features]
Y_train_same=sameRanks['Y']

X_train_seq=sequentialRanks[features]
Y_train_seq=sequentialRanks['Y']



sameTree = DecisionTreeClassifier(criterion="entropy", max_depth=3)
seqTree = DecisionTreeClassifier(criterion="entropy", max_depth=3)


sameTree = sameTree.fit(X_train_same,Y_train_same)
seqTree = seqTree.fit(X_train_seq,Y_train_seq)

trainData="poker-hand-testing.data"
df1=pd.read_csv(trainData)
sameRanksTrain=df1.loc[(df1.Num == df1.Num2) | (df1.Num == df1.Num3) | (df1.Num == df1.Num4) | (df1.Num == df1.Num5) | (df1.Num2 == df1.Num3)
 | (df1.Num2 == df1.Num4) | (df1.Num2 == df1.Num5) | (df1.Num3 == df1.Num4) | (df1.Num3 == df1.Num5) | (df1.Num4 == df1.Num5) ]

seqRanksTrain=df1.loc[-((df1.Num == df1.Num2) | (df1.Num == df1.Num3) | (df1.Num == df1.Num4) | (df1.Num == df1.Num5) | (df1.Num2 == df1.Num3)
 | (df1.Num2 == df1.Num4) | (df1.Num2 == df1.Num5) | (df1.Num3 == df1.Num4) | (df1.Num3 == df1.Num5) | (df1.Num4 == df1.Num5)) ]

X_test_same=sameRanksTrain[features]
Y_test_same=sameRanksTrain['Y']

X_test_seq=seqRanksTrain[features]
Y_test_seq=seqRanksTrain['Y']

y_pred_same = sameTree.predict(X_test_same)
y_pred_seq = seqTree.predict(X_test_seq)




k=Y_test_same.shape[0]
k1=Y_test_seq.shape[0]
accSame=metrics.accuracy_score(Y_test_same, y_pred_same )
accSeq=metrics.accuracy_score(Y_test_seq, y_pred_seq )

print("Accuracy:",((k*accSame)+(k1*accSeq)) / (k+k1) )



