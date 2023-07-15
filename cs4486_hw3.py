# CHENG HO MAN 56208961
# CHENG Hong Wai 56216309
# CHONG Chun Yu 56225263
from sklearn import tree, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import random
import tkinter as tk

def features_importances():
    cols = [0,1,2,3,4,5,10,13,16,26,29,32]
    stat.drop(stat.columns[cols],axis=1, inplace=True)

    features = stat.drop(columns = 'WINorLOSS')
    labels = stat['WINorLOSS']

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2)

    # Logistic Regression
    model = LogisticRegression(max_iter=10000,random_state=2)
    model.fit(x_train, y_train)
    y_pred_lr = model.predict(x_test)

    # Decision Tree
    clf = tree.DecisionTreeClassifier(random_state=0) 
    clf = clf.fit(x_train, y_train)
    y_pred_dt = clf.predict(x_test)

    accuracy_dt = metrics.accuracy_score(y_test, y_pred_dt)
    accuracy_lr = metrics.accuracy_score(y_test, y_pred_lr)

    text4 = ('Accuracy Score for Decision Tree: {accuracy_dt}'.format(accuracy_dt=accuracy_dt))
    text5 = ('Accuracy Score for Logistic Regression: {accuracy_lr}'.format(accuracy_lr=accuracy_lr))
    tk.Label(window, text=text4).place(relx=0.1, rely=0.85)
    tk.Label(window, text=text5).place(relx=0.1, rely=0.9)
def predict(T1,T2):

    global Team1_TP_mean, Team2_TP_mean, Team1_TP_SD, Team2_TP_SD, Team1_OP_mean, Team2_OP_mean, Team1_OP_SD, Team2_OP_SD

    Team1_stat = stat[(stat['Team']==T1)]
    Team2_stat = stat[(stat['Team']==T2)]   

    Team1_TP_mean = Team1_stat.TeamPoints.mean()
    Team2_TP_mean = Team2_stat.TeamPoints.mean()
    Team1_TP_SD = Team1_stat.TeamPoints.std()
    Team2_TP_SD = Team2_stat.TeamPoints.std()

    Team1_OP_mean = Team1_stat.OpponentPoints.mean()
    Team2_OP_mean = Team2_stat.OpponentPoints.mean()
    Team1_OP_SD = Team1_stat.OpponentPoints.std()
    Team2_OP_SD = Team2_stat.OpponentPoints.std()

    runSimulator(1000,T1,T2)

def simulation():
    Team1_points = (random.gauss(Team1_TP_mean,Team1_TP_SD)+ random.gauss(Team2_OP_mean,Team2_OP_SD))/2
    Team2_points = (random.gauss(Team2_TP_mean,Team2_TP_SD)+ random.gauss(Team1_OP_mean,Team1_OP_SD))/2
    if int(round(Team1_points)) > int(round(Team2_points)):
        return 1
    elif int(round(Team1_points)) < int(round(Team2_points)):
        return -1
    return 0

def runSimulator(times, T1, T2):
    Team1_win = 0
    Team2_win = 0
    tie = 0
    for i in range(times):
        gm = simulation()
        if gm == 1:
            Team1_win +=1 
        elif gm == -1:
            Team2_win +=1
        else: tie +=1 

        
    text1 = ('{T1} Win rate: {rate:.3f}%    '.format(T1 = T1 , rate=Team1_win/(Team1_win+Team2_win+tie)*100))
    text2 = ('{T2} Win rate: {rate:.3f}%    '.format(T2 = T2, rate=Team2_win/(Team1_win+Team2_win+tie)*100))
    text3 = ('Tie rate: {rate:.3f}%     '.format(rate=tie/(Team1_win+Team2_win+tie)*100))

    tk.Label(window, text=text1).place(relx=0.1, rely=0.7)
    tk.Label(window, text=text2).place(relx=0.1, rely=0.75)
    tk.Label(window, text=text3).place(relx=0.1, rely=0.8)



def Onclick():
    T1 = textBox1.get(1.0, "end-1c")
    T2 = textBox2.get(1.0, "end-1c")
    predict(T1,T2)
    features_importances()


if __name__=="__main__":
    stat = pd.read_csv('nba.games.stats.csv')

    window = tk.Tk()
    window.title('window')
    window.geometry('500x500')

    header_label = tk.Label(window, text='NBA Game Prediction', font=('Arial', 18)).place(relx=0.1, rely=0.1)


    label1 = tk.Label(window, text='Please input the first team name:').place(relx=0.1, rely=0.2)
    textBox1=tk.Text(window, height=2, width=10)
    textBox1.place(relx=0.55, rely=0.2)

    label2 = tk.Label(window, text='Please input the second team name:').place(relx=0.1, rely=0.3)
    textBox2=tk.Text(window, height=2, width=10)
    textBox2.place(relx=0.55, rely=0.3)

    MyButton1 = tk.Button(window, text="Submit", width=10, command=Onclick)
    MyButton1.place(relx=0.1, rely=0.5)

    window.mainloop()
