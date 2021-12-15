import numpy as np
import math as m
import pandas as pd
from scipy.optimize import minimize
from matplotlib import pyplot as plt




def DuckworthLewis20Params(file_name):

    print('Question 1 started')

    df = pd.read_csv("04_cricket_1999to2011.csv")
    dataframe = df.loc[df.Innings == 1]
    # print(dataframe)
    u = 50 - dataframe['Over'].values  ## overs to go
    w = dataframe['Wickets.in.Hand'].values  # wickets in hand
    Y_true = dataframe['Runs.Remaining'].values  # runs remaining to score

    loss = 0
    # parameter_initialization
    # Z_0=np.random.randint(0,350,10)
    Z_0 = [12, 30, 48, 80, 110, 140, 168, 210, 232, 285]
    # b=np.random.uniform(0, 1,10)
    b = [0.45, 0.4, 0.3, 0.2, 0.05, 0.1, 0.1, 0.1, 0.05, 0.1]
    parameters = np.array([Z_0, b])
    arguments = [u, w, Y_true]

    def loss_function(parameters, arguments):

        z_0 = parameters[0:10]
        b = parameters[10:]
        u, w, y_true = arguments
        loss = 0
        # for i in range(0,len(u)):
        #     current_w=w[i]
        #     current_z_0=z_0[current_w-1]
        #     b_w=b[current_w-1]
        #     predicted_score=current_z_0*(1-np.exp(-b_w*u[i]))
        #     loss=loss(predicted_score-y_true[i])**2

        # vectorization-----------
        Z = np.zeros(len(u))
        B = np.zeros(len(u))
        for i in range(1, 11):
            Z[w == i] = z_0[i - 1]
            B[w == i] = b[i - 1]
        predicted_score = Z * (1 - np.exp(-B * u))
        loss = (np.sum((predicted_score - y_true) ** 2))/len(u)
        # print('loss=',loss)


        return loss
    optimal_solution = minimize(fun=loss_function, x0=parameters, args=arguments, method='L-BFGS-B')
    loss = optimal_solution['fun']  ## need to be returned in report

    print('total error per point for Q1=', loss)
    optimal_parameters = optimal_solution['x']
    z0_final = optimal_parameters[:10]
    b_final = optimal_parameters[10:]
    print('optimal parameters calculated')
    print('ZO=', z0_final)
    print('b=',b_final)

    plt.figure()
    plt.title('Expected Runs Vs Overs to go')
    plt.xlim((0, 50))
    plt.ylim((0, 350))
    plt.xticks([0, 10, 20, 30, 40, 50])
    plt.yticks(np.arange(0, 351, 50))
    plt.xlabel('Overs to go')
    plt.ylabel('Expected Runs')

    # testing part
    for i in range(1, 11):
        u = np.arange(0, 51, 1)
        colors = ["#3399ff", '#cc99ff', '#cc0099', '#ff9966', '#4d1a00', '#66ff66', '#003300', '#cc3300', '#ff0000',
                  '#000000']
        expected_runs = z0_final[i - 1] * (1 - np.exp(-b_final[i - 1] * u))
        plt.plot(u, expected_runs, colors[i - 1], label=f'w={i}')
        plt.legend()
    plt.savefig("Expected Runs Vs Overs to go")
    print('Question  1  end')

    return z0_final, b_final

#----------------------------------------------------QUESTION2----------------------------------------------------------

def lossFunctionDL(parameters, arguments):
    df = pd.read_csv("04_cricket_1999to2011.csv")
    dataframe = df.loc[df.Innings == 1]
    # print(dataframe)
    u = 50 - dataframe['Over'].values  ## overs to go
    w = dataframe['Wickets.in.Hand'].values  # wickets in hand
    Y_true = dataframe['Runs.Remaining'].values  # runs remaining to score

    z_0 = parameters[0:10]
    L = parameters[10]
    u, w, y_true = arguments
    Z = np.zeros(len(u))
    for i in range(1, 11):
        Z[w == i] = z_0[i - 1]
    predicted_score = Z * (1 -1*np.exp(-1*L*u/(Z+1e-8)))
    loss = (np.sum((predicted_score - y_true) ** 2))/len(u)
    # print(loss)

    return loss




def DuckworthLewis11Params(file_name):
    print('Q2 started')

    df = pd.read_csv("04_cricket_1999to2011.csv")
    dataframe = df.loc[df.Innings == 1]
    # print(dataframe)
    u = 50 - dataframe['Over'].values  ## overs to go
    w = dataframe['Wickets.in.Hand'].values  # wickets in hand
    Y_true = dataframe['Runs.Remaining'].values  # runs remaining to score
    parameters= [20,30,35,55,80,110,120,180,195,250,20] # 20 is the initialized value of L here we yave taken L=20
    arguments = [u, w, Y_true]

    optimal_solution = minimize(fun=lossFunctionDL, x0=parameters, args=arguments, method='L-BFGS-B',options={'maxiter':15})
    loss = optimal_solution['fun']  ## need to be returned in report
    optimal_parameters = optimal_solution['x']
    print('total error per point for Q2=',loss)
    z0_final = optimal_parameters[:10]
    L_final = optimal_parameters[10]

    print('Zo =', z0_final)
    print('value of L=',L_final)

    plt.figure()
    plt.title('Expected Runs Vs Overs to go')
    plt.xlim((0, 50))
    plt.ylim((0, 350))
    plt.xticks([0, 10, 20, 30, 40, 50])
    plt.yticks(np.arange(0, 351, 50))
    plt.xlabel('Overs to go')
    plt.ylabel('Expected Runs')

    # testing part
    for i in range(1, 11):
        u = np.arange(0, 51, 1)
        colors = ["#3399ff", '#cc99ff', '#cc0099', '#ff9966', '#4d1a00', '#66ff66', '#003300', '#cc3300', '#ff0000',
                  '#000000']
        expected_runs = z0_final[i - 1] * (1 - np.exp((-L_final*u)/(z0_final[i-1]+1e-12)))
        plt.plot(u, expected_runs, colors[i - 1], label=f'w={i}')
        plt.legend()
    plt.savefig(" Q2--Expected Runs Vs Overs to go")
    print('Q2 end')

    return z0_final, loss

if __name__ == '__main__':
     DuckworthLewis20Params(file_name="04_cricket_1999to2011.csv")  ##----Question1----#
     DuckworthLewis11Params(file_name="04_cricket_1999to2011.csv")  ##----Question2----#
















