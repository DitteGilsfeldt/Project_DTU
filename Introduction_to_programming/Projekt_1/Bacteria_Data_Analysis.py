import numpy as np
import matplotlib.pyplot as plt
import os

#Data Load Funktion
def dataLoad(filename): #The function that loads data, filters data between the given values and removes damaged lines from the data
    data = []
    with open(filename, newline=None) as f:
        num = 1
        for i in f:  #Loops through the data
            try: #Try if data is in the correct format
                i = i.split(" ")  #Split string to list
                i[2] = i[2][0]  #Remove linebreak characters

                i = [float(h) for h in i]  #Using python list comprehensions to type cast

                if 10 < i[0] < 60 and i[1] >= 0 and 1 <= i[2] <= 4:
                    data.append(i)
                num += 1
            except: #Print if the line is damaged
                print(f"Line number {num} is damaged therefor it is skipped")

        f.close()
    return data

#Functions for data statistics
def Mean_temp(data): #Compute mean temp
    sum = 0
    for i in range(len(data)):
        sum += data[i][0]

    result = sum / len(data)
    return result


def Mean_Growth(data): #Compute mean growth rate
    sum = 0
    for i in range(len(data)):
        sum += data[i][1]

    result = sum / len(data)
    return result


def std_temp(data): #Compute standard deviation of temp
    result = np.std(
        np.delete(data, [1, 2], 1))  #Take data array, remove collums given in the array, and calculate the standard deviation.
    return result


def std_Growth(data): #Compute standard deviation of growth rate
    result = np.std(np.delete(data, [0, 2], 1))
    return result


def Mean_Cold_Growth(data): #Compute mean growth rate for the cold temperatures
    l = []
    for i in data:
        if i[0] < 20:
            l.append(i)

    result = Mean_Growth(l)
    return result


def Mean_Hot_Growth(data): #Compute mean growth rate for the hot temperatures
    l = []
    for i in data:
        if i[0] > 50:
            l.append(i)

    result = Mean_Growth(l)
    return result


def dataStatistics(data, statistic): #Function for managing the statistics
    #This function could be reduced by using a dictonary structure but this optimation was ignored beacuse it would make
    #the code less readable
    if statistic == "Mean Temperature":
        result = Mean_temp(data)

    elif statistic == "Mean Growth rate":
        result = Mean_Growth(data)

    elif statistic == "Std Temperature":
        result = std_temp(data)

    elif statistic == "Std Growth rate":
        result = std_Growth(data)

    elif statistic == "Rows":
        result = len(data)

    elif statistic == "Mean Cold Growth rate":
        result = Mean_Cold_Growth(data)

    elif statistic == "Mean Hot Growth rate":
        result = Mean_Hot_Growth(data)

    return result


#Data plot funktion
def gen_plot1(data): #Function for generating the bar plot
    arr = [0, 0, 0, 0]
    for i in data:
        arr[int(i[2] - 1)] += 1

    data = {'Salmonella enterica': arr[0], 'Bacillus cereus': arr[1], 'Listeria': arr[2],
            ' Brochothrix thermosphacta ': arr[3]}
    bacteria = list(data.keys())
    num = list(data.values())

    fig = plt.figure(1,figsize=(10, 5))

    #Creating the bar plot
    plt.bar(bacteria, num, color='maroon',
            width=0.4)

    plt.xlabel("Bacteria name")
    plt.ylabel("No. of bacteria")
    plt.title("Number of different type of bacteria")

    plt.show()

def genplot2(data): #Function for generating the scatter plots
    fig = plt.figure(2,figsize=(10, 5))
    ax1 = fig.add_subplot(111)
    datax = [[], [], [], []]
    datay = [[], [], [], []]

    for i in range(len(data)):
        datax[int(data[i][2] - 1)].append(data[i][0])
        datay[int(data[i][2] - 1)].append(data[i][1])

    ax1.scatter(datax[0], datay[0], s=10, c='b', label='Salmonella enterica')
    ax1.scatter(datax[1], datay[1], s=10, c='r', label='Bacillus cereus')
    ax1.scatter(datax[2], datay[2], s=10, c='g', label='Listeria')
    ax1.scatter(datax[3], datay[3], s=10, c='y', label='Brochothrix thermosphacta')
    plt.legend(loc='upper left')

    plt.xlabel("Temperature")
    plt.ylabel("Growth rate")
    plt.title("How growth rate relates to temperature")


    plt.show()


def dataPlot(data): #Combining the two plotting functions
    gen_plot1(data)
    genplot2(data)


#Function that are called in the main-script
def filter(data): #Filter the data from user input
    try:
        input_3=int(input("Choose a number from 1-3 to filter in temperature,growth rate or bacteria number\n"))-1
        d = data[0][input_3]
    except:
        print("This function only takes the numbers 1-3 try that")
        input_handler(data)


    inputmax=float(input("Max value \n"))
    inputmin=float(input("Min value \n"))

    inputmax,inputmin = max([inputmax,inputmin]),min([inputmax,inputmin])

    data_ny=[]
    for i in data:
        if inputmax >= i[input_3] >= inputmin:
            data_ny.append(i)

    return data_ny


def input_handler(data=None): #Function that handles the user input and manages function calls, by using conditionals
    if data:
        input("Press enter to continue")
        print('\n' * 80)
    input_1 = 5
    print(" Choose a number:  \n 1: Load data  \n 2: filter data \n 3: Show stats \n 4: Generate diagrams \n 5:Exit")
    try:
        input_1 = int(input())
        if input_1 < 0:
            input_1*-1
        if input_1 > 5:
            print("Only takes numbers from 1-5 try one of those")
            input_handler(data)


    except:
        print("Only takes numbers from 1-5 try one of those")
        input_handler(data)


    if 5 > input_1 > 1:
        if data == None:
            print("It seems that you have not loaded any data yet, try doing that before choosing option 2,3,4")

    if input_1 == 1:
        file = str(input("Enter the file adress"))
        try:
            data = dataLoad(file)
        except:
            print("The entered file cant be found or is damaged try again with another")


    elif input_1 == 2:
        data = filter(data)
    elif input_1 == 3:
        print(
            "Choose one of the folowing numbers \n 1: Mean Temperature \n 2: Mean Growth rate \n 3: Std Temperature \n 5: Rows \n 6: Mean Cold Growth rate \n 7: Mean Hot Growth rate ")
        try:
            input_2 = int(input())
            if input_2 < 0:
                input_2 * -1
            if input_2 > 7:
                print("This function only takes numbers from 1-7 try one of those")
                input_handler(data)
        except:
            print("This function only takes numbers from 1-7 try one of those")
            input_handler(data)
        dict = {
            1: "Mean Temperature",
            2: "Mean Growth rate",
            3: "Std Temperature",
            4: "Std Growth rate",
            5: "Rows",
            6: "Mean Cold Growth rate",
            7: "Mean Hot Growth rate"}
        print(dataStatistics(data, dict[input_2]))

    elif input_1 == 4:
        dataPlot(data)

    if input_1 < 5:
        if data:
            input_handler(data)
        else:
            input_handler()


#Main Script
if __name__ == "__main__": #Main loop
    input_handler() #Call the input handler

