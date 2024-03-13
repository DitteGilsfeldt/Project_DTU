import numpy as np
import matplotlib.pyplot as plt
import time
import os
import random

def getStudent_id(line):
    line = str(line)
    student_id = line[:7]
    if student_id[0] == "s" and student_id[5].isnumeric():
        return student_id
    else:
        raise Exception("Student id is in the incorrect format")

def getName(line):
    line = str(line)
    line = line[8:]
    name = ""
    for i in range(len(line)):
        if line[i].isnumeric():
            break
        elif line[i] == "," or line[i]=="-":
            break
        else:
            name += line[i]
    return name

def getGrades(line):
    line = str(line)
    line = line[8:]
    nums = ""
    for i in range(len(line)):
        if line[i].isnumeric() or line[i]=="," or line[i] == "-":
            nums += line[i]
    
    grades = []
    

    nums = nums.split(",")
    for i in range(1,len(nums)):
        grades.append(int(nums[i]))
    return grades

def loadData(filename):
    #Handles loading of the data
    final_data = []
    num = 1
    with open(filename,"r") as file:
        for i in file:
            try:
                student_id = getStudent_id(i)
                name = getName(i)
                grades = getGrades(i)
                final_data.append([student_id,name,grades])
            except:
                print(f"There was an error on line {num} and it has therefore been removed")
            num += 1

    print(f"Loaded {len(final_data)} students who have completed {len(final_data[0][2])} assignments")
    return final_data


def roundGrade(grades):
    #Rounds a number to the nearest possible grade
    grades = grades.copy()
    gradesRounded = []
    options = [-3, 0, 2, 4, 7, 10, 12]
    for grade in grades:
        new_grade = min(options, key=lambda x: abs(x - grade))
        gradesRounded.append(new_grade)
    return gradesRounded


def computeFinalGrade(grades):
    """Given a list of grades return the finalgrade"""
    grades = grades.copy()
    if -3 in grades:
        return -3
    elif len(grades) >= 2:
        ng = grades
        ng.remove(min(ng))
        finalgrade = np.mean(ng)
        return roundGrade([finalgrade])[0]

    elif len(grades) == 1:
        return grades[0]

def computeFinalGrades(grades):
    """Manages the data computeFinalGrade receives """
    grades = grades.copy()
    final_grades = []
    for i in grades:
        final_grades.append(computeFinalGrade(roundGrade(i)))
    return roundGrade(final_grades)


def numGrades(grades):
    """A function creating a dictionary of how many students recived each
        possible final grade"""
    grades = grades.copy()
    final_grades = computeFinalGrades(grades)
    final_grades = final_grades.copy()
    options = [-3, 0, 2, 4, 7, 10, 12]
    amount = {}
    for i in options:
        amount[i] = 0

    for i in final_grades:
        amount[i] += 1
    return amount


def gradesPlot(grades):
    """Functio handling the  creations of the plots using matplotlib"""
    grades = grades.copy()
    amount = numGrades(grades)
    grades_rec = list(amount.keys())
    num = list(amount.values())
    plt.figure(1)
    plt.subplot(211)
    plt.bar(range(len(amount)), num, tick_label=grades_rec)
    plt.title("Number of students who received each final grade")
    plt.xlabel("Grades")
    plt.ylabel("Number of students who received this grade")

    plt.figure(2)

    tup_list = []
    for i in grades:
        for j in range(1, len(i)+1):
            tup_list.append((j+random.uniform(-0.1,0.1),i[j-1]+random.uniform(-0.1,0.1)))


    mean = []
    mean_x = []

    for i in grades[0]:
        mean.append([])

    for i in tup_list:
        mean[round(i[0]-1)].append(round(i[1]))

    for i in range(len(mean)):
        mean[i] = sum(mean[i])/len(mean[i])
        mean_x.append(i+1)
    
    x,y = zip(*tup_list)

    plt.scatter(x,y)
    plt.plot(mean_x,mean)

    plt.title("Grades for each assignment")
    plt.xlabel("Assignment number")
    plt.ylabel("Grade")

  
    plt.yticks([0,2,4,7,10,12])

    
    plt.xticks(range(1,len(grades[0])+1))


    plt.rcParams['lines.markersize'] = 10/len(grades[0])



    plt.show()


def intInput(range1, range2):
    """The entire program uses a lot of user int inputs this function makes sure 
        That that process is streamlined. Using whileloops and try/except statement
        only valid inputs gets parsed to the functions"""
    if range1 > range2:
        range1, range2 = range2, range1

    num = 0
    while True:

        try:
            num = int(input(f"Enter an integer {range1}-{range2}: "))
        except ValueError:
            print(f"Please enter a valid integer {range1}-{range2}")
            continue
        if num >= range1 and num <= range2:
            return num
        else:
            print(f'The integer must be in the range {range1}-{range2}')

def listGrades(data):
    """ Returns the students,their grades and final grades in alphabetical order 
        by the students first name."""
    grades = []
    for i in data:
        grades.append([i[1],i[2]])
    grades.sort(key = lambda x: x[0])
    print(f"A list of students, their grades for the {len(grades[1][1])} assignments they have completed aswell as their final grade")
    for i in grades:
        print(f"{i[0]}: Grades:{i[1]}, Final grade:{computeFinalGrade(i[1])}")
    

def help():
    """A short desciription of what the users options in this program are """

    print("A short description of what your options in this program are \n")
    time.sleep(1.5)
    print(
        "Load new data: Asks for a file destination to load data into the program, handles errors and removes them \n")
    time.sleep(1.5)
    print(
        "Check for data error: Checks if any students share the same student id and if any of their grades are invalid \n")
    time.sleep(1.5)
    print("Generate plots: Requires data to be loaded and plots the data \n")
    time.sleep(1.5)
    print(
        "Display list of grades: Requires data to be loaded and returns a list of the students grades \n")
    time.sleep(1.5)
    print("Search for students: Takes a user input and lets you search for students and returns the students that match the search term and their grades, you dont have to search for full student names \n")
    time.sleep(1.5)
    print("Exit: Exits the program \n")
    time.sleep(1.5)

    print("Help: Takes you to a short description of what your options in this program are \n")

def takeToLoadData():
    """Handling user input and checking if file exists on the given path"""
    file = str(input("Enter the file adress"))
    if os.path.exists(file):
        return loadData(file)
    else:
        print("File dosent exist")
        main()


def GetStudentIdErrors(data):
    """Handles the chance of more than one student id by looping through every student
    and noting what students ids you have found. For each future student you check this list
    and if a match is a message is printed to the termminal
    """
    data = data.copy() #Problem with localscope operations effecting the global scope
                    #Creating a deep copy of the array in the local scope solved the issue
    seen = []
    id_name = {}

    flag = False
    for i in data:
        if i[0] in seen:
            print(f"{i[1]} and {id_name[i[0]]} have the same student id")
            flag = True
        else:
            seen.append(i[0])
            id_name[i[0]] = i[1]
    return flag

    
def getGradeErrors(data):
    """Handles detection of grades not on the 7-step scale.
        The function loops through all students and then loops through
        their grades, if any of them aren't in the list options a small message
        gets printed to the terminal"""

    grades = [-3,0,2,4,7,10,12]
    flag = False
    for i in data:
        for j in i[2]:
            if j not in grades:
                print(f"{i[0]} has recived the grade {j} wich is not a valid grade")
                flag = True
    return flag

def errorHandling(data):  
    """A file that manages error handling"""
    fl1 = GetStudentIdErrors(data)
    fl2 = getGradeErrors(data)
    if fl1 == False and fl2 == False:
        print("No errors found")

def searchStudents(data,name):
    """searchStudents takes the data and a search query as an input and returns
        every student who match that search query. For example if you search for S
        a list of all students starting with s will be shown aswell as their grades
    """

    if len(name) == 0:
        print("Press 1 to show no students and press 2 to show all students")
        choice = intInput(1,2)
        if choice == 1:
            print(f"0 students out of {len(data)} students found")
            return
    found_names = []

    for i in range(len(data)):
        if data[i][1][:len(name)].lower() == name.lower():
            found_names.append(data[i])
    print(f"{len(found_names)}/{len(data)} students found, these students are:")
    for i in found_names:
        print(f"{i[1]} who recived the grades {i[2]}")
    






def main(data=None): #The main script of the program
    """ The main function contols the flow of the program by using 
        int inputs from the user. Every input runs a corresponding 
        function """
    input("Press enter to continue")
    print(80 * "\n")

    print("Enter an integer from 1-7 to go to the following options",
          "1: Load new data",
          "2: Check for data errors",
          "3: Generate Plots",
          "4: Display list of grades",
          "5: Search for students",
          "6: Quit",
          "7: Help", sep="\n" 
          )

    choice = intInput(1, 7)
    if choice ==1:
        data = None
        data = takeToLoadData()
    elif choice == 2:
        if data:
            errorHandling(data)
        else:
            print("Data not loaded")
    elif choice == 3:
        if data:
            grades = []

            for i in data:
                grades.append(i[2])

            gradesPlot(grades)
        else:
            print("Data not loaded")

    elif choice == 4:
        if data:    
            listGrades(data)
        else:
            print("Data not loaded")

    elif choice == 5:
        name = input("What student do you wish to search for ")
        searchStudents(data,name)


    elif choice == 6:
        exit()

    
        
    elif choice == 7:
        help()
    
    
    main(data)


#convert text to ascii art

data = takeToLoadData()
main(data)
