"""
Created on Mon Feb 24 08:39:44 2025

@author: Ashlesha
"""

#python list
lst=["cherry", "banana","apple"]
print(lst)

##########
#list items are indexed, the  first item has index [0], the  second 1

print (lst[0])
print(lst[2])
######

#append list (task)
lst=[2,3,4,5,6,7,8]
lst1=[]
for i in lst:
    if i%2==0:
        lst1.append(i)
print(lst1) 

################
# 6 round
# 1 python
# 2 sql
# 3 basics concepts
# 4 project
# 5 major projects 
# 6 HR rounds 

################
#append list copy 
lst=["cherry","banana","apple"]
lst2=lst.copy()
print(lst2)

##################

 #clear removes all the element from the list
 
lst=["cherry","banana","apple"]
lst.clear()
print(lst)

##############

#copy() method

lst=["cherry","banana","appp"]
lst2=lst.copy()
print(lst2)

####
#count() return the  no  of time  the value "cherry" appers in list:
lst=["cherry","banana","appp"]
lst.count("cherry")

####

#extend() method add the element of cars to the frutis list:
lst=[1,2,3,4]
lst1=[5,6,7]
lst.extend(lst1)
print(lst)

#################

#insert()method the  insert the value "orange" as the second element of list

lst=["cheery","banana","apple"]
lst.insert(1,"mango")
print(lst)

###########################################################
#pop() remove the elemnt specified postiton 
lst=["cherry","cherry","banana"]
lst.pop(2)
print(lst)

#remove() remove the item with the specified value

lst=["cherry","cherry","banana"]
lst.remove("cherry")
print(lst)

###########################################################


lst=["cherry","cherry","banana"]
lst.reverse()
print(lst)


############################################################

#sort() sort the list al[habetically:
lst=["cherry","orange","banana"]
lst.sort()
print(lst)


###############

lst=[23,5,3,56,232]
lst=sorted(lst,key=int)
print(lst)

#############################################################

#25-02-25

#jupyter file  iplab
# spyder .py
#API Tools
#installing anaconda navigator


############################################################
                      #27-02-25
############################################################
                   #creating a nested list:
############################################################

nested_list = [[1,2,3], ["a","b","c"], [True,False]]
print(nested_list)
#Accessing Element in a Nested List
nested_list[0][2]

#Accessing the first inner list:
print(nested_list[0]) #output=[1,2,3]

#Accessing the  specific element  inside a sublist
print(nested_list[1][2]) # op 'c' (2nd  sublist , 3rd element)

#accessing last element 

print(nested_list[-1][-1]) #op  false
############################################################
#modifying element in Nested list

nested_list[1][1] ="z"     #changing 'b' to 'z'
print(nested_list)
############################################################

#Itereating over a Nested List:
    
for sublist in nested_list:
     print(sublist)

############################################################

#Using Two  For loops (Iterate over element)

for sublist in nested_list:
    for item in sublist:
        print(item, end ="")

#list comprehension with  nested list
#flattening of list
flat_list = [item  for sublist in nested_list for  item in sublist]
print(flat_list)


#Adding  an  Entire Sublist
nested_list = [[1,2,3],["a","b","c"], [True, False]]
nested_list.append(["New", "list"])
print(nested_list)

############################################################

#Adding an Inside Sublist
nested_list[0].append(4)
print(nested_list)

############################################################

#removing element from sublist

nested_list[1].remove('b')
print(nested_list)

############################################################


##################### Tupple #########################

tup=("cherry","cherry","banana")
print(tup)
print(tup[2])
############################################################

#once a tuple in created , you cannot change its value . Tulpe are unchange 

x = ("apple", "banana", "cherry")
print(id(x))
y = list(x)
y[1] = 'kiwi' 

#first convert into list

y = list(x)   
y[1] = "kiwi"


############################################################

#conver list to tuple 
x=tuple(y)
print(x)
print(id(x))

############################################################
#Tuple is Different data type
x = ("apple", 2, "cherry")
print(x)
############################################################

#you can access tuple items by referring to the index number, inside

x = ("apple","banana","cherry")
print(x[1])
############################################################
#to join two or more  tuples you can use the +operator
tuple1 = ("a","b","c")
tuple2= (1,2,3)
tup1= tuple1+tuple2
print(tup1)
############################################################


#######################Dictionary#####################################

dict1={"Brand":"Maruti","model":"2234", "year":2355}
print(dict1)
print(len(dict1))
print(type(dict1))

############################################################

dict1.get("model")
dict1.keys()
############################################################

car = {
"brand": "ford",
"model": "mustang",
"year": 2455
}
print(id(car))
x=car.keys()
print(x)
############################################################
 #adding one  more key and  value 
 
car["color"]="white"
car
x=car.keys()
print(x)
 
#########################################################################
#removeing  the dictionary
car={
"brand": "ford",
"model":"mustang",
"year":1943
}
car.pop("model")
print(car)

#########################################################################
#Accessing value in dictionary
car={
"brand":"ford",
"model": "mustang",
"year": 1944 
}
for x in car:
    print(car[x])
#########################################################################
                          #28-02-25
#########################################################################
#Accesing values in the dictionary
car={
"brand":"ford",
"model": "mustang",
"year": 1944 
}
for x in car:
    print(car[x])
#if you  want to access both keys  and value 
#very important

    for key, value in car.items():
        print("%s = %s" % (key, value))
    for key, value in car.items():
        print(f"{key} :{value}")
        
#########################################################################
#copying dictionary
car={
"brand":"ford",
"model": "mustang",
"year": 1964 
}
print(id (car))
car2=car.copy()
car2
print(id(car2))

#########################################################################
#Another way to copy built-in function

thisdict={
"brand":"ford",
"model": "mustang",
"year": 1964 
}
print(id(thisdict))
dict1=dict(thisdict)
dict1
print(id(dict1))

#########################################################################
#A dictionary can contain diction
#this  is called nested diction

our_family={
    "child1":{
      "Name":"ABC",
      "DOB":"23-03-2024"
    },
    "child2":{
      "Name":"xyz",
      "DOB":"23-04-2012"}
    }
our_family

#########################################################################


#Diction method
#clear(): remove all element from the car
#list

car={
"brand":"ford",
"model": "mustang",
"year": 1964 
}   
car.clear()
car  
#########################################################################
#copy
car={
"brand":"ford",
"model": "mustang",
"year": 1964 
}
x=car.copy()
print(x)

#########################################################################

#fromkey()
#create 3 dicto
#all value 0
x={'key1','key2','key3'}
y=0
thisdict=dict.fromkeys(x,y)
thisdict
#########################################################################

#get(): to get the value of dictionary

car={
"brand":"ford",
"model": "mustang",
"year": 1964 
}
car.get("model")

###########
#items()return dicton key-value pair

car={
  "brand":"ford",
  "model": "mustang",
  "year": 1964 
}
car.items()
#########################################################################

#values()
car={
 "brand":"ford",
 "model": "mustang",
 "year": 1964 
}
car.values()

#########################################################################

#sort by key

data = {'b':3,'a':4,'c':2}
data.items()
sorted_by_keys=dict(sorted(data.items()))
print(sorted_by_keys)

#sort value

data = {'b':3,'a':4,'c':2}
sorted_by_values=dict(sorted(data.items(),key=lambda items: items[1]))
print(sorted_by_values)

#########################################################################
dict1 = {"cherry": 43, "apple": 45, "Banana": 56}
sorted_data = dict(sorted(dict1.items(), key=lambda item: item[1]))
free_item_key = min(dict1, key=dict1.get)
free_item_value = dict1[free_item_key]
print(f"Free item key: {free_item_key}")
print(f"Free item value: {free_item_value}")

##########################################################################
x=1
print(x)
print(type(x))
y=100000000000000000000000
print(type(y))

#type casting
age1=12
age2=13
print(type(age1))
print(type(age2))
age=age1+age2
print(age)

#input is always string

age1=input("age: ")
age2=input("age: ")
a= age1 + age2
print(type(a))
print(a)

#input for float using type casting
num1=float(input("Enter num 1:"))
num2=float(input("Enter num 2:"))
num=num1+num2
print(num)
print(type(num))


#converting string and int in float
int_val=21
str_val="1.5"
float_val=float(int_val)
print(float_val)
print(type(float_val))
f=float(str_val)
print(f)

#Below is not aplicable

s="ab"
v=float(s)
print(v)

#complex number
c1=1
c2=2j
print("c1",c1,"c2",c2)
print(type(c1))
print(type(c2))
print(c1.real)
print(c2.imag)

#Boolean data type
a=True
print(a)
print(type(a))

#if it is true then only enter the value yes/no/true/false
#if value is false the dont enter any value for it
a=bool(input('is it okay?:'))
print(a)
print(type(a))

#arithematic operation
a=10
b=20
print(a+b)
print(type(a+b))
print(a-b)
print(type(a-b))
print(a*b)
print(type(a*b))
print(a/b)
print(type(a/b))
print(b//a)#we will get flooring value ie,2
print(type(b//a))
a=2
b=3
print(a**b)

#membership operator
wineer=None
print(wineer is None)
print(wineer is not None)
print(type(wineer))
print(wineer)


#if statement 
num=int(input("Enter the number:"))
if num > 0:
    print(num)
#IF -ELSE
num=int(input("Enter the number:"))
if num > 0:
    print("Positive number")
else:
    print("Negative number")

#IF-ELIF STATEMENTS
s=int(input("enter the savings:"))
if(s==0):
    print("No savings")
elif(s>2000) :
    print("saving greater than 2000")
elif(s<2000):
    print("Not sufficient savings")
else:
    print("Start saving")

#while loop
count = 1
print("Starting")
while count <=10:
    print(count)
    count+=1

#For loop
for i in range(2,10):
    print(i)
    #print("done")
print("done")

#anonymous loop variable
for _ in range(0,10):
    print(".",end=" ")

#break loop stetment
num=int(input("Enter the number which you want to check:"))
for i in range(0,11):
    if num==i:
        break
    print(i," ",end=" ")
    print("done")
    
    
#program for printing odd nos
start,end=4,19
for i in range(start,end):
    if i%2!=0:
        print(i,end=" ")
#printing even nos      
for i in range(4,19):
    if i%2==0:
        print(i,end=" ")
        
for i in range(4,19,2):
    if i%2==0:
        print(i,end=" ")

#assigning values
x,y,z=1,2,3
print(x)
print(y)
print(z)



#local variable and global variable 
x="awsome"
def fun():
    print("Python is",x)
fun()

x="awesome"#global variable
def fun1():
    x="fanstastic"#local variable
    print("Python is",x)
fun1()
print("Python is",x)

#dictionary

x=1
y=2.2
z=2+5j
print(type(x))
print(type(y))
print(type(z))

#Type casting
x=int(1.3)
print(x)
x=1
z=float(x)
print(z)

#string cannot concatenat with int
s="Hello"
s1=1
print(s1+s)

#f is use to concatenate string and int
s="Hello"
s2=1
print(f"{s}{s2}")

#'''is use to write 2 or more sentences in one string
x='''This is Python.It is very powerful'''
print(x)

#slicing operator
x='''This is Python.It is very powerful'''
print(x[2:8])

#slice from start
x='''This is Python.It is very powerful'''
print(x[:3])

x='''I am staying in jay colony'''
print(x[16:26])

#slicing to the end
x='''This is Python.It is very powerful'''
print(x[4:])

#negative slicing
x='''This is Python.It is very powerful'''
print(x[-5:-2])

#modifying string
x='''This is Python.It is very powerful'''
print(x.upper())
x='''This is Python.It is very powerful'''
print(x.lower())
#Reverse string
print(x[::-1])
#negative indexing
x="Python"
print(x[-1])
print(x[-5:5])
print(x[-2:-5])#Does not return anything
#Removing space fron left side
x=" This is Python"
print(x.strip())

#Removing space from right side
x="This is Python "
print(x.rstrip())

#Replace the word
x="Hello World"
print(x.replace("Hello","Hi"))

#Split the string
x="Hello- World"
print(x.split("o"))
print(x.split("-"))

x="red-green-blue"
print(x.split("-"))

x="This is python.It is difficult understand.Difficult to implement"
print(x.splite("."))

'''Write a python func that accept a hyphen seperated sequence of color as input
and return the color in a hyphen sequence after sorting'''
color="red-blue-green-yellow"
def sorted_color(color):
    string_split=color.split("-")
    string_split
    sort=sorted( string_split)
    sorted_string="_".join(sort)#string function join
    return sorted_string
sa=sorted_color(color)
print(sa)

#write program to find string is palindrom or not
s=input("Enter the sting you want to check:")
def p(s):
    s1=(s[::-1])
    if(s==s1):
        print("String is plalindrome")
    else:
        print("String is not palindeome")
p(s)

#Find operation in string
x="This is Python and it is very powerful"
print(x.find("and"))

#String concat
x="Hello"
y="World"
print(x+y)#op:HelloWorld
print(x+" "+y)#op:Hello World

#string formate
p=3
c=4
order="I want {} pizza and {} coke"
print(order.format(p,c))

p=3
c=4
order="I want {0} pizza and {1} coke"
print(order.format(p,c))

#Below is invalid declaration 
text="This is funfair it has big merry-go-round "
text="This is funfair it has big \"merry-go-round\""#use"\"escape operator
print(text)

#python boolen
print(10>9)
print(3<1)
#########################
a=20
b=10
if(a>b):
    print("a is greater than b")
else:
    print("b is greater than a")
    
#operator precedence(PEMDAS)
print(1*11+2/2-7)

#identity operators
print(a is b)
print(a is not b)
#########################################################################
               #TEST 1
#########################################################################
#Q.1. Write a program to print even numbers between 23 to 57.Each number should be printed in separate row.

for num in range(24, 58, 2):
    print(num)
#########################################################################
    #Q.2. Write program to write prime numbers between 10 to 99.
 
for i in range(10,99):   
    for j in range(2,101):
        if i%j == 0:
            break
    if i == j:
        print(i,end=",")

#########################################################################

#   Q.3. Write program to check number is palindrome.
s=input("Enter the sting you want to check:")
def p(s):
    s1=(s[::-1])
    if(s==s1):
        print("String is plalindrome")
    else:
        print("String is not palindeome")
p(s)


#########################################################################
                         # 03-03-25 test
##########################################################################
#Q.1. Given two non negative values print true if they have same last digits

def have_same_last_digit(num1, num2):
    return num1 % 10 == num2 % 10
print(have_same_last_digit(123, 43))  
print(have_same_last_digit(123, 44))  


#Q.2. Write program to write prime numbers between 10 to 99
 
for i in range(10,99):   
    for j in range(2,101):
        if i%j == 0:
            break
    if i == j:
        print(i,end=",")


#Q.3. Write program to calculate sum of all digits

def sum_of_digits(num):
    total = sum(int(digit) for digit in str(num))  
    return total
num = int(input("Enter a number: "))
print("Sum of digits:", sum_of_digits(num))

#Q.4.	Write a program to check number of occurrences
    # of specified elements in the list
def count_occurrences(lst,element):
    return lst,element
numbers = [1, 2, 3, 4, 2, 5, 2, 6, 7, 2]
element = int(input("Enter the element to count: "))
print(f"Occurrences of {element}: {count_occurrences(numbers, element)}")

#Q5Write a program to check whether an element exist in a tuple or not
tuple=(10,23,43,45)
23 in tuple
32 in tuple

#Q63.	Write a program to check number is odd or even

num = int(input("Enter the Number:"))
if (num %2) == 0:
         print("Even number")
else:
        print("odd number")



#Q7.	Write a program to add a key and value in the dictionary

book ={"book_name": "bhagavatgeeta",
"auther":"bhaktived swami",
"price":"250 Rs"
}
print(book)

#Q8 	Write a program to check number is positive or negative or zero
 
i=int(input("Enter the number:"))
if i  > 0:
    print(" positive i")
else:
    print(" negative i")

#Q9 	Write program to concatenate dictionary

"Hello" + "Python"

#Q10 Write program to append the list1 with list2 in the front	

lst1 = (12,34,54,45)
lst2 = (24,45,67,32)
lst = lst1 + lst2
print(lst)
###############################################################
            # test-2 basic python
###############################################################

#1.	Write a program to check number of occurrences of specified 
#elements in the list

a = [1, 3, 2, 6, 3, 2, 8, 2, 9, 2, 7, 3]
print(a.count(2))


#2.	Write a program to check whether an element exist in a tuple or not

tuple=(23,45,32,34)
23 in tuple
4 in tuple

#3.	Write a program to check number is odd or even

num = int(input("Enter the Number :"))
if(num%2)==0 :
    print("Number is even")
else:
     print("number is odd")

#4. Write a program to print even numbers between 23 to 57 
#Each number should be printed in separate row

for num in range(24,58,2):
    print(num)


#5.	Write program to write prime numbers between 10 to 99

for i in range(10,99):   
    for j in range(2,101):
        if i%j == 0:
            break
    if i == j:
        print(i,end=",")
###############################################################
        #4-3-25
###############################################################

dict1 = {"cherry": 43, "apple": 45, "Banana": 56}
sorted_data = dict(sorted(dict1.items(), key=lambda item: item[1]))
free_item_key = min(dict1, key=dict1.get)
free_item_value = dict1[free_item_key]
print(f"Free item key: {free_item_key}")
print(f"Free item value: {free_item_value}")

#EXAMPLE

dict = {3: "apple", 1: "banana", 2: "cherry"}
# Returns the key with the smallest value
print(min(dict, key=dict.get))  # Output: 3 (because "apple" is lexicographically smallest)
# Returns the smallest value in the dictionary
print(min(dict.values()))  # Output: "apple"
# Returns the minimum key
print(min(dict))  # Output: 1
###########################################################

my_dict = {3: "apple", 1: "banana", 2: "cherry"}
# Returns the key with the larger value
print(max(my_dict, key=my_dict.get)) 
# Returns the larger value in the dictionary
print(max(my_dict.values()))  
# Returns the max key
print(max(my_dict)) 

###########################################################
#sorting in descending order
sorted_dict = {'a': 3, 'b': 1, 'c': 2}
sorted_by_value_desc = dict(sorted(sorted_dict.items(), key=lambda item: item[1], reverse=True))
print(sorted_by_value_desc)
###########################################################
#adding value of dictionary
dict1={'apple':'123','grapes':'234','mango':'23','banana':'34'}
sum=0
for value in dict1.values():
    sum=sum+int(value)
print(sum)
###########################################################
dict1 = {'apple': '123', 'grapes': '234', 'mango': '23', 'banana': '34'}
# Convert values to integers and sum them up
total_sum = sum(int(value) for value in dict1.values())
print(total_sum) 
###########################################################
#concatenation of dictionary
dict1={1: 10,4 : 54}
dict2={5: 12,3 : 74}
dict3={5: 13,6 : 24}
dict1.update(dict2)
print(dict1)
dict1=dict1|dict2
print(dict1)

###########################################################
#write a program to check if a given key is already exists
dict1={'w':4,'b':3}
print('w'in dict1)
print('a'in dict1)

###########################################################
#python breck statment and if the while
i=1
while i<6:
    print(i)
    if(i==6):
        break
    i=i+1
###########################################################

#continue to the next iteration if i is 3
i=1
while i<6:
    i=i+1
    if(i==3):
      continue
    print(i)
    
###########################################################

#for loop
frut=["apple","banana","mango"]
for i in frut:
    print(i)
    
#breck statment
frut=["apple","banana","mango"]
for i in frut:
    print(i)
    if(i=="banana"):
        break
    #or
    frut=["apple","banana","mango"]
    for i in frut:
        if(i=="banana"):
            break
        print(i)
#contine
frut=["apple","banana","cherry"]
for x in frut:
    if x=="banana":
        continue
    print(frut)
#range sequence
for x in range(6):
 print(x)  

#range function

for x in range(6,9):
   print(x)
   
   
for x in range(2,4,79):
    print(x)  
    
#nested loop inner loop
color=["green","yellow","red"]
frut=["apple","banana","guava"]
for x in color:
    for y in frut:
        print(x,y)
        
#suppose in selling milk 100liter , and is 
#queue of customer the moment sell to 100 liter,you 
#need to inform to the customer
#that the milk is finished 
milk = 100  
while milk > 0:  
    milk -= min(milk, int(input("Enter liters required: ")))  
    print("Milk is finished!" if milk == 0 else f"Milk remaining: {milk} liters")  

#suppose you are standing in queue to auditorium where students and
#professors are in queue, if the professors are there you are allowing
#without checking but if there is student then he/she is being checked
queue = ["student", "professor", "student", "professor", "student"]  
for person in queue:  
    if person == "professor": 
        continue   
    print(f"Checking: {person}") 

############################################################
                   #5-3-25    
############################################################
#consistent
#12- month -exam-6-8
#python-simple-advance
#pythone steps-implement-writebook
#Task
num=int(input("Enter the num:"))
if (num%2)==0:
    print("even")
else:
   print("odd")
   
#Task

num=int(input("enter num:"))
if num > 0:
   print("positive")
else:
   print("negitive")
    
############################################################
#function without argument

def my_function():
    print("Hello function")
my_function()

############################################################
#function argument
def my_function(name):
    print("hello"+name)
my_function("ram")    
############################################################
#function with posotional argument
def my_function(n1,n2):
    print(n1+""+n2)
my_function("hello","world")
############################################################   
#arbitrary argu function
#if you do not know how many argu 
#will be passed into  youe argu
#add a* before parameter name
#function definition
def my_function(*args):
    print(args[0]+" "+args[2])
my_function("tappu","sonu","ram")
#############################################################
def myfun(**kwargs):
    for key,value in kwargs.items():
        #print("%s ==%s"(key,value))
        print(f'{key}:{value}')
myfun(first_name='ram', mid_name='om',last_name='gony')
############################################################    
#default parameter function
#if we call the function without argument.
#uses default value
def myfunc(country="Norway"):
    print("i am from " + country)
myfunc('india')
############################################################
#passing a list argument
#you can send any data type of argu function(string)
frut=["orange","banana","guava"]
def myfun(frut):
    for x in frut:
        print(x)
myfun(frut)
############################################################
#return values
#to let function return a value use return statement
def myfun(x):
    y=x*4
    return y
myfun(5)
#or return multiple value function
def myfun(x):
    y=x*4
    z=x*2
    return y,z
myfun(5)
############################################################
#pass function
def myfuns():
    pass#error solve not use to function
myfuns()
############################################################
#having an empty function definition
#like this would raisev error
#without pass statement
############################################################
#recursive function
#factorial of no is product all int
#form 1 to no
    #INTERVIEW QUE
def faction(x):
    if x==1:
        return 1 
    else:
        return(x*faction(x-1))
faction(2)
faction(4)
############################################################
#interview que lambda function
def add(a):
    sum=a+4 
    return sum
add(23)

add=lambda a:a+4
print(add(20))
############################################################
#lambda func can take any no argument
add=lambda a,b:a+b
print(add(2,3))   
############################################################
#finding  odd no from list
lst=[24,45,65,2,2,3,32,43,54]
odd_lst=list(filter(lambda x:(x%2 !=0),lst))
print(odd_lst)
#filter(condition)function pyhton using filter element
#form iterable (like a lst,tuple,set)based given
#filter method accept 2argu python
#filter only condititon
lst=[35,56,23]
even_lst=list(filter(lambda x:(x%2)==0,lst))
print(even_lst)    
############################################################
#map function all element oprater
#map function python built in function
#applies function each ithem iterable (like a list ,tuple , set)
ls=[35,45,6,87,34]
sqr_ls=list(map(lambda x:(x**2),ls))
print(sqr_ls)
############################################################
#split() & join () split and join string
#split() breaks a string into list
#join() combines a list string
text="app,boll,doll"
words=text.split(",")
print(words)
############################################################
new="-".join(words)
print(new)
############################################################
              #6-3-25
############################################################
#find() & index() -substring
#find() return the index 1st occurrences
#index() similar but raises error if not 
text="hello, pytthon!"
print(text.find("python"))
print(text.find("java"))
print(text.index("python"))
print(text.index("java"))
############################################################
#count() count substring occurrences
text="banana banana apple"
print(text.count("banana"))
############################################################
#startswith() and endswith () check start-end
text="python is great"
print(text.startswith("python"))
print(text.endswith("great"))
print(text.endswith("hello"))
############################################################
#isalpha () returns true if all char are letter
#isdigital() returns true if all char are digital
#isalnum() returns true if all char are letter
t="python123"
print(t.isalpha())
print(t.isdigit())
print(t.isalnum())#alpha+number(alnum)
############################################################
#isupper() check if all char are uppercase
#islower() check if all char lowercase
#isupper
t="HELLO"
t2="HELLO1234"
t3="234"
count1=0
for i in range(len(t)):
    if t[i].isupper():
        count1=count1+1
print(count1)
#islower
count2=0
for i in range(len(t)):
    if t[i].islower():
        count2=count2+1
print(count2)

a="12345"
count3=0
for i in range(len(a)):
    if a[i].isdigit():
        count3=count3+1
print(count3)
############################################################
#fullstop(.)
str = str if str.endswith(".") else str + "."
############################################################
#pyramid
for i in range(4):
    for j in range(3):
        print("#", end=" ")
    print()

for i in range(4):
       for j in range(i+1):
           print("#",end=" ")
       print()

for i in range(5):
    for j in range(4-i):
        print("#", end=" ")
    print()
############################################################
#duplicate function
a=[6,8,5,6,7]
a.sort()
a
def is_duplicate(a):
    for i in range(len(a)-1):
        if(a[i]==a[i+1]):
            return True
    return False
print(is_duplicate(a))
############################################################    
#anagram 
str='Elbow'
str.replace(' ', ' ').lower()
a=list(str.replace("", " ").lower())
sorted(a)    

def are_anagram(str1,str2):
    a=list(str1.replace("","").lower())
    b=list(str2.prplace("","").lower())
    if(len(a)!=len(b)):
        return False
    else:
        return(sorted(a)==sorted(b))
print(are_anagram("elbow","below"))
############################################################
            #7-3-25
############################################################
#type error handling exercise
#nameerror
#print(hii)
#syntax
print("hello")
#Typrerror
str="hello"
str=str+5
#indexerror
str="hello"
print(str[10])
#keyerror
dict={1:22,2:22,3:44}
print(dict[5])
#Attributeerror
str="hello"
str.reverse()
#valueerror
str="moose"
ans_1=int(str)
############################################################












