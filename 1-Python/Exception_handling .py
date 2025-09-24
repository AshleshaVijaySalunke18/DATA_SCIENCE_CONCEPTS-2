# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 08:13:50 2025

@author: Ashlesha
"""
###########################################
#17-3-25
###########################################
a=10
b=0
result=a/b #//Throws ArithmaticException
###########################################
#Zero Divison Error
try:
    result=a/b
except ZeroDivisionError:
    print("Cannot divide by Zero!")
###########################################
#index Error
num=[1,2,3]
print(num[5])
######
try:
    print(num[5])
except IndexError:
    print("Index out of range!")
###########################################
#Handling exception without naming them
try:
    numerator=50
    denom=int(input("Enter the denominator:")) #enter 10.3
    quotient=(numerator/denom)
    print("Division Performing Successfully")
except ValueError:
    print("only INTEGERS should be entered")
except:
    print("OOPS...SOME EXCEPTION RAISED")
###########################################
#Handling exception using try..except..else
try:
    numerator=50
    denom=int(input("Enter the Denominator:"))
    quotient=(numerator/denom)
    print("Division performing successfully")
except ZeroDivisionError:
    print("Denominator as Zero is not allowed")
except ValueError:
    print("only INTEGERS should be Entered")
else:
    print("The result of Division operation is", quotient)
###########################################
#Handling exception using try..except..else..finally
try:
    numerator=50
    denom=int(input("Enter the Denominator"))
    quotient=(numerator/denom)
    print("Divison performed successfully")
except ZeroDivisionError:
    print("Denominator as Zero is not allowed")
except ValueError:
    print("Only INTEGER should be Entered")
else:
    print("The result of Division operation is", quotient)
finally:
    print("OVER AND OUT")
#####################################
#fileNOTFoundError output show
with open('c:/1-python/pi_digit.txt','r')as file:
    contents=file.read()
print(contents.rstrip())
##fileNOTFoundError this error show
with open('c:/1-python/pi_digits.txt','r')as file:
    contents=file.read()
print(contents.rstrip())
#####################################
try:
    with open("c:/1-python/pi_digits.txt",'r')as file:
        contents=file.read()
except FileNotFoundError:
    print("File Not Found!")
#####################################
#permissionError
with open('c:/1-python/pi_digit.txt','r')as file:
    contents=file.read()
print(contents.rstrip())
#################################
try:
    with open('c:/1-python/pi_digit_new.txt')as file:
        contents=file.read()
    print(contents.rstrip())
except PermissionError:
    print("You don't have permission to access this file!")
#########################################
#AttributeError
obj=None
print(obj.some_attribute) #raises attributeError
######################
if obj is not None:
    print(obj.some_attribute)
else:
    print("Object Is None!")
##################################
#memoryError
huge_list=[1]*(10**10)#raises memoryerror
########
#handling using generator
def gennum():
    for i in range(10**10):
        yield i  #yields num 1 by 1  preventing memory overload
gen = gennum()
print(next(gen))
print(next(gen))
##################################
#18-3-25
##################################
def recfun():
    return recfun()#infinite recursion
recfun()  #raises recursionerror
##################################
import sys
sys.setrecursionlimit(1000)#set recusion limit to a
def saferecfun(depth=0,max_depth=10):
    if depth>=max_depth:
        return "DONE"
    return saferecfun(depth+1,max_depth)
print(saferecfun())#works safely
##################################
#1st and last line output
with open('c:/1-python/pi_digit.txt','r')as file:
    contents=file.read()
print(contents.rstrip())
##################################
with open('c:/1-python/pi_digit.txt','r')as file:
    lines=file.readlines()
    if lines:
        print("fisrt line:",lines[0].strip())
        print("Last line:",lines[-1].strip())
##################################
#write a program to accept input from user and append
filename ='c:/1-python/programming.txt'
with open(filename,'w')as file:
    file.write("I am  code.\n")
    file.write("I love creating  new game.\n")
    in_line=input("Enter the line:")
    file.write(in_line)
##################################
# WAP to read content from a txt file line
#by line and stored each line into list
filename='c:/1-python/pi_digit.txt'
with open(filename,'r')as file:
    lines=file.readlines()
    pi_string=[]
    for line in lines:
        pi_string.append(line.rstrip())
        #pi_string+=line.rstrip()
        print(pi_string)
    print(len(pi_string))
##################################
'''WAP find the longest word from the txt file
contents, assuming that the file will
have only oen longest'''
filename='c:/1-python/programming.txt'
with open (filename,'r')as file:
    line=file.readline()
    longest_word=''
    for line in file:
        word=line.split()
        for word in word:
            if len(word)> len(longest_word):
                longest_word=word
print("The longest word:",longest_word)
##################################
'''WAP find the shortest word from the txt file
contents, assuming that the file will
have only oen shortest'''

filename = 'c:/1-python/programming.txt'
with open(filename, 'r') as file:
    short_word = None  # Initialize as None to properly compare lengths
    for line in line:  
        words = line.split()  # Split line into words
        for word in words:
            if short_word is None or len(word) < len(short_word):  # Correct comparison
                short_word = word  
print("The shortest word:", short_word)
##################################
#WAP to count the frequency of a user-entered word txt file

filename='c:/1-python/programming.txt'
input_line=input('Enter the  text:')
words=input_line.split()
word_count=len(words)

#user input the file
with open(filename,'w')as file:
    file.write(input_line)

#display the word count
print("The total word entered:",word_count)

######################################################

sentence = 'world wide web'
sentence = sentence.upper()  # Convert to uppercase
word = sentence.split()  # Split into words

# Convert each word into a list of characters
char_list = [list(w) for w in word]  
print(char_list)

# Extract the first and last character of the first word
first_char = char_list[0][0]  # First character of 'WORLD'
last_char = char_list[0][-1]  # Last character of 'WORLD'

# Get ASCII values
ascii_first = ord(first_char)
ascii_last = ord(last_char)

# Print results
print(f"First char: {first_char}, ASCII: {ascii_first}")
print(f"Last char: {last_char}, ASCII: {ascii_last}")

#########################################################
'''Concatenate the sums of 
each word to form the result.'''
sentence = 'world wide web'
sentence = sentence.upper()  # Convert to uppercase
words = sentence.split()  # Split into words

# Convert each word into a list of characters
char_list = [list(word) for word in words]

# Process each word and compute ASCII differences
ascii_diff = []
for chars in char_list:  # Rename loop variable to 'chars'
    first_char = chars[0]  # First character
    last_char = chars[-1]  # Last character
    ascii_first = ord(first_char)
    ascii_last = ord(last_char)

    diff = abs(ascii_first - ascii_last)
    ascii_diff.append(diff)

# Print result
for word, diff in zip(words, ascii_diff):
    print(f"Word: {word}, ASCII Diff: {diff}")

##########################################
''' In each word, find the Sum of the 
Difference between the first letter and
the last letter,second letter and the 
penultimate letter, and so on till the center of the word.
WORLD = [W-D]+[O-L]+[R] = [23-4]+[15-12]+[18] = [19]+[3]+[18] = [40]
WIDE = [W-E]+[I-D] = [23-5]+[9-4] = [18]+[5] = [23]'''
sentence='world wide web'
sentence=sentence.upper()
words= sentence.split()
#process each word

word_sum =[]

for word in words:
    length=len(word)
    total=0#sum of diff for this word
    
    for i in range((length+1)//2):#iterate till center
       first_char=word[i]
       last_char=word[length-1-i]
       total +=abs(ord(first_char) - ord(last_char))
    
    word_sum.append(str(total))

final_result = int("".join(word_sum))

for word, word_sum in zip (word,word_sum):
    print(f"word: {word},sum of diff:{word_sum}")
    
print(f"final output:{final_result}")

######################################
#20-3-25
######################################
'''WAP to accept  2num from the user
and perform division.If any exception
occurs, print an error message or else
print the result.'''
try:
    num1=float(input('Enter Number1:'))
    num2=float(input('Enter number2:'))
    num=num1/num2
    print(('The result is',num))
except ZeroDivisionError:
    print("Error:Division by zero is not allowed.")
except ValueError:
    print(("Error:Please enter numeric values only."))
    num1=float(input(""))
######################################
'''WAP to accept a num from the user &
check whether it`s  prime or not .If user
enters anything other than number, handle the
exception and print an error msg'''

def is_prime (num):
    if num<2:
        return False
    for i in range (2,int(num**0.5)+1):
        if num %  1==0:
            return False
    return True

try:
    num=int(input('Please Enter the number:'))
    if is_prime():
        print("The enter  number is prime")
    else:
        print("The Enter number is not prime")
except ValueError:
    print('Error:please the valid integer')
#########################
'''wap accept the file  to be 
opened from the uses, if file exist print the contents
of the file in title case or else handle
the exception and print error msg '''
try:
    file_name = input("Please enter the file name with absolute address: ")  # Get user input
    with open(file_name, 'r') as file:  # Open file in read mode
        contents = file.read()
        print(contents.title())  # Convert text to title case
except FileNotFoundError:
    print("Error: File not found. Please enter the correct file name with an absolute path.")
except PermissionError:
    print("Error: You don't have permission to access this file.")
##################################
''' Declare a list with 10 integers and ask the user to enter an index.
Check whether the number in that index is
positive or negative number. 
If any invalid index is entered,
 handle the exception and print 
 an error message.  '''
num=[5,-3,-7,-1 ,12,-8,9,-6,15,2] 
try:
    index=int(input("Enter the Index(0-9):"))
    value=num[index]
    if value>0:
        print(f"the number at index {index} is positive.")
    else:
        print(f"The number at index{index}is negative")
except IndexError:
  print("Error:Index out of rang.Please enter a value:")
except ValueError:
  print("Error:Please enter a valid integer.")        
##############################################

 
 
 
 
 
 
 
 
 