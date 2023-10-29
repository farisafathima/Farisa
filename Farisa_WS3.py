import time
import difflib


#You are given a list called fruits =  ['mango', 'kiwi', 'strawberry', 'guava', 'pineapple', 'mandarin orange'].
# Create a variable named capitalized_fruits and use list comprehension syntax to produce output like
# ['Mango', 'Kiwi', 'Strawberry', etc...]

def list_comprehension1(l):
    capitalized_fruits=[fruit.capitalize() for fruit in l]
    print(capitalized_fruits)

#You are given a list called fruits =  ['mango', 'kiwi', 'strawberry', 'guava', 'pineapple', 'mandarin orange'].
# Make a variable named fruits_with_only_two_vowels. Use list comprehension to produce
# ['mango', 'kiwi', 'strawberry'], a list of fruits with only two vowels.

def list_comprehension2(l):
    fruits_with_only_two_vowels=[fruit for fruit in l if (
        fruit.count("a") +
        fruit.count("e")+
        fruit.count("i")+
        fruit.count("o")+
        fruit.count("u")) == 2]
    print(fruits_with_only_two_vowels)

#Given org1, org2. find all similar pairs of genome sequences  using list comprehension.
# “Similar” means: similarity(seq1, seq2) > threshold


def similarity(seq1, seq2, threshold):
    # Calculate the similarity between seq1 and seq2 using the SequenceMatcher
    seq_matcher = difflib.SequenceMatcher(None,seq1, seq2)
    similarity = seq_matcher.ratio()
    return similarity > threshold

org1 = ["ACGTTTCA", "AGGCCTTA", "AAAACCTG"]
org2 = ["AGCTTTGA", "GCCGGAAT", "GCTACTGA"]
threshold = 0.6  # Define your desired similarity threshold

similar_pairs = [(seq1, seq2) for seq1 in org1 for seq2 in org2 if similarity(seq1, seq2, threshold)]
print(similar_pairs)



#Given numbers=[1,2,3,4,5,6,7,8,9,10]. Create a dictionary of numbers and their squares, excluding odd numbers using
# dictionary comprehension.

def dict_comprehension(numbers):
    squares_even={num:num** 2 for num in numbers if num % 2 == 1}
    print(squares_even)

#sentence="Hello, how are you?" . Write a dictionary comprehension to ma words to their reverse in a sentence.
#The output should be - {'Hello,':',olleH','how':'woh','are':'era','you?':'?ouy'}

def dict_comprehension2(sentence):
    sentence1=sentence.split(" ")
    reverse_words={word:word[::-1] for word in sentence1 }
    print(reverse_words)




#Write  a lambda function to sort a list of strings by the last character

def lambda_func1(li):
    j=sorted(li,key=lambda i : i[::-1])
    print(j)

#Write a Python program to rearrange positive and negative numbers in a given array using Lambda.

def lambda_func2(array):
    neg= list(filter(lambda i : i < 0 , array))
    pos=list(filter(lambda i : i>=0 , array))
    print(neg,pos)
#Create a logging decorator to record function calls, arguments, and return values.
    # create a decorator that prints the following:
	   #Calling add with args: (2, 3), kwargs: {}
        #add returned: 5

def decorator_func(func):
    def arguments(*args,**kwargs):
        print('Calling add with args',args , 'kwargs',kwargs)
        result=func(*args,**kwargs)
        print('add returned:',result)
    return arguments

@decorator_func
def add(a, b):
    return a + b


#Create a decorator to measure the execution time of a function. Please demonstrate using a
# sample function (add sleep for checking) and a decorator for this sample function that
# measures the execution time.

def decorating(func):
    def inner_func(*args,**kwargs):
        start=time.time()
        print("Starting time:",start)
        print("Function executing")
        func(*args,**kwargs)
        end=time.time()
        print("End time:",end)
        print("Execution time:", end-start)
    return inner_func

@decorating
def sleep_func(x):
    time.sleep(x)
@decorating
def another_func(n):
    for i in range(n):
        print(i)

#Write a function division() that accepts two arguments. The function should be able to
# catch an exception such as ZeroDivisionError, ValueError, or any unknown error you
# might come across when you are doing a division operation. Also, add a “finally” construct.


def division(num1,num2):
    try:
        res=num1/num2
        print(res)
    except ZeroDivisionError as z:
        print('Denominator should not be zero: ',z)
    except ValueError as v:
        print('Value error: ', v)
    except Exception as e :
        print('Other error: ',e)
    finally:
        print('This statement will execute')



# interactive calculator- User input is  a formula that consists of a number,
# an operator (at least + and -), and another number, separated by white space
#If the input does not consist of 3 elements, raise a FormulaError, which is a custom Exception.
#Try to convert the first and third input to a float (like so: float_value = float(str_value)).
# Catch any ValueError that occurs, and instead raise a FormulaError
#If the second input is not '+' or '-', again raise a FormulaError
#If the input is valid, perform the calculation and print out the result.
# The user is then prompted to provide new input, and so on, until the user enters quit.

class FormulaError(Exception):
    pass

def interactive_calculator():
    while True:
        exp = input("Enter an expression in the format- num1 operaator num2  [or 'quit' to exit]: ")
        if exp == 'quit':
            break

        exp1 = exp.split()
        operator = {'+', '-', '/', '*'}

        if len(exp) != 3 or exp[1] not in operator:
            raise FormulaError("Invalid input")

        num1 = int(exp[0])
        num2 = int(exp[2])

        if exp[1] == '+':
            print(num1 + num2)
        elif exp[1] == '-':
            print(num1 - num2)
        elif exp[1] == '/':
            if num2 == 0:
                raise FormulaError("Division by zero")
            print(num1 / num2)
        elif exp[1] == '*':
            print(num1 * num2)



def main():
    l= ['mango', 'kiwi', 'strawberry', 'guava', 'pineapple', 'mandarin orange']
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sentence = "Hello, how are you?"
    li = ["hello", "how", "are", "you"]
    array = [2, 5, -4, 7, -9]


   # list_comprehension1(l)
    #list_comprehension2(l)
    #dict_comprehension(numbers)
    #dict_comprehension2(sentence)
    #lambda_func1(li)
    #lambda_func2(array)
    #add(3, 5)
    #sleep_func(10)
    #another_func(6)
    #division(4,0)
    #division(4,'a')
    '''
    try:
        interactive_calculator()
    except FormulaError as e:
        print(f"Error: {e}")
    '''

if __name__=="__main__":
    main()
