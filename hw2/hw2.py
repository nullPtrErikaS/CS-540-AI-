import sys
import math

# I used chatGPT for some help
# can you break this assignment down into smaller parts
# is this outline a good start
# explain bayes theorem
# why am i getting NameError
# how to do log probabilities
# how to do conditional probabilities
# why is my Q4 results different
# can you help me think of every edge case 


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    # f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    # f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment

    # using list to store letter for A-Z
    X = [0] * 26

    # X=dict()
    with open (filename, 'r', encoding='utf-8') as f:
        # TODO: add your code here
        for line in f: 
            for char in line:
                char = char.upper()
                if 'A' <= char <= 'Z':

                    # increment the index
                    X[ord(char) - ord('A')] += 1

    # check if count = 0
    if sum(X) == 0:
        print("Q1")
        for i in  range(26):
            letter = chr(i + ord('A'))
            count = X[i]  
            print(letter, count) 
        return X

    print ("Q1")
    for i in range(26):
        # get corresponding letter
        letter = chr(i + ord('A'))  
        # get count for letter
        count = X[i]  
        print(letter, count) 

    return X

# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!

def compute_Q2(X, e, s):

    # A count
    X1 = X[0]
    # probability A in english
    e1 = e[0]
    # probability A in spanih
    s1 = s[0]

    # calculate log probabilities for both
    if e1 > 0:
        val_english = X1 * math.log(e1)  

    else:
        val_english = 0  

    if s1 > 0:
        val_spanish = X1 * math.log(s1) 

    else:
        val_spanish = 0  

    print("Q2")
    print(f"{val_english:.4f}")
    print(f"{val_spanish:.4f}")

# compute and print overall log probabilities
def compute_Q3(X, e, s, prior_english, prior_spanish):

    # prior prob
    F_english = math.log(prior_english)
    F_spanish = math.log(prior_spanish)

    for i in range(26): 
        if X[i] > 0:  # only add letters in text
            if e[i] > 0:  # only include prob in English is non-zero
                F_english += X[i] * math.log(e[i])
            # else:
            #     # small probabilities
            #     F_english += X[i] * math.log(1e-10) 

            if s[i] > 0:  # only if the prob in Spanish is non-zero
                F_spanish += X[i] * math.log(s[i])
            # else:
            #     # handle small probabilities
            #     F_spanish += X[i] * math.log(1e-10)

    print("Q3")
    print(f"{F_english:.4f}")
    print(f"{F_spanish:.4f}")

    return F_english, F_spanish

# compute and print prob for text being english
def compute_Q4(F_english, F_spanish):

    # likely spanish
    if F_spanish - F_english >= 100:
        P_english = 0

    # likely english
    elif F_spanish - F_english <= -100:
        P_english = 1
    
    # Sigmoid funciton 
    else:
        P_english = 1 / (1 + math.exp(F_spanish - F_english))

    print("Q4")
    print(f"{P_english:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("usage: ...")
        sys.exit(1)

    filename = sys.argv[1]

    prior_english = float(sys.argv[2])
    prior_spanish = float(sys.argv[3])

    # get char probs
    e, s = get_parameter_vectors()

    # count letters
    X = shred(filename)

    # compute and print log prob
    compute_Q2(X, e, s)

    # compute overall log
    F_english, F_spanish = compute_Q3(X, e, s, prior_english, prior_spanish)

    # compute and print prob being english
    compute_Q4(F_english, F_spanish)