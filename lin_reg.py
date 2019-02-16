import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#X represents the year
#Y represents the number of transistors on a chip
X = []
Y = []

# reading the data
M = pd.read_csv("moore.csv" , delimiter="\t" ,header=None)
    

# finding commas and brackets
match_brack_commas = re.compile(r"\[\d+\]|,")

# finding anything that is not a number
match_non_numbers = re.compile(r"[^\d]+")

#M[1] is the number of transistors 
#cleaning M1 from brackets and commas
M1_without_commas_and_brackets = M[1].str.replace(match_brack_commas,'')

#cleaning columns and letting just numbers
M1_with_clean_numbers = M1_without_commas_and_brackets.str.replace(match_non_numbers,'')
M2_with_clean_numbers = M[2].str.replace(match_brack_commas,'')


# transform pandas series to np array then cast to float(resp integer)
Y = M1_with_clean_numbers.values.astype(np.float)
X = M2_with_clean_numbers.values.astype(np.integer)

plt.scatter(X,Y)
plt.show()

# transformin Y to a log function which is correlated with X
Y = np.log(Y)
plt.scatter(X,Y)
plt.show()


# applyinf the linear regression formulas to find ab , wth yhat = a*X + b
denominator = X.dot(X) - X.sum()*X.mean()
a = (X.dot(Y) - X.sum()*Y.mean())/denominator
b = (X.dot(X)*Y.mean() -X.mean()*X.dot(Y))/denominator 

yhat = a*X + b
plt.scatter(X,Y)
plt.plot(X,yhat)

plt.show()


# Now let's interprete the results
# We want to know how many years it 
# takes for the number of transistors to double?
# we  have y = exp(a*x + b)
# 2*y = exp(a*(X+ eps) + b)
# 2*exp(aX + b) = exp(a*(X+eps) + b)
# eps = log(2)/a

print('It takes ', np.log(2)/a ,'years to double the number of transistores on a chip')