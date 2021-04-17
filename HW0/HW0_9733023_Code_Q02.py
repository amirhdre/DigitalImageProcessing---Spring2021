'''
___OVERVIEW___
CODE FOR QUESTION 2 - HW 0

___SUMMARY___
Rank transform scores in a game. Then show final result in a dictionary.

___AUTHOR___
Amirhossein Daraie — 9733023

___DATE___
2/26/2021

'''

import numpy as np

# create array of 100 elements between -2 and 7
# note: upper bound is excluded
nums = np.random.randint(-2,8,100)

# create counter and ranks dictionary
ranks = dict()
i = 1

# loop over the numbers in array, then save into the dictionary 
for num in np.unique(nums)[::-1]:
    ranks[i] = [num]*sum(nums==num)
    i+=sum(nums==num)


# print nums array
nums = np.sort(nums)
nums = nums[::-1]
print()
print('number array = ',nums)

# print ranks dictionary 
print('ranks = ',ranks)
print()



