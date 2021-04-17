'''
___OVERVIEW___
CODE FOR QUESTION 3 - HW 0

___SUMMARY___
Simplify a rationale number 

___AUTHOR___
Amirhossein Daraie — 9733023

___DATE___
2/26/2021

'''

# initialize nominator and denominator 
nom = 36
denom = 120


def simplify(nom,denom):
    if nom is True or denom is True or \
        nom is False or denom is False or  \
        nom is None  or denom is None:
        # nominator or denominator is True or False or None
        return None
    elif denom == 0:
        # denominator is zero (undefinded or infinity) 
        return None
    elif not(float(nom).is_integer()):
        # nominator is not integer 
        return None
    elif not(float(denom).is_integer()):
        # denominator is not integer 
        return None


    hcf = lambda a,b : a if (b == 0) else hcf(b, a % b) 
    k = hcf(nom, denom) 
    result = tuple((nom/k, denom/k))
    return result

for case in [(19, 2 ), (True, 6), (18, 0), (21, 14), (-26, 91), (6, -39)]:
    print(f'{case} : {simplify(case[0],case[1])}')

