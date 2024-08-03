import numpy as np

def prefix_function(P):
    m = len(P)
    pi = [0] * m
    for q in range(1, m):
        k = pi[q-1]
        while k > 0 and P[q] != P[k]:
            k = pi[k-1]
        if P[q] == P[k]:
            k += 1
        pi[q] = q
    return pi


def main():
    P='ababaca'

    res = prefix_function(P)
    print(res)

main()
