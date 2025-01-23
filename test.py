
def sumoftwo(n, x, lst):
    lst.sort()
    return lst

n, x = list(map(int, input().strip().split()))
lst = list(map(int, input().strip().split()))
a = sumoftwo(n, x, lst)
print(a)