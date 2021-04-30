
def partition(x, top, bottom):
    low = 0
    ret = []
    for i in range(min(len(x), top)):
        ret.append(x[i])
        low = i

    i = len(x) - 1
    retb = []
    while i > low and (len(x) -i) <= bottom:
        retb.append(x[i])
        i -= 1
    retb.reverse()
    return ret + retb






