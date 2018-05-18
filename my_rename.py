import os


A = os.listdir("TFTB") #11
j = 0
for i in A:
    #print(A)
    aa = os.listdir("TFTB/" + i)
    print(aa)
    for ii in aa:
        print(ii)
        print(type(ii))
        lowerdir = ii.lower()
        print(lowerdir)
        os.rename("TFTB/" + A[j] + "/" + ii, "TFTB/" + A[j] + "/" + lowerdir)
        print(j)
    j += 1
