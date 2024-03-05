count = 0
for i in range(10000):
    if 2**(i - 1) % 7 == 0:
        count+=1
        print(i)
print(count)