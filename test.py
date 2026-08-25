
def countdown(n):
    while n > 0:
        yield n
        n -= 1

for num in countdown(5):
    print(num)   # 依次输出 5 4 3 2 1



"""
def infinite():
    i = 0
    while True and i>10:
        yield i
        i += 1
"""

