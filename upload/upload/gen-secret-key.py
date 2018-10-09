import os

b = os.urandom(16)
print(b)
with open('flask-secret-key', 'wb') as f:
    f.write(b)

with open('flask-secret-key', 'rb') as f:
    b2 = f.read()
    print(b2)
