from scipy.sparse import csr_matrix
import math

A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])

diag = A.diagonal()
print(diag)
row = []
for i in range(len(diag)):
    row.append(i)
print(row)

Degree = csr_matrix((diag, (row, row)), shape=A.shape)
print(Degree)

diag_inverse = []
for element in diag:
    if element != 0:
        diag_inverse.append(1 / element)
    else:
        diag_inverse.append(element)
Degree_inverse = csr_matrix((diag_inverse, (row, row)), shape=A.shape)
print(Degree_inverse)

diag_log_inverse = []
for element in diag:
    if element == 1 or element == 0:
        diag_log_inverse.append(0)
    else:
        diag_log_inverse.append(1 / math.log(element))
Degree_log_inverse = csr_matrix((diag_log_inverse, (row, row)), shape=A.shape)
print(Degree_log_inverse)