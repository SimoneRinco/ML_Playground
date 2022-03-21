import numpy as np

def shape_rank_examples():

  def print_info(a):
    # ndim is the rank of the numpy array
    print(a.ndim)
    # shape is the number of dimensions along each axis
    # Observe that there can be confusion: a 5D array (rank=1 shape=(5)) can be confused with
    # a 5D tensor (rank=5)
    print(a.shape)
    print(a.dtype)

  print('=== scalar example')
  s = np.array(12)
  print_info(s)
  # Observe that s cannot be set
  # s[0] = 32 does not work as s has dimension 0

  print('=== vector example')
  v = np.array([1, 2, 3, 4])
  print_info(v)
  print(v)

  print('=== multi dim example')
  a = np.array(np.linspace(0, 100, 30), dtype=np.float32).reshape(2, 3, 5)
  print_info(a)
  print(a)

def slicing_example():

  # This example contains also broadcasting examples

  a = np.arange(0, 12, 1).reshape(3, 4)

  b = np.array([20, 30, 60, 70])
  print('=== a[2,:] = b')
  a[2,:] = b
  print(a)

  c = np.array([10, 20, 30])
  print('=== a[:, 2] = c')
  a[:, 2] = c
  print(a)

  d = np.array([[11, 21, 31]])
  # this works even if a is (3, 4) and c is (1, 3) instead of (3, 1)
  print('=== a[:, 1] = d')
  print(d.shape)
  a[:, 1] = d
  print(a)

  # Curiously, a (3, 1) matrix does not work
  #e = d.transpose()
  #print('=== a[:, 3] = e')
  #a[:, 3] = e
  #print(a)

  # In higher dimensions
  a = np.arange(0, 4*5*6).reshape(4, 5, 6)
  b = 999 * np.ones((3, 2))
  # b's rank is smaller than a's ---> b needs broadcasting
  a[:3, 2:, 3:5] = b
  print(a)



def element_wise_ops_examples():

  a = np.array(np.arange(0, 10, 1), dtype=np.int32)
  b = np.array(np.arange(2, 22, 2), dtype=np.int32)

  print('=== element-wise sum')
  print(a + b)
  print('=== element-wise multiplication')
  print(a * b)
  print('=== multiplication by a scalar')
  print(5.4 * b)

def broadcasting_examples():

  a = np.array([[2, 3, 4], [1, 0, 3], [4, 0, -1]])
  b = np.array([1, 3, 5])
  print('=== (1)')
  # a and b have different ranks (a has rank 2 and b has rank 1)
  # 1) the shape of b is changed adding dimensions at the front: (3) --> (1, 3)
  # 2) b is repeated in the front axis: b_new[x, :] = b
  print(a + b)

  b = np.array([[1, 3, 5]])
  print('=== (2)')
  # Equivalent to the above, but step 1) is not necessary.
  print(a + b)

  b = np.array([[1], [3], [5]])
  print('=== (3)')
  # In this case a and b have same rank but different size (b is (3, 1)).
  # Step 1) is not necessary
  # 2) b is reapeated in the last axis: b_new[:, x] = b
  print(a + b)

  ### In higher dimensions:

  a = np.arange(0, 2*3*4*5).reshape(2, 3, 4, 5)
  b = 10 + np.arange(0, 4*5).reshape(4, 5)
  print('=== (4)')
  # 1) b: (4, 5) --> (1, 1, 4, 5)
  # 2) b_new[x, y, :, :] = b
  print(a + b)

  b = 100 * np.ones((2, 3))
  print('=== (5)')
  print(a + b)
  
def dot_product_examples():

  def print_dot(a, b):
    dotp = np.dot(a, b)
    print('a:', a)
    print('b:', b)
    print(dotp)
    print('a.b shape:', dotp.shape)


  a = np.array([3, 5, 7])
  b = np.array([1, 2, 3])

  print('=== (1)')
  # dot product of two vectors
  print_dot(a, b)

  a = np.arange(0, 2*3).reshape(2, 3)
  print('=== (2)')
  # Matrix-vector multiplication:
  # (2, 3) . (3,) -> (2,)
  print_dot(a, b)

  b = 2 * np.arange(0, 3*5).reshape(3, 5)
  print('=== (3)')
  # Matrix-matrix multiplication:
  # (2, 3) . (3, 5) -> 2, 5
  print_dot(a, b)

  a = np.arange(0, 2*3*4*5).reshape(2, 3, 4, 5)
  b = np.arange(0, 6*5*7).reshape(6, 5, 7)
  print('=== (4)')
  # Tensor-tensor multiplication:
  # (2, 3, 4, 5) . (6, 5, 7) -> (2, 3, 4, 6, 7)
  print_dot(a, b)



if __name__ == "__main__":
  #shape_rank_examples()
  #slicing_example()
  #element_wise_ops_examples()
  #broadcasting_examples()
  dot_product_examples()