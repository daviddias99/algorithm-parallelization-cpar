experiments_lu_seq = [

  # LU Seq
  ['lu_omp_1', 1, 3, 1],

  # LU Blocks
  ['lu_omp_1', 2, 3, 128],
  ['lu_omp_1', 2, 3, 256],
  ['lu_omp_1', 2, 3, 512],
  ['lu_omp_2', 2, 3, 128],
  ['lu_omp_2', 2, 3, 256],
  ['lu_omp_2', 2, 3, 512],
  ['lu_omp_3', 2, 3, 128],
  ['lu_omp_3', 2, 3, 256],
  ['lu_omp_3', 2, 3, 512],
  ['lu_omp_4', 2, 3, 128],
  ['lu_omp_4', 2, 3, 256],
  ['lu_omp_4', 2, 3, 512],
]

experiments_lu_par_data = [
  # LU OMP Data
  ['lu_omp_1', 3, 3, 128],
  ['lu_omp_1', 3, 3, 256],
  ['lu_omp_1', 3, 3, 512],
  ['lu_omp_2', 3, 3, 128],
  ['lu_omp_2', 3, 3, 256],
  ['lu_omp_2', 3, 3, 512],
  ['lu_omp_3', 3, 3, 128],
  ['lu_omp_3', 3, 3, 256],
  ['lu_omp_3', 3, 3, 512],
  ['lu_omp_4', 3, 3, 128],
  ['lu_omp_4', 3, 3, 256],
  ['lu_omp_4', 3, 3, 512],
]

experiments_lu_par_func = [
  # LU OMP Func
  ['lu_omp_1', 4, 3, 128],
  ['lu_omp_1', 4, 3, 256],
  ['lu_omp_1', 4, 3, 512],
  ['lu_omp_2', 4, 3, 128],
  ['lu_omp_2', 4, 3, 256],
  ['lu_omp_2', 4, 3, 512],
  ['lu_omp_3', 4, 3, 128],
  ['lu_omp_3', 4, 3, 256],
  ['lu_omp_3', 4, 3, 512],
  ['lu_omp_4', 4, 3, 128],
  ['lu_omp_4', 4, 3, 256],
  ['lu_omp_4', 4, 3, 512],
]

experiments_lu_sycl_cpu = [
  # Sycl CPU
  ['lu_sycl_cpu', 8 , 1, 'cpu', 3],
  ['lu_sycl_cpu', 16, 1, 'cpu', 3],
  ['lu_sycl_cpu', 32, 1, 'cpu', 3],
]

experiments_lu_sycl_gpu = [
  # Sycl GPU
  ['lu_sycl_gpu', 8 , 1, 'gpu', 3],
  ['lu_sycl_gpu', 16, 1, 'gpu', 3],
  ['lu_sycl_gpu', 32, 1, 'gpu', 3],
]

experiments_mm_omp = [

  # OMP Seq Block
  ['matmul_omp_1', 1, 3, 1],

  # OMP Collapse
  ['matmul_omp_1', 3, 3, 128],
  ['matmul_omp_1', 3, 3, 256],
  ['matmul_omp_1', 3, 3, 512],
  ['matmul_omp_2', 3, 3, 128],
  ['matmul_omp_2', 3, 3, 256],
  ['matmul_omp_2', 3, 3, 512],
  ['matmul_omp_3', 3, 3, 128],
  ['matmul_omp_3', 3, 3, 256],
  ['matmul_omp_3', 3, 3, 512],
  ['matmul_omp_4', 3, 3, 128],
  ['matmul_omp_4', 3, 3, 256],
  ['matmul_omp_4', 3, 3, 512],
]
experiments_mm_cuda = [

  # CUDA Block
  ['matmul_cuda', 2, 3, 8],
  ['matmul_cuda', 2, 3, 16],
  ['matmul_cuda', 2, 3, 32],

  # CUDA Block Local Mem
  ['matmul_cuda', 3, 3, 8],
  ['matmul_cuda', 3, 3, 16],
  ['matmul_cuda', 3, 3, 32],
]

experiments_mm_sycl_cpu = [

  # Sycl CPU Naive
  ['matmul_sycl_cpu', 8 , 1, 'cpu', 3],
  ['matmul_sycl_cpu', 16, 1, 'cpu', 3],
  ['matmul_sycl_cpu', 32, 1, 'cpu', 3],

  # Sycl CPU Block
  ['matmul_sycl_cpu', 8 , 2, 'cpu', 3],
  ['matmul_sycl_cpu', 16, 2, 'cpu', 3],
  ['matmul_sycl_cpu', 32, 2, 'cpu', 3],

  # Sycl CPU Block Local Mem
  ['matmul_sycl_cpu', 8 , 3, 'cpu', 3],
  ['matmul_sycl_cpu', 16, 3, 'cpu', 3],
  ['matmul_sycl_cpu', 32, 3, 'cpu', 3],
]

experiments_mm_sycl_gpu = [

  # Sycl CPU Naive
  ['matmul_sycl_gpu', 8 , 1, 'gpu', 3],
  ['matmul_sycl_gpu', 16, 1, 'gpu', 3],
  ['matmul_sycl_gpu', 32, 1, 'gpu', 3],

  # Sycl CPU Block
  ['matmul_sycl_gpu', 8 , 2, 'gpu', 3],
  ['matmul_sycl_gpu', 16, 2, 'gpu', 3],
  ['matmul_sycl_gpu', 32, 2, 'gpu', 3],

  # Sycl CPU Block Local Mem
  ['matmul_sycl_gpu', 8 , 3, 'gpu', 3],
  ['matmul_sycl_gpu', 16, 3, 'gpu', 3],
  ['matmul_sycl_gpu', 32, 3, 'gpu', 3],
]