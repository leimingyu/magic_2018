#!/usr/bin/env python
'''application list, application dir'''

app = []
app2dir_dd = {}

#------------------------------------------------------------------------------
# 
#------------------------------------------------------------------------------

# cuda sdk 8.0:  ``$./run.sh devid``
app.append('cudasdk_matrixMul')
app2dir_dd['cudasdk_matrixMul']              = '../apps/devid_cudasdk80/0_Simple/matrixMul'

app.append('cudasdk_vectorAdd')
app2dir_dd['cudasdk_vectorAdd']              = '../apps/devid_cudasdk80/0_Simple/vectorAdd'

app.append('cudasdk_convolutionFFT2D')
app2dir_dd['cudasdk_convolutionFFT2D']       = '../apps/devid_cudasdk80/3_Imaging/convolutionFFT2D'

app.append('cudasdk_convolutionSeparable')
app2dir_dd['cudasdk_convolutionSeparable']   = '../apps/devid_cudasdk80/3_Imaging/convolutionSeparable'

app.append('cudasdk_convolutionTexture')
app2dir_dd['cudasdk_convolutionTexture']     = '../apps/devid_cudasdk80/3_Imaging/convolutionTexture'

app.append('cudasdk_dct8x8')
app2dir_dd['cudasdk_dct8x8']                 = '../apps/devid_cudasdk80/3_Imaging/dct8x8'

app.append('cudasdk_dwtHaar1D')
app2dir_dd['cudasdk_dwtHaar1D']              = '../apps/devid_cudasdk80/3_Imaging/dwtHaar1D'

app.append('cudasdk_dxtc')
app2dir_dd['cudasdk_dxtc']                   = '../apps/devid_cudasdk80/3_Imaging/dxtc'

app.append('cudasdk_stereoDisparity')
app2dir_dd['cudasdk_stereoDisparity']        = '../apps/devid_cudasdk80/3_Imaging/stereoDisparity'

app.append('cudasdk_binomialOptions')
app2dir_dd['cudasdk_binomialOptions']        = '../apps/devid_cudasdk80/4_Finance/binomialOptions'

app.append('cudasdk_BlackScholes')
app2dir_dd['cudasdk_BlackScholes']           = '../apps/devid_cudasdk80/4_Finance/BlackScholes'

app.append('cudasdk_quasirandomGenerator')
app2dir_dd['cudasdk_quasirandomGenerator']   = '../apps/devid_cudasdk80/4_Finance/quasirandomGenerator'

app.append('cudasdk_SobolQRNG')
app2dir_dd['cudasdk_SobolQRNG']              = '../apps/devid_cudasdk80/4_Finance/SobolQRNG'

app.append('cudasdk_c++11Cuda')
app2dir_dd['cudasdk_c++11Cuda']              = '../apps/devid_cudasdk80/6_Advanced/c++11_cuda'

app.append('cudasdk_concurrentKernels')
app2dir_dd['cudasdk_concurrentKernels']      = '../apps/devid_cudasdk80/6_Advanced/concurrentKernels'

app.append('cudasdk_eigenvalues')
app2dir_dd['cudasdk_eigenvalues']            = '../apps/devid_cudasdk80/6_Advanced/eigenvalues'

app.append('cudasdk_fastWalshTransform')
app2dir_dd['cudasdk_fastWalshTransform']     = '../apps/devid_cudasdk80/6_Advanced/fastWalshTransform'

app.append('cudasdk_FDTD3d')
app2dir_dd['cudasdk_FDTD3d']                 = '../apps/devid_cudasdk80/6_Advanced/FDTD3d'

app.append('cudasdk_interval')
app2dir_dd['cudasdk_interval']               = '../apps/devid_cudasdk80/6_Advanced/interval'

app.append('cudasdk_lineOfSight')
app2dir_dd['cudasdk_lineOfSight']            = '../apps/devid_cudasdk80/6_Advanced/lineOfSight'

app.append('cudasdk_mergeSort')
app2dir_dd['cudasdk_mergeSort']              = '../apps/devid_cudasdk80/6_Advanced/mergeSort'

app.append('cudasdk_radixSortThrust')
app2dir_dd['cudasdk_radixSortThrust']        = '../apps/devid_cudasdk80/6_Advanced/radixSortThrust'

app.append('cudasdk_reduction')
app2dir_dd['cudasdk_reduction']              = '../apps/devid_cudasdk80/6_Advanced/reduction'

app.append('cudasdk_scalarProd')
app2dir_dd['cudasdk_scalarProd']             = '../apps/devid_cudasdk80/6_Advanced/scalarProd'

app.append('cudasdk_scan')
app2dir_dd['cudasdk_scan']                   = '../apps/devid_cudasdk80/6_Advanced/scan'

app.append('cudasdk_segmentationTreeThrust')
app2dir_dd['cudasdk_segmentationTreeThrust'] = '../apps/devid_cudasdk80/6_Advanced/segmentationTreeThrust'

app.append('cudasdk_shflscan')
app2dir_dd['cudasdk_shflscan']               = '../apps/devid_cudasdk80/6_Advanced/shfl_scan'

app.append('cudasdk_sortingNetworks')
app2dir_dd['cudasdk_sortingNetworks']        = '../apps/devid_cudasdk80/6_Advanced/sortingNetworks'

app.append('cudasdk_threadFenceReduction')
app2dir_dd['cudasdk_threadFenceReduction']   = '../apps/devid_cudasdk80/6_Advanced/threadFenceReduction'

app.append('cudasdk_transpose')
app2dir_dd['cudasdk_transpose']              = '../apps/devid_cudasdk80/6_Advanced/transpose'

app.append('cudasdk_batchCUBLAS')
app2dir_dd['cudasdk_batchCUBLAS']            = '../apps/devid_cudasdk80/7_CUDALibraries/batchCUBLAS'

app.append('cudasdk_boxFilterNPP')
app2dir_dd['cudasdk_boxFilterNPP']           = '../apps/devid_cudasdk80/7_CUDALibraries/boxFilterNPP'

app.append('cudasdk_MCEstimatePiInlineP')
app2dir_dd['cudasdk_MCEstimatePiInlineP']    = '../apps/devid_cudasdk80/7_CUDALibraries/MC_EstimatePiInlineP'

app.append('cudasdk_MCEstimatePiInlineQ')
app2dir_dd['cudasdk_MCEstimatePiInlineQ']    = '../apps/devid_cudasdk80/7_CUDALibraries/MC_EstimatePiInlineQ'

app.append('cudasdk_MCEstimatePiP')
app2dir_dd['cudasdk_MCEstimatePiP']          = '../apps/devid_cudasdk80/7_CUDALibraries/MC_EstimatePiP'

app.append('cudasdk_MCEstimatePiQ')
app2dir_dd['cudasdk_MCEstimatePiQ']          = '../apps/devid_cudasdk80/7_CUDALibraries/MC_EstimatePiQ'

app.append('cudasdk_MCSingleAsianOptionP')
app2dir_dd['cudasdk_MCSingleAsianOptionP']   = '../apps/devid_cudasdk80/7_CUDALibraries/MC_SingleAsianOptionP'

app.append('cudasdk_simpleCUBLAS')
app2dir_dd['cudasdk_simpleCUBLAS']           = '../apps/devid_cudasdk80/7_CUDALibraries/simpleCUBLAS'

app.append('cudasdk_simpleCUFFTcallback')
app2dir_dd['cudasdk_simpleCUFFTcallback']    = '../apps/devid_cudasdk80/7_CUDALibraries/simpleCUFFT_callback'


# polybench 
app.append('poly_2dconv')
app2dir_dd['poly_2dconv']      = '../apps/devid_poly/CUDA/2DCONV'

app.append('poly_3dconv')
app2dir_dd['poly_3dconv']      = '../apps/devid_poly/CUDA/3DCONV'

app.append('poly_3mm')
app2dir_dd['poly_3mm']         = '../apps/devid_poly/CUDA/3MM'

app.append('poly_atax')
app2dir_dd['poly_atax']        = '../apps/devid_poly/CUDA/ATAX'

app.append('poly_bicg')
app2dir_dd['poly_bicg']        = '../apps/devid_poly/CUDA/BICG'

app.append('poly_correlation')
app2dir_dd['poly_correlation'] = '../apps/devid_poly/CUDA/CORR'

app.append('poly_covariance')
app2dir_dd['poly_covariance']  = '../apps/devid_poly/CUDA/COVAR'

app.append('poly_fdtd2d')
app2dir_dd['poly_fdtd2d']      = '../apps/devid_poly/CUDA/FDTD-2D'

app.append('poly_gemm')
app2dir_dd['poly_gemm']        = '../apps/devid_poly/CUDA/GEMM'

app.append('poly_gesummv')
app2dir_dd['poly_gesummv']     = '../apps/devid_poly/CUDA/GESUMMV'

app.append('poly_mvt')
app2dir_dd['poly_mvt']         = '../apps/devid_poly/CUDA/MVT'

app.append('poly_syr2k')
app2dir_dd['poly_syr2k']       = '../apps/devid_poly/CUDA/SYR2K'

app.append('poly_syrk')
app2dir_dd['poly_syrk']        = '../apps/devid_poly/CUDA/SYRK'

# lonestar 
app.append('lonestar_bh')
app2dir_dd['lonestar_bh']      = '../apps/devid_lonestar/apps/bh'

app.append('lonestar_dmr')
app2dir_dd['lonestar_dmr']     = '../apps/devid_lonestar/apps/dmr'

app.append('lonestar_mst')
app2dir_dd['lonestar_mst']     = '../apps/devid_lonestar/apps/mst'

app.append('lonestar_sssp')
app2dir_dd['lonestar_sssp']    = '../apps/devid_lonestar/apps/sssp'

# parboil 
app.append('parboil_bfs')
app2dir_dd['parboil_bfs']      = '../apps/devid_parboil/benchmarks/bfs'

app.append('parboil_cutcp')
app2dir_dd['parboil_cutcp']    = '../apps/devid_parboil/benchmarks/cutcp'

app.append('parboil_lbm')
app2dir_dd['parboil_lbm']      = '../apps/devid_parboil/benchmarks/lbm'

app.append('parboil_mriq')
app2dir_dd['parboil_mriq']     = '../apps/devid_parboil/benchmarks/mri-q'

app.append('parboil_sgemm')
app2dir_dd['parboil_sgemm']    = '../apps/devid_parboil/benchmarks/sgemm'

app.append('parboil_stencil')
app2dir_dd['parboil_stencil']  = '../apps/devid_parboil/benchmarks/stencil'

# rodinia
app.append('rodinia_backprop')
app2dir_dd['rodinia_backprop']   = '../apps/devid_rodinia/backprop'

app.append('rodinia_b+tree')
app2dir_dd['rodinia_b+tree']     = '../apps/devid_rodinia/b+tree'

app.append('rodinia_dwt2d')
app2dir_dd['rodinia_dwt2d']      = '../apps/devid_rodinia/dwt2d'

app.append('rodinia_gaussian')
app2dir_dd['rodinia_gaussian']   = '../apps/devid_rodinia/gaussian'

app.append('rodinia_heartwall')
app2dir_dd['rodinia_heartwall']  = '../apps/devid_rodinia/heartwall'

app.append('rodinia_hybridsort')
app2dir_dd['rodinia_hybridsort'] = '../apps/devid_rodinia/hybridsort'

app.append('rodinia_hotspot')
app2dir_dd['rodinia_hotspot']    = '../apps/devid_rodinia/hotspot'

app.append('rodinia_lud')
app2dir_dd['rodinia_lud']        = '../apps/devid_rodinia/lud'

app.append('rodinia_lavaMD')
app2dir_dd['rodinia_lavaMD']     = '../apps/devid_rodinia/lavaMD'

app.append('rodinia_needle')
app2dir_dd['rodinia_needle']     = '../apps/devid_rodinia/nw'

app.append('rodinia_pathfinder')
app2dir_dd['rodinia_pathfinder'] = '../apps/devid_rodinia/pathfinder'

# shoc 
app.append('shoc_lev1BFS')
app2dir_dd['shoc_lev1BFS']       = '../apps/devid_shoc/src/cuda/level1/bfs'

app.append('shoc_lev1sort')
app2dir_dd['shoc_lev1sort']      = '../apps/devid_shoc/src/cuda/level1/sort'

app.append('shoc_lev1fft')
app2dir_dd['shoc_lev1fft']       = '../apps/devid_shoc/src/cuda/level1/fft'

app.append('shoc_lev1GEMM')
app2dir_dd['shoc_lev1GEMM']      = '../apps/devid_shoc/src/cuda/level1/gemm'

app.append('shoc_lev1md5hash')
app2dir_dd['shoc_lev1md5hash']   = '../apps/devid_shoc/src/cuda/level1/md5hash'

app.append('shoc_lev1reduction')
app2dir_dd['shoc_lev1reduction'] = '../apps/devid_shoc/src/cuda/level1/reduction'
