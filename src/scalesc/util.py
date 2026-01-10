import os 
from gc import collect
from typing import TYPE_CHECKING, Union
import sys
import glob
import time 
import scipy
import math
import warnings
import cupy as cp
import numpy as np
import rapids_singlecell as rsc
import scanpy as sc
import anndata as ad
import scipy.sparse
import cupyx.scipy.sparse
from cupyx.scipy.sparse import issparse, isspmatrix_csc, isspmatrix_csr, spmatrix
# import harmonypy_gpu
from scalesc import harmonypy_gpu, kernels
# comment when debug
warnings.filterwarnings('ignore', 'Expected ')
warnings.simplefilter('ignore')

# define const
GPU_MEMORY_LIMIT = 80
GPU_MEMORY_USAGE = GPU_MEMORY_LIMIT * 0.6 # 0.2 for test now 
GPU_ARRAY_TYPE = Union[cp.ndarray, cupyx.scipy.sparse.csr_matrix, cupyx.scipy.sparse.csc_matrix]
CPU_ARRAY_TYPE = Union[np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]

class AnnDataBatchReader():
    """Chunked dataloader for extremely large single-cell dataset. Return a data chunk each time for further processing.""" 
    def __init__(self, 
                 data_dir, 
                 preload_on_cpu=True, 
                 preload_on_gpu=False, 
                 gpus=None, 
                 max_cell_batch=100000, 
                 max_gpu_memory_usage=GPU_MEMORY_USAGE, 
                 return_anndata=True):
        assert os.path.exists(data_dir), 'data dir not exists.'
        assert os.path.isdir(data_dir), 'target is not a folder.'
        self.files = {}
        self.total_size = 0
        self.first = True
        for file in glob.glob(f'{data_dir}/*.h5*'):
            size = os.path.getsize(file)
            self.files[file] = {'size': size}
            self.total_size += size
        self.NUM_GENE_PARTATION = np.ceil(self.total_size / max_gpu_memory_usage)
        self.MAX_GPU_MEM_USAGE = max_gpu_memory_usage
        self.max_cell_batch = max_cell_batch
        self.preload_on_cpu = preload_on_cpu
        self.preload_on_gpu = preload_on_gpu
        self.have_looped_once = False
        self.anndata = None
        self.adata = None
        self.n_cell = -1
        self.n_cell_origin = -1
        self.n_gene = -1
        self.n_gene_origin = -1
        self.genes_filter = None
        self.cells_filter = None
        self.set_genes_filter(None)
        self.set_cells_filter(None)

        if preload_on_gpu:  # use multiple cards, if nothing specified, use gpu:0 as default
            if gpus is not None:
                self.gpus = gpus
            else:
                self.gpus = [0]

        if preload_on_cpu or preload_on_gpu:  # preload data in chunks on CPU or GPU
            self._preload()

    def clear(self):
        if self.batches:
            self.batches.clear()
        if self.adata is not None:
            self.adata = None
        if self.anndata:
            self.anndata = None
        del(self)
            
    def batch_to_CPU(self):
        if self.batches:
            for i in range(len(self.batches)):
                rsc.get.anndata_to_CPU(self.batches[i])
        
    def batch_to_GPU(self):
        if self.batches:
            for i in range(len(self.batches)):
                rsc.get.anndata_to_GPU(self.batches[i])
        
    def get_merged_adata_with_X(self): # merge on CPU only
        if self.batches:
            assert isinstance(self.batches[0].X, CPU_ARRAY_TYPE), "call 'batch_to_CPU()' to transfer all batches to CPU." 
            adata_merged = ad.concat(self.batches, merge='same', uns_merge='same')
            return adata_merged

    def _batch_criteria(self, criteria, **kwargs):
        # TODO: indptr batch instead of limiting on GPU memory.
        pass

    def _get_size(self, obj, unit='G'):
        size = sys.getsizeof(obj) 
        if unit == 'G':
            size /= (1024*1024*1024)
        elif unit == 'M':
            size /= (1024*1024)
        return size
    
    def _get_anndata_obj(self):
        assert self.anndata is not None, "anndata object is empty! if preload_on_cpu is False, anndata will be automatically initialized after calling 'batchify()'."
        obj = ad.concat(self.anndata, merge='same', uns_merge='same')
        if self.cells_filter is not None:
            filter_flatten = []
            for f in self.cells_filter:
                filter_flatten += list(f)
            obj = obj[filter_flatten]
        if self.genes_filter is not None:
            obj = obj[:, self.genes_filter]
        return obj

    @property
    def shape(self):
        return self.n_cell, self.n_gene
        # return self.n_cell_origin, self.n_gene_origin

    def set_cells_filter(self, filter, update=True):
        """ Update cells filter and applied on data chunks if `update` set to `True`, otherwise, update filter only."""
        if filter is not None:
            if self.preload_on_cpu and update:
                self.update_by_cells_filter(filter)
            if self.cells_filter is not None:
                for _cells_filter, _filter in zip(self.cells_filter, filter):
                    assert sum(_cells_filter) == len(_filter), "current cell filter doesn't match to the previous one."
                    _cells_filter[_cells_filter == 1] = _filter
            else:
                self.cells_filter = filter
            self.n_cell = sum([sum(_) for _ in filter])
            self.n_cell_origin = sum([len(_) for _ in filter])

    # def set_cells_filter(self, filter, update=True):
    #     """
    #         Update cells filter and applied on data chunks if update set to True,
    #         otherwise, update filter only. 
    #     """
    #     if filter is not None:
    #         self.n_cell = sum([sum(_) for _ in filter])
    #         self.n_cell_origin = sum([len(_) for _ in filter])
    #     self.cells_filter = filter
    #     if self.preload_on_cpu and update:
    #         self.update_by_cells_filter(self.cells_filter)

    def set_genes_filter(self, filter, update=True):
        """ Update genes filter and applied on data chunks if `update` set to True, otherwise, update filter only. 

            Note:  
                Genes filter can be set sequentially, a new filter should be always compatible with the previous 
                filtered data.
        """
        if filter is not None:
            if self.preload_on_cpu and update:
                self.update_by_genes_filter(filter)
            if self.genes_filter is not None:
                assert self.n_gene == len(filter), "current gene filter doesn't match to the previous one."
                self.genes_filter[self.genes_filter == 1] = filter
            else:
                self.genes_filter = filter
            self.n_gene = sum(self.genes_filter)
            self.n_gene_origin = len(self.genes_filter)

    # def set_genes_filter(self, filter, update=True):
    #     """
    #         Update genes filter and applied on data chunks if update set to True,
    #         otherwise, update filter only. 
    #         Notes:  genes filter can be set sequentially, a new filter should be always compatible with the previous filtered data.
    #     """
    #     if filter is not None:
    #         if self.preload_on_cpu and update:
    #             self.update_by_genes_filter(filter)
    #         if self.genes_filter is not None:
    #             assert self.n_gene == len(filter), "current gene filter doesn't match the previous one."
    #             _genes_filter = np.zeros(len(self.genes_filter), dtype=bool)
    #             _genes_filter[np.where(self.genes_filter==1)[0][filter]] = True
    #             filter = _genes_filter
    #         self.n_gene = sum(filter)
    #         self.n_gene_origin = len(filter)
    #     self.genes_filter = filter
         
    def update_by_cells_filter(self, filter):
        if filter is not None:
            for i in range(len(self.batches)):
                self.batches[i] = self.batches[i][filter[i]].copy()
            if self.adata is not None:
                self.adata = self.adata[np.concatenate(filter)].copy()

    def update_by_genes_filter(self, filter):
        if filter is not None:
            for i in range(len(self.batches)):
                self.batches[i] = self.batches[i][:, filter].copy()
            if self.adata is not None:
                self.adata = self.adata[:, filter].copy()

    def gpu_wrapper(self, generator):
        for item in generator:
            rsc.get.anndata_to_GPU(item)
            yield item

    def read(self, fname):
        SUPPORT_EXT = {'.h5ad': sc.read_h5ad, '.h5': sc.read_10x_h5}
        ext = os.path.splitext(fname)[1]
        assert ext in SUPPORT_EXT, "only .h5ad or .h5 file is supported"
        return SUPPORT_EXT[ext](fname)

    def _preload(self):
        """ Read files from disk and preload all chunks on GPU if `preload_on_gpu` set to True, otherwise put on CPU."""
        fnames = list(self.files.keys())
        anndata = []
        fid = 0
        bid = 0
        current_cells = 0
        self.batches = []
        batch = []
        batch_total_size = 0
        # assume can be loaded entirly on CPU right now
        while fid < len(fnames):
            # print(f'loading {fnames[fid]}')
            # sample = sc.read_h5ad(fnames[fid])
            sample = self.read(fnames[fid])
            sample.var_names_make_unique()
            sample.obs_names_make_unique()
            nc, ng = sample.shape
            size_cpu = self._get_size(sample, unit='G')
            if current_cells + nc > self.max_cell_batch and batch:
                if len(batch) == 1:
                    d = batch.pop()
                else:
                    d = ad.concat(batch, merge='same', uns_merge='same')
                check_dtype(d)
                # multiple GPUs enabled
                if self.preload_on_gpu:
                    gpu_id = fid % len(self.gpus)
                    with cp.cuda.Device(gpu_id):
                        rsc.get.anndata_to_GPU(d)
                anndata.append(sc.AnnData(obs=d.obs, 
                                          var=d.var,
                                          uns=d.uns,
                                          obsm=d.obsm,
                                          varm=d.varm,))                
                self.batches.append(d)
                batch = []
                batch_total_size = 0
                # gc()
                current_cells = 0
                bid += 1
            batch.append(sample)
            batch_total_size += size_cpu
            current_cells += nc
            fid += 1
        if batch:
            if len(batch) == 1:
                d = batch.pop()
            else:
                d = ad.concat(batch, merge='same', uns_merge='same')
            check_dtype(d)
            if self.preload_on_gpu:
                rsc.get.anndata_to_GPU(d)
            anndata.append(sc.AnnData(obs=d.obs,
                                      var=d.var,
                                      uns=d.uns,
                                      obsm=d.obsm,
                                      varm=d.varm,))   
            self.batches.append(d)
            bid += 1
            batch = []
            # gc()
        self.anndata = anndata
        self.adata = self._get_anndata_obj()

    def batchify(self, axis='cell'):
        """ Return a data generator if `preload_on_cpu` is set as `True`. """
        if self.preload_on_cpu:
            return self._batchify_from_ram(axis=axis)
        else:
            return self._batchify_from_disk(axis=axis) 
        
    def _batchify_from_ram(self, axis='cell'):
        """ Return a generator if `preload_on_cpu` is set as `True`. """
        t_total = 0
        axes = ['cell', 'gene']
        assert axis in axes, "axis should be either 'cell' or 'gene'."
        if axis == 'cell':
            for bid, batch in enumerate(self.batches):
                t_start = time.time()
                # batch = batch.copy()
                # if not self.preload_on_cpu:
                #     if self.genes_filter is not None: # TODO: can be copy here then delete 'sample' to reduce mem further
                #         batch = batch[:, self.genes_filter]
                #     if self.cells_filter is not None:
                #         batch = batch[self.cells_filter[bid]]
                #     batch = batch.copy()
                # if self.genes_filter is not None: # TODO: can be copy here then delete 'sample' to reduce mem further
                #     batch = batch[:, self.genes_filter]
                # if self.cells_filter is not None:
                #     batch = batch[self.cells_filter[bid]]
                # gc()
                # print(batch.shape)
                # transfer a single batch to GPU
                if not self.preload_on_gpu:
                    # batch = batch.copy() # make it a copy and transfer to GPU, otherwise OOM on GPU
                    rsc.get.anndata_to_GPU(batch) 
                # make sure transfer data back to gpu0 for computing
                # with cp.cuda.Device(0):
                #     rsc.get.anndata_to_GPU(batch)
                t_total += time.time() - t_start
                yield batch
                # transfer that batch back to CPU
                t_start = time.time()
                if not self.preload_on_gpu:
                    rsc.get.anndata_to_CPU(batch) 
                t_total += time.time() - t_start
                del(batch)
        elif axis == 'gene':
            raise NotImplementedError("batchify along genes hasn't been implemented yet")
                # gc()
        # print(f'single loop loading time:  {t_total}')

    def _batchify_from_disk(self, axis='cell'):
        """ Chunk loader when there is no preload on neither CPU nor GPU.

            Note: 
                Usually this is disabled, we assume there is always enough space on CPU
        """
        axes = ['cell', 'gene']
        assert axis in axes, "axis should be either 'cell' or 'gene'."
        if axis == 'cell':
            fnames = list(self.files.keys())
            anndata = []
            fid = 0
            bid = 0
            current_cells = 0
            batch = []
            batch_total_size = 0
            # assume can be loaded entirly on CPU right now
            while fid < len(fnames):
                # print(f'loading {fnames[fid]}')
                # sample = sc.read_h5ad(fnames[fid])
                sample = self.read(fnames[fid])
                sample.var_names_make_unique()
                sample.obs_names_make_unique()
                nc, ng = sample.shape
                size_cpu = self._get_size(sample, unit='G')
                if current_cells + nc > self.max_cell_batch and batch:
                    if len(batch) == 1:
                        d = batch.pop()
                    else:
                        d = ad.concat(batch, merge='same', uns_merge='same')
                    if not self.have_looped_once:
                        anndata.append(sc.AnnData(obs=d.obs, 
                                                var=d.var,
                                                uns=d.uns,
                                                obsm=d.obsm,
                                                varm=d.varm,))    
                    if self.genes_filter is not None:
                        d = d[:, self.genes_filter].copy()
                    if self.cells_filter is not None:
                        d = d[self.cells_filter[bid]].copy()
                    check_dtype(d)
                    rsc.get.anndata_to_GPU(d)            
                    yield d
                    del(d)
                    batch = []
                    batch_total_size = 0
                    current_cells = 0
                    bid += 1
                batch.append(sample)
                batch_total_size += size_cpu
                current_cells += nc
                fid += 1
            if batch:
                if len(batch) == 1:
                    d = batch.pop()
                else:
                    d = ad.concat(batch, merge='same', uns_merge='same')
                if not self.have_looped_once:
                    anndata.append(sc.AnnData(obs=d.obs,
                            var=d.var,
                            uns=d.uns,
                            obsm=d.obsm,
                            varm=d.varm,))   
                if self.genes_filter is not None:
                    d = d[:, self.genes_filter].copy()
                if self.cells_filter is not None:
                    d = d[self.cells_filter[bid]].copy()
                check_dtype(d)
                rsc.get.anndata_to_GPU(d)
                yield d
                bid += 1
                batch = []
            self.anndata = anndata
            self.have_looped_once = True
        # if axis == 'cell':
        #     fid = 0
        #     bid = 0
        #     while fid < len(fnames):
        #         current_size = 0
        #         batch = []
        #         while fid < len(fnames) and current_size + self.files[fnames[fid]]['size'] <= self.MAX_GPU_MEM_USAGE:
        #             # sample = sc.read_h5ad(fnames[fid])
        #             sample = self.read(fnames[fid])
        #             sample.var_names_make_unique()
        #             sample.obs_names_make_unique()
        #             if self.genes_filter is not None:
        #                 sample = sample[:, self.genes_filter].copy()
        #             batch.append(sample)
        #             current_size += self.files[fnames[fid]]['size']
        #             fid += 1
        #         batched_data = ad.concat(batch)
        #         if self.cells_filter is not None:
        #             batched_data = batched_data[self.cells_filter[bid]].copy()
        #         batch = []      
        #         current_size = 0 
        #         bid += 1
        #         # gc()
        #         check_dtype(batched_data)
        #         rsc.get.anndata_to_GPU(batched_data)
        #         yield batched_data
        #         del(batched_data)
        #         # gc()
        elif axis == 'gene':
            bid = 0
            cells_filter = np.concatenate(self.cells_filter)
            while bid < self.NUM_GENE_PARTATION:
                batch = []
                for i, fname in enumerate(fnames):
                    # sample = sc.read_h5ad(fname) # not sure if can do chuncked read, like specifying rows to read
                    sample = self.read(fname)
                    sample.var_names_make_unique()
                    sample.obs_names_make_unique()
                    if self.genes_filter is not None:
                        sample = sample[:, self.genes_filter].copy()
                    _, n_gene = sample.shape
                    batch_size = np.ceil(n_gene / self.NUM_GENE_PARTATION)
                    start_index = int(batch_size * bid) 
                    end_index = int(min(n_gene, start_index+batch_size))
                    sample_slice = sample[:, start_index:end_index]
                    batch.append(sample_slice.copy())
                    del(sample_slice)
                    del(sample)
                    # gc()
                batched_data = ad.concat(batch, merge='same', uns_merge='same') 
                if cells_filter is not None:
                    batched_data_copy = batched_data[cells_filter].copy()
                bid += 1
                batch = []  
                # del(batched_data)
                gc()
                yield batched_data_copy
                del(batched_data_copy)
                # gc()


                
def filter_cells(
    adata,
    *,
    qc_var,
    min_count,
    max_count):
    """ Cell filtering according to min and max gene counts.

        Note:  
            `filter_cells` in rsc doesn't support filter out cells 
            by min and max counts at the same time. a modification 
            is made here for dealing with both together.
    """    
    if qc_var in adata.obs.keys():
        if min_count is not None and max_count is not None:
            inter = (adata.obs[qc_var] <= max_count) & (min_count <= adata.obs[qc_var])
        elif min_count is not None:
            inter = adata.obs[qc_var] >= min_count
        elif max_count is not None:
            inter = adata.obs[qc_var] <= max_count
        else:
            print("Please specify a cutoff to filter against")
    return inter.values
    

def svd_flip(pcs):
    """ Flip the signs of loading according to sign(max(abs(loadings))).

        Note: this function is used to match up scanpy's results of PCA.

        Args: 
            pcs (:obj:`np.ndarray`|`cp.ndarray`): PC loadings.
        
        Returns:
            pcs_adjusted (:obj:`np.ndarray`|`cp.ndarray`): Flipped loadings.
    """
    # set the largest loading of each PC as positive
    _min, _max = pcs.min(axis=0), pcs.max(axis=0)
    pcs_adjusted = cp.sign(_min+_max) * pcs    
    return pcs_adjusted


def check_dtype(adata):
    """ Convert dtype to `float32` or `float64`.

        Note: 
            `rapids-singlecell` doesn't support sparse matrix under `float16`.
    """
    if isinstance(adata.X, scipy.sparse.csr_matrix):
        if not adata.X.dtype == np.float32 or not adata.X.dtype == np.float64:
            adata.X = adata.X.astype(np.float32) 
    else:
        raise Exception('data X is not a sparse matrix')
    

def gc(): 
    """Release CPU and GPU RAM"""
    collect()
    cp.get_default_memory_pool().free_all_blocks()


def _mean_var_major(X, major, minor):
    """ Mean and variance kernels for csr_matrix along the major axis.

        Note: 
            Not used for now.
    """
    # from _kernels._mean_var_kernel import _get_mean_var_major
    mean = cp.zeros(major, dtype=cp.float64)
    var = cp.zeros(major, dtype=cp.float64)
    block = (64,)
    grid = (major,)
    get_mean_var_major = kernels.get_mean_var_major(X.data.dtype)
    get_mean_var_major(
        grid, block, (X.indptr, X.indices, X.data, mean, var, major, minor)
    )
    mean = mean / minor
    var = var / minor
    var -= cp.power(mean, 2)
    var *= minor / (minor - 1)
    return mean, var


def _mean_var_minor(X, major, minor):
    """ Mean and variance kernels for csr_matrix along the minor axis.

        Note:  
            Modified so that it returns sum(x) and sq_sum(x) instead of mean and variance.
    """
    # from _kernels._mean_var_kernel import _get_mean_var_minor
    sums = cp.zeros(minor, dtype=cp.float64)
    sq_sums = cp.zeros(minor, dtype=cp.float64)
    block = (32,)
    grid = (int(math.ceil(X.nnz / block[0])),)
    get_mean_var_minor = kernels.get_mean_var_minor(X.data.dtype)
    get_mean_var_minor(grid, block, (X.indices, X.data, sums, sq_sums, major, X.nnz))
    # var = (var - mean**2) * (major / (major - 1)) # rapids use N-1 to scale
    return sums, sq_sums


def _mean_var_dense(X, axis):
    """ Mean and variance kernels for dense matrix. """
    # from _kernels._mean_var_kernel import mean_sum, sq_sum
    var = kernels.sq_sum(X, axis=axis)
    mean = kernels.mean_sum(X, axis=axis)
    mean = mean / X.shape[axis]
    var = var / X.shape[axis]
    var -= cp.power(mean, 2)
    var *= X.shape[axis] / (X.shape[axis] - 1)
    return mean, var


def get_mean_var(X, axis=0):
    """ Calculating mean and variance of a given matrix based on customized kernels.

        Note: 
            No such methods implemented yet for `csr_matrix`.
    """
    if issparse(X):
        if axis == 0:
            if isspmatrix_csr(X):
                major = X.shape[0]
                minor = X.shape[1]
                mean, var = _mean_var_minor(X, major, minor)
            elif isspmatrix_csc(X):
                major = X.shape[1]
                minor = X.shape[0]
                mean, var = _mean_var_major(X, major, minor)
        elif axis == 1:
            if isspmatrix_csr(X):
                major = X.shape[0]
                minor = X.shape[1]
                mean, var = _mean_var_major(X, major, minor)
            elif isspmatrix_csc(X):
                major = X.shape[1]
                minor = X.shape[0]
                mean, var = _mean_var_minor(X, major, minor)
    else:
        mean, var = _mean_var_dense(X, axis)
    return mean, var


def check_nonnegative_integers(X):
    """ Check if `X` is a nonnegative integer matrix.

        Note:  
            Check values of data to ensure it is count data.
    """
    if issparse(X):
        data = X.data
    else:
        data = X
    # Check no negatives
    if cp.signbit(data).any():
        return False
    elif cp.any(~cp.equal(cp.mod(data, 1), 0)):
        return False
    else:
        return True


def harmony(
    adata,
    key,
    *,
    basis = "X_pca",
    adjusted_basis = "X_pca_harmony",
    init_seeds = None,
    n_init = 1,
    dtype = cp.float32,
    max_iter_harmony = 10,
    random_state = 0,
    **kwargs):
    """ Harmony GPU version. """
    X = adata.obsm[basis].astype(float)
    res = harmonypy_gpu.run_harmony(X, adata.obs, key, init_seeds=init_seeds, n_init=n_init, max_iter_harmony=max_iter_harmony, dtype=dtype)
    adata.obsm[adjusted_basis] = res.result()
    return


def correct_leiden(adata):
    df_tmp = adata.obs.leiden.value_counts()
    old2new = {df_tmp.index[i]:str(i) for i in range(df_tmp.shape[0])}
    adata.obs['leiden'] = adata.obs['leiden'].astype(str).apply(lambda x: old2new[x])


def find_indices(A, indptr, out_rows):
    find_indices_kernel = kernels.get_find_indices()
    N = A.size
    M = indptr.size
    threads_per_block = 512
    blocks = (N + threads_per_block - 1) // threads_per_block
    find_indices_kernel((blocks,), (threads_per_block,), (A, indptr, out_rows, N, M))


def csr_indptr_to_coo_rows(nnz, Bp):
    out_rows = cp.empty(nnz, dtype=np.int32)
    find_indices(cp.arange(nnz), Bp, out_rows)
    return out_rows


def csr_row_index(Ax, Aj, Ap, rows):
    """Populate indices and data arrays from the given row index.

    Args:
        Ax (`cupy.ndarray`): data array from input sparse matrix
        Aj (`cupy.ndarray`): indices array from input sparse matrix
        Ap (`cupy.ndarray`): indptr array from input sparse matrix
        rows (`cupy.ndarray`): index array of rows to populate

    Returns:
        Bx (`cupy.ndarray`): data array of output sparse matrix
        Bj (`cupy.ndarray`): indices array of output sparse matrix
        Bp (`cupy.ndarray`): indptr array for output sparse matrix
    """
    row_nnz = cp.diff(Ap)
    Bp = cp.empty(rows.size + 1, dtype=np.int64)
    Bp[0] = 0
    cp.cumsum(row_nnz[rows], out=Bp[1:])
    nnz = int(Bp[-1])
    out_rows = csr_indptr_to_coo_rows(nnz, Bp)
    Bj, Bx = kernels.csr_row_index_kernel(out_rows, rows, Ap, Aj, Ax, Bp)
    return Bx, Bj, Bp, out_rows


def csr_col_index(Ax, Aj, Ai, cols, shape):
    col_ind  = cp.empty_like(Aj, dtype=cp.bool_)
    kernels.check_in_cols_kernel(Aj, cols.size, cols, col_ind)
    coo = scipy.sparse.coo_matrix((Ax[col_ind].get(), (Ai[col_ind].get(), Aj[col_ind].get())), shape=shape)
    return coo


def write_to_disk(adata, output_dir, data_name, batch_name=None):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if batch_name is None:
            out_name = f'{output_dir}/{data_name}.h5ad'
        else:
            out_name = f'{output_dir}/{data_name}.{batch_name}.h5ad'
        adata.write_h5ad(out_name)


# if __name__ == 'scalesc.util':
#     # example usage for 70k cells in human lung
#     reader = AnnDataBatchReader('/edgehpc/dept/compbio/projects/scaleSC/haotian/batch/data_dir/70k_human_lung', preload_on_cpu=True, preload_on_gpu=False)
#     # reader.batch_to_CPU()
#     x = reader.get_merged_adata_with_X()
#     print(x)

        
        





