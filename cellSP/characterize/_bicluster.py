from sklearn.preprocessing import scale
from scipy.stats import norm
from operator import itemgetter
from sklearn.utils.validation import check_array
import multiprocessing as mp
import bottleneck as bn
import numpy as np
import math
import random


class Bicluster:
    """This class models a bicluster. Adapter from https://github.com/padilha/biclustlib/blob/master/biclustlib/algorithms/las.py

    Parameters
    ----------
    rows : numpy.array
        Rows of the bicluster (assumes that row indexing starts at 0).

    cols : numpy.array
        Columns of the bicluster (assumes that column indexing starts at 0).
    score : float
        Score of the bicluster
    data : numpy.ndarray
        bla
    """

    def __init__(self, rows, cols, score, data=None):
        if isinstance(rows, np.ndarray) and rows.dtype == bool and cols.dtype == bool:
            self.rows = np.nonzero(rows)[0]
            self.cols = np.nonzero(cols)[0]
        elif isinstance(cols, np.ndarray) and rows.dtype == int and cols.dtype == int:
            self.rows = rows
            self.cols = cols
        else:
            raise ValueError("rows and cols must be bool or int numpy.arrays")
        self.score = score
        if data is not None:
            n, m = len(self.rows), len(self.cols)

            if isinstance(data, np.ndarray) and (data.shape == (n, m) or (len(data) == 0 and n == 0)):
                self.data = data
            else:
                raise ValueError("")

    def intersection(self, other):
        """Returns a bicluster that represents the area of overlap between two biclusters."""
        rows_intersec = np.intersect1d(self.rows, other.rows)
        cols_intersec = np.intersect1d(self.cols, other.cols)
        return Bicluster(rows_intersec, cols_intersec)

    def union(self, other):
        rows_union = np.union1d(self.rows, other.rows)
        cols_union = np.union1d(self.cols, other.cols)
        return Bicluster(rows_union, cols_union)

    def overlap(self, other):
        min_area = min(self.area, other.area)
        return self.intersection(other).area / min_area

    @property
    def area(self):
        """Calculates the number of matrix elements of the bicluster."""
        return len(self.rows) * len(self.cols)

    def sort(self):
        """Sorts the array of row and the array of column indices of the bicluster."""
        self.rows.sort()
        self.cols.sort()

    def __str__(self):
        return 'Bicluster(rows={0}, cols={1})'.format(self.rows, self.cols)


class Biclustering:
    """This class models a biclustering. Adaped from https://github.com/padilha/biclustlib/blob/master/biclustlib/algorithms/las.py.

    Parameters
    ----------
    biclusters : list
        A list of instances from the Bicluster class.
    """

    def __init__(self, biclusters):
        if all(isinstance(b, Bicluster) for b in biclusters):
            self.biclusters = biclusters
        else:
            raise ValueError("biclusters list contains an element that is not a Bicluster instance")

    def __str__(self):
        return '\n'.join(str(b) for b in self.biclusters)
    
class LargeAverageSubmatrices():
    """Large Average Submatrices (LAS)

    LAS searches for submatrices by optimizing a significance score that trades off between
    submatrix size and average value. Added parallelization.
    Adatped from https://github.com/padilha/biclustlib/blob/master/biclustlib/algorithms/las.py.

    Reference
    ----------
    Shabalin, A. A., Weigman, V. J., Perou, C. M., and Nobel, A. B. (2009). Finding large
    average submatrices in high dimensional data. The Annals of Applied Statistics, 3(3):
    985-1012.

    Parameters
    ----------
    num_biclusters : int, default: 10
        Number of biclusters to be found.

    score_threshold : float, default: 1.0
        Significance score threshold.

    randomized_searches : int, default: 1000
        Number of executions of the search procedure for each bicluster.

    transform : bool, default: False
        If True, applies the transformation f(x) = sign(x) * log(1 + |x|) to each entry of the
        input dataset before performing the algorithm (recommended by the original authors for
        datasets that exhibit heavy tails).

    tol : float, default: 1e-6
        The maximum absolute difference among the scores of two consecutive iterations to declare convergence.
    """

    def __init__(self, num_biclusters=10, score_threshold=1.0, randomized_searches=1000, scale_data=True, transform=False, tol=1e-6, threads=1):
        self.num_biclusters = num_biclusters
        self.score_threshold = score_threshold
        self.randomized_searches = randomized_searches
        self.scale_data = scale_data
        self.transform = transform
        self.tol = tol
        self.threads = threads

    def run(self, data):
        """Compute biclustering.

        Parameters
        ----------
        data : numpy.ndarray
        """
        data = check_array(data, dtype=np.double, copy=True)
        self._validate_parameters()

        if self.scale_data:
            data = scale(data)

        if self.transform:
            data = np.sign(data) * np.log(1 + np.abs(data))

            if self.scale_data:
                data = scale(data)

        biclusters = []
        data_matrix = data.copy(order='C')
        shared_data = mp.RawArray('d', data_matrix.shape[0] * data_matrix.shape[1])
        shared_data_np = np.frombuffer(shared_data).reshape(data_matrix.shape)
        np.copyto(shared_data_np, data_matrix)
        del data
        for i in range(self.num_biclusters):
            with mp.Pool(processes=self.threads, initializer=self._initializer_func, initargs=(data_matrix, data_matrix.shape)) as pool:
                best, avg, score = max((pool.starmap(self._find_bicluster, [() for _ in range(self.randomized_searches)])), key=itemgetter(-1))
            if score < self.score_threshold:
                break
            shared_data_np[np.ix_(best.rows, best.cols)] -= avg
            best.score = score
            biclusters.append(best)
            data_matrix = shared_data_np.copy(order='C')
            shared_data = mp.RawArray('d', data_matrix.shape[0] * data_matrix.shape[1])
            shared_data_np = np.frombuffer(shared_data).reshape(data_matrix.shape)
            np.copyto(shared_data_np, data_matrix)
        return Biclustering(biclusters)

    def _initializer_func(self, X, X_shape):
        global matrix, matrix_shape
        matrix = X
        matrix_shape = X_shape

    def _find_bicluster(self):
        """The basic bicluster search procedure. Each run of this method returns a submatrix
        that is a local maximum of the score function adopted.
        """
        b = self._find_constrained_bicluster()
        return self._improve_bicluster(b)

    def _find_constrained_bicluster(self):
        """Find a k x l bicluster."""
        data = np.frombuffer(matrix).reshape(matrix_shape)
        num_rows, num_cols = data.shape

        k = random.randint(1, math.ceil(num_rows / 2))
        l = random.randint(1, math.ceil(num_cols / 2))

        cols = np.random.choice(num_cols, size=l, replace=False)

        old_avg, avg = float('-inf'), 0.0
        while abs(avg - old_avg) > self.tol:
            old_avg = avg

            row_sums = np.sum(data[:, cols], axis=1)
            rows = bn.argpartition(row_sums, num_rows - k)[-k:] # this is usually faster than rows = np.argsort(row_sums)[-k:]

            col_sums = np.sum(data[rows, :], axis=0)
            cols = bn.argpartition(col_sums, num_cols - l)[-l:] # this is usually faster than cols = np.argsort(col_sums)[-l:]

            avg = np.mean(data[np.ix_(rows, cols)])
        return Bicluster(rows, cols, avg)

    def _improve_bicluster(self, b):
        """Relaxes the k x l bicluster constraint in order to maximize the score function locally."""
        data = np.frombuffer(matrix).reshape(matrix_shape)
        num_rows, num_cols = data.shape

        row_range = np.arange(1, num_rows + 1)
        col_range = np.arange(1, num_cols + 1)

        row_log_combs = self._log_combs(num_rows)[1:] # self._log_combs(num_rows)[1:] discards the case where the bicluster has 0 rows
        col_log_combs = self._log_combs(num_cols)[1:] # self._log_combs(num_cols)[1:] discards the case where the bicluster has 0 columns

        old_score, score = float('-inf'), 0.0

        while abs(score - old_score) > self.tol:
            old_score = score

            row_sums = np.sum(data[:, b.cols], axis=1)
            order = np.argsort(-row_sums)
            row_cumsum = np.cumsum(row_sums[order])
            row_scores = self._scores(row_range, len(b.cols), row_cumsum, row_log_combs, col_log_combs)
            rmax = np.argmax(row_scores) # searches for the number of rows that maximizes the score
            b.rows = order[:rmax+1]

            col_sums = np.sum(data[b.rows, :], axis=0)
            order = np.argsort(-col_sums)
            col_cumsum = np.cumsum(col_sums[order])
            col_scores = self._scores(col_range, len(b.rows), col_cumsum, col_log_combs, row_log_combs)
            cmax = np.argmax(col_scores) # searches for the number of columns that maximizes the score
            b.cols = order[:cmax+1]

            avg = col_cumsum[cmax] / b.area
            score = col_scores[cmax]

        return b, avg, score

    def _scores(self, range_, k, cumsum, m_log_combs, n_log_combs):
        """Calculates the score function for all possible numbers of rows (or columns)."""
        avgs = cumsum / (range_ * k)
        log_probs = norm.logcdf(-avgs * np.sqrt(range_ * k))
        return - log_probs - m_log_combs - n_log_combs[k-1]

    def _log_combs(self, n):
        """Calculates the log of n choose k for k ranging from 0 to n."""
        log_facts = self._cum_log_factorial(n)
        return log_facts[n] - (log_facts + log_facts[::-1])

    def _cum_log_factorial(self, n):
        """Calculates the log of the factorials from 0 to n."""
        log_cumsum = np.cumsum(np.log(np.arange(1, n + 1)))
        return np.append(0, log_cumsum) # 0 because log 0! = log 1 = 0, so this array will have size n + 1

    def _validate_parameters(self):
        if self.num_biclusters <= 0:
            raise ValueError("num_biclusters must be > 0, got {}".format(num_biclusters))

        if self.randomized_searches <= 0:
            raise ValueError("randomized_searches must be > 0, got {}".format(self.randomized_searches))

        if not isinstance(self.transform, bool):
            raise ValueError("transform must be either True or False, got {}".format(self.transform))

        if self.tol <= 0.0:
            raise ValueError("tol must be a small double > 0.0, got {}".format(self.tol))


def _modified_jaccard_similarity(list1, list2):
    #overlap coefficient
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    len_smaller = min(len(set1), len(set2))
    return intersection / len_smaller

def _cum_log_factorial(n):
    """Calculates the log of the factorials from 0 to n."""
    log_cumsum = np.cumsum(np.log(np.arange(1, n + 1)))
    return np.append(0, log_cumsum) # 0 because log 0! = log 1 = 0, so this array will have size n + 1

def _log_combs(n):
    """Calculates the log of n choose k for k ranging from 0 to n."""
    log_facts = _cum_log_factorial(n)
    return log_facts[n] - (log_facts + log_facts[::-1])

def _scores(range_, k, cumsum, m_log_combs, n_log_combs):
        """Calculates the score function for all possible numbers of rows (or columns)."""
        avgs = cumsum / (range_ * k)
        log_probs = norm.logcdf(-avgs * np.sqrt(range_ * k))
        return - log_probs - m_log_combs - n_log_combs[k-1]

def _expand_bicluster_rows(scores, rows, columns):
    '''
        Expands bicluster cells by adding cells based on their score
    '''
    expanded_rows = list(rows)
    for n, i in enumerate(scores):
        if n not in rows:
            if np.mean(i[columns]) > np.mean(scores[rows][:, columns]):
                expanded_rows.append(n)
    return expanded_rows

def _get_sprawl_score(p1, scores, col_range, col_log_combs, row_log_combs):
    col_sums = np.sum(scores[[int(x) for x in list(p1)]], axis=0)
    order = np.argsort(-col_sums)
    col_cumsum = np.cumsum(col_sums[order])
    col_scores = _scores(col_range, len(p1), col_cumsum, col_log_combs, row_log_combs)
    cmax = np.argmax(col_scores) # searches for the number of columns that maximizes the score
    score = col_scores[cmax]
    return score

def _string_to_tuples(tuple_string):
    values = tuple_string.split(', ')
    return [f"({pair.split(',')[0].strip('()')},{pair.split(',')[1].strip('()')})" for pair in values]

def pair_to_geneset(gene_pairs):
    genepairs = set(_string_to_tuples(gene_pairs))
    genes = []
    for x in genepairs:
        genes.append(x.split(",")[0].strip("()"))
        genes.append(x.split(",")[1].strip("()"))
    genes = set(genes)
    return genepairs, genes

def _expand_bicluster(df, scores, scores_scaled, uids, score_issues, col_range, col_log_combs, row_log_combs, mode = "sprawl", gene_pairs = None, df_sprawl = None):
    '''
        Expands bicluster by combining two biclusters if their gene set have a jaccard similarity of 2/3 or more.
    '''
    combined = False
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            if mode == "sprawl":
                genes1 = set(df.at[i, 'genes'].split(','))
                genes2 = set(df.at[j, 'genes'].split(','))
            else:
                genepairs1, genes1 = pair_to_geneset(df.at[i, 'gene-pairs'])
                genepairs2, genes2 = pair_to_geneset(df.at[j, 'gene-pairs'])
                genepair2_index = [gene_pairs.index(x) for x in genepairs2]
                genepair1_index = [gene_pairs.index(x) for x in genepairs1]
            jaccard = _modified_jaccard_similarity(genes1, genes2)
            confirm_method = True
            if mode == "sprawl":
                confirm_method = df.at[i, 'method'] == df.at[j, 'method']
                df_scores = df_sprawl[df.at[i, 'method']]
                df_sp_scaled = df_scores.copy()
                df_sp_scaled[:] = scale(df_sp_scaled)
                scores_scaled = df_sp_scaled.values
            if jaccard >= 2/3 and confirm_method and (i,j) not in score_issues and (j,i) not in score_issues:
                positions1 = [uids.index(x) for x in df.at[i, 'uIDs'].split(',')]
                positions2 = [uids.index(x) for x in df.at[j, 'uIDs'].split(',')]
                cells1 = set(df.at[i, 'uIDs'].split(','))
                cells2 = set(df.at[j, 'uIDs'].split(','))
                positions_union = [uids.index(x) for x in list(cells1.union(cells2))]
                if df.at[j, 'combined'] > 0:
                    pre_score = _get_sprawl_score(positions2, scores_scaled, col_range, col_log_combs, row_log_combs)
                else:
                    pre_score = _get_sprawl_score(positions1, scores_scaled, col_range, col_log_combs, row_log_combs)
                post_score = _get_sprawl_score(positions_union, scores_scaled, col_range, col_log_combs, row_log_combs)
                if post_score < 1:
                    score_issues.append((i,j))
                    score_issues.append((j,i))
                    break
                if mode == "instant":
                    df.at[i, 'gene-pairs'] = ','.join(list(genepairs1.union(genepairs2)))
                df.at[i, 'genes'] = ','.join(list(genes1.union(genes2)))
                df.at[i, 'uIDs'] = ','.join(list(cells1.union(cells2)))
                df.at[i, '#cells'] = len(list(cells1.union(cells2)))
                df.at[i, 'combined'] = 1 + df.at[i, 'combined'] + df.at[j, 'combined']
                combined = True
                if mode == "sprawl":
                    df.at[i, f'{mode} average'] = f"{df.at[i, f'{mode} average'].split(':')[0]},{np.mean(df_scores.iloc[[int(x) for x in list(positions2)]][list(genes2)])}:{np.mean(df_scores.iloc[positions_union][list(genes1.union(genes2))])}"
                else:
                    df.at[i, f'{mode} average'] = f"{df.at[i, f'{mode} average'].split(':')[0]},{np.mean(scores[positions2][:, genepair2_index])}:{np.mean(scores[positions_union][:, list(set(genepair1_index).union(set(genepair2_index)))])}"
                df.at[i, f'{mode} score'] = f"{df.at[i, f'{mode} score'].split(':')[0]},{pre_score}:{post_score}"
                df = df.drop(j).reset_index(drop=True)
                break
        if combined:
            break
    return df, score_issues
