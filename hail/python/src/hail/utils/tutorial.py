import hail as hl
from .java import FatalError, Env, info
from .misc import local_path_uri, new_local_temp_dir
import os
import zipfile
from urllib.request import urlretrieve

__all__ = [
    'get_1kg',
    'get_movie_lens'
]

resources = {
    '1kg_annotations': 'https://storage.googleapis.com/hail-tutorial/1kg_annotations.txt',
    '1kg_matrix_table': 'https://storage.googleapis.com/hail-tutorial/1kg.vcf.bgz',
    'ensembl_gene_annotations': 'https://storage.googleapis.com/hail-tutorial/ensembl_gene_annotations.txt',
    'movie_lens_100k': 'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
}

tmp_dir: str = None


def init_temp_dir():
    global tmp_dir
    if tmp_dir is None:
        tmp_dir = new_local_temp_dir()


def _mkdir(jhc, path):
    if not Env.jutils().dirExists(jhc, path):
        r = Env.jutils().mkdir(jhc, path)
        if not r:
            raise IOError(f'could not mkdir {path}')


def get_1kg(output_dir, overwrite: bool = False):
    """Download subset of the `1000 Genomes <http://www.internationalgenome.org/>`__
    dataset and sample annotations.

    Notes
    -----
    The download is about 15M.

    Parameters
    ----------
    output_dir
        Directory in which to write data.
    overwrite
        If ``True``, overwrite any existing files/directories at `output_dir`.
    """
    jhc = Env.hc()._jhc

    _mkdir(jhc, output_dir)

    matrix_table_path = os.path.join(output_dir, '1kg.mt')
    vcf_path = os.path.join(output_dir, '1kg.vcf.bgz')
    sample_annotations_path = os.path.join(output_dir, '1kg_annotations.txt')
    gene_annotations_path = os.path.join(output_dir, 'ensembl_gene_annotations.txt')

    if (overwrite
            or not Env.jutils().dirExists(jhc, matrix_table_path)
            or not Env.jutils().fileExists(jhc, sample_annotations_path)
            or not Env.jutils().fileExists(jhc, vcf_path)
            or not Env.jutils().fileExists(jhc, gene_annotations_path)):
        init_temp_dir()
        tmp_vcf = os.path.join(tmp_dir, '1kg.vcf.bgz')
        source = resources['1kg_matrix_table']
        info(f'downloading 1KG VCF ...\n'
             f'  Source: {source}')
        urlretrieve(resources['1kg_matrix_table'], tmp_vcf)
        cluster_readable_vcf = Env.jutils().copyToTmp(jhc, local_path_uri(tmp_vcf), 'vcf')
        info('importing VCF and writing to matrix table...')
        hl.import_vcf(cluster_readable_vcf, min_partitions=16).write(matrix_table_path, overwrite=True)

        tmp_sample_annot = os.path.join(tmp_dir, '1kg_annotations.txt')
        source = resources['1kg_annotations']
        info(f'downloading 1KG annotations ...\n'
             f'  Source: {source}')
        urlretrieve(source, tmp_sample_annot)

        tmp_gene_annot = os.path.join(tmp_dir, 'ensembl_gene_annotations.txt')
        source = resources['ensembl_gene_annotations']
        info(f'downloading Ensembl gene annotations ...\n'
             f'  Source: {source}')
        urlretrieve(source, tmp_gene_annot)

        hl.hadoop_copy(local_path_uri(tmp_sample_annot), sample_annotations_path)
        hl.hadoop_copy(local_path_uri(tmp_gene_annot), gene_annotations_path)
        hl.hadoop_copy(local_path_uri(tmp_vcf), vcf_path)
        info('Done!')
    else:
        info('1KG files found')


def get_movie_lens(output_dir, overwrite: bool = False):
    """Download public Movie Lens dataset.

    Notes
    -----
    The download is about 6M.

    See the
    `MovieLens website <https://grouplens.org/datasets/movielens/100k/>`__
    for more information about this dataset.

    Parameters
    ----------
    output_dir
        Directory in which to write data.
    overwrite
        If ``True``, overwrite existing files/directories at those locations.
    """

    jhc = Env.hc()._jhc

    _mkdir(jhc, output_dir)

    paths = [os.path.join(output_dir, x) for x in ['movies.ht', 'ratings.ht', 'users.ht']]
    if overwrite or any(not Env.jutils().dirExists(jhc, f) for f in paths):
        init_temp_dir()
        source = resources['movie_lens_100k']
        tmp_path = os.path.join(tmp_dir, 'ml-100k.zip')
        info(f'downloading MovieLens-100k data ...\n'
             f'  Source: {source}')
        urlretrieve(source, tmp_path)
        with zipfile.ZipFile(tmp_path, 'r') as z:
            z.extractall(tmp_dir)

        user_table_path = os.path.join(os.path.join(tmp_dir, 'ml-100k', 'u.user'))
        movie_table_path = os.path.join(os.path.join(tmp_dir, 'ml-100k', 'u.item'))
        ratings_table_path = os.path.join(os.path.join(tmp_dir, 'ml-100k', 'u.data'))
        assert (os.path.exists(user_table_path))
        assert (os.path.exists(movie_table_path))
        assert (os.path.exists(ratings_table_path))

        user_cluster_readable = Env.jutils().copyToTmp(jhc, local_path_uri(user_table_path), 'txt')
        movie_cluster_readable = Env.jutils().copyToTmp(jhc, local_path_uri(movie_table_path), 'txt')
        ratings_cluster_readable = Env.jutils().copyToTmp(jhc, local_path_uri(ratings_table_path), 'txt')

        [movies_path, ratings_path, users_path] = paths

        genres = ['Action', 'Adventure', 'Animation',
                  "Children's", 'Comedy', 'Crime',
                  'Documentary', 'Drama', 'Fantasy',
                  'Film-Noir', 'Horror', 'Musical',
                  'Mystery', 'Romance', 'Sci-Fi',
                  'Thriller', 'War', 'Western']

        # utility functions for importing movies
        def field_to_array(ds, field):
            return hl.cond(ds[field] != 0, hl.array([field]), hl.empty_array(hl.tstr))

        def fields_to_array(ds, fields):
            return hl.flatten(hl.array([field_to_array(ds, f) for f in fields]))

        def rename_columns(ht, new_names):
            return ht.rename({k: v for k, v in zip(ht.row, new_names)})

        info(f'importing users table and writing to {users_path} ...')

        users = rename_columns(
            hl.import_table(user_cluster_readable, key=['f0'], no_header=True, impute=True, delimiter='|'),
            ['id', 'age', 'sex', 'occupation', 'zipcode'])
        users.write(users_path, overwrite=True)

        info(f'importing movies table and writing to {movies_path} ...')

        movies = hl.import_table(movie_cluster_readable, key=['f0'], no_header=True, impute=True, delimiter='|')
        movies = rename_columns(movies,
                                ['id', 'title', 'release date', 'video release date', 'IMDb URL', 'unknown'] + genres)
        movies = movies.drop('release date', 'video release date', 'unknown', 'IMDb URL')
        movies = movies.transmute(genres=fields_to_array(movies, genres))
        movies.write(movies_path, overwrite=True)

        info(f'importing ratings table and writing to {ratings_path} ...')

        ratings = hl.import_table(ratings_cluster_readable, no_header=True, impute=True)
        ratings = rename_columns(ratings,
                                 ['user_id', 'movie_id', 'rating', 'timestamp'])
        ratings = ratings.drop('timestamp')
        ratings.write(ratings_path, overwrite=True)

    else:
        info('Movie Lens files found!')
