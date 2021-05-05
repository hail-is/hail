import hailtop.batch as hb
import hail as hl
import pandas as pd
from typing import Tuple
from sklearn.ensemble import RandomForestRegressor


def random_forest(df_x_path: str, df_y_path: str, window_name: str, cores: int = 1) -> Tuple[str, float, float]:
    # read in data
    df_x = pd.read_table(df_x_path, header=0, index_col=0)
    df_y = pd.read_table(df_y_path, header=0, index_col=0)

    # split training and testing data for the current window
    x_train = df_x[df_x.index != window_name]
    x_test = df_x[df_x.index == window_name]

    y_train = df_y[df_y.index != window_name]
    y_test = df_y[df_y.index == window_name]

    # run random forest
    max_features = 3 / 4
    rf = RandomForestRegressor(n_estimators=100,
                               n_jobs=cores,
                               max_features=max_features,
                               oob_score=True,
                               verbose=False)

    rf.fit(x_train, y_train)

    # apply the trained random forest on testing data
    y_pred = rf.predict(x_test)

    # store obs and pred values for this window
    obs = y_test["oe"].to_list()[0]
    pred = y_pred[0]

    return (window_name, obs, pred)


def as_tsv(input: Tuple[str, float, float]) -> str:
    return '\t'.join(str(i) for i in input)


def checkpoint_path(window):
    return f'gs://my_bucket/checkpoints/random-forest/{window}'


def main(df_x_path, df_y_path, output_path, python_image):
    backend = hb.ServiceBackend()
    b = hb.Batch(name='rf-loo', default_python_image=python_image)

    with hl.hadoop_open(df_y_path) as f:
        local_df_y = pd.read_table(f, header=0, index_col=0)

    df_x_input = b.read_input(df_x_path)
    df_y_input = b.read_input(df_y_path)

    results = []

    for window in local_df_y.index.to_list():
        checkpoint = checkpoint_path(window)
        if hl.hadoop_exists(checkpoint):
            result = b.read_input(checkpoint)
            results.append(result)
            continue

        j = b.new_python_job()

        result = j.call(random_forest, df_x_input, df_y_input, window)
        tsv_result = j.call(as_tsv, result)
        tsv_result = tsv_result.as_str()

        b.write_output(tsv_result, checkpoint)
        results.append(tsv_result)

    output = hb.concatenate(b, results)
    b.write_output(output, output_path)

    b.run(wait=False)
    backend.close()
