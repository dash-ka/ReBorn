
from time import time
import re, itertools, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sqlalchemy import text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score


def get_memory_usage(model, items_table=None):
    if items_table is None:
        items_table = "items_table"

    explain_query = "EXPLAIN ANALYZE " + model.query.format(items_table = items_table)
    with model.engine.connect() as con:
        stmt = con.execute(text(explain_query)).fetchall()
          
    merged_stmt = "".join(itertools.chain.from_iterable(stmt))
    mem_usages = re.findall(r"(Memory( Usage)?: (\d+)\w+)", merged_stmt)
    memory = sum([int(usage[-1]) for usage in mem_usages])
    return memory


def db2bow(db_table):

    """ Convert a relational table to a triple format `item-subject-feature` """

    table = db_table.\
        groupby(["item", "subject"])["feature"].\
        apply(lambda x: "|".join(x)).\
        reset_index()

    return table


def bow(x, names):
    bow = []
    for i in range(x.shape[0]):
        item = x[i].tocoo()
        data = item.data
        keys = [names[c] for c in item.col]
        bow.append(dict(zip(keys, data)))
    return bow


Q_prepare_tables = """ 

    SELECT subject, item, keyword, authid, pubname INTO test_table
    FROM (
        WITH ids AS (
            SELECT id as pubid
            FROM publications.publication
            WHERE pubname is not null
            INTERSECT
            SELECT pubid
            FROM publications.keyword
            INTERSECT
            SELECT pubid FROM publications.pub_author
        ),
        sample as (
            SELECT pubid, subject_area, pubname, row_number() OVER (partition by subject_area order by random()) AS row
            FROM ids JOIN publications.publication on ids.pubid = id
            )
        SELECT pubid AS item, subject_area AS subject, keyword, authid, pubname
        FROM sample
            JOIN publications.keyword AS k USING (pubid)
            JOIN publications.pub_author AS a USING (pubid)
        WHERE row <= 200
    ) F;

    SELECT COUNT(distinct item) 
    FROM test_table;

"""

Q_create_featurizer = """ 
    
        CREATE OR REPLACE FUNCTION featurize(table_name text, columns VARIADIC text[])
        RETURNS TABLE(subject varchar, item int, feature text) as
        $$
            BEGIN
            return query EXECUTE
            'SELECT DISTINCT t.subject, t.item, attributes.feature FROM ('||
                'SELECT *, row_to_json('|| table_name ||'.*) as json_data ' ||
                'FROM '|| table_name||') as t, ' ||
            'LATERAL ('||
                        'SELECT concat_ws('':'', $1[i], json_extract_path(t.json_data, $1[i])) as feature '||
                        'FROM generate_subscripts($1, 1) as indices(i)) as attributes;' USING columns;
            END
        $$ language plpgsql;
   """

Q_sample_from_db = """ 

        DROP TABLE IF EXISTS {output_table} CASCADE;

        CREATE TABLE {output_table} as (
            with sample as (
                SELECT item, row_number() OVER (order by random()) AS row
                FROM train_table
                GROUP BY item),
            ids as (
                SELECT item
                FROM sample
                WHERE row <= {sample_size}
                )
            SELECT * 
            FROM train_table JOIN ids using(item)
        );

        """
Q_compute_counts = """

        CREATE OR REPLACE VIEW counts_view as (
           WITH data as (
                SELECT DISTINCT te.subject, te.item, feature::varchar
                FROM featurize('{test_table}', {mutate_cols}) AS te
                JOIN featurize('{train_table}', {mutate_cols}) AS tr
                USING (feature)
                ),
            term_freq AS (
                SELECT subject, item, feature,
                COUNT(*) OVER (PARTITION BY (item, feature)) AS weight
                FROM data ORDER BY item, feature
                )
            SELECT *
            FROM term_freq
        );

        SELECT * FROM counts_view;

"""

Q_compute_tfidf = """

        CREATE OR REPLACE VIEW idf_view AS (
            WITH train_data as (
                SELECT DISTINCT item, feature::varchar 
                FROM featurize('{train_table}', {mutate_cols})
                ),
            n_docs as (
                SELECT count(distinct item) AS n
                FROM train_data
                ),
            doc_freq AS (
                SELECT item, feature, n_docs.n AS num_samples,
                COUNT(item) OVER (PARTITION BY feature) AS doc_frequency
                FROM train_data, n_docs
                ),
            idf AS (
                SELECT DISTINCT feature, ( ln((1+num_samples) / (1+doc_frequency)) + 1) AS idf
                FROM doc_freq
                ),
            test_data AS (
                SELECT subject, item, feature::VARCHAR 
                FROM featurize('{test_table}', {mutate_cols})
                ),
            normalized_score as (
                    SELECT subject, item, feature, idf, SUM(idf * idf) OVER (PARTITION BY item) AS l2
                    FROM test_data JOIN idf USING(feature)
                    )
            SELECT subject, item, feature, idf/sqrt(l2) AS weight
            FROM normalized_score
            );

        SELECT * FROM idf_view;
"""

def featurize_table(engine, table_name, mutate_cols):

    mutate_cols = ", ".join("'"+col+"'" for col in mutate_cols)

    with engine.connect() as con:
        return pd.read_sql_query(f""" 
                                 SELECT subject, item, feature
                                 FROM featurize('{table_name}', {mutate_cols})
                                 """, con
                                 )

def compute_weights(engine, train_table, test_table, mutate_cols, use_idf=False):

    mutate_cols = ", ".join("'" + col + "'" for col in mutate_cols)

    with engine.connect() as con:
        with con.begin():
            if use_idf:
                return con.execute(
                    text(Q_compute_tfidf.format(
                                    mutate_cols=mutate_cols,
                                    train_table= train_table,
                                    test_table=test_table
                                    )
                    )
                ).fetchall()

            else:
                return con.execute(
                    text(Q_compute_counts.format(
                                    mutate_cols=mutate_cols,
                                    train_table= train_table,
                                    test_table=test_table
                                    )
                    )
                ).fetchall()

    

def get_train_test_sample(engine, train_size=100, use_idf=False):

    mutate_cols = ["keyword", "pubname", "authid"]

    with engine.connect() as con:
        with con.begin():

            # create or replace `featurize` procedure
            con.execute(
                text(
                    Q_create_featurizer
                    )
                )
            
            # sample a table of a given size for training 
            con.execute(
                text(
                    Q_sample_from_db.format(
                        output_table="sample_table",
                        sample_size=train_size
                        )
                    )
                )
            
    tr = featurize_table(engine, 'sample_table', mutate_cols)
    te = featurize_table(engine, 'test_table', mutate_cols)
    weights = compute_weights(engine, "sample_table", "test_table", mutate_cols, use_idf=use_idf)


    # convert each db item to a bag-of-words
    train = db2bow(tr)
    test = db2bow(te)

    # estimate tf-idf scores and build feature vectors for training and testing
    if use_idf:
        vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split("|"))
    else:
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split("|"))


    vectorizer.fit(train["feature"])
    feature_names = vectorizer.get_feature_names_out()

    x_tr = vectorizer.transform(train["feature"])
    y_tr = train["subject"].values.astype(np.int32)

    x_te = vectorizer.transform(test["feature"])
    rows_to_keep = np.ravel(x_te.sum(axis=1) != 0)
    x_te = x_te[rows_to_keep, :]
    y_te = test["subject"].iloc[rows_to_keep].values.astype(np.int32)

    return x_tr, y_tr, x_te, y_te, feature_names


def timing_cpu(engine, models, use_idf=False, runs=1):

        times = []
        file =  "./scopus_timing_idf.csv"

        for run in range(runs):
            for train_size in (np.arange(10)+1)/10:

                n_rows = int(7000 * train_size)
                x_tr, y_tr, x_te, y_te, feature_names = get_train_test_sample(engine, n_rows, use_idf)
                b_te = bow(x_te, feature_names)
                b_tr = bow(x_tr, feature_names)
                table_name = "idf_view" if use_idf else "counts_view"

                for name, (model_sql, model_skl) in models.items():
                    print(f"Run {run + 1}/{runs}: executing {name} with train_size={train_size}")

                    memory_usage = 0

                    if name == "BC":

                        fit_start = time()
                        model_sql.fit(b_tr, y_tr)
                        fit_end = time()

                        predict_start = time()
                        y_pred = model_sql.predict(table_name)
                        predict_end = time()

                        # numpy implementation
                        model_skl.fit(x_tr, y_tr)
                        sk_pred = model_skl.predict(x_te)

                    else:
                        
                        fit_start = time()
                        model_sql.fit(x_tr, y_tr, feature_names)
                        fit_end = time()

                        predict_start = time()
                        y_pred = model_sql.predict(table_name)
                        predict_end = time()

                        # sklearn implementation
                        sk_pred = model_sql.model.predict(x_te)

                        memory_usage = get_memory_usage(model_sql, items_table=table_name)

                    times.append({
                        'run': run+1,
                        'model': name,
                        'train_size': train_size,
                        'fit_time': fit_end - fit_start,
                        'predict_time': predict_end - predict_start,
                        'memory_usage': memory_usage,
                        'score_sklearn': accuracy_score(y_te, sk_pred),
                        'score_sql': accuracy_score(y_te, y_pred)
                    })
                    print("writing to file", file)
                    pd.DataFrame(times).to_csv(file, index=False)

        return times


def plot_timing(  score_label='Score'):
        timing = []
        file = f"./scopus_timing_idf.csv"
        if os.path.exists(file):
            timing.append(pd.read_csv(file))

        df = pd.concat(timing).groupby(['model', 'train_size']).describe().reset_index()
        df.columns = [' '.join(col).strip() for col in df.columns]
        for column in ['fit_time', 'predict_time', 'score_sql', "score_sklearn"]:
            df[f"{column} err"] = df[f"{column} std"] / np.sqrt(df["run count"])

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3.3))
        plt.tight_layout(pad=3, rect=(0, 0, 1, 0.95))

        for key, group in df.groupby('model'):
            args = {'label': key, 'legend': False, 'marker': ".", 'capsize': 2, 'linewidth': 1, 'elinewidth': 0.5}
            if key.startswith("BC"):
                args['color'] = 'black'
            if key == "BC (GPU)":
                args['linestyle'] = 'dotted'
                args['marker'] = 'x'
            group.plot(x='train_size', y='fit_time mean', yerr='fit_time err', ax=ax1, **args)
            group.plot(x='train_size', y='predict_time mean', yerr='predict_time err', ax=ax2, **args)
            group.plot(x='train_size', y='score_sql mean', yerr='score_sql err', ax=ax3, **args)
            #group.plot(x='train_size', y='score_sklearn mean', yerr='score_sklearn err', ax=ax3, **args)

        for ax, label in [(ax1, "Training Time (s)"), (ax2, "Prediction Time (s)"), (ax3, score_label)]:
            ax.set_xlabel('Dataset Size\n(fraction of training data)', fontsize=14)
            ax.set_ylabel(label, fontsize=14)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            plt.setp(ax.spines.values(), linewidth=0.5)
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontweight("bold")

        for ax in [ax1, ax2]:
            ax.set_yscale('log')

        handles, labels = ax3.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=len(df.model.unique()), prop={"size": 12})

        file = f"./scopus_timing.png"
        fig.savefig(file, bbox_inches='tight', format='png', dpi=300)
        print(f"Image saved in {file}")