from time import time
import re, itertools, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sqlalchemy import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

def db2bow(db_table):

    """ Convert a relational table to a triple format `item-subject-feature` """

    table = db_table.\
        groupby(["item", "subject"])["feature"].\
        apply(lambda x: "$".join(x)).\
        reset_index()

    return table

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

def model_accuracy(predictions, test, rows_to_keep):
    merged = predictions.merge(test.iloc[rows_to_keep], on="item") 
    return sum(merged["class"].astype(np.int32)==merged["subject"].astype(np.int32)) / len(merged)

def accuracy_score_born(engine, predictions):
    with engine.connect() as con:
        y_true = pd.read_sql_query("select distinct item, subject from test_table", con)
    y_pred = pd.DataFrame(predictions).reset_index()
    y_pred.columns =["item", "prediction"]
    tbl = y_true.merge(y_pred, on="item")
    return sum(tbl["subject"].astype(np.int64) == tbl["prediction"]) / len(tbl)



Q_sample_rows_from_db = """ 

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
            SELECT subject::integer , item, keyword, pubname, authid
            FROM train_table JOIN ids using(item)
        );

        """

Q_featurize_table = """ 
    
        CREATE OR REPLACE FUNCTION featurize_table(table_name text, columns VARIADIC text[])
        RETURNS TABLE(subject varchar, item int, feature text) as
        $$
            BEGIN
            return query EXECUTE
            'SELECT DISTINCT t.subject, t.item, replace(attributes.feature, ''"'', '''') FROM ('||
                'SELECT *, row_to_json('|| table_name ||'.*) as json_data ' ||
                'FROM '|| table_name||') as t, ' ||
            'LATERAL ('||
                        'SELECT concat_ws(''|'', $1[i], json_extract_path(t.json_data, $1[i])) as feature '||
                        'FROM generate_subscripts($1, 1) as indices(i)) as attributes;' USING columns;
            END
        $$ language plpgsql;
   """

def featurize_table(engine, table_name, mutate_cols):

    """ Convert a single table into the following format (subject, item, feature).
        This function is used to formated data from db for the out-of-db training.

        Args:
                train_table::str = name of denormalized table with training data
                mutate_cols::List[str] = a list of column names to be used as features

        """

    mutate_cols = ", ".join("'"+col+"'" for col in mutate_cols)


    with engine.connect() as con:
        with con.begin():
             con.execute(text(Q_featurize_table))

        return pd.read_sql_query(f""" 
                                 SELECT distinct subject, item, feature
                                 FROM featurize_table('{table_name}', {mutate_cols})
                                 """, con
                                 )
    


def get_train_test_sample(engine, train_size=100):

    mutate_cols = ["keyword", "pubname", "authid"]

    with engine.connect() as con:
        with con.begin():

            # sample a table of training items from denormalized scopus db 
            con.execute(
                text(
                    Q_sample_rows_from_db.format(
                        output_table="sample_table",
                        sample_size=train_size
                        )
                    )
                )
            
    # read featurized train/test tables from db for out-of-db training of sklearn models

    tr = featurize_table(engine, 'sample_table', mutate_cols)
    te = featurize_table(engine, 'test_table', mutate_cols)

    # convert each table item to a bag-of-words 

    train = db2bow(tr)
    test = db2bow(te)

    # count feature occurences to compute feature weights

    vectorizer = CountVectorizer(tokenizer=lambda x: x.split("$"))
    vectorizer.fit(train["feature"])
    feature_names = vectorizer.get_feature_names_out()

    x_tr = vectorizer.transform(train["feature"])
    y_tr = train["subject"].values.astype(np.int32)

    x_te = vectorizer.transform(test["feature"])
    rows_to_keep = np.ravel(x_te.sum(axis=1) != 0)
    x_te = x_te[rows_to_keep, :]
    y_te = test["subject"].iloc[rows_to_keep].values.astype(np.int32)

    return train, test, rows_to_keep, x_tr, y_tr, x_te, y_te, feature_names



def timing_sql(engine, models, runs=1):
    times = []
    file =  "./results.csv"

    for run in range(runs):
            for train_size in (np.arange(10)+1)/10:

                n_rows = int(7000 * train_size)
                old_train, old_test, rows_to_keep, x_tr, y_tr, x_te, y_te, feature_names = get_train_test_sample(engine, n_rows)
                table_name = "items_table" 

                for name, (model_sql, model_skl) in models.items():
                    print(f"Run {run + 1}/{runs}: executing {name} with train_size={train_size}")

                    memory_usage = 0

                    if "BC" in name:

                            fit_start = time()
                            model_sql.fit("SELECT DISTINCT item FROM sample_table", y=None)
                            fit_end = time()

                            predict_start = time()
                            y_pred = model_sql.predict("SELECT DISTINCT item FROM test_table")
                            predict_end = time()

                            # numpy implementation
                            model_skl.fit(x_tr, y_tr)
                            sk_pred = model_skl.predict(x_te)

                            acc_skl = accuracy_score(y_te, sk_pred)
                            acc_sql = accuracy_score_born(engine, y_pred)
                            
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
                        acc_skl = accuracy_score(y_te, sk_pred)
                        acc_sql = model_accuracy(y_pred, old_test, rows_to_keep)

                    times.append({
                        'run': run+1,
                        'model': name,
                        'train_size': train_size, 
                        'fit_time': fit_end - fit_start,
                        'predict_time': predict_end - predict_start,
                        'memory_usage': memory_usage, 
                        'score_sklearn': acc_skl,
                        'score_sql': acc_sql
                    })
                    print("writing to file", file)
                    pd.DataFrame(times).to_csv(file, index=False)

    return times


def plot_timing( score_label='Score'):
        timing = []
        file = f"./results.csv"
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
            if key.startswith("BC_sql"):
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

        file = f"./results.png"
        fig.savefig(file, bbox_inches='tight', format='png', dpi=300)
        print(f"Image saved in {file}")
