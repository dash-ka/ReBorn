import pandas as pd
import numpy as np
from scipy.sparse._csr import csr_matrix
from sqlalchemy import  inspect, text, MetaData, Table, Column, Integer, Float, String

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB


class SQLModel(object):

    def __init__(self, engine):
        self.engine = engine

    @classmethod
    def from_sklearn(cls, sklearn_model:str, engine):

        MODELS = {
        "LogisticRegression": LogisticRegressionSQL,
        "MultinomialNB": MultinomialNBSQL,
        "DecisionTreeClassifier": DecisionTreeSQL,
        "RandomForestClassifier": RandomForestSQL
    }
        
        try:
            model = MODELS[sklearn_model]
            return model(engine)
        
        except KeyError as err:
            print("Model not suported.")
        

    def write(self, con, table, records):
        
        if inspect(con).has_table(table.name):
            table.drop(con)
        table.create(con)

        keys = records[0].keys()
        values = f'VALUES(:{", :".join(keys)})'
        q = f""" 
            INSERT INTO {table.name} ({','.join(keys)})
            {values}
            """
        con.execute(text(q), records)
        return table 


    def write_items(self, connection, items, feature_names=None):

        table = Table(
            "items_table", MetaData(),
            Column("item", Integer, primary_key=True),
            Column("feature", String, primary_key=True),
            Column("weight", Float)
            )

        records = []
        for i, features in enumerate(items):

            if isinstance(features, dict):
                for j, w in features.items():
                    records.append({"item":i, "feature":j, "weight":w})

            elif isinstance(features, csr_matrix):
                for (j, w) in zip(features.indices, features.data):
                    records.append({"item":i, "feature":feature_names[j] 
                                    if feature_names is not None else j, "weight":w})
            else:
                for j, w in enumerate(features):
                    assert not isinstance(w, csr_matrix)
                    records.append({"item":i, "feature":j, "weight":w})

        return self.write(connection, table, records)
            
           

    def fit(self, X_train, y_train, feature_names=None):

        model = self.sklearn_model()
        self.model = model.fit(X_train, y_train)
        self._model2sql(model, feature_names)

        return self
    

    def predict(self, items, feature_names=None):
     
        with self.engine.connect() as con:
            with con.begin():
                if not isinstance(items, str):
                    items = self.write_items(con, items, feature_names)

                predictions = pd.read_sql(
                    self.query.format(items_table = items), con
                    )#["class"].values
                
        targets = self.model.classes_
        predictions["class"] = targets[predictions["class"].values]

        return predictions
    


class LogisticRegressionSQL(SQLModel):

    def __init__(self, engine):
        super().__init__(engine)
        self.sklearn_model = LogisticRegression

    query =  """ 
        
        WITH
            activation AS(
            SELECT item, class,
                SUM({items_table}.weight * logistic_table.weight) + bias AS z
                FROM {items_table}, logistic_table
                WHERE {items_table}.feature = logistic_table.feature
                GROUP BY item, class, bias
            ),
            summation AS (
                SELECT item, SUM(EXP(z)) AS value
                FROM activation
                GROUP BY item
            ),
            predictions AS (
                SELECT activation.item, class, EXP(z)/summation.value AS p
                FROM activation, summation
                WHERE activation.item = summation.item
                ),
            ranking AS (
                    SELECT item, class, row_number() OVER (PARTITION BY item ORDER BY p DESC) AS rank
                    FROM predictions
                )
            SELECT item, class
            FROM ranking
            WHERE rank = 1

        """

    def _model2sql(self, model, feature_names=None):

            # extract weights and biases
            weights, bias = model.coef_, model.intercept_

            table = Table(
                "logistic_table", MetaData(),
                Column("class", Integer, primary_key=True),
                Column("feature", String, primary_key=True),
                Column("weight", Float),
                Column("bias", Float)
                )     
            # manage binary classification
            if weights.shape[0] < 2:
                weights = np.vstack([-weights, weights])
                bias = np.vstack([-bias, bias]).reshape(2,)
            
            if feature_names is not None:
                params = [{"class":c, "feature":feature_names[f], "weight":weights[c][f], "bias":bias[c]}
                        for c in range(weights.shape[0])
                        for f in range(weights.shape[1])]
            else:
                params = [{"class":c, "feature":f, "weight":weights[c][f], "bias":bias[c]}
                        for c in range(weights.shape[0])
                        for f in range(weights.shape[1])]
            
            # create the table and insert fitted params
            with self.engine.connect() as con:
                with con.begin():
                    self.write(con, table, params)  


class MultinomialNBSQL(SQLModel):
    
    def __init__(self, engine):
        super().__init__(engine)
        self.sklearn_model = MultinomialNB

    query = """
        WITH
        feature_probs AS(
            SELECT item, class, SUM({items_table}.weight * feature_log_probs.weight) AS p
            FROM {items_table}, feature_log_probs 
            WHERE {items_table}.feature = feature_log_probs.feature
            GROUP BY item, class
            ),

        posterior_probs as (
            SELECT item, feature_probs.class, p + weight as p
            FROM feature_probs, class_priors
            WHERE class_priors.class = feature_probs.class
            ),

        ranking AS (
                SELECT item, class, row_number() OVER (PARTITION BY item ORDER BY p DESC) AS rank
                FROM posterior_probs
            )
        SELECT item, class
        FROM ranking
        WHERE rank = 1
        """

    def _model2sql(self, model, feature_names=None):

        # extract parameters
        feature_log_probs, class_prior = model.feature_log_prob_, model.class_log_prior_
        
        table_feature_probs = Table(
            "feature_log_probs", MetaData(),
            Column("class", Integer, primary_key=True),
            Column("feature", String, primary_key=True),
            Column("weight", Float)
            )  

        table_priors = Table(
            "class_priors", MetaData(),
            Column("class", Integer, primary_key=True),
            Column("weight", Float)
        )

        classes, features = feature_log_probs.shape
        params_posterior = [{"class":k, "feature":feature_names[j], "weight":feature_log_probs[k][j]} 
                            if feature_names is not None
                            else {"class":k, "feature":j, "weight":feature_log_probs[k][j]}
                for k in range(classes)
                for j in range(features)]

        params_priors = [
            {"class":k,  "weight": class_prior[k]}
            for k in range(classes)
        ]
        
        with self.engine.connect() as con:
            with con.begin():
                self.write(con, table_feature_probs, params_posterior) 
                self.write(con, table_priors, params_priors)  


class DecisionTreeSQL(SQLModel):

    def __init__(self, engine):
        super().__init__(engine)
        self.sklearn_model = DecisionTreeClassifier

    """
    This class implements the SQL wrapper for a Sklearn Decision Tree Model
    """

    query= """ 
            
            WITH recursive Q as (

            -- ancor term:
            SELECT item, is_leaf, CASE WHEN direction.right = 1
                THEN tree_table.right_child
                ELSE tree_table.left_child 
                END AS split_node_id
            FROM (
                -- set a flag on items that satisfy the splitting rule (i.e., feature[j] > threshold)
                SELECT DISTINCT item, COUNT(threshold) OVER (PARTITION BY item) AS right
                FROM {items_table} LEFT JOIN tree_table ON 
                    {items_table}.feature = tree_table.feature AND
                    {items_table}.weight > tree_table.threshold AND
                    depth = 0
                ORDER BY item
                ) AS direction 
            -- append to each row (i.e., item) ids of the right and left child
            JOIN tree_table ON tree_table.depth=0

            UNION 

            -- recursive term:
            SELECT item, is_leaf, CASE WHEN direction.right=1
                THEN tree_table.right_child
                ELSE tree_table.left_child 
                END AS split_node_id
            FROM (
                SELECT DISTINCT item, Q.split_node_id, COUNT(threshold) OVER (PARTITION BY item) AS right
                FROM Q JOIN {items_table} USING (item)                    -- extend each item with its features
                    LEFT JOIN tree_table                                  -- complete only matching rows (i.e., item features) with tree info
                    ON tree_table.node = Q.split_node_id AND             
                    {items_table}.feature = tree_table.feature  AND
                    {items_table}.weight > tree_table.threshold
                    ORDER BY item
                ) AS direction 
            -- append to each row (i.e., item) ids of the right and left child of the splitting node
            JOIN tree_table ON tree_table.node = direction.split_node_id

            -- filter out row (i.e., items) that reach the leaf node
            WHERE is_leaf != 1

        ), leaf_nodes AS (
            SELECT item, split_node_id, MAX(split_node_id) OVER(PARTITION BY item) AS leaf
            FROM Q

        ), ranking as (
            SELECT item, split_node_id, class, value,
                sum(value) over(partition by item, class) AS k_support,
                sum(value) over(partition by item) AS tot_leaf_support
            FROM leaf_nodes JOIN partition_table ON split_node_id = node
            WHERE split_node_id = leaf
            ORDER BY item, class
            
        ), probs AS (
            SELECT DISTINCT item, class, k_support/CAST(tot_leaf_support as float) AS class_probability,
                row_number() OVER (PARTITION BY item ORDER BY k_support / CAST(tot_leaf_support as float) DESC) AS n
            FROM ranking
            ORDER BY item, n
        )
        SELECT item, class
        FROM probs
        WHERE n=1;
    """

   

    @staticmethod
    def get_tree_info(clf: DecisionTreeClassifier):
        features = clf.tree_.feature
        thresholds = clf.tree_.threshold
        node_partition = clf.tree_.value
        left_child = clf.tree_.children_left
        right_child = clf.tree_.children_right

        # retrieving depth 
        node_ids = np.arange(clf.tree_.node_count, dtype=int)
        node_depth = np.zeros(clf.tree_.node_count, dtype=int)
        is_leaf = np.zeros(clf.tree_.node_count, dtype=int)

        stack = [(0, 0)]
        while len(stack) >0 :
            node_id, depth = stack.pop()
            node_depth[node_id] = depth

            if left_child[node_id]!=right_child[node_id]:
                stack.append((left_child[node_id], depth+1))
                stack.append((right_child[node_id], depth+1))
            else:
                is_leaf[node_id] = 1
        node_depth = node_depth.astype(int)

        return node_ids, node_depth, node_partition, features, thresholds, left_child, right_child, is_leaf
    

    
    def extract_params(self, model: DecisionTreeClassifier, feature_names=None, model_id=None):

        model_id = 0 if model_id is None else model_id
        tree_params = []
        class_partitions = []
        for node_id, depth, node_partition, feature, threshold, left, right, flag in zip(*DecisionTreeSQL.get_tree_info(model)):
            tree_params.append(
                {
                    "tree_id":model_id, 
                    "node":node_id,
                    "depth":depth,
                    "feature":feature if feature_names is None else feature_names[feature], 
                    "threshold":threshold, 
                    "left_child":left, 
                    "right_child":right, 
                    "is_leaf":flag
                } 
            )
            
            for class_k, items_in_k in enumerate(node_partition.reshape(-1)):
                class_partitions.append(
                    {
                        "tree_id": model_id, 
                        "node": node_id, 
                        "class": class_k,
                        "value": int(items_in_k)
                    }
                )   
        
        return tree_params, class_partitions
    

    def _model2sql(self, model, feature_names=None):

        tree_params, class_partitions = self.extract_params(model, feature_names)

        tree_table = Table(
            "tree_table", 
            MetaData(),
        
            Column("tree_id", Integer, primary_key=True),
            Column("node", Integer, primary_key=True),
            Column("depth", Integer),
            Column("feature", String),
            Column("threshold", Float),
            Column("left_child", Integer),
            Column("right_child", Integer),
            Column("is_leaf", Integer)
            )

        partition_table = Table(
            "partition_table",
            MetaData(),
            Column("tree_id", Integer, primary_key=True),
            Column("node", Integer, primary_key=True),
            Column("class", Integer, primary_key=True),
            Column("value", Integer),
        )

        with self.engine.connect() as connection:
            with connection.begin():
                self.write(connection, tree_table, tree_params)
                self.write(connection, partition_table, class_partitions)



class RandomForestSQL(DecisionTreeSQL):

    def __init__(self, engine):
        super().__init__(engine)
        self.sklearn_model = RandomForestClassifier

    query = """
        
            WITH RECURSIVE Q AS (
                SELECT item, partition.tree_id, is_leaf,
                CASE WHEN partition.right = 1 then right_child
                ELSE left_child END AS split_node
                FROM (
                    WITH help as (select distinct tree_id from forest_table)
                    SELECT distinct item, help.tree_id, count(forest_table.tree_id) over (partition by item, help.tree_id) as right
                    FROM {items_table} CROSS JOIN help
                        LEFT JOIN forest_table ON forest_table.tree_id = help.tree_id AND
                                            forest_table.feature = {items_table}.feature AND
                                            {items_table}.weight > forest_table.threshold AND 
                                            forest_table.depth = 0
                    ORDER BY item, help.tree_id
                    ) AS partition
                JOIN forest_table on partition.tree_id = forest_table.tree_id AND depth = 0

                UNION

                SELECT DISTINCT item, direction.tree_id, is_leaf,
                CASE WHEN direction.right = 1 then right_child
                ELSE left_child END AS split_node
                FROM (
                    SELECT DISTINCT item, Q.tree_id, Q.split_node,
                                    COUNT(forest_table.tree_id) OVER (PARTITION BY item, Q.tree_id) AS right
                    FROM Q
                    JOIN {items_table} USING (item)
                    LEFT JOIN forest_table ON forest_table.tree_id = Q.tree_id AND
                                        forest_table.node = Q.split_node AND
                                        forest_table.feature = {items_table}.feature AND
                                        forest_table.threshold < {items_table}.weight
                    ) AS direction
                JOIN forest_table on direction.tree_id = forest_table.tree_id AND
                                    direction.split_node = forest_table.node
                WHERE is_leaf != 1
                ),
                
                leaf_nodes AS (
                    SELECT item, tree_id, split_node, MAX(split_node) OVER(PARTITION BY item, tree_id) AS leaf
                    FROM Q
                    ),
                ranking as (
                    SELECT item, leaf_nodes.tree_id, split_node, class, value,
                        sum(value) over(partition by item, class) AS k_support,
                        sum(value) over(partition by item) AS tot_leaf_support
                    FROM leaf_nodes JOIN forest_partition_table ON
                        leaf_nodes.tree_id = forest_partition_table.tree_id AND leaf = node
                    WHERE split_node = leaf
                    ORDER BY item, class
                    ),
                probs as (
                    SELECT DISTINCT item, class, k_support / CAST(tot_leaf_support AS float) AS class_probability
                    FROM ranking
                    ORDER BY item, class
                    ),
                prediction as (
                    SELECT item, class, class_probability, row_number() over (PARTITION BY item ORDER BY class_probability desc) as n
                    FROM probs
                )
                select item, class
                from prediction
                where n=1
                order by item;
                
        """  
    
    def _model2sql(self, model):

        forest_params, class_partitions = [],[]
        for tree_id, tree_classifier in enumerate(model.estimators_):
            t_params, c_partitions = self.extract_params(tree_classifier, tree_id)
            forest_params.extend(t_params), class_partitions.extend(c_partitions)

        forest_table = Table(
            "forest_table", 
            MetaData(),
        
            Column("tree_id", Integer, primary_key=True),
            Column("node", Integer, primary_key=True),
            Column("depth", Integer),
            Column("feature", String),
            Column("threshold", Float),
            Column("left_child", Integer),
            Column("right_child", Integer),
            Column("is_leaf", Integer)
            )

        forest_partition_table = Table(
            "forest_partition_table",
            MetaData(),
            Column("tree_id", Integer, primary_key=True),
            Column("node", Integer, primary_key=True),
            Column("class", Integer, primary_key=True),
            Column("value", Integer),
        )

        with self.engine.connect() as connection:
            with connection.begin():
                self.write(connection, forest_table, forest_params)
                self.write(connection, forest_partition_table, class_partitions)
