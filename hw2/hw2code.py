import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    order = np.argsort(feature_vector)
    values = np.asarray(feature_vector, dtype=float)[order]
    labels = np.asarray(target_vector)[order]
    n = len(values)

    leftSize = np.flatnonzero(values[:-1] < values[1:]) + 1
    if leftSize.size == 0:
        return np.array([]), np.array([]), None, None

    thresholds = values[leftSize - 1] / 2 + values[leftSize] / 2
    rightSize = n - leftSize

    prefixOnes = np.cumsum(labels == 1)
    leftOnes = prefixOnes[leftSize - 1]
    rightOnes = prefixOnes[-1] - leftOnes

    leftZeros = leftSize - leftOnes
    rightZeros = rightSize - rightOnes

    ginis = -2 / n * (
        leftOnes * (leftZeros / leftSize)
        + rightOnes * (rightZeros / rightSize)
    )

    bestMask = np.isclose(ginis, ginis.max(), rtol=0, atol=1e-15)
    bestIndex = np.flatnonzero(bestMask)[0]

    return thresholds, ginis, thresholds[bestIndex], ginis[bestIndex]


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        minSplit = 2 if self._min_samples_split is None else self._min_samples_split
        minLeaf = 1 if self._min_samples_leaf is None else self._min_samples_leaf

        isPure = np.all(sub_y == sub_y[0])
        depthLimit = self._max_depth is not None and depth >= self._max_depth

        if isPure or depthLimit or len(sub_y) < minSplit:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        bestScore = -np.inf
        bestMask = None
        bestRule = {}

        for feature, featureType in enumerate(self._feature_types):
            column = sub_X[:, feature]

            if featureType == "real":
                values = column.astype(float)
            else:
                counts = Counter(column)
                ones = Counter(column[sub_y == 1])
                categories = sorted(
                    counts,
                    key=lambda category: ones[category] / counts[category],
                )
                ranks = {
                    category: rank
                    for rank, category in enumerate(categories)
                }
                values = np.array([ranks[category] for category in column])

            thresholds, ginis, threshold, score = find_best_split(values, sub_y)

            if threshold is None:
                continue

            if minLeaf > 1:
                leftSizes = np.searchsorted(
                    np.sort(values), thresholds, side="left"
                )
                rightSizes = len(values) - leftSizes
                allowed = (leftSizes >= minLeaf) & (rightSizes >= minLeaf)

                if not allowed.any():
                    continue

                allowedThresholds = thresholds[allowed]
                allowedGinis = ginis[allowed]
                best = np.isclose(
                    allowedGinis, allowedGinis.max(), rtol=0, atol=1e-15
                )
                bestIndex = np.flatnonzero(best)[0]
                threshold = allowedThresholds[bestIndex]
                score = allowedGinis[bestIndex]

            if score <= bestScore:
                continue

            bestScore = score
            bestMask = values < threshold
            bestRule = {"feature_split": feature}

            if featureType == "real":
                bestRule["threshold"] = threshold
            else:
                bestRule["categories_split"] = {
                    category
                    for category, rank in ranks.items()
                    if rank < threshold
                }

        if bestMask is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node.update(bestRule)
        node["type"] = "nonterminal"
        node["left_child"] = {}
        node["right_child"] = {}

        self._fit_node(sub_X[bestMask], sub_y[bestMask], node["left_child"], depth + 1)
        self._fit_node(sub_X[~bestMask], sub_y[~bestMask], node["right_child"], depth + 1)

    def fit(self, X, y):
        self._tree = {}
        self._fit_node(X, y, self._tree)
        return self

    def _predict_node(self, x, node):
        while node["type"] != "terminal":
            feature = node["feature_split"]

            if self._feature_types[feature] == "real":
                goLeft = float(x[feature]) < node["threshold"]
            else:
                goLeft = x[feature] in node["categories_split"]

            node = node["left_child"] if goLeft else node["right_child"]

        return node["class"]

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
