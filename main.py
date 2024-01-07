from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np


class Metrics:
    name = ['cosine', 'minkowski', 'minkowski', 'chebyshev', 'manhattan', 'euclidean', 'jaccard', 'mahalanobis']
    p = [None, 3, 4, None, None, None, None, None]


class Knn:

    def __init__(self, input_data, division: float, n: int):
        ftrs = input_data[:, :-1]
        tgt = input_data[:, -1]
        tts = train_test_split(ftrs, tgt, test_size=division)
        (train_ftrs, test_ftrs, train_tgt, test_tgt) = tts
        self.train_ftrs = train_ftrs
        self.train_tgt = train_tgt.astype(int)
        self.test_ftrs = test_ftrs
        self.test_tgt = test_tgt.astype(int)
        self.n = n
        self.metrics = Metrics

    def preds(self, metric: str, p=None):
        if metric == 'mahalanobis':
            covariance_matrix = np.cov(self.train_ftrs, rowvar=False)
            knn = KNeighborsClassifier(n_neighbors=self.n, metric=metric, metric_params={'V': covariance_matrix})
        elif p is not None:
            knn = KNeighborsClassifier(n_neighbors=self.n, metric=metric, p=p)
        else:
            knn = KNeighborsClassifier(n_neighbors=self.n, metric=metric)

        fit = knn.fit(self.train_ftrs, self.train_tgt)
        preds = fit.predict(self.test_ftrs)
        return preds

    def __preds_for_metrics(self):
        metrics_preds = []
        for m_name, m_p in zip(self.metrics.name, self.metrics.p):
            metrics_preds.append(self.preds(m_name, m_p))
        return metrics_preds

    def __accuracy_for_metrics(self):
        preds = self.__preds_for_metrics()
        preds_for_i_metrics = []
        accuracy_for_metrics = []
        combined_accuracy_for_metrics = []
        for i in range(len(self.metrics.name)):
            final_preds = []
            for k in range(len(preds[0])):
                preds_for_j = []
                for j in range(i + 1):
                    preds_for_j.append(preds[j][k])
                counters = np.bincount(preds_for_j)
                most_common_pred = np.argmax(counters)
                final_preds.append(most_common_pred)
            preds_for_i_metrics.append(final_preds)
        for m, n in zip(preds_for_i_metrics, preds):
            combined_accuracy_for_metrics.append(accuracy_score(self.test_tgt, m, normalize=True))
            accuracy_for_metrics.append(accuracy_score(self.test_tgt, n))
        return {'combined_accuracy': np.array(combined_accuracy_for_metrics),
                'accuracy': np.array(accuracy_for_metrics)}

    def show_accuracy(self):
        accuracy = self.__accuracy_for_metrics()
        x = self.metrics.name
        y1 = accuracy['combined_accuracy']
        y2 = accuracy['accuracy']
        fig, ax1 = plt.subplots()

        line1, = ax1.plot(x, y1, color='#F2AF5C', label='combined accuracy')
        ax1.set_xlabel('Metric')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', labelsize=8)

        ax2 = ax1.twinx()
        line2, = ax2.plot(x, y2, color='#508BBF', label='accuracy for metric')

        ax1.set_ylim(min(min(y1), min(y2)) - 0.1, max(max(y1), max(y2)) + 0.1)
        ax2.set_ylim(min(min(y1), min(y2)) - 0.1, max(max(y1), max(y2)) + 0.1)

        lines = [line1, line2]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper left')

        plt.title('Accuracy of metrics')

        plt.show()


file_path = './australian.txt'
data = np.loadtxt(file_path, delimiter=' ')

newKnn = Knn(data, 0.25, 3)
newKnn.show_accuracy()
