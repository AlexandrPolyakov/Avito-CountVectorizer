# -*- coding: utf-8 -*-
from typing import List, Set


class CountVectorizer:
    """Формирование словаря, подсчет количества вхождений по корпусу.

    :param encoding: кодировка
    :type encoding: str
    :param is_lower: приводить ли слова к нижнему регистру
    :type is_lower: bool
    """

    def __init__(self, encoding: str = 'utf-8', is_lower: bool = True):
        self.encoding = encoding
        self.is_lower = is_lower
        self.feature_names = set()
        self.vocabulary = {}

    def fit(self, corpus: List[List[str]]):
        """Обучение, формирование уникальных слов, словаря.

        :param corpus: обучающий корпус
        :type corpus: List[List[str]]
        """

        for text in corpus:
            for word in text.split(' '):
                if self.is_lower:
                    word = word.lower()
                self.feature_names.add(word)

        for index, word in enumerate(sorted(list(self.feature_names))):
            self.vocabulary[word] = index

    def transform(self, corpus: List[List[str]]) -> List[List[int]]:
        """Преобразование входного корпуса в матрицу количества вхождений слов.

        :param corpus: входной корпус
        :type corpus: List[List[str]]

        :rtype: List[List[int]]
        :return: матрица количества вхождений слов
        """

        count_matrix = [[0 for word in self.vocabulary] for text in corpus]
        for text_index, text in enumerate(corpus):
            counter = {}
            for word in text.split(' '):
                if self.is_lower:
                    word = word.lower()
                if word in self.vocabulary:
                    if word in counter:
                        counter[word] += 1
                    else:
                        counter[word] = 1
            for word, count in counter.items():
                word_index = self.vocabulary.get(word)
                if word_index is not None:
                    count_matrix[text_index][word_index] = count
        return count_matrix

    def fit_transform(self, corpus: List[List[str]]) -> List[List[int]]:
        """Обучение и преобразование корпуса в матрицу количества вхождений слов.

        :param corpus: входной корпус
        :type corpus: List[List[str]]

        :rtype: List[List[int]]
        :return: матрица количества вхождений слов
        """

        self.fit(corpus)
        return self.transform(corpus)

    def get_feature_names(self) -> Set[str]:
        """Возвращение уникальных слов обучающего корпуса.

        :rtype: Set[str]
        :return: уникальные слова обучающего корпуса
        """

        return self.feature_names


if __name__ == "__main__":
    count_vectorizer = CountVectorizer()
    corpus = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste'
    ]
    count_matrix = count_vectorizer.fit_transform(corpus)
    print('feature_names: ', count_vectorizer.get_feature_names())
    print('count_matrix: ', count_matrix)
