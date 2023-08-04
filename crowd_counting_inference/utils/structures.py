from typing import List


class Point:
    def __init__(self, x: float, y: float, score: float = 0.0, label: str = "") -> None:
        self._x: float = x
        self._y: float = y
        self._label: str = label
        self._score: float = score

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        self._score = value

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value


class HeadLocalizationResult:
    def __init__(self) -> None:
        self._points: List[Point] = []

    def add(self, point: Point):
        self._points.append(point)

    @property
    def points(self):
        return self._points
