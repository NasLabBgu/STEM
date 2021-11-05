import time


class IncrementableInt:
    def __init__(self, init_value: int = 0):
        self.value = init_value

    def increment(self):
        self.value += 1

    def __repr__(self):
        return repr(self.value)

    def __str__(self):
        return str(self.value)


class TimeMeasure:
    def __init__(self):
        self.__start = None
        self.__end = None

    def start(self):
        if self.__start is None:
            self.__start = time.time()

        return self.__start

    def end(self):
        if self.__end is None:
            self.__end = time.time()

        return self.__end

    def duration(self):
        if self.__start is None or self.__end is None:
            raise AssertionError("Time not taken yet")

        return self.__end - self.__start

