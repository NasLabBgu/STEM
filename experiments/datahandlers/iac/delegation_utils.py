from abc import ABCMeta


def _make_delegator_method(name):
    def delegator(self, *args, **kwargs):
        return getattr(self._delegate, name)(*args, **kwargs)
    return delegator


class DelegatingMeta(ABCMeta):
    def __new__(mcs, name, bases, dct):
        abstract_method_names = frozenset.union(*(base.__abstractmethods__
                                                  for base in bases))
        for name in abstract_method_names:
            if name not in dct:
                dct[name] = _make_delegator_method(name)

        return super(DelegatingMeta, mcs).__new__(mcs, name, bases, dct)
