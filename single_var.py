#!/usr/bin/env python3

"""
Find the smallest 3 value logic formulas that utilize a single variable and produce one of the 27 possible mappings
"""

from sys import flags, argv
from itertools import product, islice, takewhile
from abc import abstractmethod, ABC

LAND = "∧"
LOR = "∨"
LNEG = "¬"
LVALUES = "tuf"

# A unique mapping is triple of three pairs. E.g. the identity mapping is ((t, t), (u, u), (f, f))
def mappings():
    a = product(LVALUES, repeat=2)
    x, y, z = islice(a, 3), islice(a, 3), islice(a, 3)
    return product(x, y, z)


def to_int(lval):
    if lval == "f":
        return 0
    elif lval == "u":
        return 1
    elif lval == "t":
        return 2
    else:
        raise TypeError()


def from_int(num):
    if num == 0:
        return "f"
    elif num == 1:
        return "u"
    elif num == 2:
        return "t"
    else:
        raise TypeError()


class Expr(ABC):
    @staticmethod
    def enumerate(budget: int):
        for size in range(1, budget + 1):
            for ExprType in LANGUAGE:
                yield from (
                    x
                    for x in takewhile(
                        lambda e: e.size() <= size,
                        ExprType.enumerate(budget - ExprType.min_size()),
                    )
                    if x.size() == size
                )

    @abstractmethod
    def size() -> int:
        pass

    @abstractmethod
    def eval(self) -> str:
        pass

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o)

    @abstractmethod
    def __str__(self) -> str:
        pass

    def wrap(self):
        return f"({self})"

    @classmethod
    def leaf_classes(cls):
        wl = set(cls.__subclasses__())
        result = set()
        while len(wl) != 0:
            item = wl.pop()
            if len(item.__subclasses__()) == 0:
                result.add(item)
            else:
                wl.update(item.__subclasses__())
        return result

    @staticmethod
    def min_size():
        return 1


class BareExpr(Expr):
    def size(self):
        return 1

    def wrap(self):
        return str(self)


class Var(BareExpr):
    @staticmethod
    def enumerate(budget: int):
        yield Var()

    def __eq__(self, o: object) -> bool:
        return type(o) == Var

    def __hash__(self) -> int:
        return 0

    def __str__(self) -> str:
        return "x"

    def eval(self) -> str:
        return (("t", "t"), ("u", "u"), ("f", "f"))


class Lit(BareExpr):
    def __init__(self, v) -> None:
        self.v = v

    def __eq__(self, o: object) -> bool:
        if type(o) != Lit:
            return False
        return self.v == o.v

    def __hash__(self) -> int:
        return hash(self.v)

    def __str__(self) -> str:
        return self.v

    @staticmethod
    def enumerate(budget: int):
        return (Lit(v) for v in "tuf")

    def eval(self) -> str:
        return (("t", self.v), ("u", self.v), ("f", self.v))


class Neg(BareExpr):
    def __init__(self, e: Expr) -> None:
        self.e = e

    def __eq__(self, o: object) -> bool:
        if type(o) != Neg:
            return False
        return self.e == o.e

    def __hash__(self) -> int:
        return hash(self.e)

    @staticmethod
    def enumerate(budget: int):
        for e in Expr.enumerate(budget - 1):
            yield Neg(e)

    def size(self):
        return 1 + self.e.size()

    def eval(self) -> str:
        v = (p[1] for p in self.e.eval())
        return tuple((l, from_int(2 - to_int(v))) for l, v in zip(LVALUES, v))

    def __str__(self) -> str:
        return f"{LNEG}{self.e.wrap()}"


class BinExpr(Expr):
    def __init__(self, edges):
        edges = tuple(edges)
        assert len(edges) == 2
        self.edges = edges

    @classmethod
    def enumerate(Cls, budget: int):
        seen = set()
        for a in Expr.enumerate(budget - 2):
            asize = a.size()
            for b in Expr.enumerate(budget - asize):
                if b.size() > asize:
                    break
                e = Cls((a, b))
                if e in seen:
                    continue
                seen.add(e)
                yield e

    def __eq__(self, o: object) -> bool:
        if type(self) != type(o):
            return False
        return set(self.edges) == set(o.edges)

    def __hash__(self) -> int:
        return hash(frozenset(self.edges))

    def size(self):
        return 1 + sum(edge.size() for edge in self.edges)

    def eval_helper(self, op):
        a = self.edges[0].eval()
        b = self.edges[1].eval()
        return tuple(
            (f1, from_int(op(to_int(t1), to_int(t2))))
            for (f1, t1), (_, t2) in zip(a, b)
        )

    @staticmethod
    def min_size():
        return 3


class Conj(BinExpr):
    def eval(self) -> str:
        return self.eval_helper(min)

    def __str__(self) -> str:
        return f"{self.edges[0].wrap()} {LAND} {self.edges[1].wrap()}"


class Disj(BinExpr):
    def eval(self) -> str:
        return self.eval_helper(max)

    def __str__(self) -> str:
        return f"{self.edges[0].wrap()} {LOR} {self.edges[1].wrap()}"


class Eq(BinExpr):
    def eval(self) -> str:
        return self.eval_helper(lambda a, b: 2 if a == b else 0)

    def __str__(self) -> str:
        return f"{self.edges[0].wrap()} = {self.edges[1].wrap()}"


class Lt(BinExpr):
    def eval(self) -> str:
        return self.eval_helper(lambda a, b: 2 if a < b else 0)

    def __str__(self) -> str:
        return f"{self.edges[0].wrap()} < {self.edges[1].wrap()}"


def query_size_optimal(mapping, size_limit):
    best_size = -1
    found = False
    for expr in Expr.enumerate(size_limit):
        s = expr.size()
        if found and s > best_size:
            break
        evaled = expr.eval()
        if evaled == mapping:
            found = True
            best_size = expr.size()
            yield expr
    if not found:
        print(mapping_str(mapping))


def query_all_mappings():
    for mapping in mappings():
        for expr in query_size_optimal(mapping, 9):
            # print(str(expr))
            pass


T = "t"
U = "u"
F = "f"


def mapping_str(mapping):
    return "".join(m[1] for m in mapping)


def m(t, u, f):
    return (("t", t), ("u", u), ("f", f))


def interactive():
    global result
    i = input('Enter desired mapping (e.g. the identity mapping is "T U F"\n')
    i = (eval(i) for i in i.split())
    results = query_size_optimal(m(*i), 10)
    for result in results:
        print(result)
        if input("Another? [Y/n] ").lower() == "n":
            break
    else:
        print("All done.")


LANGUAGE = [Neg, Lit, Var, Eq, Lt, Conj, Disj]

if __name__ == "__main__" and not flags.interactive:
    if len(argv) == 2 and argv[1] == "--all":
        query_all_mappings()
    else:
        try:
            interactive()
        except EOFError:
            pass
