# abandoned :()

from abc import abstractmethod, abstractproperty, abstractstaticmethod
from dataclasses import dataclass
from functools import wraps
from itertools import count, product
from textwrap import wrap
from typing import Callable

lvalues = ("b", "st", "t", "f", "cf", "u")
b, st, t, f, cf, u = lvalues


def fixpoint(func: Callable, a):
    while (b := func(a)) != a:
        a = b
    return b


poset = fixpoint(
    lambda inset: inset
    | {(a, d) for (a, b), (c, d) in product(inset, repeat=2) if b == c},
    {(u, t), (b, t), (cf, u), (f, cf), (st, b), (f, st)},
)
assert all((r, l) not in poset for (l, r) in poset)

implication_1 = list()
for body, head in product(lvalues, repeat=2):
    if body in (f, cf):
        result = t
    elif body == u:
        result = f if head in (st, f) else t
    elif head in (b, t, f):
        result = head
    elif head == st:
        result = f if body == t else t
    else:
        result = f
    implication_1.append((body, head, result))
del result

assert len(implication_1) == 36
assert len({(body, head) for body, head, _ in implication_1}) == 36

implication_goal = {(body, head): result for body, head, result in implication_1}

cols = "".join(f"{v:3}" for v in lvalues)
print(f"    {cols}")
for row in lvalues:
    y = ""
    for col in lvalues:
        y += f"{implication_goal[(row, col)]:3}"
    print(f"{row:4}" + y)


def lt(l, r):
    return (l, r) in poset


def le(l, r):
    return lt(l, r) or l == r


def gt(l, r):
    return lt(r, l)


def ge(l, r):
    return le(r, l)


lt_d = {
    t: 3,
    u: 2,
    b: 2,
    cf: 1,
    st: 1,
    f: 0,
}


def lower_bounds(l, r):
    for v in lvalues:
        if le(v, l) and le(v, r):
            yield v


def upper_bounds(l, r):
    for v in lvalues:
        if le(l, v) and le(r, v):
            yield v


# We assume GUB and LUB are unique
def meet(l, r):
    return max(lower_bounds(l, r), key=lt_d.get)


def join(l, r):
    return min(upper_bounds(l, r), key=lt_d.get)


assert all(join(l, r) == t for l, r in product((u, cf), (b, st)))
assert all(meet(l, r) == f for l, r in product((u, cf), (b, st)))
assert all(meet(t, o) == o for o in lvalues)
assert all(join(t, o) == t for o in lvalues)
assert all(meet(f, o) == f for o in lvalues)
assert all(join(f, o) == o for o in lvalues)


class BinOp:
    @abstractstaticmethod
    def op(l, r):
        pass

    @classmethod
    def enumerate(Cls):
        yield Cls()

    def eval(self, stack: list):
        a, b = stack.pop(), stack.pop()
        stack.append(self.op(a, b))

    def __repr__(self) -> str:
        return self.symbol


def bool_to_lvalue(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return t if func(*args, **kwargs) else f

    return wrapper


class MeetOp(BinOp):
    op = staticmethod(meet)
    symbol = "&"


class JoinOp(BinOp):
    op = staticmethod(join)
    symbol = "|"


class LtOp(BinOp):
    op = staticmethod(bool_to_lvalue(lt))
    symbol = "<"


class LeOp(BinOp):
    op = staticmethod(bool_to_lvalue(le))
    symbol = "<="


class GtOp(BinOp):
    op = staticmethod(bool_to_lvalue(gt))
    symbol = ">"


class GeOp(BinOp):
    op = staticmethod(bool_to_lvalue(ge))
    symbol = ">="


BinOps = (MeetOp, JoinOp, LtOp, LeOp, GtOp, GeOp)


@dataclass(frozen=True)
class PartialBinOp:
    constant: str
    binop: BinOp

    @classmethod
    def enumerate(Cls):
        for v in lvalues:
            for Binop in BinOps:
                yield Cls(v, Binop())

    def eval(self, stack: list):
        stack.append(self.constant)
        return self.binop.eval(stack)

    def __repr__(self) -> str:
        return f"{self.constant}{self.binop}"


class DupeStackOp:
    @classmethod
    def enumerate(Cls):
        yield Cls()

    def eval(self, stack: list):
        stack.extend([*stack])


def instruction():
    for BinOp in BinOps:
        yield from BinOp.enumerate()
    yield from PartialBinOp.enumerate()
    yield from DupeStackOp.enumerate()


@dataclass(frozen=True)
class Program:
    goal: dict
    instructions: tuple

    @classmethod
    def enumerate(Cls, goal: dict, size: int):
        return (
            Program(goal, instructions)
            for instructions in product(instruction(), repeat=size)
        )

    def eval(self, stack: list):
        for instr in self.instructions:
            instr.eval(stack)

    def verify(self) -> bool:
        for a, b in product(lvalues, repeat=2):
            stack = [a, b]
            try:
                self.eval(stack)
            except IndexError:
                return False
            if stack != [self.goal[(a, b)]]:
                return False
        return True


def synthesize(goal):
    for size in range(1, 6):
        print(f"Trying programs of size {size}")
        solutions = tuple(p for p in Program.enumerate(goal, size) if p.verify())
        if len(solutions) == 0:
            continue
        print("Solution(s) found!")
        for solution in solutions:
            print(solution.instructions)
        break
    print(":(")


always_top_goal = {(l, r): t for l, r in product(lvalues, repeat=2)}
some_goal = {
    (l, r): join(l, r) if not (l == r == f) else f
    for l, r in product(lvalues, repeat=2)
}
# p = Program(always_top_goal, (JoinOp(), PartialBinOp(t, JoinOp())))

# p.verify()

# synthesize(implication_goal)


for a, b in product(lvalues, repeat=2):
    c = join(a, b)
    print(f":- not join({a}, {b}, {c}).")
    c = meet(a, b)
    print(f":- not meet({a}, {b}, {c}).")
