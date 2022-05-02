
#const size = 6.

instruction(1..size).

lvalue(t;u;cf;b;st;f).
lvar(t), lvar(u), lvar(cf), lvar(b), lvar(st), lvar(f).
rvar(t), rvar(u), rvar(cf), rvar(b), rvar(st), rvar(f).

lt(u, t).
lt(cf, u).
lt(f, cf).
lt(b, t).
lt(st, b).
lt(f, st).
lt(X, Z) :- lt(X, Y), lt(Y, Z).
le(L, R) :- lt(L, R).
le(L, L) :- lvalue(L).

lower_bound(L, R, M) :- le(M, L), le(M, R).
lower_bound_not_glb(L, R, M) :- lower_bound(L, R, M), lower_bound(L, R, M2), not le(M2, M).
upper_bound(L, R, M) :- le(L, M), le(R, M).
upper_bound_not_lub(L, R, M) :- upper_bound(L, R, M), upper_bound(L, R, M2), not le(M, M2).

meet(L, R, M) :- lower_bound(L, R, M), not lower_bound_not_glb(L, R, M).
:- meet(L, R, M), meet(L, R, M2), M != M2.
join(L, R, M) :- upper_bound(L, R, M), not upper_bound_not_lub(L, R, M).
:- join(L, R, M), join(L, R, M2), M != M2.


impl(b, b, b).
impl(b, st, t).
impl(b, t, t).
impl(b, f, f).
impl(b, cf, f).
impl(b, u, f).
impl(st, b, b).
impl(st, st, t).
impl(st, t, t).
impl(st, f, f).
impl(st, cf, f).
impl(st, u, f).
impl(t, b, b).
impl(t, st, f).
impl(t, t, t).
impl(t, f, f).
impl(t, cf, f).
impl(t, u, f).
impl(f, b, t).
impl(f, st, t).
impl(f, t, t).
impl(f, f, t).
impl(f, cf, t).
impl(f, u, t).
impl(cf, b, t).
impl(cf, st, t).
impl(cf, t, t).
impl(cf, f, t).
impl(cf, cf, t).
impl(cf, u, t).
impl(u, b, t).
impl(u, st, f).
impl(u, t, t).
impl(u, f, f).
impl(u, cf, t).
impl(u, u, t).

component(L) :- lvalue(L).
component(lvar).
component(rvar).

operation(lt).
operation(le).
operation(meet).
operation(join).

component((Op, ID)) :- operation(Op), ID = 0..5.

{ arc_left(Src, (Op, ID)) : component(Src) } = 1 :- component((Op, ID)).
{ arc_right(Src, (Op, ID)) : component(Src) } = 1 :- component((Op, ID)).

path_to(A, B) :- arc_left(A, B).
path_to(A, B) :- arc_right(A, B).
path_to(A, C) :- path_to(A, B), path_to(B, C).
:- path_to(A, A).

component_value(L, L) :- lvalue(L).
component_value(lvar, V) :- lvar(V).
component_value(rvar, V) :- rvar(V).
component_value((Op, ID), t) :- 
    component((Op, ID)),
    arc_left(Left, (Op, ID)),
    component_value(Left, LeftValue),
    arc_right(Right, (Op, ID)),
    component_value(Right, RightValue),
    Op = lt,
    lt(LeftValue, RightValue).
component_value((Op, ID), f) :- 
    component((Op, ID)),
    arc_left(Left, (Op, ID)),
    component_value(Left, LeftValue),
    arc_right(Right, (Op, ID)),
    component_value(Right, RightValue),
    Op = lt,
    not lt(LeftValue, RightValue).
component_value((Op, ID), t) :- 
    component((Op, ID)),
    arc_left(Left, (Op, ID)),
    component_value(Left, LeftValue),
    arc_right(Right, (Op, ID)),
    component_value(Right, RightValue),
    Op = le,
    le(LeftValue, RightValue).
component_value((Op, ID), f) :- 
    component((Op, ID)),
    arc_left(Left, (Op, ID)),
    component_value(Left, LeftValue),
    arc_right(Right, (Op, ID)),
    component_value(Right, RightValue),
    Op = le,
    not le(LeftValue, RightValue).
component_value((Op, ID), t) :- 
    component((Op, ID)),
    arc_left(Left, (Op, ID)),
    component_value(Left, LeftValue),
    arc_right(Right, (Op, ID)),
    component_value(Right, RightValue),
    Op = lt,
    lt(LeftValue, RightValue).
component_value((Op, ID), Value) :- 
    component((Op, ID)),
    arc_left(Left, (Op, ID)),
    component_value(Left, LeftValue),
    arc_right(Right, (Op, ID)),
    component_value(Right, RightValue),
    Op = meet,
    meet(LeftValue, RightValue, Value).
component_value((Op, ID), Value) :- 
    component((Op, ID)),
    arc_left(Left, (Op, ID)),
    component_value(Left, LeftValue),
    arc_right(Right, (Op, ID)),
    component_value(Right, RightValue),
    Op = join,
    join(LeftValue, RightValue, Value).    


{ output_component((Op, ID)) : component((Op, ID)) } = 1.
output((Op, ID), Value) :- component_value((Op, ID), Value).

% arc_left(lvar, (join, 0)).
% arc_right(rvar, (join, 0)).


:- not sat.
sat :- lvar(LV), rvar(RV), output(_, Value), impl(LV, RV, Value).
lvar(L) :- sat, lvalue(L).
rvar(L) :- sat, lvalue(L).

#show output/2.
% #show arc_left/2.
% #show arc_right/2.


% Debug
% has_meet(L, R) :- meet(L, R, _).
% :- not has_meet(L, R), lvalue(L), lvalue(R).
% has_join(L, R) :- join(L, R, _).
% :- not has_join(L, R), lvalue(L), lvalue(R).
% :- not join(b, b, b).
% :- not meet(b, b, b).
% :- not join(b, st, b).
% :- not meet(b, st, st).
% :- not join(b, t, t).
% :- not meet(b, t, b).
% :- not join(b, f, b).
% :- not meet(b, f, f).
% :- not join(b, cf, t).
% :- not meet(b, cf, f).
% :- not join(b, u, t).
% :- not meet(b, u, f).
% :- not join(st, b, b).
% :- not meet(st, b, st).
% :- not join(st, st, st).
% :- not meet(st, st, st).
% :- not join(st, t, t).
% :- not meet(st, t, st).
% :- not join(st, f, st).
% :- not meet(st, f, f).
% :- not join(st, cf, t).
% :- not meet(st, cf, f).
% :- not join(st, u, t).
% :- not meet(st, u, f).
% :- not join(t, b, t).
% :- not meet(t, b, b).
% :- not join(t, st, t).
% :- not meet(t, st, st).
% :- not join(t, t, t).
% :- not meet(t, t, t).
% :- not join(t, f, t).
% :- not meet(t, f, f).
% :- not join(t, cf, t).
% :- not meet(t, cf, cf).
% :- not join(t, u, t).
% :- not meet(t, u, u).
% :- not join(f, b, b).
% :- not meet(f, b, f).
% :- not join(f, st, st).
% :- not meet(f, st, f).
% :- not join(f, t, t).
% :- not meet(f, t, f).
% :- not join(f, f, f).
% :- not meet(f, f, f).
% :- not join(f, cf, cf).
% :- not meet(f, cf, f).
% :- not join(f, u, u).
% :- not meet(f, u, f).
% :- not join(cf, b, t).
% :- not meet(cf, b, f).
% :- not join(cf, st, t).
% :- not meet(cf, st, f).
% :- not join(cf, t, t).
% :- not meet(cf, t, cf).
% :- not join(cf, f, cf).
% :- not meet(cf, f, f).
% :- not join(cf, cf, cf).
% :- not meet(cf, cf, cf).
% :- not join(cf, u, u).
% :- not meet(cf, u, cf).
% :- not join(u, b, t).
% :- not meet(u, b, f).
% :- not join(u, st, t).
% :- not meet(u, st, f).
% :- not join(u, t, t).
% :- not meet(u, t, u).
% :- not join(u, f, u).
% :- not meet(u, f, f).
% :- not join(u, cf, u).
% :- not meet(u, cf, cf).
% :- not join(u, u, u).
% :- not meet(u, u, u).