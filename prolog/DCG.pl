:- set_prolog_flag(double_quotes, chars).
:- use_module(library(clpfd)).
:- style_check(-singleton).


reverse_list([], []).

reverse_list([Head|Tail], Reversed) :-
    reverse_list(Tail, ReversedTail),
    append(ReversedTail, [Head], Reversed).

element_at(Index, Ls, Element) :-
    nth0(Index, Ls, Element).

elements_at_diff_position(Index1, Ls) :-
    Index1>0,
    Index2 is Index1-1,
    element_at(Index1, Ls, Element1),
    element_at(Index2, Ls, Element2),
    Element1 #\= Element2.

constrain(Ls, []).
constrain(Ls, [Num|Index1]) :-
    Num>0,
    elements_at_diff_position(Num,Ls),
    constrain(Ls, Index1).





f1(1) --> [1].

f1(Y) --> [1],f1(X).

f2(Y) --> [2],f1(X).
f2(Y) --> [2],f2(X).

f3(Y) --> [3],f2(X).
f3(Y) --> [3],f3(X).

f4(Y) --> [4],f3(X).

f5(Y) --> [5],f3(X).
f5(Y) --> [5],f2(X).
f5(Y) --> [5],f1(X).

fun0(N,C,Ls):-
    length(Ls1,N),
    phrase(f5(N),Ls1),
    reverse_list(Ls1,Ls),
    constrain(Ls,C).

fun1(N,C,Ls):-
    length(Ls1,N),
    phrase(f4(N),Ls1),
    reverse_list(Ls1,Ls),
    constrain(Ls,C).

