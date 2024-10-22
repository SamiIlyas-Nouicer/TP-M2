% appartient/2 predicate to check if an element belongs to a list
appartient([X|_], [X|_]) :- !. 
appartient([X|_], [_|L]) :- appartient(X, L).


% hello_world/0 predicate to print 'Hello, World!'
hello_world :- write('Hello, sami!'), nl.

%premier predicate to get the first element of a list
premier( X, [X|_]).

%dernier predicate to get the last element of a list
dernier(X, [X]) :- !.
dernier(X ,[_,Y|L]) :- dernier(X, [Y|L]).

%avant dernier element of a list

avantdernier(X,[X,_]).
avantdernier(X,[_,Y,Z|L]) :- write('checking 1st element of a 2 size list !'), nl, avantdernier(X,[Y,Z|L]) .

%suppk/3
suppk(1,[_|L],L) :- !.
suppk(N, [X|L], [X|L1]) :- N1 is N-1, suppk(N1, L, L1).