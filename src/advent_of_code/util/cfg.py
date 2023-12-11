from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Collection, Generic, Hashable, Iterator, Optional, Sequence, TypeVar

T = TypeVar("T", bound=Hashable)
log = logging.getLogger("aoc")


@dataclass(frozen=True)
class Rule(Generic[T]):
    symbol: T
    replacement: tuple[T]
    weight: int = 1


@dataclass
class ContextFreeGrammar(Generic[T]):
    rules: Collection[Rule[T]]
    start_symbol: T

    @cached_property
    def non_terminals(self) -> Sequence[T]:
        result = {r.symbol for r in self.rules if r.symbol != self.start_symbol}
        return [self.start_symbol] + list(result)

    def is_cnf(self) -> bool:
        for rule in self.rules:
            match rule:
                case Rule(sym, [], _):
                    if sym != self.start_symbol:
                        return False
                case Rule(_, [r], _):
                    if r in self.non_terminals:
                        return False
                case Rule(_, repl, _) if len(repl) == 2:
                    if any(r not in self.non_terminals for r in repl):
                        return False
                case _:
                    return False
        return True

    def to_cnf(
        self, terminal_symbols: set[T], symbol_generator: Iterator[T]
    ) -> ContextFreeGrammar[T]:
        cnf_rules = list(self.rules)
        cnf_start_sym = self.start_symbol

        # Eliminate terminal symbols from LHS
        non_terminals = set(self.non_terminals)
        new_syms: dict[T, T] = {}
        new_rules: list[Rule[T]] = []
        for t_sym in non_terminals & terminal_symbols:
            new_sym = next(symbol_generator)
            new_syms[t_sym] = new_sym
            new_rules.append(Rule(new_sym, (t_sym,), 0))
            non_terminals.remove(t_sym)
            non_terminals.add(new_sym)
        for rule in cnf_rules:
            match rule:
                case Rule(sym, repl, weight):
                    new_rule = Rule(
                        new_syms.get(sym, sym),
                        tuple(new_syms.get(r, r) for r in repl),
                        0 if sym in new_syms else weight,
                    )
                    new_rules.append(new_rule)
        cnf_rules = new_rules

        # Eliminate the start symbol from RHS
        if any(cnf_start_sym in r.replacement for r in self.rules):
            cnf_start_sym = next(symbol_generator)
            cnf_rules.append(Rule(cnf_start_sym, (self.start_symbol,), 0))

        # Eliminate rules with non-solitary terminals
        new_rules = []
        for rule in cnf_rules:
            match rule:
                case Rule(sym, repl, weight) if len(repl) > 1:
                    for r in repl:
                        if r in terminal_symbols:
                            new_sym = next(symbol_generator)
                            new_syms[r] = new_sym
                            new_rules.append(Rule(new_sym, (r,), 0))
                    new_rules.append(Rule(sym, tuple(new_syms.get(r, r) for r in repl), weight))
                case _:
                    new_rules.append(rule)
        cnf_rules = new_rules

        # Eliminate RHS with more than 2 non-terminals
        new_rules = []
        for rule in cnf_rules:
            if len(rule.replacement) < 3:
                new_rules.append(rule)
                continue
            match rule:
                case Rule(sym, repl, weight):
                    prev_sym = sym
                    for r in repl[:-2]:
                        next_sym = next(symbol_generator)
                        new_rules.append(Rule(prev_sym, (r, next_sym), 0))
                        prev_sym = next_sym
                    new_rules.append(Rule(prev_sym, repl[-2:], weight))
        cnf_rules = new_rules

        # TODO: Eliminate nullable rules

        # Eliminate unit rules
        rule_map: dict[T, Collection[Sequence[T]]] = defaultdict(list)
        for rule in cnf_rules:
            rule_map[rule.symbol].append((rule.replacement, rule.weight))

        def _find_non_unit_repls(sym) -> Iterator[tuple[Sequence[T], int]]:
            for repl, weight in rule_map[sym]:
                match repl:
                    case [r] if r in terminal_symbols:
                        yield repl, weight
                    case [r]:
                        for r, w in _find_non_unit_repls(r):
                            yield r, weight + w
                    case _:
                        yield repl, weight

        new_rules = []
        for rule in cnf_rules:
            match rule:
                case Rule(sym, (repl,), weight) if repl in non_terminals:
                    new_rules.extend(Rule(sym, r, weight + w) for r, w in _find_non_unit_repls(repl))
                case _:
                    new_rules.append(rule)
        cnf_rules = new_rules

        return ContextFreeGrammar(cnf_rules, cnf_start_sym)


def cyk(
    text: list[T], grammar: ContextFreeGrammar
) -> Optional[list[list[set[Rule[T]]]]]:
    if not grammar.is_cnf():
        raise ValueError("Grammar must be in CNF.")

    text_len = len(text)
    result = [[set() for _ in range(text_len)] for _ in range(text_len)]

    for i, char in enumerate(text):
        for rule in grammar.rules:
            match rule:
                case Rule(_, (r,), _) if r == char:
                    result[0][i].add(rule)
    if not all(result[0][i] for i in range(text_len)):
        return None

    for span_length in range(2, text_len + 1):
        for span_start in range(0, text_len - span_length + 1):
            for partition_length in range(1, span_length):
                target_rbs = {r.symbol for r in result[span_length - 2 - span_start][span_start]}
                target_rcs = {r.symbol for r in result[span_length - partition_length - 1][span_start + partition_length]}
                if not target_rbs or not target_rcs:
                    continue
                for rule in grammar.rules:
                    match rule:
                        case Rule(_, [rb, rc], _) if rb in target_rbs and rc in target_rcs:
                            result[span_length - 1][span_start].append(rule)

    return bool(result[text_len - 1][0])
