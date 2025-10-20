from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from functools import cache, cached_property
from typing import Collection, Generic, Hashable, Iterator, Optional, Sequence, TypeVar

T = TypeVar("T", bound=Hashable)
log = logging.getLogger("aoc")


@dataclass(frozen=True)
class Rule(Generic[T]):
    symbol: T
    replacement: tuple[T, ...]
    weight: int = 1


@dataclass
class ContextFreeGrammar(Generic[T]):
    rules: Collection[Rule[T]]
    start_symbol: T

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, ContextFreeGrammar):
            return False
        return self.start_symbol == __value.start_symbol and set(self.rules) == set(__value.rules)

    @cached_property
    def non_terminals(self) -> Sequence[T]:
        result = {r.symbol for r in self.rules if r.symbol != self.start_symbol}
        return [self.start_symbol, *result]

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
                        weight,
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

        def _find_non_unit_repls(sym: T) -> Iterator[tuple[Sequence[T], int]]:
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
                case Rule(sym, [repl], weight) if repl in non_terminals:
                    new_rules.extend(
                        Rule(sym, r, weight + w) for r, w in _find_non_unit_repls(repl)
                    )
                case _:
                    new_rules.append(rule)
        cnf_rules = new_rules

        return ContextFreeGrammar(cnf_rules, cnf_start_sym)


CykResult = list[list[set[tuple[Rule[T], int]]]]


def cyk(text: list[T], grammar: ContextFreeGrammar[T]) -> Optional[CykResult[T]]:
    if not grammar.is_cnf():
        raise ValueError("Grammar must be in CNF.")

    text_len = len(text)
    result = [[set() for _ in range(text_len - i)] for i in range(text_len)]

    for i, char in enumerate(text):
        for rule in grammar.rules:
            match rule:
                case Rule(_, [r], _) if r == char:
                    result[0][i].add((rule, None))
    if not all(result[0][i] for i in range(text_len)):
        return None

    for span_length in range(2, text_len + 1):
        for span_start in range(0, text_len - span_length + 1):
            for partition_length in range(1, span_length):
                target_rbs = {r.symbol for r, _ in result[partition_length - 1][span_start]}
                target_rcs = {
                    r.symbol
                    for r, _ in result[span_length - partition_length - 1][
                        span_start + partition_length
                    ]
                }
                if not target_rbs or not target_rcs:
                    continue
                for rule in grammar.rules:
                    match rule:
                        case Rule(_, [rb, rc], _) if rb in target_rbs and rc in target_rcs:
                            result[span_length - 1][span_start].add((rule, partition_length))

    if any(r.symbol == grammar.start_symbol for r, _ in result[-1][0]):
        return result
    return None


def min_parse_weight(cyk_result: CykResult[T], start_symbol: T) -> int:
    @cache
    def _min_weight(span_length, span_start, symbol) -> int:
        result = None
        for rule, partition_length in cyk_result[span_length - 1][span_start]:
            if rule.symbol != symbol:
                continue
            if partition_length is None:
                result = rule.weight if result is None else min(result, rule.weight)
            else:
                left_weight = _min_weight(partition_length, span_start, rule.replacement[0])
                right_weight = _min_weight(
                    span_length - partition_length,
                    span_start + partition_length,
                    rule.replacement[1],
                )
                total_weight = rule.weight + left_weight + right_weight
                result = total_weight if result is None else min(result, total_weight)
        return result

    return _min_weight(len(cyk_result), 0, start_symbol)
