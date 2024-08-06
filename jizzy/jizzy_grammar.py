from __future__ import annotations

from jizzy.grammar import Terminal, NonTerminal, Rule, GrammarMeta
from jizzy.engine import Engine

from functools import cache


class Symbols(metaclass=GrammarMeta[Engine]):
    ENGINE = Engine

    IF = Terminal(pattern=r"if")
    ELSE = Terminal(pattern=r"else")
    FOR = Terminal(pattern=r"for")
    IN = Terminal(pattern=r"in")
    WHILE = Terminal(pattern=r"while")
    STRUCT = Terminal(pattern=r"struct")
    RETURN = Terminal(pattern=r"return")

    DOUBLECOLON = Terminal(pattern=r"::")
    SEMICOLON = Terminal(pattern=r";")
    COLON = Terminal(pattern=r":")
    DOT = Terminal(pattern=r"\.")
    COMMA = Terminal(pattern=r",")

    OP = Terminal(pattern=r"\(")
    CP = Terminal(pattern=r"\)")
    OB = Terminal(pattern=r"\[")
    CB = Terminal(pattern=r"\]")
    OC = Terminal(pattern=r"\{")
    CC = Terminal(pattern=r"\}")

    ASS_IGN = Terminal(pattern=r"=")
    ASS_ADD = Terminal(pattern=r"\+=")
    ASS_SUB = Terminal(pattern=r"-=")
    ASS_MUL = Terminal(pattern=r"\*=")
    ASS_DIV = Terminal(pattern=r"/=")
    ASS_SHL = Terminal(pattern=r"<<=")
    ASS_SHR = Terminal(pattern=r">>=")

    BIT_LEFT = Terminal(pattern=r"<<")
    BIT_RIGHT = Terminal(pattern=r">>")

    CMP_IE = Terminal(pattern=r"<=>")

    CMP_LT = Terminal(pattern=r"<")
    CMP_GT = Terminal(pattern=r">")
    CMP_LE = Terminal(pattern=r"<=")
    CMP_GE = Terminal(pattern=r">=")
    CMP_EQ = Terminal(pattern=r"==")
    CMP_NE = Terminal(pattern=r"!=")

    BIT_AND = Terminal(pattern=r"&")
    BIT_XOR = Terminal(pattern=r"\^")
    BIT_OR = Terminal(pattern=r"\|")

    LOG_AND = Terminal(pattern=r"&&")
    LOG_OR = Terminal(pattern=r"\|\|")

    MATH_ADD = Terminal(pattern=r"\+")
    MATH_SUB = Terminal(pattern=r"-")

    MATH_MUL = Terminal(pattern=r"\*")
    MATH_DIV = Terminal(pattern=r"/")
    MATH_MOD = Terminal(pattern=r"%")

    DEC = Terminal(pattern=r"--")
    INC = Terminal(pattern=r"\+\+")

    IDENTIFIER = Terminal(pattern=r"\w+")

    LIT_S8 = Terminal(pattern=r"u8\"([^\\\"]|\\.)*\"")
    LIT_S16 = Terminal(pattern=r"u16\"([^\\\"]|\\.)*\"")
    LIT_S32 = Terminal(pattern=r"u32\"([^\\\"]|\\.)*\"")

    LIT_C8 = Terminal(pattern=r"u8\'([^\\\']|\\.)*\'")
    LIT_C16 = Terminal(pattern=r"u16\'([^\\\']|\\.)*\'")
    LIT_C32 = Terminal(pattern=r"u32\'([^\\\']|\\.)*\'")

    LIT_I8 = Terminal(pattern=r"\d+i8")
    LIT_I16 = Terminal(pattern=r"\d+i16")
    LIT_I32 = Terminal(pattern=r"\d+i32")
    LIT_I64 = Terminal(pattern=r"\d+i64")

    LIT_U8 = Terminal(pattern=r"\d+u8")
    LIT_U16 = Terminal(pattern=r"\d+u16")
    LIT_U32 = Terminal(pattern=r"\d+u32")
    LIT_U64 = Terminal(pattern=r"\d+u64")

    LIT_F32 = Terminal(pattern=r"(\d+.\d+|\d+.|.\d+|\d+)f32")
    LIT_F64 = Terminal(pattern=r"(\d+.\d+|\d+.|.\d+|\d+)f64")

    program = NonTerminal()
    expression_list = NonTerminal()
    expression = NonTerminal()
    assignment = NonTerminal()
    mapping = NonTerminal()
    bit_shift = NonTerminal()
    spaceship = NonTerminal()
    comparison = NonTerminal()
    equality = NonTerminal()
    bit_and = NonTerminal()
    bit_xor = NonTerminal()
    bit_or = NonTerminal()
    log_and = NonTerminal()
    log_or = NonTerminal()
    sum = NonTerminal()
    product = NonTerminal()
    paren_block = NonTerminal()
    brace_block = NonTerminal()
    curly_block = NonTerminal()
    base_expression = NonTerminal()

    @classmethod
    @cache
    def start(cls) -> NonTerminal:
        return cls.program

    @classmethod
    @cache
    def rules(cls) -> list[Rule]:
        return [
            Rule(
                callback=cls.ENGINE.finalize_program,
                lhs=cls.program,
                rhs=[*cls.expression_list]
            ),

            Rule(
                callback=cls.ENGINE.expand_expression_list,
                lhs=cls.expression_list,
                rhs=[*cls.expression_list, *cls.expression]
            ),
            Rule(
                callback=cls.ENGINE.make_expression_list,
                lhs=cls.expression_list,
                rhs=[*cls.expression]
            ),

            # Top level expression
            Rule(
                callback=cls.ENGINE.make_statement,
                lhs=cls.expression,
                rhs=[*cls.assignment, *cls.SEMICOLON]
            ),
            Rule(
                callback=cls.ENGINE.make_statement,
                lhs=cls.expression,
                rhs=[*cls.assignment, *cls.COMMA]
            ),
            Rule(
                callback=cls.ENGINE.make_statement,
                lhs=cls.expression,
                rhs=[*cls.assignment]
            ),

            # Assignment operators ::= += -= /= *= >>= <<=
            Rule(
                callback=cls.ENGINE.make_binary_expression,
                lhs=cls.assignment,
                rhs=[*cls.expression, *cls.ASS_IGN, *cls.mapping]
            ),
            Rule(
                callback=cls.ENGINE.make_binary_expression,
                lhs=cls.assignment,
                rhs=[*cls.expression, *cls.ASS_ADD, *cls.mapping]
            ),
            Rule(
                callback=cls.ENGINE.make_binary_expression,
                lhs=cls.assignment,
                rhs=[*cls.expression, *cls.ASS_SUB, *cls.mapping]
            ),
            Rule(
                callback=cls.ENGINE.make_binary_expression,
                lhs=cls.assignment,
                rhs=[*cls.expression, *cls.ASS_MUL, *cls.mapping]
            ),
            Rule(
                callback=cls.ENGINE.make_binary_expression,
                lhs=cls.assignment,
                rhs=[*cls.expression, *cls.ASS_DIV, *cls.mapping]
            ),
            Rule(
                callback=cls.ENGINE.make_binary_expression,
                lhs=cls.assignment,
                rhs=[*cls.expression, *cls.ASS_SHL, *cls.mapping]
            ),
            Rule(
                callback=cls.ENGINE.make_binary_expression,
                lhs=cls.assignment,
                rhs=[*cls.expression, *cls.ASS_SHR, *cls.mapping]
            ),
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.assignment,
                rhs=[*cls.mapping]
            ),

            # Mapping operator :
            Rule(
                callback=cls.ENGINE.make_binary_expression,
                lhs=cls.mapping,
                rhs=[*cls.mapping, *cls.COLON, *cls.bit_shift]
            ),
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.mapping,
                rhs=[*cls.bit_shift]
            ),

            # Bitwise << >>
            Rule(
                callback=cls.ENGINE.make_binary_expression,
                lhs=cls.bit_shift,
                rhs=[*cls.bit_shift, *cls.BIT_LEFT, *cls.spaceship]
            ),
            Rule(
                callback=cls.ENGINE.make_binary_expression,
                lhs=cls.bit_shift,
                rhs=[*cls.bit_shift, *cls.BIT_RIGHT, *cls.spaceship]
            ),
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.bit_shift,
                rhs=[*cls.spaceship]
            ),

            # Spaceship <->
            Rule(
                callback=cls.ENGINE.make_binary_expression,
                lhs=cls.spaceship,
                rhs=[*cls.spaceship, *cls.CMP_IE, *cls.comparison]
            ),
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.spaceship,
                rhs=[*cls.comparison]
            ),

            # Normal comparison < > <= >=
            Rule(
                callback=cls.ENGINE.make_binary_expression,
                lhs=cls.comparison,
                rhs=[*cls.comparison, *cls.CMP_LT, *cls.equality]
            ),
            Rule(
                callback=cls.ENGINE.make_binary_expression,
                lhs=cls.comparison,
                rhs=[*cls.comparison, *cls.CMP_GT, *cls.equality]
            ),
            Rule(
                callback=cls.ENGINE.make_binary_expression,
                lhs=cls.comparison,
                rhs=[*cls.comparison, *cls.CMP_LE, *cls.equality]
            ),
            Rule(
                callback=cls.ENGINE.make_binary_expression,
                lhs=cls.comparison,
                rhs=[*cls.comparison, *cls.CMP_GE, *cls.equality]
            ),
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.comparison,
                rhs=[*cls.equality]
            ),

            # Equality ::== !=
            Rule(
                callback=cls.ENGINE.make_binary_expression,
                lhs=cls.equality,
                rhs=[*cls.equality, *cls.CMP_EQ, *cls.bit_and]
            ),
            Rule(
                callback=cls.ENGINE.make_binary_expression,
                lhs=cls.equality,
                rhs=[*cls.equality, *cls.CMP_NE, *cls.bit_and]
            ),
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.equality,
                rhs=[*cls.bit_and]
            ),

            # Bitwise &
            Rule(
                callback=cls.ENGINE.make_binary_expression,
                lhs=cls.bit_and,
                rhs=[*cls.bit_and, *cls.BIT_AND, *cls.bit_xor]
            ),
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.bit_and,
                rhs=[*cls.bit_xor]
            ),

            # Bitwise ^
            Rule(
                callback=cls.ENGINE.make_binary_expression,
                lhs=cls.bit_xor,
                rhs=[*cls.bit_xor, *cls.BIT_XOR, *cls.bit_or]
            ),
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.bit_xor,
                rhs=[*cls.bit_or]
            ),

            # Bitwise |
            Rule(
                callback=cls.ENGINE.make_binary_expression,
                lhs=cls.bit_or,
                rhs=[*cls.bit_or, *cls.BIT_OR, *cls.log_and]
            ),
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.bit_or,
                rhs=[*cls.log_and]
            ),

            # Logical &&
            Rule(
                callback=cls.ENGINE.make_binary_expression,
                lhs=cls.log_and,
                rhs=[*cls.log_and, *cls.LOG_AND, *cls.log_or]
            ),
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.log_and,
                rhs=[*cls.log_or]
            ),

            # Logical ||
            Rule(
                callback=cls.ENGINE.make_binary_expression,
                lhs=cls.log_or,
                rhs=[*cls.log_or, *cls.LOG_OR, *cls.sum]
            ),
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.log_or,
                rhs=[*cls.sum]
            ),

            # Arithmetic + -
            Rule(
                callback=cls.ENGINE.make_binary_expression,
                lhs=cls.sum,
                rhs=[*cls.sum, *cls.MATH_ADD, *cls.product]
            ),
            Rule(
                callback=cls.ENGINE.make_binary_expression,
                lhs=cls.sum,
                rhs=[*cls.sum, *cls.MATH_SUB, *cls.product]
            ),
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.sum,
                rhs=[*cls.product]
            ),

            # Arithmetic * / %
            Rule(
                callback=cls.ENGINE.make_binary_expression,
                lhs=cls.product,
                rhs=[*cls.product, *cls.MATH_MUL, *cls.base_expression]
            ),
            Rule(
                callback=cls.ENGINE.make_binary_expression,
                lhs=cls.product,
                rhs=[
                    *cls.product, *cls.MATH_DIV, *cls.base_expression]
            ),
            Rule(
                callback=cls.ENGINE.make_binary_expression,
                lhs=cls.product,
                rhs=[*cls.product, *cls.MATH_MOD, *cls.base_expression]
            ),
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.product,
                rhs=[*cls.base_expression]
            ),

            Rule(
                callback=cls.ENGINE.make_paren_block,
                lhs=cls.paren_block,
                rhs=[cls.OP, *cls.expression_list, cls.CP]
            ),
            Rule(
                callback=cls.ENGINE.make_paren_block,
                lhs=cls.paren_block,
                rhs=[cls.OP, cls.CP]
            ),

            Rule(
                callback=cls.ENGINE.make_brace_block,
                lhs=cls.brace_block,
                rhs=[cls.OB, *cls.expression_list, cls.CB]
            ),
            Rule(
                callback=cls.ENGINE.make_brace_block,
                lhs=cls.brace_block,
                rhs=[cls.OB, cls.CB]
            ),

            Rule(
                callback=cls.ENGINE.make_curly_block,
                lhs=cls.curly_block,
                rhs=[cls.OC, *cls.expression_list, cls.CC]
            ),
            Rule(
                callback=cls.ENGINE.make_curly_block,
                lhs=cls.curly_block,
                rhs=[cls.OC, cls.CC]
            ),

            # Keywords
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.base_expression,
                rhs=[*cls.IF]
            ),
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.base_expression,
                rhs=[*cls.ELSE]
            ),
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.base_expression,
                rhs=[*cls.FOR]
            ),
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.base_expression,
                rhs=[*cls.IN]
            ),
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.base_expression,
                rhs=[*cls.WHILE]
            ),
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.base_expression,
                rhs=[*cls.STRUCT]
            ),
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.base_expression,
                rhs=[*cls.RETURN]
            ),

            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.base_expression,
                rhs=[*cls.IDENTIFIER]
            ),

            # Unicode string literals
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.base_expression,
                rhs=[*cls.LIT_S8]
            ),
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.base_expression,
                rhs=[*cls.LIT_S16]
            ),
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.base_expression,
                rhs=[*cls.LIT_S32]
            ),

            # Unicode char literals
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.base_expression,
                rhs=[*cls.LIT_C8]
            ),
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.base_expression,
                rhs=[*cls.LIT_C16]
            ),
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.base_expression,
                rhs=[*cls.LIT_C32]
            ),

            # Signed integer literals
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.base_expression,
                rhs=[*cls.LIT_I8]
            ),
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.base_expression,
                rhs=[*cls.LIT_I16]
            ),
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.base_expression,
                rhs=[*cls.LIT_I32]
            ),
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.base_expression,
                rhs=[*cls.LIT_I64]
            ),

            # Unsigned integer literals
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.base_expression,
                rhs=[*cls.LIT_U8]
            ),
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.base_expression,
                rhs=[*cls.LIT_U16]
            ),
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.base_expression,
                rhs=[*cls.LIT_U32]
            ),
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.base_expression,
                rhs=[*cls.LIT_U64]
            ),

            # Floating point literals
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.base_expression,
                rhs=[*cls.LIT_F32]
            ),
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.base_expression,
                rhs=[*cls.LIT_F64]
            ),
            # Unary * & + -
            Rule(
                callback=cls.ENGINE.make_prefix_expression,
                lhs=cls.base_expression,
                rhs=[*cls.BIT_AND, *cls.base_expression]
            ),
            Rule(
                callback=cls.ENGINE.make_prefix_expression,
                lhs=cls.base_expression,
                rhs=[*cls.MATH_ADD, *cls.base_expression]
            ),
            Rule(
                callback=cls.ENGINE.make_prefix_expression,
                lhs=cls.base_expression,
                rhs=[*cls.MATH_SUB, *cls.base_expression]
            ),
            Rule(
                callback=cls.ENGINE.make_prefix_expression,
                lhs=cls.base_expression,
                rhs=[*cls.MATH_MUL, *cls.base_expression]
            ),
            # Post and pre increment/decrement
            Rule(
                callback=cls.ENGINE.make_prefix_expression,
                lhs=cls.base_expression,
                rhs=[*cls.INC, *cls.base_expression]
            ),
            Rule(
                callback=cls.ENGINE.make_prefix_expression,
                lhs=cls.base_expression,
                rhs=[*cls.DEC, *cls.base_expression]
            ),
            Rule(
                callback=cls.ENGINE.make_postfix_expression,
                lhs=cls.base_expression,
                rhs=[*cls.base_expression, *cls.INC]
            ),
            Rule(
                callback=cls.ENGINE.make_postfix_expression,
                lhs=cls.base_expression,

                rhs=[*cls.base_expression, *cls.DEC]
            ),

            Rule(
                callback=cls.ENGINE.make_binary_expression,
                lhs=cls.base_expression,
                rhs=[*cls.base_expression, *cls.DOT, *cls.IDENTIFIER]
            ),
            Rule(
                callback=cls.ENGINE.make_binary_expression,
                lhs=cls.base_expression,
                rhs=[*cls.base_expression, *cls.DOUBLECOLON, *cls.IDENTIFIER]
            ),
            # Function call stuff () [] {}
            Rule(
                callback=cls.ENGINE.make_block_expression,
                lhs=cls.base_expression,
                rhs=[*cls.base_expression, *cls.paren_block]
            ),
            Rule(
                callback=cls.ENGINE.make_block_expression,
                lhs=cls.base_expression,
                rhs=[*cls.base_expression, *cls.brace_block]
            ),
            Rule(
                callback=cls.ENGINE.make_block_expression,
                lhs=cls.base_expression,
                rhs=[*cls.base_expression, *cls.curly_block]
            ),
            # Parenthesized expression ()
            Rule(
                callback=cls.ENGINE.identity,
                lhs=cls.base_expression,
                rhs=[cls.OP, *cls.expression, cls.CP]
            )
        ]
