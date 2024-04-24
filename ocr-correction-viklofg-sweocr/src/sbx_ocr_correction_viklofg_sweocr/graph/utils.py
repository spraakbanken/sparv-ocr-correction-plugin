import re
from typing import List, Tuple, TypedDict

import diff_match_patch as dmp_module

dmp = dmp_module.diff_match_patch()

ENDING_WHITESPACE = re.compile(r"\s$")


def end_with_space(s: str) -> str:
    if not s:
        return s
    # print(f"{s[-1]=}")
    # print(f"{ENDING_WHITESPACE.fullmatch(s[-1])=}")
    return f"{s} " if (ENDING_WHITESPACE.fullmatch(s[-1]) is None) else s


def token_diff(s1: str, s2: str) -> List[Tuple[int, str]]:
    d = dmp.diff_main(s1, s2)
    dmp.diff_cleanupSemantic(d)
    return d


EditRange = TypedDict("EditRange", {"from": int, "to": int, "insert": str})


def edit_range(s0: str, s: str) -> EditRange:
    """
    >>> edit_range('0123456789', '0189')
    {from: 2, to: 8, insert: ''}

    edit_range('0123456789', '01') // => {from: 2, to: 10, insert: ''}
    edit_range('0123456789', '89') // => {from: 0, to: 8, insert: ''}
    edit_range('0123456789', '') // => {from: 0, to: 10, insert: ''}

    edit_range('0123456789', '01xyz89') // => {from: 2, to: 8, insert: 'xyz'}
    edit_range('0123456789', '01xyz') // => {from: 2, to: 10, insert: 'xyz'}
    edit_range('0123456789', 'xyz89') // => {from: 0, to: 8, insert: 'xyz'}
    edit_range('0123456789', 'xyz') // => {from: 0, to: 10, insert: 'xyz'}

    edit_range('', '01') // => {from: 0, to: 0, insert: '01'}
    """
    patches = token_diff(s0, s)  # noqa: F841
    # const patches = token_diff(s0, s)
    # const pre = R.takeWhile<[number, string]>(i => i[0] == 0, patches)
    # const post = R.takeLastWhile<[number, string]>(i => i[0] == 0, R.drop(pre.length, patches))  # noqa: E501
    # const from = pre.map(i => i[1]).join('').length
    # const postlen = post.map(i => i[1]).join('').length
    # const to = s0.length - postlen
    # const insert = s.slice(from, s.length - (s0.length - to))
    # return {from, to, insert}


def uniq(xs: List[str]) -> List[str]:
    used = set()
    return [x for x in xs if x not in used and (used.add(x) or True)]
