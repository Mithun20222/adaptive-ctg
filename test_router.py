from AdaptiveRouter import AdaptiveRouter

kws = [[1], [2, 3], [4], [5]]
r = AdaptiveRouter(kws, L=25)

assert r.check(0, []) is None                       # 25/4 = 6.25 > 5: plenty of room
assert r.check(4, [9] * 4) is None                  # 21/4 = 5.25 > 5
assert r.check(5, [9] * 5) == ([1], 4)              # 20/4 = 5.0 <= 5: escalate, k = 1+3
assert r.check(15, [1, 2, 3]) == ([4], 4)           # 2 remaining: 10/2 = 5, next missing is [4]
assert r.check(20, [1, 2, 3, 4, 5]) is None         # all satisfied
assert r.check(24, [9] * 24) is None                # 1 token left, 4 keywords: nothing fits... see below
print("router tests passed")