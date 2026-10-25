from AdaptiveRouter import AdaptiveRouter

kws = [[1], [2, 3], [4], [5]]
r = AdaptiveRouter(kws, L=25)

assert r.check(0, []) is None                         # 25/4 = 6.25 > 5: plenty of room
assert r.check(4, [9] * 4) is None                    # 21/4 = 5.25 > 5
# 20/4 = 5.0 <= 5: escalate with ALL remaining keywords, k = (1+2+1+1) + 3 = 8
assert r.check(5, [9] * 5) == ([[1], [2, 3], [4], [5]], 8)
# generated has [1] and [2,3]; remaining [4],[5]: 10/2 = 5.0 -> escalate, k = (1+1) + 3 = 5
assert r.check(15, [1, 2, 3]) == ([[4], [5]], 5)
assert r.check(20, [1, 2, 3, 4, 5]) is None           # all satisfied
# the old "trap": 1 token left. Only keywords of length 1 can still fit ([2,3] cannot),
# so the router escalates with those three and k = min(3 + 3, 1) = 1
assert r.check(24, [9] * 24) == ([[1], [4], [5]], 1)

# the crash case: a 5-token keyword with only 3 tokens of room must be dropped
r2 = AdaptiveRouter([[1, 2, 3, 4, 5], [6]], L=25)
assert r2.check(22, [9] * 22) == ([[6]], 3)           # room 3: only [6] fits, k = min(1+3, 3)
r3 = AdaptiveRouter([[1, 2, 3, 4, 5]], L=25)
assert r3.check(22, [9] * 22) is None                 # nothing fits: stay with BOLT
print("router tests passed")