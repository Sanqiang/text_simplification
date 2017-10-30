import copy as cp


class Solution:
    """
    @param: n: Given the range of numbers
    @param: k: Given the numbers of combinations
    @return: All the combinations of k numbers out of 1..n
    """

    def combine(self, n, k):
        # write your code here
        self.results = []
        for ni in range(1, n + 1):
            self.dfs(n, k, ni, 0, [])
        return self.results

    def dfs(self, n, k, ni, ki, cur):
        if ki == k:
            self.results.append(cp.deepcopy(cur))
            return
        for ni in range(ni, n + 1):
            cur.append(ni)
            self.dfs(n, k, ni + 1, ki + 1, cur)
            del cur[-1]

x = Solution().combine(4, 2)
print(x)