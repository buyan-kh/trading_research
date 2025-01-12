def max_profit(coins):
    m, n = len(coins), len(coins[0])
    max_coins = float('-inf')
    
    def dfs(i, j, neutralizations, current_coins):
        nonlocal max_coins
        
        if i == m-1 and j == n-1:
            max_coins = max(max_coins, current_coins)
            return
        
        if i >= m or j >= n:
            return
        
        if j + 1 < n:
            next_val = coins[i][j+1]
            if next_val >= 0:
                dfs(i, j+1, neutralizations, current_coins + next_val)
            else:
                dfs(i, j+1, neutralizations, current_coins + next_val)
                if neutralizations > 0:
                    dfs(i, j+1, neutralizations-1, current_coins + abs(next_val))
        
        if i + 1 < m:
            next_val = coins[i+1][j]
            if next_val >= 0:
                dfs(i+1, j, neutralizations, current_coins + next_val)
            else:
                dfs(i+1, j, neutralizations, current_coins + next_val)
                if neutralizations > 0:
                    dfs(i+1, j, neutralizations-1, current_coins + abs(next_val))
    
    initial = coins[0][0]
    dfs(0, 0, 2, initial)
    return max_coins

coins1 = [[0,1,-1],[1,-2,3],[2,-3,4]]
print(max_profit(coins1))

coins2 = [[10,10,10],[10,10,10]]
print(max_profit(coins2))
