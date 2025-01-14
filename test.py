from collections import defaultdict
import sys

def dfs(graph, node, parent, values, subtree_sum, start, end):
    # Initialize subtree sum with current node's value
    subtree_sum[node] = values[node-1]
    start[node] = end[0]
    
    # Process all children
    for child in graph[node]:
        if child != parent:
            end[0] += 1
            dfs(graph, child, node, values, subtree_sum, start, end)
            subtree_sum[node] += subtree_sum[child]
    
    end[node] = end[0]

def main():
    # Read input
    n, q = map(int, input().split())
    values = list(map(int, input().split()))
    
    # Create adjacency list representation of tree
    graph = defaultdict(list)
    for _ in range(n-1):
        a, b = map(int, input().split())
        graph[a].append(b)
        graph[b].append(a)
    
    # Arrays to store subtree information
    subtree_sum = [0] * (n + 1)
    start = [0] * (n + 1)
    end = [0]
    
    # Process the tree using DFS
    dfs(graph, 1, 0, values, subtree_sum, start, end)
    
    # Process queries
    for _ in range(q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            # Update value
            _, s, x = query
            diff = x - values[s-1]
            values[s-1] = x
            # Update all ancestors
            curr = s
            while curr > 0:
                subtree_sum[curr] += diff
                curr = next((p for p in graph[curr] if start[p] < start[curr] < end[p]), 0)
        else:
            # Query subtree sum
            _, s = query
            print(subtree_sum[s])

if __name__ == "__main__":
    main()