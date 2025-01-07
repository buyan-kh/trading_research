class Solution:
    def calculateScore(self, s: str) -> int:
        myarr = [[] for _ in range(26)]
        point = 0
        for idx, char in enumerate(s):
            mirror_index = 219 - ord(char) - 97  # Calculate the mirror index
            if myarr[mirror_index]:
                point += idx - myarr[mirror_index].pop()
            else:
                myarr[ord(char) - 97].append(idx)  # Correct the append method
        return point

# Example usage
solution = Solution()
s = "aczzx"
print(solution.calculateScore(s))  # Expected Output: 18