import unittest
from math_quiz import random_num_generation, generate_operators, implement_operators
class TestMathGame(unittest.TestCase):

    def test_random_num_generation(self):
        # Test if random numbers generated are within the specified range
        min_val = 1
        max_val = 10
        for _ in range(1000):  # Test a large number of random values
            rand_num = random_num_generation(min_val, max_val)
            self.assertTrue(min_val <= rand_num <= max_val)

    def generate_operators(self):
         for _ in range(1000):  # Test a large number of random values
            operator = generate_operators()
            self.assertTrue(operator in ['+', '-', '*'])
        

    def implement_operators(self):
            test_cases = [
                (5, 2, '+', '5 + 2', 7),
                (9, 3, '-', '9 - 3', 4),
                (7, 7, '*', '7 * 7', 49),
            ]
            for num1, num2, operator, expected_problem, expected_answer in test_cases:
                problem, answer = implement_operators(num1, num2, operator)
                self.assertEqual(problem, expected_problem)
                self.assertEqual(answer, expected_answer)
                

if __name__ == "__main__":
    unittest.main()
