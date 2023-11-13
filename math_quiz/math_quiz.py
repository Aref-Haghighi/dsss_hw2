import random


def random_num_generation(min_num, max_num):
    """
    Generate Random integer.
    min_num: Minimum number.
    max_num: Max number.
    return: Random integer between min number and max number.
    """
    return random.randint(min_num, max_num)


def generate_operators():
    """
    Generating different operators.
    return: Choosing different random operators
    
    """
    return random.choice(['+', '-', '*'])


def implement_operators(num1, num2, operator):
    """
    Implement the operators

    num1: Operand 1
    num2: Operand 2
    operator: Operators(+, -, *)
    return:Plot of the all operation with its answer by 3 operators. 
    
    """
    plot = f"{num1} {operator} {num2}"
    if operator == '+': result = num1 + num2
    elif operator == '-': result = num1 - num2
    else: result = num1 * num2
    return plot, result

def math_quiz():
    score = 0
    total_questions = 3
    print("Welcome to the Math Quiz Game!")
    print("You will be presented with math problems, and you need to provide the correct answers.")

    for _ in range(total_questions):
        num1 = random_num_generation(1, 10); num2 = random_num_generation(1, 5); operator = generate_operators()

        PROBLEM, ANSWER = implement_operators(num1, num2, operator)
        print(f"\nQuestion: {PROBLEM}")

        try:
            user_answer = int(input("your answer: "))
        except ValueError:
            print("invalid input. Please enter right integer :)")
            continue

        if user_answer == ANSWER:
            print("Correct! You earned a point.")
            score += 1
        else:
            print(f"Wrong answer. The correct answer is {ANSWER}.")

    print(f"\nGame over! Your score is: {score}/{total_questions}")
if __name__ == "__main__":
    math_quiz()
