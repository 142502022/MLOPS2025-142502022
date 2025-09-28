def add(num1, num2):
    return num1 + num2

def sub(num1, num2):
    return num1 - num2

def multiply(num1, num2):
    return num1 * num2

def div(num1, num2):
    return num1 / num2

def main():
    print("Select operation:")
    print("1. Addition")
    print("2. Subtraction")
    print("3. Multiplication")
    print("4. Division")
    select = int(input("Enter choice (1/2/3/4): "))
    a = float(input("Enter first number: "))
    b = float(input("Enter second number: "))

    if select == 1:
        print("Result:", add(a, b))
    elif select == 2:
        print("Result:", sub(a, b))
    elif select == 3:
        print("Result:", multiply(a, b))
    elif select == 4:
        if b != 0:
            print("Result:", div(a, b))
        else:
            print("Error: Division by zero not allowed")
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()


