def FiscGPT():
    """
    Simple function that takes user input and returns it as a string.
    """
    # Get user input
    user_input = input("FiscGPT > ")
    
    # Return the input as is
    return user_input

if __name__ == "__main__":
    result = FiscGPT()
    print(f"Vous avez saisi: {result}") 