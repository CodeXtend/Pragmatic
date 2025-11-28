from fact_extracter import FactExtracter


def main():
    """CLI interface for the FactExtracter."""
    extracter = FactExtracter()
    
    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        response = extracter.run(user_input)
        print("Agent:", response)


if __name__ == "__main__":
    main()