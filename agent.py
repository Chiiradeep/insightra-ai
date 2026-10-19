from google import genai

from tools import (
    calculate_average,
    calculate_sum,
    calculate_product,
    find_maximum,
    analyze_students,
    get_subject_average,
    get_top_student,
    get_dataset_summary,
    load_dataset
)


# Create Gemini client
client = genai.Client()


# Tools available to the agent
tools = [
    calculate_average,
    calculate_sum,
    calculate_product,
    find_maximum,
    analyze_students,
    get_subject_average,
    get_top_student,
    get_dataset_summary,
    load_dataset
]


# Create chat
chat = client.chats.create(
    model="gemini-3.6-flash",
    config={
        "tools": tools
    }
)


print("===================================")
print("     AI DATA ANALYST AGENT")
print("===================================")
print("Type 'exit' to stop.")


while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Agent: Goodbye!")
        break

    try:

        response = chat.send_message(user_input)

        print("\nAgent:", response.text)

    except Exception as e:

        print("\nError:", e)