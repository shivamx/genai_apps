import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

def find_movies(description: str, location: str = ""):
    """find movie titles currently playing in theaters based on any description, genre, title words, etc.

    Args:
        description: Any kind of description including category or genre, title words, attributes, etc.
        location: The city and state, e.g. San Francisco, CA or a zip code e.g. 95616
    """
    return ["Barbie", "Oppenheimer"]


def find_theaters(location: str, movie: str = ""):
    """Find theaters based on location and optionally movie title which are is currently playing in theaters.

    Args:
        location: The city and state, e.g. San Francisco, CA or a zip code e.g. 95616
        movie: Any movie title
    """
    return ["Googleplex 16", "Android Theatre"]


def get_showtimes(location: str, movie: str, theater: str, date: str):
    """
    Find the start times for movies playing in a specific theater.

    Args:
      location: The city and state, e.g. San Francisco, CA or a zip code e.g. 95616
      movie: Any movie title
      thearer: Name of the theater
      date: Date for requested showtime
    """
    return ["10:00", "11:00"]



functions = {
    "find_movies": find_movies,
    "find_theaters": find_theaters,
    "get_showtimes": get_showtimes,
}


def call_function(function_call, functions):
    function_name = function_call.name
    function_args = function_call.args

    return functions[function_name](**function_args)



model = genai.GenerativeModel(
    model_name="gemini-1.0-pro", tools=[functions.values()]
)


response = model.generate_content(
    "Which theaters in Mountain View show the Barbie movie?"
)
""" response.candidates[0].content.parts """


part = response.candidates[0].content.parts[0]

# Check if it's a function call; in real use you'd need to also handle text responses as you won't know what the model will respond with.
if part.function_call:
    result = call_function(part.function_call, functions)

print(result)



""" Finally, pass the response plus the message history to the next generate_content() call to get a final text response from the model.
For implementation :
https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Function_calling.ipynb?authuser=1#scrollTo=gdb62GstAD_3 """