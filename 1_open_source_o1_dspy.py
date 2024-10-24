import re
from dotenv import load_dotenv
from colorama import init, Fore, Style
import dspy

# Initialize colorama
init(autoreset=True)

# Load environment variables
load_dotenv()

# Constants for prompts
SYSTEM_PROMPT = """
You are a large reasoning model. You are given a question and you need
to reason step by step to find the answer. Do not provide the answer unless
you have reasoned through the question.
You can use Python code and write it in <python> and </python> tags.
The Python code could help you solve some of the steps.
Write it in a way that it can be parsed and executed using the exec() function.
Only use Python standard libraries.
Add explicit print statements so the output could be explained easily.
Only use Python code if it is absolutely necessary.
Be mutually exclusive and collectively exhaustive in your reasoning.
"""

STEPS_PROMPT = """
First, I need to think about the question and the thinking steps I need to take.
Put those steps into an opening <thinking> and closing </thinking> tag.
Each step should be in a <step> tag.
Example: steps to check if a number is prime:
<thinking>
    <step>Check if the number is greater than 1.</step>
    <step>Check if the number is divisible by any number other than 1 and itself.</step>
</thinking>
Do not provide an answer yet.
"""

ANSWER_PROMPT = """
Now I will use the thinking steps to reason and craft the complete final answer.
The answer should contain all the logical steps I took to arrive
at the answer.
Put the answer in an opening <answer> and closing </answer> tag.
Example: <answer>The number is prime because it is greater than 1 and
not divisible by any number other than 1 and itself.</answer>
"""


class Message:
    """Represents a message in the conversation."""

    def __init__(self, role, content):
        self.role = role
        self.content = content

    def to_dict(self):
        return {"role": self.role, "content": self.content}

    def truncate(self, max_tokens=200, mode: str = "start"):
        tokens = self.content.split()
        if len(tokens) > max_tokens:
            if mode == "start":
                self.content = " ".join(tokens[:max_tokens])
            elif mode == "end":
                self.content = " ".join(tokens[-max_tokens:])
            elif mode == "mix":
                self.content = " ".join(tokens[:max_tokens // 2] + tokens[-max_tokens // 2:])
            elif mode == "random":
                import random
                self.content = " ".join(random.sample(tokens, max_tokens))
            self.content += "..."
        return self


class Memory:
    """Manages the conversation history and context."""

    def __init__(self, context_length=200):
        self.context_length = int(context_length * 0.9)  # Apply a 10% buffer
        self.history = []

    def add_message(self, message: Message):
        """Adds a message to the conversation history and prints the updated history."""
        self.history.append(message.to_dict())
        self.pretty_print_history()
        self._update_context()

    def _update_context(self):
        total_length = sum(len(msg["content"].split()) for msg in self.history)
        if total_length <= self.context_length:
            return
        # Truncate messages from the start except for system messages
        new_history = []
        remaining_length = self.context_length
        for message in reversed(self.history):
            msg_length = len(message["content"].split())
            if message["role"] == "system" or remaining_length - msg_length >= 0:
                new_history.insert(0, message)
                remaining_length -= msg_length
            else:
                break
        self.history = new_history

    def pretty_print_history(self):
        """Prints the last message in the conversation history in a readable format with colors."""
        if not self.history:
            return  # No messages to print

        msg = self.history[-1]  # Get the last message
        role = msg['role']
        content = msg['content']
        if role == 'system':
            color = Fore.YELLOW + Style.BRIGHT
        elif role == 'user':
            color = Fore.GREEN + Style.BRIGHT
        elif role == 'assistant':
            color = Fore.CYAN + Style.BRIGHT
        else:
            color = Style.RESET_ALL

        print(f"{color}Role: {role}{Style.RESET_ALL}")
        print(f"{color}Content:\n{content}{Style.RESET_ALL}")
        print(f"{'-' * 100}\n")

    def clear(self):
        self.history = []


class LargeReasoningModel:
    """Simulates a large reasoning model that processes instructions step by step."""

    def __init__(self, model_name="openai/gpt-4o", context_length=128000):
        self.model_name = model_name
        self.context_length = context_length
        self.memory = Memory(context_length=self.context_length)
        self.client = self._initialize_client()
        self.memory.add_message(Message(role="system", content=SYSTEM_PROMPT))

    def _initialize_client(self):
        """Initializes the API client."""

        return dspy.LM(self.model_name)

    def generate_response(self, messages, temperature=0.2, logprobs=False):
        """Generates a response from the model."""
        completion = self.client(
            messages=messages,
            temperature=temperature,
            logprobs=logprobs
        )
        # completion is a list with one element, so we extract the first element
        return completion[0]

    @staticmethod
    def extract_python_code(response_content):
        """Extracts Python code enclosed in <python> tags."""
        pattern = re.compile(r'<python>(.*?)</python>', re.DOTALL)
        match = pattern.search(response_content)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def extract_steps(response_content):
        """Extracts Python code enclosed in <step> tags."""
        pattern = re.compile(r'<step>(.*?)</step>', re.DOTALL)
        match = pattern.search(response_content)
        if match:
            # Extract all steps and return as a list
            steps = pattern.findall(response_content)
            return steps
        return None

    @staticmethod
    def execute_python_code(code):
        """Executes Python code and returns the output."""
        try:
            # Preload necessary modules
            import io
            import contextlib
            import math
            import random

            # Create an isolated global environment for exec
            exec_globals = {
                '__builtins__': __import__('builtins'),  # Allow basic built-ins
                'math': math,  # Allow math functions
                'random': random,  # Allow random functions
            }

            local_vars = {}

            # Capture output
            with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                exec(code, exec_globals, local_vars)
                output = buf.getvalue()

            return f"The following Python code was executed successfully:\n{code}\nOutput:\n{output}"
        except ImportError as imp_err:
            return f"ImportError occurred: {str(imp_err)}"
        except Exception:
            import traceback
            return f"An error occurred while executing the code:\n{traceback.format_exc()}"

    def reasoning(self, instruction, temperature=0.2, logprobs=False, max_retries=5):
        """Performs step-by-step reasoning based on the instruction."""
        self.memory.add_message(Message(role="user", content=instruction))
        self.memory.add_message(Message(role="assistant", content=STEPS_PROMPT))

        # Attempt to parse the assistant's response
        for _ in range(max_retries):
            response = self.generate_response(self.memory.history, temperature=temperature, logprobs=logprobs)
            try:
                steps = self.extract_steps(response)
                break
            except Exception:
                continue
        else:
            raise ValueError("Failed to parse steps from the assistant's response.")

        self.memory.add_message(Message(role="assistant", content=response))
        max_retries = 3  # Define the maximum number of retries

        for i, step in enumerate(steps, 1):
            step_message = f"Let's do step {i}: {step}"
            self.memory.add_message(Message(role="assistant", content=step_message))
            response = self.generate_response(self.memory.history, temperature=temperature, logprobs=logprobs)
            self.memory.add_message(Message(role="assistant", content=response))

            # Extract and execute any Python code in the response
            python_code = self.extract_python_code(response)

            if python_code:
                retries = 0
                success = False

                while retries < max_retries and not success:
                    python_output = self.execute_python_code(python_code)

                    # Check for errors in the output
                    if "error" in python_output.lower():
                        # If error, add error message
                        error_message = f"An error occurred while executing step {i}: {python_output}"
                        self.memory.add_message(Message(role="assistant", content=error_message))

                        # Retry message
                        retry_message = f"Retrying step {i} to ensure no further errors (Attempt {retries + 1} of {max_retries})"
                        self.memory.add_message(Message(role="assistant", content=retry_message))
                        retries += 1
                    else:
                        # If successful, proceed
                        self.memory.add_message(Message(role="assistant", content=python_output))
                        success = True

                if not success:
                    final_error_message = f"Step {i} encountered errors after {max_retries} retries. Moving on to the next step."
                    self.memory.add_message(Message(role="assistant", content=final_error_message))

        # Ask for the final answer
        self.memory.add_message(Message(role="assistant", content=ANSWER_PROMPT))
        final_response = self.generate_response(self.memory.history, temperature=temperature, logprobs=logprobs)
        self.memory.add_message(Message(role="assistant", content=final_response))

        return final_response


if __name__ == "__main__":
    # Example usage
    model = LargeReasoningModel()
    # response = model.reasoning("Is 1253 a prime number?")
    response = model.reasoning("is 125849302233 a prime ?")
    print(f"{Fore.BLUE + Style.BRIGHT}Final Answer:{Style.RESET_ALL}")
    print(response)
