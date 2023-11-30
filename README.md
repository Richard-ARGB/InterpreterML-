# InterpreterML

## Description

This project implements a basic interpreter for a Machine Learning model definition language. It allows users to define models by specifying their name, family, type, description, as well as inputs, outputs, and layers that compose the model.

## Features

- **Lexer and Parser:** Uses PLY (Python Lex-Yacc) for lexical and syntactic analysis of the language defined for creating Machine Learning models.

- **Model Execution:** The `Model` class interprets the model definition and provides an `execute` method to print the model information, including inputs, outputs, and layers.

- **MIT License:** This project is distributed under the [MIT License](LICENSE).

## How to Use

1. Clone the repository:

    ```bash
    git clone https://github.com/your_username/InterpreterML.git
    cd InterpreterML
    ```

2. Run the interpreter:

    ```bash
    python interpreterML.py
    ```

3. Enter the model definition when prompted.

## Examples

Example model definitions are provided in the `test_cases.txt` file. You can run these test cases to verify the interpreter's functionality.


## License

This project is under the [MIT License](LICENSE).
