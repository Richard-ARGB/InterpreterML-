# Definiciones de la gramática:
# G = (NT, T, S, P)
#
# T = {TOKEN_MODEL_NAME, TOKEN_MODEL_FAMILY, TOKEN_MODEL_TYPE, TOKEN_MODEL_DESCRIPTION, TOKEN_INPUTS, TOKEN_INPUT, 
#      TOKEN_INPUT_NAME, TOKEN_INPUT_TYPE, TOKEN_OUTPUTS, TOKEN_OUTPUT, TOKEN_OUTPUT_NAME, TOKEN_OUTPUT_TYPE, 
#      TOKEN_MODEL, TOKEN_LAYERS, TOKEN_LAYER, TOKEN_LAYER_NAME, TOKEN_LAYER_PARAMS, TOKEN_COLON, TOKEN_LCURLY, 
#      TOKEN_RCURLY, TOKEN_LSQUARE, TOKEN_RSQUARE, TOKEN_COMMA, TOKEN_NUMBER, TOKEN_VARIABLE}
#
# NT = {start, model_section, inputs_section_opt, inputs_section, input_block, outputs_section_opt, outputs_section, 
#       output_block, model_layers_section_opt, model_layers_section, layers_block, layer_block, numbers_list}
#
# S = start
#
# P = 
#   start ::= model_section inputs_section_opt outputs_section_opt model_layers_section_opt
#   model_section ::= TOKEN_MODEL_NAME TOKEN_COLON TOKEN_VARIABLE TOKEN_MODEL_FAMILY TOKEN_COLON TOKEN_VARIABLE TOKEN_MODEL_TYPE TOKEN_COLON TOKEN_VARIABLE TOKEN_MODEL_DESCRIPTION TOKEN_COLON TOKEN_VARIABLE
#   inputs_section_opt ::= inputs_section | ε
#   inputs_section ::= TOKEN_INPUTS TOKEN_LCURLY input_block TOKEN_RCURLY
#   input_block ::= TOKEN_INPUT TOKEN_LCURLY TOKEN_INPUT_NAME TOKEN_COLON TOKEN_VARIABLE TOKEN_INPUT_TYPE TOKEN_COLON TOKEN_VARIABLE TOKEN_RCURLY
#   outputs_section_opt ::= outputs_section | ε
#   outputs_section ::= TOKEN_OUTPUTS TOKEN_LCURLY output_block TOKEN_RCURLY
#   output_block ::= TOKEN_OUTPUT TOKEN_LCURLY TOKEN_OUTPUT_NAME TOKEN_COLON TOKEN_VARIABLE TOKEN_OUTPUT_TYPE TOKEN_COLON TOKEN_VARIABLE TOKEN_RCURLY
#   model_layers_section_opt ::= model_layers_section | ε
#   model_layers_section ::= TOKEN_MODEL TOKEN_LCURLY layers_block TOKEN_RCURLY
#   layers_block ::= TOKEN_LAYERS TOKEN_LCURLY layer_block TOKEN_RCURLY
#   layer_block ::= TOKEN_LAYER TOKEN_LCURLY TOKEN_LAYER_NAME TOKEN_COLON TOKEN_VARIABLE TOKEN_LAYER_PARAMS TOKEN_COLON TOKEN_LSQUARE numbers_list TOKEN_RSQUARE TOKEN_RCURLY
#   numbers_list ::= TOKEN_NUMBER | numbers_list TOKEN_COMMA TOKEN_NUMBER

from ply import lex, yacc
import os
# Tokens
# Lista de nombres de tokens
reserved = {
    'Model_Name': 'TOKEN_MODEL_NAME',
    'Model_Family': 'TOKEN_MODEL_FAMILY',
    'Model_Type': 'TOKEN_MODEL_TYPE',
    'Model_Description': 'TOKEN_MODEL_DESCRIPTION',
    'Inputs': 'TOKEN_INPUTS',
    'Input': 'TOKEN_INPUT',
    'Input_Name': 'TOKEN_INPUT_NAME',
    'Input_Type': 'TOKEN_INPUT_TYPE',
    'Outputs': 'TOKEN_OUTPUTS',
    'Output': 'TOKEN_OUTPUT',
    'Output_Name': 'TOKEN_OUTPUT_NAME',
    'Output_Type': 'TOKEN_OUTPUT_TYPE',
    'Model': 'TOKEN_MODEL',
    'Layers': 'TOKEN_LAYERS',
    'Layer': 'TOKEN_LAYER',
    'Layer_Name': 'TOKEN_LAYER_NAME',
    'Layer_Params': 'TOKEN_LAYER_PARAMS'
}

# Tokens
tokens = list(reserved.values()) + [
    'TOKEN_COLON', 'TOKEN_LCURLY', 'TOKEN_RCURLY', 'TOKEN_LSQUARE', 'TOKEN_RSQUARE', 'TOKEN_COMMA', 'TOKEN_NUMBER', 'TOKEN_VARIABLE'
]

# Expresiones regulares para tokens simples
t_TOKEN_COLON = r':'
t_TOKEN_LCURLY = r'{'
t_TOKEN_RCURLY = r'}'
t_TOKEN_LSQUARE = r'\['
t_TOKEN_RSQUARE = r'\]'
t_TOKEN_COMMA = r','

# Expresión regular para TOKEN_NUMBER
def t_TOKEN_NUMBER(t):
    r'\d+(\.\d+)?'
    try:
        t.value = float(t.value)  # Intenta convertir a float
    except ValueError:
        t.value = int(t.value)  # Si falla, convierte a int
    return t

# Expresión regular para TOKEN_VARIABLE
def t_TOKEN_VARIABLE(t):
    r'[a-zA-Z0-9_ .]+'
    t.value = t.value.strip()
    t.type = reserved.get(t.value, 'TOKEN_VARIABLE')  # Usa 'TOKEN_VARIABLE' como valor predeterminado
    return t

def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

def t_error(t):
    error_message = f"TOKEN_ERROR: Illegal character '{t.value[0]}' at position {t.lexpos} - {t.lexpos + len(t.value)}"
    raise Exception(error_message)

t_ignore = ' \t'

# Build the lexer
lexer = lex.lex()

# Parsing rules
def p_start(p):
    '''start : model_section inputs_section_opt outputs_section_opt model_layers_section_opt'''
    try:
        p[0] = ('start', p[1], p[2], p[3], p[4])
    except Exception as e:
        raise SyntaxError(f"Error in start rule: {e}")

def p_model_section(p):
    '''model_section : TOKEN_MODEL_NAME TOKEN_COLON TOKEN_VARIABLE TOKEN_MODEL_FAMILY TOKEN_COLON TOKEN_VARIABLE TOKEN_MODEL_TYPE TOKEN_COLON TOKEN_VARIABLE TOKEN_MODEL_DESCRIPTION TOKEN_COLON TOKEN_VARIABLE'''
    try:
        p[0] = ('model_section', p[3], p[6], p[9], p[12])
    except Exception as e:
        raise SyntaxError(f"Error in model_section rule: {e}")

def p_inputs_section_opt(p):
    '''inputs_section_opt : inputs_section
                          | '''
    try:
        p[0] = ('inputs_section_opt', p[1]) if len(p) > 1 else ('inputs_section_opt', None)
    except Exception as e:
        raise SyntaxError(f"Error in inputs_section_opt rule: {e}")

def p_inputs_section(p):
    '''inputs_section : TOKEN_INPUTS TOKEN_LCURLY input_block TOKEN_RCURLY'''
    try:
        p[0] = ('inputs_section', p[3])
    except Exception as e:
        raise SyntaxError(f"Error in inputs_section rule: {e}")

def p_input_block(p):
    '''input_block : TOKEN_INPUT TOKEN_LCURLY TOKEN_INPUT_NAME TOKEN_COLON TOKEN_VARIABLE TOKEN_INPUT_TYPE TOKEN_COLON TOKEN_VARIABLE TOKEN_RCURLY'''
    try:   
        p[0] = {'name': p[5], 'type': p[8]}
    except Exception as e:
        raise SyntaxError(f"Error in input_block rule: {e}")

def p_outputs_section_opt(p):
    '''outputs_section_opt : outputs_section
                           | '''
    try:
        p[0] = ('outputs_section_opt', p[1]) if len(p) > 1 else ('outputs_section_opt', None)
    except Exception as e:
        raise SyntaxError(f"Error in outputs_section_opt rule: {e}")

def p_outputs_section(p):
    '''outputs_section : TOKEN_OUTPUTS TOKEN_LCURLY output_block TOKEN_RCURLY'''
    try:  
        p[0] = ('outputs_section', p[3])
    except Exception as e:
        raise SyntaxError(f"Error in outputs_section rule: {e}")

def p_output_block(p):
    '''output_block : TOKEN_OUTPUT TOKEN_LCURLY TOKEN_OUTPUT_NAME TOKEN_COLON TOKEN_VARIABLE TOKEN_OUTPUT_TYPE TOKEN_COLON TOKEN_VARIABLE TOKEN_RCURLY'''
    try:    
        p[0] = {'name': p[5], 'type': p[8]}
    except Exception as e:
        raise SyntaxError(f"Error in output_block rule: {e}")

def p_model_layers_section_opt(p):
    '''model_layers_section_opt : model_layers_section
                                | '''
    try:
        p[0] = ('model_layers_section_opt', p[1]) if len(p) > 1 else ('model_layers_section_opt', None)
    except Exception as e:
        raise SyntaxError(f"Error in model_layers_section_opt rule: {e}")

def p_model_layers_section(p):
    '''model_layers_section : TOKEN_MODEL TOKEN_LCURLY layers_block TOKEN_RCURLY'''
    try:
        p[0] = ('model_layers_section', p[3])
    except Exception as e:
        raise SyntaxError(f"Error in model_layers_section rule: {e}")

def p_layers_block(p):
    '''layers_block : TOKEN_LAYERS TOKEN_LCURLY layer_block TOKEN_RCURLY'''
    try:
        p[0] = ('layers_block', p[3])
    except Exception as e:
        raise SyntaxError(f"Error in layers_block rule: {e}")

def p_layer_block(p):
    '''layer_block : TOKEN_LAYER TOKEN_LCURLY TOKEN_LAYER_NAME TOKEN_COLON TOKEN_VARIABLE TOKEN_LAYER_PARAMS TOKEN_COLON TOKEN_LSQUARE numbers_list TOKEN_RSQUARE TOKEN_RCURLY'''
    try: 
        p[0] = {'name': p[5], 'params': p[10]}
    except Exception as e:
        raise SyntaxError(f"Error in layer_block rule: {e}")

def p_numbers_list(p):
    '''numbers_list : TOKEN_NUMBER
                        | numbers_list TOKEN_COMMA TOKEN_NUMBER'''
    try:
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[1].append(p[3])
            p[0] = p[1]
    except Exception as e:
        raise SyntaxError(f"Error in numbers_list rule: {e}")

def p_error(p):
    global current_test_success
    current_test_success = False
    if p:
        print(f"PARSER_ERROR: Syntax error at '{p.value}' at position {p.lexpos} - {p.lexpos + len(p.value)}")
    else:
        print(f"PARSER_ERROR: Syntax error at EOF")

# Build the parser
parser = yacc.yacc()

# Semantic classes
class Model:
    def __init__(self, model_name, model_family, model_type, model_description, inputs=None, outputs=None, layers=None):
        self.model_name = model_name
        self.model_family = model_family
        self.model_type = model_type
        self.model_description = model_description
        self.inputs = self.extract_blocks(inputs, 'TOKEN_INPUT') if inputs else None
        self.outputs = self.extract_blocks(outputs, 'TOKEN_OUTPUT') if outputs else None
        self.layers = self.extract_blocks(layers, 'TOKEN_LAYER') if layers else None

    def extract_blocks(self, section, block_type):
        if section and len(section) > 2:
            return [self.extract_block(block, block_type) for block in section[2]]

    def extract_block(self, block, block_type):
        return {
            'name': self.get_value(block, 'TOKEN_' + block_type + '_NAME'),
            'type': self.get_value(block, 'TOKEN_' + block_type + '_TYPE'),
            'params': self.get_value(block, 'TOKEN_' + block_type + '_PARAMS')
        }


    def execute(self):
        print(f"\nExecuting model: {self.model_name}")
        print(f"Model Family: {self.model_family}")
        print(f"Model Type: {self.model_type}")
        print(f"Model Description: {self.model_description}")

        if self.inputs:
            print("\nInputs:")
            for input_block in self.inputs:
                print(f"  Input Name: {input_block['name']}")
                print(f"  Input Type: {input_block['type']}")

        if self.outputs:
            print("\nOutputs:")
            for output_block in self.outputs:
                print(f"  Output Name: {output_block['name']}")
                print(f"  Output Type: {output_block['type']}")

        if self.layers:
            print("\nLayers:")
            for layer_block in self.layers:
                print(f"  Layer Name: {layer_block['name']}")
                print(f"  Layer Params: {layer_block['params']}")

# Function to extract the value from a block
def get_value(block, token_type):
    if block is None or block[0] != token_type:
        return None

    if len(block) > 1 and isinstance(block[1], tuple):
        return [get_value(inner_block, token_type) for inner_block in block[1]]
    else:
        return block[1]

def execute_model_from_test_case(test_case):
    result = parser.parse(test_case, lexer=lexer)
    if result:
        model_instance = Model(*result[1:])
        model_instance.execute()
        compare_blocks("Output", result[3][2], model_instance.outputs)
        compare_blocks("Layer", result[4][2][0][2], model_instance.layers)

        # Comparing output blocks
        if model_instance.outputs and len(model_instance.outputs) > 2:
            expected_outputs = [(get_value(output_block, 'TOKEN_OUTPUT_NAME'), get_value(output_block, 'TOKEN_OUTPUT_TYPE'))
                                for output_block in model_instance.outputs[2]]
            actual_outputs = [(get_value(output_block, 'TOKEN_OUTPUT_NAME'), get_value(output_block, 'TOKEN_OUTPUT_TYPE'))
                              for output_block in result[3][2]]

            compare_blocks("Output", expected_outputs, actual_outputs)

        # Comparing layer blocks
        if model_instance.layers and len(model_instance.layers) > 2:
            expected_layers = [(get_value(layer_block, 'TOKEN_LAYER_NAME'), get_value(layer_block, 'TOKEN_LAYER_PARAMS'))
                               for layer_block in model_instance.layers[2][0][2]]
            actual_layers = [(get_value(layer_block, 'TOKEN_LAYER_NAME'), get_value(layer_block, 'TOKEN_LAYER_PARAMS'))
                             for layer_block in result[4][2][0][2]]

            compare_blocks("Layer", expected_layers, actual_layers)
            
def compare_blocks(block_type, expected_blocks, actual_blocks):
    if len(expected_blocks) == len(actual_blocks):
        for i, (expected_name, expected_params) in enumerate(expected_blocks):
            actual_name, actual_params = actual_blocks[i]
            if expected_name == actual_name and expected_params == actual_params:
                print(f"{block_type} block {i + 1}: Equivalent")
            else:
                print(f"{block_type} block {i + 1}: Not equivalent")
    else:
        print(f"Number of {block_type.lower()} blocks is different. Cannot compare.")
                
def get_user_input(input_name):
    while True:
        try:
            user_input = float(input(f"Enter value for {input_name}: "))
            return user_input
        except ValueError:
            print("Please enter a valid number.")        
#Tokenizador
def test_lexer(data):
    lexer.input(data)
    while True:
        token = lexer.token()
        if not token:
            break
        token_end = token.lexpos + len(token.value) if isinstance(token.value, str) else token.lexpos + 1
        print(f"Tipo: {token.type}, Valor: {token.value}, Posicion Inicial: {token.lexpos}, Posicion Final: {token_end}")
def read_test_cases_from_file(file_path):
    with open(file_path, 'r') as file:
        test_cases = file.read().split('\n\n')
    return test_cases
# Modificar si se quiere hacer alguna otra prueba con la gramatica       
test_cases = [
    '''
    Model_Name: Cat or Dog Classifier
    Model_Family: CLASSIFIER
    Model_Type: ANN
    Model_Description: A model to classify dogs and cats.
    ''',
    '''
    Model_Name: Another Model
    Model_Family: REGRESSOR
    Model_Type: CNN
    Model_Description: A model for regression tasks.
    ''',
    '''
    Model_Name: Cat or Dog Classifier
    Model_Family: CLASSIFIER
    Model_Type: ANN
    Model_Description: A model to classify dogs and cats.
    Inputs{
    Input{
    Input_Name: Input1
    Input_Type: NUMBER
    }
    }
    ''',
    '''
    Model_Name: Cat or Dog Classifier
    Model_Family: CLASSIFIER
    Model_Type: ANN
    Model_Description: A model to classify dogs and cats.
    Inputs{
    Input{
    Input_Name: Input1
    Input_Type: NUMBER
    }
    }
    Outputs{
    Output{
    Output_Name: Output1
    Output_Type: CATEGORICAL
    }
    }
    Model{
    Layers{
        Layer{
        Layer_Name: Layer1
        Layer_Params: [6,8.9,7.8]
        }
    }
    }
'''
]

for i, test_case in enumerate(test_cases):
    print(f"\nCaso de Prueba {i+1}:")
    test_lexer(test_case)
        
# Function to execute the model for each test case
def execute_model_from_test_case(test_case):
    result = parser.parse(test_case, lexer=lexer)
    if result:
        model_instance = Model(*result[1:])
        model_instance.execute()
                
# Test cases
parse_success = []

       
# Execute lexer, parser, and model for each test case
for i, test_case in enumerate(test_cases):
    print(f"\nExecuting Test Case {i + 1}:")
    
    # Lexer
    test_lexer(test_case)
    
    # Parser
    try:
        result = parser.parse(test_case, lexer=lexer)
        parse_success.append(True)
        print("Parser passed successfully.")
    except Exception as e:
        parse_success.append(False)
        print(f"Parser failed: {e}")

    # Model execution
    if parse_success[-1]:
        execute_model_from_test_case(test_case)
        print("Model executed successfully.")
    else:
        print("Model execution skipped due to parser failure.")

if all(parse_success):
    print("\nAll test cases passed successfully.")
else:
    for index, success in enumerate(parse_success):
        if not success:
            print(f"\nTest case {index + 1} did not pass.")
       
# Ejecutar el modelo a partir del test case ingresado por el usuario
# Obtén la ruta del directorio actual del script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Construye la ruta completa al archivo de casos de prueba
file_path = os.path.join(current_directory, 'test_cases.txt')

# Imprime la ruta para verificar
print(f"File path: {file_path}")
test_cases = read_test_cases_from_file(file_path)
parse_success = []

# Execute lexer, parser, and model for each test case
for i, test_case in enumerate(test_cases):
    print(f"\nExecuting Test Case {i + 1} from File:")
    
    # Lexer
    test_lexer(test_case)
    
    # Parser
    try:
        result = parser.parse(test_case, lexer=lexer)
        parse_success.append(True)
        print("Parser passed successfully.")
    except Exception as e:
        parse_success.append(False)
        print(f"Parser failed: {e}")

    # Model execution
    if parse_success[-1]:
        execute_model_from_test_case(test_case)
        print("Model executed successfully.")
    else:
        print("Model execution skipped due to parser failure.")




