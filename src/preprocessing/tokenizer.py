import re
from typing import List

class SourceCodeTokenizer:
    """
    A source-code-aware tokenizer for C/C++ source code.
    It isolates identifiers, control keywords, operators, string literals, 
    and numeric literals while safely removing line and block comments.
    """
    def __init__(self) -> None:
        # Regex components:
        # Group 1: Comments (single-line // or block /* ... */)
        # Group 2: Double-quoted string literals (e.g. "hello\n")
        # Group 3: Single-quoted character literals (e.g. 'c')
        # Group 4: Numeric constants (hexadecimal, decimals, floats)
        # Group 5: Multi-character and standard C/C++ operators
        # Group 6: Word tokens (keywords, function names, variable names, APIs)
        # Group 7: Single punctuations / brackets
        self.pattern = re.compile(
            r'(/\*[\s\S]*?\*/|//.*)|'              # Comments
            r'("(?:\\.|[^\\"])*")|'                # String literals
            r'(\'(?:\\.|[^\\\'])\')|'              # Character literals
            r'(0[xX][0-9a-fA-F]+|\d+(?:\.\d+)?)|'  # Numbers
            r'(->|\+\+|--|<<|>>|<=|>=|==|!=|&&|\|\||[+\-*/%&|^!=<>]=?)|'  # Multi-char operators
            r'(\w+)|'                              # Word characters
            r'([{}()\[\].;,?~:])'                  # Punctuations and Brackets
        )

    def tokenize(self, code_str: str) -> List[str]:
        """
        Tokenize a string of C/C++ source code.
        
        Args:
            code_str (str): The raw C/C++ code function.
            
        Returns:
            List[str]: List of extracted tokens.
        """
        if not isinstance(code_str, str) or not code_str.strip():
            return []
            
        tokens = []
        for match in self.pattern.finditer(code_str):
            # If Group 1 matches (comments), we skip it to remove them.
            if match.group(1):
                continue
            
            # Find the active matching group (Groups 2 through 7)
            for group_idx in range(2, 8):
                matched_val = match.group(group_idx)
                if matched_val is not None:
                    tokens.append(matched_val)
                    break
                    
        return tokens
