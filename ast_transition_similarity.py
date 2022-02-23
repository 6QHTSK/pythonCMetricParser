import numpy
from pycparser import c_ast, c_parser

import parser
from skimage.metrics import structural_similarity as ssim


class AstTransitionParser(c_ast.NodeVisitor, parser.Parser):
    @staticmethod
    def get_parser_name():
        return "transition"

    _nodeTypeName = ['ArrayDecl', 'ArrayRef', 'Assignment', 'Alignas', 'BinaryOp', 'Break', 'Case', 'Cast', 'Compound',
                     'CompoundLiteral', 'Constant', 'Continue', 'Decl', 'DeclList', 'Default', 'DoWhile',
                     'EllipsisParam', 'EmptyStatement', 'Enum', 'Enumerator', 'EnumeratorList', 'ExprList', 'FileAST',
                     'For', 'FuncCall', 'FuncDecl', 'FuncDef', 'Goto', 'ID', 'IdentifierType', 'If', 'InitList',
                     'Label', 'NamedInitializer', 'ParamList', 'PtrDecl', 'Return', 'StaticAssert', 'Struct',
                     'StructRef', 'Switch', 'TernaryOp', 'TypeDecl', 'Typedef', 'Typename', 'UnaryOp', 'Union', 'While',
                     'Pragma']

    def __init__(self, **kwargs):
        self.metrix = []
        self._nodeTypeLen = len(self._nodeTypeName)
        for i in range(0, self._nodeTypeLen):
            self.metrix.append([0] * self._nodeTypeLen)

    def add_edge(self, father, son):
        father_num = self._nodeTypeName.index(father)
        son_num = self._nodeTypeName.index(son)
        # Add Edge to father_node to son_node
        self.metrix[father_num][son_num] += 1

    def generic_visit(self, node, **kwargs):
        """ Visit a node.
            Since all the node was visited by generic_visit
            visit(node) is not used
        """
        node_name = type(node).__name__
        # Add an edge to its father
        father = kwargs.get("father", "")
        if father != "":
            # not the root node
            self.add_edge(father, node_name)

        for next_node in node:
            self.generic_visit(next_node, father=node_name)

    def get_probability_transition_matrix(self, multiplier=1):
        p_matrix = []
        for i in range(0, self._nodeTypeLen):
            p_metrix_row = [0] * self._nodeTypeLen
            row_sum = sum(self.metrix[i])
            if row_sum != 0:
                for j in range(0, self._nodeTypeLen):
                    p_metrix_row[j] = self.metrix[i][j] / row_sum * multiplier
            p_matrix.append(p_metrix_row)

        return p_matrix

    def parse(self, code):
        cpp_code = self.cpp(code)

        ast = c_parser.CParser().parse(text=cpp_code)
        self.generic_visit(ast)

    def get_array(self) -> numpy.ndarray:
        return numpy.array(self.get_probability_transition_matrix())

    @staticmethod
    def similarity(arr1, arr2) -> float:
        return ssim(arr1, arr2)
