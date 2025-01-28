from dreamcoder.program import *
from dreamcoder.type import arrow, baseType, tint, tlist, t0, t1, t2, tboolean
from dreamcoder.task import Task
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive

from scipy.ndimage import binary_dilation, label
from scipy import ndimage
from scipy.ndimage import measurements
from math import sin,cos,radians,copysign
import numpy as np
from collections import deque
from statistics import mode

from typing import Tuple, NewType, List, Callable, Dict, Type, FrozenSet, Union, Container

tgrid = baseType("grid")
tsize = baseType("size")
Size = NewType("Size", Tuple[Tuple[int]])

class Grid():
    """
    Represents an ARC grid, along with a position.
    For an unshifted grid (e.g. the input), the position is (0,0).
    Position holds the canonical index of the top-left grid pixel.
    This encompasses the Input, Grid and Object types from Alford.

    Instantiation with cutout=True will remove any background.
    """

    def __init__(self, grid: np.ndarray, position: Tuple[int, int]=(0, 0), cutout=False):
        self.position = position

        if grid.shape[0] > 30 or grid.shape[1] > 30:
            raise PrimitiveException(f"Grid size {grid.shape} too large")

        self.cutout = cutout
        if cutout:
            self.grid, (xpos, ypos) = Grid.cutout(grid)
            self.position = (self.position[0] + xpos, self.position[1] + ypos)
        else:
            self.grid = grid

    @property
    def size(self) -> Size:
        return self.grid.shape
    
    @classmethod
    def cutout(grid: np.ndarray) -> np.ndarray:
        xr, yr = np.nonzero(grid)
        xpos, ypos = min(xr), min(yr)
        return grid[xpos:max(xr)+1, ypos:max(yr)+1], (xpos, ypos)
    
    def newgrid(self, grid: np.ndarray, offset=None, cutout=False) -> "Grid":
        """
        Return a new Grid object containing a new userprovided grid.
        The position is cloned from the parent.
        """
        position = self.position
        if offset:
            position = (position[0] + offset[0], position[1] + offset[1])

        if grid.shape[0] > 30 or grid.shape[1] > 30:
            raise PrimitiveException(f"Grid size {grid.shape} too large")

        return Grid(grid, position, cutout)
    
    def count(self) -> int:
        """
        Returns the number of non-zero elements in the grid
        """
        return np.count_nonzero(self.grid)

    def __eq__(self, other) -> bool:
        """
        Score a grid. Returns True iff the two grids are equal, ignoring position
        """
        if isinstance(other, Grid):
            return self.size == other.size and (self.grid == other.grid).all()
        return False

    def __repr__(self):
        return f"Grid({self.grid.shape[0]}x{self.grid.shape[1]} at {self.position})"





tboolean = baseType("boolean")
Boolean = NewType("Boolean", bool)
tinteger = baseType("integer")
Integer = NewType("Integer", int)
tcord = baseType("cord")
Cord = NewType("Cord", Tuple[Integer, Integer])

tcontainer = baseType("container")

tfrozenset = baseType("frozenset")

ttuple = baseType("tuple")

tcallable = baseType("callable")

tcontainercontainer = baseType("containercontainer")
ContainerContainer = NewType("ContainerContainer", Container[Container])

tintegerset = baseType("integerset")
IntegerSet = NewType("IntegerSet", FrozenSet[Integer])

tdslGrid = baseType("dslGrid")
dslGrid = NewType("dslGrid", Tuple[Tuple[int]])

ttupletuple = baseType("tupletuple")
TupleTuple = NewType("TupleTuple", Tuple[Tuple])

tcell = baseType("cell")
Cell = NewType("Cell", Tuple[Integer, Cord])
#Cell = Tuple[Integer, Cord]

tobject = baseType("object")
Object = NewType("Object", FrozenSet[Cell])
#Object = FrozenSet[Cell]

tindices = baseType("indices")
Indices = NewType("Indices", FrozenSet[Cord])
#Indices = FrozenSet[Cord]

tindicesset = baseType("indicesset")
IndicesSet = NewType("IndicesSet", FrozenSet[Indices])

tobjects = baseType("objects")
Objects = NewType("Objects", FrozenSet[Object])

typemap: Dict[Type, TypeConstructor] = {
    Size: tsize,
    Boolean: tboolean,
    Container: tcontainer,
    FrozenSet: tfrozenset,
    Tuple: ttuple,
    Callable: tcallable,
    ContainerContainer: tcontainercontainer,
    IntegerSet: tintegerset,
    TupleTuple: ttupletuple,
    IndicesSet: tindicesset,
    Indices: tindices,
    Objects: tobjects,
    Object: tobject,
    Cell: tcell,
    Grid: tgrid,
    dslGrid: tdslGrid,
    Integer: tinteger,
    Boolean: tboolean,
    Cord: tcord
}

def primitive_assert(boolean, message=None):
    """
    Raise a PrimitiveException if the condition is false.
    This stops execution on the current program and does not raise an error.
    """
    if not boolean:
        raise PrimitiveException(message)

import inspect, typing

class DSL:
    def __init__(self, typemap: Dict[Type, TypeConstructor], verbose=False):
        self.typemap = typemap
        self.primitives = {}
        self.verbose = verbose

    def cvt_type(self, anno):
        # Handle list types natively
        # These annotations have type typing._GenericAlias
        # __origin__ attr is list, __args__ attr is a tuple of constituent types
        if hasattr(anno, '__origin__'):
            if anno.__origin__ == list:
                # We recursively convert the constituent
                return tlist(self.cvt_type(anno.__args__[0]))
        
        if anno in self.typemap:
            return self.typemap[anno]
        
        raise TypeError(f"Annotation {anno} has no corresponding DreamCoder type")

    def register(self, f: Callable, name: str=None, typesig: List[TypeConstructor]=None, autocurry: bool=True):
        if not isinstance(f, typing.Callable):
            # This is a value, not a function
            if len(typesig) != 1:
                raise TypeError('Value passed to Primitive constructor, typesig must be of length 1')
            if name is None:
                raise ValueError('Value passed to Primitive constructor, name must be specified')
            dc_type = typesig[-1]
            primitive = Primitive(name, dc_type, f)
            primitive.typesig = typesig # Allow later reflection
            self.primitives[name] = primitive
            print(primitive.value)
            if self.verbose:
                print(f"Registered value {name} of type {dc_type}.")

            return

        if name is None:
            name = f.__name__
            if name == '<lambda>':
                raise ValueError('<lambda> passed to Primitive constructor, name must be specified')
            
        fn_sig = inspect.signature(f)
        params = list(fn_sig.parameters.items())
        param_count = len(params)
        if typesig is None:
            # Generate a DreamCoder type signature for this function by inspection
            arrow_args = []

            for arg, argtype in params:
                anno = argtype.annotation
                arrow_args.append(self.cvt_type(anno))

            typesig = arrow_args + [self.cvt_type(fn_sig.return_annotation)    ]

        dc_type = arrow(*typesig)

        # This function has more than 1 input and needs to be curried
        # We have special cases for 2/3 params because these are significantly faster
        if autocurry and param_count > 1:
            if param_count == 2:
                func = lambda x: lambda y: f(x, y)
            elif param_count == 3:
                func = lambda x: lambda y: lambda z: f(x, y, z)
            else:
                def curry(f, n, args):
                    if n:
                        return lambda x: curry(f, n-1, args + [x])
                    return f(*args)
                func = curry(f, param_count, [])
        else:
            func = f

        if self.verbose:
            print(f"Registered {name} with inferred type {dc_type}.")

        primitive = Primitive(name, dc_type, func)
        primitive.typesig = typesig # Allow later reflection
        self.primitives[name] = primitive
    
    def registerMany(self, funcs: List[Callable]):
        for func in funcs:
            try:
                self.register(func)
            except KeyboardInterrupt:
                raise
            except:
                print(f"Error occured on {f}")
                raise

    # Decorator function to define a primitive
    def primitive(self, func: Callable=None, name: str=None, typesig: List[TypeConstructor]=None, autocurry: bool=True):
        # First, we define a decorator factory
        def decorator(func):
            self.register(func, name, typesig, autocurry)
            return func
        
        # If we are called as a decorator factory, return the decorator
        if func is None:
            return decorator
        else:
            return decorator(func)

    @staticmethod
    def parse_primitive_names(ocaml_contents):
        contents = ''.join([c[:-1] for c in ocaml_contents if c[0:2] + c[-3:-1] != '(**)'])
        contents = contents.split('primitive "')[1:]
        primitives = [p[:p.index('"')] for p in contents if '"' in p]
        return primitives

    def generate_ocaml_primitives(self):
        primitives = list(self.primitives.values())

        with open("solvers/program.ml", "r") as f:
            contents = f.readlines()

        start_ix = min([i for i in range(len(contents)) if contents[i][0:7] == '(* AUTO'])
        end_ix = min([i for i in range(len(contents)) if contents[i][0:11] == '(* END AUTO'])

        non_auto_contents = contents[0:start_ix+1] + contents[end_ix:]
        # get the existing primitive names. We won't auto-create any primitives
        # whose name matches an existing name.
        existing_primitives = self.parse_primitive_names(non_auto_contents)

        lines = [p.ocaml_string() + '\n' for p in primitives
                if p.name not in existing_primitives]

        for p in primitives:
            if p.name in existing_primitives:
                print('Primitive {} already exists, skipping ocaml code generation for it'.format(p.name))

        contents = contents[0:start_ix+1] + lines + contents[end_ix:]

        with open("solvers/program.ml", "w") as f:
            f.write(''.join(contents))

dsl = DSL(typemap, verbose=False)

### DSL ###

def gridToTT(x:Grid) -> dslGrid:
    grid = x.grid
    result = tuple(map(tuple, grid))
    return result

def check_ragged(x:dslGrid) -> Boolean:
    t = len(x[0])
    for row in x:
        if len(row) != t:
            return True
    return False

def ttToGrid(x:dslGrid) -> Grid:
    if check_ragged(x):
        primitive_assert(False, "This is not of the right shape for Grid")
    return Grid(np.array(x))


dsl.register((0,0), "ORIGIN", [tcord])
dsl.register((2,2), "TWO_BY_TWO", [tcord])
dsl.register((3,3), "THREE_BY_THREE", [tcord])
dsl.register(0, "ZERO", [tinteger])
dsl.register(1, "ONE", [tinteger])
dsl.register(2, "TWO", [tinteger])
dsl.register(3, "THREE", [tinteger])
dsl.register(4, "FOUR", [tinteger])
dsl.register(5, "FIVE", [tinteger])
dsl.register(6, "SIX", [tinteger])
dsl.register(7, "SEVEN", [tinteger])
dsl.register(8, "EIGHT", [tinteger])
dsl.register(9, "NINE", [tinteger])
dsl.register(10, "TEN", [tinteger])

dsl.register(True, "T", [tboolean])
dsl.register(False, "F", [tboolean])

dsl.register(-1, "NEG_ONE", [tinteger])
dsl.register(-2, "NEG_TWO", [tinteger])

dsl.register((1,0), "DOWN", [tcord])
dsl.register((0,1), "RIGHT", [tcord])
dsl.register((-1,0), "UP", [tcord])
dsl.register((0,-1), "LEFT", [tcord])


dsl.register((1,1), "UNITY", [tcord])
dsl.register((-1,-1), "NEG_UNITY", [tcord])
dsl.register((-1,1), "UP_RIGHT", [tcord])
dsl.register((1,-1), "DOWN_LEFT", [tcord])

dsl.register((0,2), "ZERO_BY_TWO", [tcord])
dsl.register((2,0), "TWO_BY_ZERO", [tcord])

@dsl.primitive
def vmirrordslGrid(
    piece: Grid
) -> Grid:
    """ mirroring along vertical """
    temp = gridToTT(piece)
    piece = temp
    return ttToGrid(tuple(row[::-1] for row in piece))

@dsl.primitive
def vmirrorObject(
    piece: Object
) -> Object:
    """ mirroring along vertical """
    d = ulcornerObject(piece)[1] + lrcornerObject(piece)[1]
    return frozenset((v, (i, d - j)) for v, (i, j) in piece)

@dsl.primitive
def vmirrorIndices(
    piece: Indices
) -> Indices:
    """ mirroring along vertical """
    d = ulcornerIndices(piece)[1] + lrcornerIndices(piece)[1]
    return frozenset((i, d - j) for i, j in piece)

@dsl.primitive
def ulcornerObject(
    patch: Object
) -> Cord:
    """ index of upper left corner """
    return tuple(map(min, zip(*toindicesObject(patch))))

@dsl.primitive
def ulcornerIndices(
    patch: Indices
) -> Cord:
    """ index of upper left corner """
    return tuple(map(min, zip(*toindicesIndices(patch))))

@dsl.primitive
def lrcornerObject(
    patch: Object
) -> Cord:
    """ index of lower right corner """
    return tuple(map(max, zip(*toindicesObject(patch))))

@dsl.primitive
def lrcornerIndices(
    patch: Indices
) -> Cord:
    """ index of lower right corner """
    return tuple(map(max, zip(*toindicesIndices(patch))))

@dsl.primitive
def toindicesObject(
    patch: Object
) -> Indices:
    """ indices of object cells """
    if len(patch) == 0:
        return frozenset()
    return frozenset(index for value, index in patch)

@dsl.primitive
def toindicesIndices(
    patch: Indices
) -> Indices:
    """ indices of object cells """
    if len(patch) == 0:
        return frozenset()
    return patch

@dsl.primitive
def hmirrordslGrid(
    piece: Grid
) -> Grid:
    """ mirroring along horizontal """
    temp = gridToTT(piece)
    piece = temp
    return ttToGrid(piece[::-1])
    
@dsl.primitive
def hmirrorObject(
    piece: Object
) -> Object:
    """ mirroring along horizontal """
    d = ulcornerObject(piece)[0] + lrcornerObject(piece)[0]
    return frozenset((v, (d - i, j)) for v, (i, j) in piece)

@dsl.primitive
def hmirrorIndices(
    piece: Indices
) -> Indices:
    """ mirroring along horizontal """
    d = ulcornerIndices(piece)[0] + lrcornerIndices(piece)[0]
    return frozenset((d - i, j) for i, j in piece)

@dsl.primitive
def dmirrordslGrid(
    piece: Grid
) -> Grid:
    temp = gridToTT(piece)
    piece = temp
    """ mirroring along diagonal """
    if isinstance(piece, tuple):
        return ttToGrid(tuple(zip(*piece)))

@dsl.primitive
def dmirrorObject(
    piece: Object
) -> Object:
    """ mirroring along diagonal """
    if isinstance(piece, tuple):
        return tuple(zip(*piece))
    a, b = ulcornerObject(piece)
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (j - b + a, i - a + b)) for v, (i, j) in piece)
    return frozenset((j - b + a, i - a + b) for i, j in piece)

@dsl.primitive
def dmirrorIndices(
    piece: Indices
) -> Indices:
    """ mirroring along diagonal """
    if isinstance(piece, tuple):
        return tuple(zip(*piece))
    a, b = ulcornerIndices(piece)
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (j - b + a, i - a + b)) for v, (i, j) in piece)
    return frozenset((j - b + a, i - a + b) for i, j in piece)

@dsl.primitive
def rot180(
    grid: Grid
) -> Grid:
    """ half rotation """
    temp = gridToTT(grid)
    grid = temp
    return ttToGrid(tuple(tuple(row[::-1]) for row in grid[::-1]))

@dsl.primitive
def upscaledslGrid(
    element: Grid,
    factor: Integer
) -> Grid:
    """ upscale object or grid """
    temp = gridToTT(element)
    element = temp
    g = tuple()
    for row in element:
        upscaled_row = tuple()
        for value in row:
            upscaled_row = upscaled_row + tuple(value for num in range(factor))
        g = g + tuple(upscaled_row for num in range(factor))
    return ttToGrid(g)

@dsl.primitive
def upscaleObject(
    element: Object,
    factor: Integer
) -> Object:
    """ upscale object or grid """
    if len(element) == 0:
        return frozenset()
    di_inv, dj_inv = ulcornerObject(element)
    di, dj = (-di_inv, -dj_inv)
    normed_obj = shiftObject(element, (di, dj))
    o = set()
    for value, (i, j) in normed_obj:
        for io in range(factor):
            for jo in range(factor):
                o.add((value, (i * factor + io, j * factor + jo)))
    return shiftObject(frozenset(o), (di_inv, dj_inv))
  
@dsl.primitive
def shiftObject(
    patch: Object,
    directions: Cord
) -> Object:
    """ shift patch """
    if len(patch) == 0:
        return patch
    di, dj = directions
    return frozenset((value, (i + di, j + dj)) for value, (i, j) in patch)

@dsl.primitive
def hconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids horizontally """
    x = gridToTT(a)
    a = x
    y = gridToTT(b)
    b = y
    return ttToGrid(tuple(i + j for i, j in zip(a, b)))

@dsl.primitive
def replace(
    grid: Grid,
    replacee: Integer,
    replacer: Integer
) -> Grid:
    """ color substitution """
    temp = gridToTT(grid)
    grid = temp
    return ttToGrid(tuple(tuple(replacer if v == replacee else v for v in r) for r in grid))

@dsl.primitive
def crop(
    grid: Grid,
    start: Cord,
    dims: Cord
) -> Grid:
    """ subgrid specified by start and dimension """
    temp = gridToTT(grid)
    grid = temp
    return ttToGrid(Tuple(r[start[1]:start[1]+dims[1]] for r in grid[start[0]:start[0]+dims[0]]))

@dsl.primitive
def switch(
    grid: Grid,
    a: Integer,
    b: Integer
) -> Grid:
    """ color switching """
    temp = gridToTT(grid)
    grid = temp
    return ttToGrid(tuple(tuple(v if (v != a and v != b) else {a: b, b: a}[v] for v in r) for r in grid))

@dsl.primitive
def rot270(
    grid: Grid
) -> Grid:
    """ quarter anticlockwise rotation """
    temp = gridToTT(grid)
    grid = temp
    return ttToGrid(tuple(tuple(row[::-1]) for row in zip(*grid[::-1]))[::-1])

@dsl.primitive
def vconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids vertically """
    x = gridToTT(a)
    a = x
    y = gridToTT(b)
    b = y
    return ttToGrid(a + b)

@dsl.primitive
def downscale(
    grid: Grid,
    factor: Integer
) -> Grid:
    temp = gridToTT(grid)
    grid = temp
    if factor == 0: #added to prevent division by zero errors
        return grid
    """ downscale grid """
    h, w = len(grid), len(grid[0])
    g = tuple()
    for i in range(h):
        r = tuple()
        for j in range(w):
            if j % factor == 0:
                r = r + (grid[i][j],)
        g = g + (r, )
    h = len(g)
    dsg = tuple()
    for i in range(h):
        if i % factor == 0:
            dsg = dsg + (g[i],)
    return ttToGrid(dsg)

@dsl.primitive
def tojvec(
    j: Integer
) -> Cord:
    """ vector pointing horizontally """
    return (0, j)

@dsl.primitive
def mostcolorObject(
    element: Object
) -> Integer:
    """ most common color """
    values = [v for r in element for v in r]
    return max(set(values), key=values.count)

@dsl.primitive
def mostcolordslGrid(
    element: Grid
) -> Integer:
    """ most common color """
    temp = gridToTT(element)
    element = temp
    values = [v for v, _ in element]
    return max(set(values), key=values.count)

@dsl.primitive
def canvas(
    value: Integer,
    dimensions: Cord
) -> Grid:
    """ grid construction """
    return ttToGrid(tuple(tuple(value for j in range(dimensions[1])) for i in range(dimensions[0])))

@dsl.primitive
def hsplit(
    grid: Grid,
    n: Integer
) -> Tuple:
    temp = gridToTT(grid)
    grid = temp
    if n == 0:
        primitive_assert(False, 'Function not designed for 0, n')
    """ split grid horizontally """
    h, w = len(grid), len(grid[0]) // n
    offset = len(grid[0]) % n != 0
    return tuple(crop(ttToGrid(grid), (0, w * i + i * offset), (h, w)) for i in range(n))

@dsl.primitive
def firstTuple(
    container: Container
) -> Tuple:
    """ first item of container """
    return next(iter(container))

@dsl.primitive
def firstInt(
    container: Container
) -> Integer:
    """ first item of container """
    return next(iter(container))

@dsl.primitive
def firstBool(
    container: Container
) -> Boolean:
    """ first item of container """
    return next(iter(container))

@dsl.primitive
def firstCord(
    container: Container
) -> Cord:
    """ first item of container """
    return next(iter(container))

@dsl.primitive
def firstContainer(
    container: Container
) -> Container:
    """ first item of container """
    return next(iter(container))

@dsl.primitive
def firstFrozenSet(
    container: Container
) -> FrozenSet:
    """ first item of container """
    return next(iter(container))

@dsl.primitive
def firstCallable(
    container: Container
) -> Callable:
    """ first item of container """
    return next(iter(container))

@dsl.primitive
def firstCC(
    container: Container
) -> ContainerContainer:
    """ first item of container """
    return next(iter(container))

@dsl.primitive
def firstIntegerSet(
    container: Container
) -> IntegerSet:
    """ first item of container """
    return next(iter(container))

@dsl.primitive
def firstTT(
    container: Container
) -> TupleTuple:
    """ first item of container """
    return next(iter(container))

@dsl.primitive
def firstCell(
    container: Container
) -> Cell:
    """ first item of container """
    return next(iter(container))

@dsl.primitive
def firstObject(
    container: Container
) -> Object:
    """ first item of container """
    return next(iter(container))

@dsl.primitive
def firstIndices(
    container: Container
) -> Indices:
    """ first item of container """
    return next(iter(container))

@dsl.primitive
def firstdslGrid(
    container: Container
) -> Grid:
    """ first item of container """
    if isinstance(next(iter(container)), Grid):
        return next(iter(container))
    return ttToGrid(next(iter(container)))

@dsl.primitive
def firstIndicesSet(
    container: Container
) -> IndicesSet:
    """ first item of container """
    return next(iter(container))

@dsl.primitive
def firstObjects(
    container: Container
) -> Objects:
    """ first item of container """
    return next(iter(container))




print(f"Registered {len(dsl.primitives)} total primitives.")

p = dsl


### comment everything down ###


'''
@dsl.primitive
def dmirrordslGrid(
    piece: Grid
) -> Grid:
    temp = gridToTT(piece)
    piece = temp
    """ mirroring along diagonal """
    if isinstance(piece, tuple):
        return ttToGrid(tuple(zip(*piece)))

@dsl.primitive
def dmirrorObject(
    piece: Object
) -> Object:
    """ mirroring along diagonal """
    if isinstance(piece, tuple):
        return tuple(zip(*piece))
    a, b = ulcornerObject(piece)
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (j - b + a, i - a + b)) for v, (i, j) in piece)
    return frozenset((j - b + a, i - a + b) for i, j in piece)

@dsl.primitive
def dmirrorIndices(
    piece: Indices
) -> Indices:
    """ mirroring along diagonal """
    if isinstance(piece, tuple):
        return tuple(zip(*piece))
    a, b = ulcornerIndices(piece)
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (j - b + a, i - a + b)) for v, (i, j) in piece)
    return frozenset((j - b + a, i - a + b) for i, j in piece)

@dsl.primitive
def identityCord(
    x: Cord
) -> Cord:
    """ identity function """
    return x

@dsl.primitive
def identityInt(
    x: Integer
) -> Integer:
    """ identity function """
    return x

@dsl.primitive
def identityBool(
    x: Boolean
) -> Boolean:
    """ identity function """
    return x

@dsl.primitive
def identityContainer(
    x: Container
) -> Container:
    """ identity function """
    return x

@dsl.primitive
def identityFrozen(
    x: FrozenSet
) -> FrozenSet:
    """ identity function """
    return x

@dsl.primitive
def identityTuple(
    x: Tuple
) -> Tuple:
    """ identity function """
    return x

@dsl.primitive
def identityCallable(
    x: Callable
) -> Callable:
    """ identity function """
    return x

@dsl.primitive
def identityCC(
    x: ContainerContainer
) -> ContainerContainer:
    """ identity function """
    return x

@dsl.primitive
def identityIntegerSet(
    x: IntegerSet
) -> IntegerSet:
    """ identity function """
    return x

@dsl.primitive
def identityTT(
    x: TupleTuple
) -> TupleTuple:
    """ identity function """
    return x

@dsl.primitive
def identityCell(
    x: Cell
) -> Cell:
    """ identity function """
    return x

@dsl.primitive
def identityObject(
    x: Object
) -> Object:
    """ identity function """
    return x

@dsl.primitive
def identityIndices(
    x: Indices
) -> Indices:
    """ identity function """
    return x

@dsl.primitive
def identitydslGrid(
    x: Grid
) -> Grid:
    """ identity function """
    return x

@dsl.primitive
def identityIndicesSet(
    x: IndicesSet
) -> IndicesSet:
    """ identity function """
    return x

@dsl.primitive
def identityObjects(
    x: Objects
) -> Objects:
    """ identity function """
    return x

@dsl.primitive
def addInt(
    a: int,
    b: int
) -> int:
    """ addition """
    return a + b

@dsl.primitive
def addCordInt(
    a: Cord,
    b: int
) -> Cord:
    """ addition """
    return (a[0] + b, a[1] + b)

@dsl.primitive
def addIntCord(
    a: int,
    b: Cord
) -> Cord:
    """ addition """
    return (a + b[0], a + b[1])

@dsl.primitive
def addCord(
    a: Cord,
    b: Cord
) -> Cord:
    """ addition """
    return (a[0] + b[0], a[1] + b[1])

@dsl.primitive
def subtractInt(
    a: int,
    b: int
) -> int:
    """ subtraction """
    return a - b

@dsl.primitive
def subtractCord(
    a: Cord,
    b: Cord
) -> Cord:
    """ subtraction """
    return (a[0] - b[0], a[1] - b[1])

@dsl.primitive
def subtractIntCord(
    a: int,
    b: Cord
) -> Cord:
    """ subtraction """
    return (a - b[0], a - b[1])

@dsl.primitive
def subtractCordInt(
    a: Cord,
    b: int
) -> Cord:
    """ subtraction """
    return (a[0] - b, a[1] - b)

@dsl.primitive
def multiplyInt(
    a: int,
    b: int
) -> int:
    """ multiplication """
    return a * b

@dsl.primitive
def multiplyCord(
    a: Cord,
    b: Cord
) -> Cord:
    """ multiplication """
    return (a[0] * b[0], a[1] * b[1])

@dsl.primitive
def multiplyIntCord(
    a: int,
    b: Cord
) -> Cord:
    """ multiplication """
    return (a * b[0], a * b[1])

@dsl.primitive
def multiplyCordInt(
    a: Cord,
    b: int
) -> Cord:
    """ multiplication """
    return (a[0] * b, a[1] * b)
    
@dsl.primitive
def divideInt(
    a: int,
    b: int
) -> int:
    """ floor division """
    if b == 0:
        primitive_assert(False, 'Function not designed for 0, b')
    return a // b

@dsl.primitive
def divideCord(
    a: Cord,
    b: Cord
) -> Cord:
    """ floor division """
    return (a[0] // b[0], a[1] // b[1])

@dsl.primitive
def divideIntCord(
    a: int,
    b: Cord
) -> Cord:
    """ floor division """
    return (a // b[0], a // b[1])

@dsl.primitive
def divideCordInt(
    a: Cord,
    b: int
) -> Cord:
    if b == 0:
        primitive_assert(False, 'Function not designed for 0, b')
    """ floor division """
    return (a[0] // b, a[1] // b)

@dsl.primitive
def invertInt(
    n: int
) -> int:
    """ inversion with respect to addition """
    return -n

@dsl.primitive
def invertCord(
    n: Cord
) -> Cord:
    """ inversion with respect to addition """
    return (-n[0], -n[1])

@dsl.primitive
def even(
    n: Integer
) -> Boolean:
    """ evenness """
    return n % 2 == 0

@dsl.primitive
def doubleInt(
    n: int
) -> int:
    """ scaling by two """
    # why? already have multi
    return n * 2

@dsl.primitive
def doubleCord(
    n: Cord
) -> Cord:
    """ scaling by two """
    # why? already have multi
    return (n[0] * 2, n[1] * 2)

@dsl.primitive
def halveInt(
    n: int
) -> int:
    """ scaling by one half """
    # why? already have div
    return n // 2

@dsl.primitive
def halveCord(
    n: Cord
) -> Cord:
    """ scaling by one half """
    # why? already have div
    return (n[0] // 2, n[1] // 2)

@dsl.primitive
def flip(
    b: Boolean
) -> Boolean:
    """ logical not """
    return not b

@dsl.primitive
def equalityInt(
    a: int,
    b: int
) -> Boolean:
    """ equality """
    return a == b

@dsl.primitive
def equalityBool(
    a: Boolean,
    b: Boolean
) -> Boolean:
    """ equality """
    return a == b

@dsl.primitive
def equalityCord(
    a: Cord,
    b: Cord
) -> Boolean:
    """ equality """
    return a == b

@dsl.primitive
def equalityContainer(
    a: Container,
    b: Container
) -> Boolean:
    """ equality """
    return a == b

@dsl.primitive
def equalityFrozenSet(
    a: FrozenSet,
    b: FrozenSet
) -> Boolean:
    """ equality """
    return a == b

@dsl.primitive
def equalityTuple(
    a: Tuple,
    b: Tuple
) -> Boolean:
    """ equality """
    return a == b

@dsl.primitive
def equalityCallable(
    a: Callable,
    b: Callable
) -> Boolean:
    """ equality """
    return a == b

@dsl.primitive
def equalityCC(
    a: ContainerContainer,
    b: ContainerContainer
) -> Boolean:
    """ equality """
    return a == b

@dsl.primitive
def equalityIntegerSet(
    a: IntegerSet,
    b: IntegerSet
) -> Boolean:
    """ equality """
    return a == b

@dsl.primitive
def equalityTT(
    a: TupleTuple,
    b: TupleTuple
) -> Boolean:
    """ equality """
    return a == b

@dsl.primitive
def equalityCell(
    a: Cell,
    b: Cell
) -> Boolean:
    """ equality """
    return a == b

@dsl.primitive
def equalityObject(
    a: Object,
    b: Object
) -> Boolean:
    """ equality """
    return a == b

@dsl.primitive
def equalityIndices(
    a: Indices,
    b: Indices
) -> Boolean:
    """ equality """
    return a == b

@dsl.primitive
def equalitydslGrid(
    a: Grid,
    b: Grid
) -> Boolean:
    """ equality """
    x = gridToTT(a)
    y = gridToTT(b)
    return x == y

@dsl.primitive
def equalityIndicesSet(
    a: IndicesSet,
    b: IndicesSet
) -> Boolean:
    """ equality """
    return a == b

@dsl.primitive
def equalityObjects(
    a: Objects,
    b: Objects
) -> Boolean:
    """ equality """
    return a == b

@dsl.primitive
def containedInt(
    value: int,
    container: Container
) -> Boolean:
    """ element of """
    return value in container

@dsl.primitive
def containedBool(
    value: Boolean,
    container: Container
) -> Boolean:
    """ element of """
    return value in container

@dsl.primitive
def containedCord(
    value: Tuple[int ,int],
    container: Container
) -> Boolean:
    """ element of """
    return value in container

@dsl.primitive
def containedContainer(
    value: Container,
    container: Container
) -> Boolean:
    """ element of """
    return value in container

@dsl.primitive
def containedFrozenSet(
    value: FrozenSet,
    container: Container
) -> Boolean:
    """ element of """
    return value in container

@dsl.primitive
def containedTuple(
    value: Tuple,
    container: Container
) -> Boolean:
    """ element of """
    return value in container

@dsl.primitive
def containedCallable(
    value: Callable,
    container: Container
) -> Boolean:
    """ element of """
    return value in container

@dsl.primitive
def containedCC(
    value: ContainerContainer,
    container: Container
) -> Boolean:
    """ element of """
    return value in container

@dsl.primitive
def containedIntegerSet(
    value: IntegerSet,
    container: Container
) -> Boolean:
    """ element of """
    return value in container

@dsl.primitive
def containedTT(
    value: TupleTuple,
    container: Container
) -> Boolean:
    """ element of """
    return value in container

@dsl.primitive
def containedCell(
    value: Cell,
    container: Container
) -> Boolean:
    """ element of """
    return value in container

@dsl.primitive
def containedObject(
    value: Object,
    container: Container
) -> Boolean:
    """ element of """
    return value in container

@dsl.primitive
def containedIndices(
    value: Indices,
    container: Container
) -> Boolean:
    """ element of """
    return value in container

@dsl.primitive
def containeddslGrid(
    value: Grid,
    container: Container
) -> Boolean:
    """ element of """
    temp = gridToTT(value)
    value = temp
    return value in container

@dsl.primitive
def containedIndicesSet(
    value: IndicesSet,
    container: Container
) -> Boolean:
    """ element of """
    return value in container

@dsl.primitive
def containedObjects(
    value: Objects,
    container: Container
) -> Boolean:
    """ element of """
    return value in container

@dsl.primitive
def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

@dsl.primitive
def intersection(
    a: FrozenSet,
    b: FrozenSet
) -> FrozenSet:
    """ returns the intersection of two containers """
    return a & b

@dsl.primitive
def difference(
    a: FrozenSet,
    b: FrozenSet
) -> FrozenSet:
    """ set difference """
    return type(a)(e for e in a if e not in b)

@dsl.primitive
def dedupe(
    tup: Tuple
) -> Tuple:
    """ remove duplicates """
    return tuple(e for i, e in enumerate(tup) if tup.index(e) == i)

@dsl.primitive
def order(
    container: Container,
    compfunc: Callable
) -> Tuple:
    """ order container by custom key """
    return tuple(sorted(container, key=compfunc))

@dsl.primitive
def repeatInt(
    item: Integer,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))

@dsl.primitive
def repeatBool(
    item: Boolean,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))

@dsl.primitive
def repeatCord(
    item: Tuple[int ,int],
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))

@dsl.primitive
def repeatContainer(
    item: Container,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))

@dsl.primitive
def repeatFrozenSet(
    item: FrozenSet,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))

@dsl.primitive
def repeatTuple(
    item: Tuple,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))

@dsl.primitive
def repeatCallable(
    item: Callable,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))

@dsl.primitive
def repeatCC(
    item: ContainerContainer,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))

@dsl.primitive
def repeatIntegerSet(
    item: IntegerSet,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))

@dsl.primitive
def repeatTT(
    item: TupleTuple,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))

@dsl.primitive
def repeatCell(
    item: Cell,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))

@dsl.primitive
def repeatObject(
    item: Object,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))

@dsl.primitive
def repeatIndices(
    item: Indices,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))

@dsl.primitive
def repeatdslGrid(
    item: Grid,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    temp = gridToTT(item)
    item = temp
    return tuple(item for i in range(num))

@dsl.primitive
def repeatIndicesSet(
    item: IndicesSet,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))

@dsl.primitive
def repeatObjects(
    item: Objects,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))

@dsl.primitive
def greater(
    a: Integer,
    b: Integer
) -> Boolean:
    """ greater """
    return a > b

@dsl.primitive
def size(
    container: Container
) -> Integer:
    """ cardinality """
    return len(container)

@dsl.primitive
def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

@dsl.primitive
def maximum(
    container: IntegerSet
) -> Integer:
    """ maximum """
    return max(container, default=0)

@dsl.primitive
def minimum(
    container: IntegerSet
) -> Integer:
    """ minimum """
    return min(container, default=0)

@dsl.primitive
def valmax(
    container: Container,
    compfunc: Callable
) -> Integer:
    """ maximum by custom function """
    return compfunc(max(container, key=compfunc, default=0))

@dsl.primitive
def valmin(
    container: Container,
    compfunc: Callable
) -> Integer:
    """ minimum by custom function """
    return compfunc(min(container, key=compfunc, default=0))

@dsl.primitive
def argmaxInt(
    container: Container,
    compfunc: Callable
) -> Integer:
    """ largest item by custom order """
    return max(container, key=compfunc)

@dsl.primitive
def argmaxBool(
    container: Container,
    compfunc: Callable
) -> Boolean:
    """ largest item by custom order """
    return max(container, key=compfunc)

@dsl.primitive
def argmaxCord(
    container: Container,
    compfunc: Callable
) -> Tuple[int ,int]:
    """ largest item by custom order """
    return max(container, key=compfunc)

@dsl.primitive
def argmaxContainer(
    container: Container,
    compfunc: Callable
) -> Container:
    """ largest item by custom order """
    return max(container, key=compfunc)

@dsl.primitive
def argmaxFrozenSet(
    container: Container,
    compfunc: Callable
) -> FrozenSet:
    """ largest item by custom order """
    return max(container, key=compfunc)

@dsl.primitive
def argmaxTuple(
    container: Container,
    compfunc: Callable
) -> Tuple:
    """ largest item by custom order """
    return max(container, key=compfunc)

@dsl.primitive
def argmaxCallable(
    container: Container,
    compfunc: Callable
) -> Callable:
    """ largest item by custom order """
    return max(container, key=compfunc)

@dsl.primitive
def argmaxCC(
    container: Container,
    compfunc: Callable
) -> ContainerContainer:
    """ largest item by custom order """
    return max(container, key=compfunc)

@dsl.primitive
def argmaxIntegerSet(
    container: Container,
    compfunc: Callable
) -> IntegerSet:
    """ largest item by custom order """
    return max(container, key=compfunc)

@dsl.primitive
def argmaxTT(
    container: Container,
    compfunc: Callable
) -> TupleTuple:
    """ largest item by custom order """
    return max(container, key=compfunc)

@dsl.primitive
def argmaxCell(
    container: Container,
    compfunc: Callable
) -> Cell:
    """ largest item by custom order """
    return max(container, key=compfunc)

@dsl.primitive
def argmaxObject(
    container: Container,
    compfunc: Callable
) -> Object:
    """ largest item by custom order """
    return max(container, key=compfunc)

@dsl.primitive
def argmaxIndices(
    container: Container,
    compfunc: Callable
) -> Indices:
    """ largest item by custom order """
    return max(container, key=compfunc)

@dsl.primitive
def argmaxdslGrid(
    container: Container,
    compfunc: Callable
) -> Grid:
    """ largest item by custom order """
    if isinstance(max(container, key=compfunc), Grid):
        return max(container, key=compfunc)
    return ttToGrid(max(container, key=compfunc))

@dsl.primitive
def argmaxIndicesSet(
    container: Container,
    compfunc: Callable
) -> IndicesSet:
    """ largest item by custom order """
    return max(container, key=compfunc)

@dsl.primitive
def argmaxObjects(
    container: Container,
    compfunc: Callable
) -> Objects:
    """ largest item by custom order """
    return max(container, key=compfunc)

@dsl.primitive
def argminInt(
    container: Container,
    compfunc: Callable
) -> Integer:
    """ largest item by custom order """
    return min(container, key=compfunc)

@dsl.primitive
def argminBool(
    container: Container,
    compfunc: Callable
) -> Boolean:
    """ largest item by custom order """
    return min(container, key=compfunc)

@dsl.primitive
def argminCord(
    container: Container,
    compfunc: Callable
) -> Tuple[int ,int]:
    """ largest item by custom order """
    return min(container, key=compfunc)

@dsl.primitive
def argminContainer(
    container: Container,
    compfunc: Callable
) -> Container:
    """ largest item by custom order """
    return min(container, key=compfunc)

@dsl.primitive
def argminFrozenSet(
    container: Container,
    compfunc: Callable
) -> FrozenSet:
    """ largest item by custom order """
    return min(container, key=compfunc)

@dsl.primitive
def argminTuple(
    container: Container,
    compfunc: Callable
) -> Tuple:
    """ largest item by custom order """
    return min(container, key=compfunc)

@dsl.primitive
def argminCallable(
    container: Container,
    compfunc: Callable
) -> Callable:
    """ largest item by custom order """
    return min(container, key=compfunc)

@dsl.primitive
def argminCC(
    container: Container,
    compfunc: Callable
) -> ContainerContainer:
    """ largest item by custom order """
    return min(container, key=compfunc)

@dsl.primitive
def argminIntegerSet(
    container: Container,
    compfunc: Callable
) -> IntegerSet:
    """ largest item by custom order """
    return min(container, key=compfunc)

@dsl.primitive
def argminTT(
    container: Container,
    compfunc: Callable
) -> TupleTuple:
    """ largest item by custom order """
    return min(container, key=compfunc)

@dsl.primitive
def argminCell(
    container: Container,
    compfunc: Callable
) -> Cell:
    """ largest item by custom order """
    return min(container, key=compfunc)

@dsl.primitive
def argminObject(
    container: Container,
    compfunc: Callable
) -> Object:
    """ largest item by custom order """
    return min(container, key=compfunc)

@dsl.primitive
def argminIndices(
    container: Container,
    compfunc: Callable
) -> Indices:
    """ largest item by custom order """
    return min(container, key=compfunc)

@dsl.primitive
def argmindslGrid(
    container: Container,
    compfunc: Callable
) -> Grid:
    """ largest item by custom order """
    if isinstance(min(container, key=compfunc), Grid):
        return min(container, key=compfunc)
    return ttToGrid(min(container, key=compfunc))

@dsl.primitive
def argminIndicesSet(
    container: Container,
    compfunc: Callable
) -> IndicesSet:
    """ largest item by custom order """
    return min(container, key=compfunc)

@dsl.primitive
def argminObjects(
    container: Container,
    compfunc: Callable
) -> Objects:
    """ largest item by custom order """
    return min(container, key=compfunc)

@dsl.primitive
def mostcommonInt(
    container: Container
) -> Integer:
    """ most common item """
    return max(set(container), key=container.count)

@dsl.primitive
def mostcommonBool(
    container: Container
) -> Boolean:
    """ most common item """
    return max(set(container), key=container.count)

@dsl.primitive
def mostcommonCord(
    container: Container
) -> Cord:
    """ most common item """
    return max(set(container), key=container.count)

@dsl.primitive
def mostcommonContainer(
    container: Container
) -> Container:
    """ most common item """
    return max(set(container), key=container.count)

@dsl.primitive
def mostcommonFrozenSet(
    container: Container
) -> FrozenSet:
    """ most common item """
    return max(set(container), key=container.count)

@dsl.primitive
def mostcommonTuple(
    container: Container
) -> Tuple:
    """ most common item """
    return max(set(container), key=container.count)

@dsl.primitive
def mostcommonCallable(
    container: Container
) -> Callable:
    """ most common item """
    return max(set(container), key=container.count)

@dsl.primitive
def mostcommonCC(
    container: Container
) -> ContainerContainer:
    """ most common item """
    return max(set(container), key=container.count)

@dsl.primitive
def mostcommonIntegerSet(
    container: Container
) -> IntegerSet:
    """ most common item """
    return max(set(container), key=container.count)

@dsl.primitive
def mostcommonTT(
    container: Container
) -> TupleTuple:
    """ most common item """
    return max(set(container), key=container.count)

@dsl.primitive
def mostcommonCell(
    container: Container
) -> Cell:
    """ most common item """
    return max(set(container), key=container.count)

@dsl.primitive
def mostcommonObject(
    container: Container
) -> Object:
    """ most common item """
    return max(set(container), key=container.count)

@dsl.primitive
def mostcommonIndices(
    container: Container
) -> Indices:
    """ most common item """
    return max(set(container), key=container.count)

@dsl.primitive
def mostcommondslGrid(
    container: Container
) -> Grid:
    """ most common item """
    if isinstance(max(set(container), key=container.count), Grid):
        return max(set(container), key=container.count)
    return ttToGrid(max(set(container), key=container.count))

@dsl.primitive
def mostcommonIndicesSet(
    container: Container
) -> IndicesSet:
    """ most common item """
    return max(set(container), key=container.count)

@dsl.primitive
def mostcommonObjects(
    container: Container
) -> Objects:
    """ most common item """
    return max(set(container), key=container.count)

@dsl.primitive
def leastcommonInt(
    container: Container
) -> Integer:
    """ most common item """
    return min(set(container), key=container.count)

@dsl.primitive
def leastcommonBool(
    container: Container
) -> Boolean:
    """ most common item """
    return min(set(container), key=container.count)

@dsl.primitive
def leastcommonCord(
    container: Container
) -> Cord:
    """ most common item """
    return min(set(container), key=container.count)

@dsl.primitive
def leastcommonContainer(
    container: Container
) -> Container:
    """ most common item """
    return min(set(container), key=container.count)

@dsl.primitive
def leastcommonFrozenSet(
    container: Container
) -> FrozenSet:
    """ most common item """
    return min(set(container), key=container.count)

@dsl.primitive
def leastcommonTuple(
    container: Container
) -> Tuple:
    """ most common item """
    return min(set(container), key=container.count)

@dsl.primitive
def leastcommonCallable(
    container: Container
) -> Callable:
    """ most common item """
    return min(set(container), key=container.count)

@dsl.primitive
def leastcommonCC(
    container: Container
) -> ContainerContainer:
    """ most common item """
    return min(set(container), key=container.count)

@dsl.primitive
def leastcommonIntegerSet(
    container: Container
) -> IntegerSet:
    """ most common item """
    return min(set(container), key=container.count)

@dsl.primitive
def leastcommonTT(
    container: Container
) -> TupleTuple:
    """ most common item """
    return min(set(container), key=container.count)

@dsl.primitive
def leastcommonCell(
    container: Container
) -> Cell:
    """ most common item """
    return min(set(container), key=container.count)

@dsl.primitive
def leastcommonObject(
    container: Container
) -> Object:
    """ most common item """
    return min(set(container), key=container.count)

@dsl.primitive
def leastcommonIndices(
    container: Container
) -> Indices:
    """ most common item """
    return min(set(container), key=container.count)

@dsl.primitive
def leastcommondslGrid(
    container: Container
) -> Grid:
    """ most common item """
    if isinstance(min(set(container), key=container.count), Grid):
        return min(set(container), key=container.count)
    return ttToGrid(min(set(container), key=container.count))

@dsl.primitive
def leastcommonIndicesSet(
    container: Container
) -> IndicesSet:
    """ most common item """
    return min(set(container), key=container.count)

@dsl.primitive
def leastcommonObjects(
    container: Container
) -> Objects:
    """ most common item """
    return min(set(container), key=container.count)

@dsl.primitive
def initsetInt(
    value: Integer
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

@dsl.primitive
def initsetBool(
    value: Boolean
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

@dsl.primitive
def initsetCord(
    value: Cord
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

@dsl.primitive
def initsetContainer(
    value: Container
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

@dsl.primitive
def initsetFrozenSet(
    value: FrozenSet
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

@dsl.primitive
def initsetTuple(
    value: Tuple
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

@dsl.primitive
def initsetCallable(
    value: Callable
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

@dsl.primitive
def initsetCC(
    value: ContainerContainer
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

@dsl.primitive
def initsetIntegerSet(
    value: IntegerSet
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

@dsl.primitive
def initsetTT(
    value: TupleTuple
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

@dsl.primitive
def initsetCell(
    value: Cell
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

@dsl.primitive
def initsetObject(
    value: Object
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

@dsl.primitive
def initsetIndices(
    value: Indices
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

@dsl.primitive
def initsetdslGrid(
    value: Grid
) -> FrozenSet:
    """ initialize container """
    temp = gridToTT(value)
    value = temp
    return frozenset({value})

@dsl.primitive
def initsetIndicesSet(
    value: IndicesSet
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

@dsl.primitive
def initsetObjects(
    value: Objects
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

@dsl.primitive
def both(
    a: Boolean,
    b: Boolean
) -> Boolean:
    """ logical and """
    return a and b

@dsl.primitive
def either(
    a: Boolean,
    b: Boolean
) -> Boolean:
    """ logical or """
    return a or b

@dsl.primitive
def incrementInt(
    x: Integer
) -> Integer:
    """ incrementing """
    return x + 1

@dsl.primitive
def incrementCord(
    x: Cord
) -> Cord:
    """ incrementing """
    return (x[0] + 1, x[1] + 1)

@dsl.primitive
def decrementInt(
    x: Integer
) -> Integer:
    """ decrementing """
    return x - 1

@dsl.primitive
def decrementCord(
    x: Cord
) -> Cord:
    """ decrementing """
    return (x[0] - 1, x[1] - 1)

@dsl.primitive
def crementInt(
    x: Integer
) -> Integer:
    """ incrementing positive and decrementing negative """
    return 0 if x == 0 else (x + 1 if x > 0 else x - 1)

@dsl.primitive
def crementCord(
    x: Cord
) -> Cord:
    """ incrementing positive and decrementing negative """
    return (
        0 if x[0] == 0 else (x[0] + 1 if x[0] > 0 else x[0] - 1),
        0 if x[1] == 0 else (x[1] + 1 if x[1] > 0 else x[1] - 1)
    )

@dsl.primitive
def signInt(
    x: Integer
) -> Integer:
    """ sign """
    return 0 if x == 0 else (1 if x > 0 else -1)

@dsl.primitive
def signCord(
    x: Cord
) -> Cord:
    """ sign """
    return (
        0 if x[0] == 0 else (1 if x[0] > 0 else -1),
        0 if x[1] == 0 else (1 if x[1] > 0 else -1)
    )


@dsl.primitive
def positive(
    x: Integer
) -> Boolean:
    """ positive """
    return x > 0

@dsl.primitive
def toivec(
    i: Integer
) -> Cord:
    """ vector pointing vertically """
    return (i, 0)



@dsl.primitive
def sfilter(
    container: Container,
    condition: Callable
) -> Container:
    """ keep elements in container that satisfy condition """
    return type(container)(e for e in container if condition(e))

@dsl.primitive
def mfilter(
    container: Container,
    function: Callable
) -> FrozenSet:
    """ filter and merge """
    return merge(sfilter(container, function))

# from here

@dsl.primitive
def extractInt(
    container: Container,
    condition: Callable
) -> Integer:
    """ first element of container that satisfies condition """
    return next(e for e in container if condition(e))

@dsl.primitive
def extractBool(
    container: Container,
    condition: Callable
) -> Boolean:
    """ first element of container that satisfies condition """
    return next(e for e in container if condition(e))

@dsl.primitive
def extractCord(
    container: Container,
    condition: Callable
) -> Cord:
    """ first element of container that satisfies condition """
    return next(e for e in container if condition(e))

@dsl.primitive
def extractContainer(
    container: Container,
    condition: Callable
) -> Container:
    """ first element of container that satisfies condition """
    return next(e for e in container if condition(e))

@dsl.primitive
def extractFrozenSet(
    container: Container,
    condition: Callable
) -> FrozenSet:
    """ first element of container that satisfies condition """
    return next(e for e in container if condition(e))

@dsl.primitive
def extractTuple(
    container: Container,
    condition: Callable
) -> Tuple:
    """ first element of container that satisfies condition """
    return next(e for e in container if condition(e))

@dsl.primitive
def extractCallable(
    container: Container,
    condition: Callable
) -> Callable:
    """ first element of container that satisfies condition """
    return next(e for e in container if condition(e))

@dsl.primitive
def extractCC(
    container: Container,
    condition: Callable
) -> ContainerContainer:
    """ first element of container that satisfies condition """
    return next(e for e in container if condition(e))

@dsl.primitive
def extractIntegerSet(
    container: Container,
    condition: Callable
) -> IntegerSet:
    """ first element of container that satisfies condition """
    return next(e for e in container if condition(e))

@dsl.primitive
def extractTT(
    container: Container,
    condition: Callable
) -> TupleTuple:
    """ first element of container that satisfies condition """
    return next(e for e in container if condition(e))

@dsl.primitive
def extractCell(
    container: Container,
    condition: Callable
) -> Cell:
    """ first element of container that satisfies condition """
    return next(e for e in container if condition(e))

@dsl.primitive
def extractObject(
    container: Container,
    condition: Callable
) -> Object:
    """ first element of container that satisfies condition """
    return next(e for e in container if condition(e))

@dsl.primitive
def extractIndices(
    container: Container,
    condition: Callable
) -> Indices:
    """ first element of container that satisfies condition """
    return next(e for e in container if condition(e))

@dsl.primitive
def extractdslGrid(
    container: Container,
    condition: Callable
) -> Grid:
    """ first element of container that satisfies condition """
    if isinstance(next(e for e in container if condition(e)), Grid):
        return next(e for e in container if condition(e))
    return ttToGrid(next(e for e in container if condition(e)))

@dsl.primitive
def extractIndicesSet(
    container: Container,
    condition: Callable
) -> IndicesSet:
    """ first element of container that satisfies condition """
    return next(e for e in container if condition(e))

@dsl.primitive
def extractObjects(
    container: Container,
    condition: Callable
) -> Objects:
    """ first element of container that satisfies condition """
    return next(e for e in container if condition(e))

@dsl.primitive
def totuple(
    container: FrozenSet
) -> Tuple:
    """ conversion to tuple """
    return tuple(container)



@dsl.primitive
def lastInt(
    container: Container
) -> Integer:
    """ first item of container """
    return max(enumerate(container))[1]

@dsl.primitive
def lastBool(
    container: Container
) -> Boolean:
    """ first item of container """
    return max(enumerate(container))[1]

@dsl.primitive
def lastCord(
    container: Container
) -> Cord:
    """ first item of container """
    return max(enumerate(container))[1]

@dsl.primitive
def lastContainer(
    container: Container
) -> Container:
    """ first item of container """
    return max(enumerate(container))[1]

@dsl.primitive
def lastFrozenSet(
    container: Container
) -> FrozenSet:
    """ first item of container """
    return max(enumerate(container))[1]

@dsl.primitive
def lastTuple(
    container: Container
) -> Tuple:
    """ first item of container """
    return max(enumerate(container))[1]

@dsl.primitive
def lastCallable(
    container: Container
) -> Callable:
    """ first item of container """
    return max(enumerate(container))[1]

@dsl.primitive
def lastCC(
    container: Container
) -> ContainerContainer:
    """ first item of container """
    return max(enumerate(container))[1]

@dsl.primitive
def lastIntegerSet(
    container: Container
) -> IntegerSet:
    """ first item of container """
    return max(enumerate(container))[1]

@dsl.primitive
def lastTT(
    container: Container
) -> TupleTuple:
    """ first item of container """
    return max(enumerate(container))[1]

@dsl.primitive
def lastCell(
    container: Container
) -> Cell:
    """ first item of container """
    return max(enumerate(container))[1]

@dsl.primitive
def lastObject(
    container: Container
) -> Object:
    """ first item of container """
    return max(enumerate(container))[1]

@dsl.primitive
def lastIndices(
    container: Container
) -> Indices:
    """ first item of container """
    return max(enumerate(container))[1]

@dsl.primitive
def lastdslGrid(
    container: Container
) -> Grid:
    """ first item of container """
    if isinstance(max(enumerate(container))[1], Grid):
        return max(enumerate(container))[1]
    return ttToGrid(max(enumerate(container))[1])

@dsl.primitive
def lastIndicesSet(
    container: Container
) -> IndicesSet:
    """ first item of container """
    return max(enumerate(container))[1]

@dsl.primitive
def lastObjects(
    container: Container
) -> Objects:
    """ first item of container """
    return max(enumerate(container))[1]

@dsl.primitive
def insertInt(
    value: Integer,
    container: FrozenSet
) -> FrozenSet:
    """ insert item into container """
    return container.union(frozenset({value}))

@dsl.primitive
def insertBool(
    value: Boolean,
    container: FrozenSet
) -> FrozenSet:
    """ insert item into container """
    return container.union(frozenset({value}))

@dsl.primitive
def insertCord(
    value: Cord,
    container: FrozenSet
) -> FrozenSet:
    """ insert item into container """
    return container.union(frozenset({value}))

@dsl.primitive
def insertContainer(
    value: Container,
    container: FrozenSet
) -> FrozenSet:
    """ insert item into container """
    return container.union(frozenset({value}))

@dsl.primitive
def insertFrozenSet(
    value: FrozenSet,
    container: FrozenSet
) -> FrozenSet:
    """ insert item into container """
    return container.union(frozenset({value}))

@dsl.primitive
def insertTuple(
    value: Tuple,
    container: FrozenSet
) -> FrozenSet:
    """ insert item into container """
    return container.union(frozenset({value}))

@dsl.primitive
def insertCallable(
    value: Callable,
    container: FrozenSet
) -> FrozenSet:
    """ insert item into container """
    return container.union(frozenset({value}))

@dsl.primitive
def insertCC(
    value: ContainerContainer,
    container: FrozenSet
) -> FrozenSet:
    """ insert item into container """
    return container.union(frozenset({value}))

@dsl.primitive
def insertIntegerSet(
    value: IntegerSet,
    container: FrozenSet
) -> FrozenSet:
    """ insert item into container """
    return container.union(frozenset({value}))

@dsl.primitive
def insertTT(
    value: TupleTuple,
    container: FrozenSet
) -> FrozenSet:
    """ insert item into container """
    return container.union(frozenset({value}))

@dsl.primitive
def insertCell(
    value: Cell,
    container: FrozenSet
) -> FrozenSet:
    """ insert item into container """
    return container.union(frozenset({value}))

@dsl.primitive
def insertObject(
    value: Object,
    container: FrozenSet
) -> FrozenSet:
    """ insert item into container """
    return container.union(frozenset({value}))

@dsl.primitive
def insertIndices(
    value: Indices,
    container: FrozenSet
) -> FrozenSet:
    """ insert item into container """
    return container.union(frozenset({value}))

@dsl.primitive
def insertdslGrid(
    value: Grid,
    container: FrozenSet
) -> FrozenSet:
    """ insert item into container """
    temp = gridToTT(value)
    value = temp
    return container.union(frozenset({value}))

@dsl.primitive
def insertIndicesSet(
    value: IndicesSet,
    container: FrozenSet
) -> FrozenSet:
    """ insert item into container """
    return container.union(frozenset({value}))

@dsl.primitive
def insertObjects(
    value: Objects,
    container: FrozenSet
) -> FrozenSet:
    """ idnsert item into container """
    return container.union(frozenset({value}))

@dsl.primitive
def removeInt(
    value: Integer,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)

@dsl.primitive
def removeBool(
    value: Boolean,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)

@dsl.primitive
def removeCord(
    value: Cord,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)

@dsl.primitive
def removeContainer(
    value: Container,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)

@dsl.primitive
def removeFrozenSet(
    value: FrozenSet,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)

@dsl.primitive
def removeTuple(
    value: Tuple,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)

@dsl.primitive
def removeCallable(
    value: Callable,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)

@dsl.primitive
def removeCC(
    value: ContainerContainer,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)

@dsl.primitive
def removeIntegerSet(
    value: IntegerSet,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)

@dsl.primitive
def removeTT(
    value: TupleTuple,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)

@dsl.primitive
def removeCell(
    value: Cell,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)

@dsl.primitive
def removeObject(
    value: Object,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)

@dsl.primitive
def removeIndices(
    value: Indices,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)

@dsl.primitive
def removedslGrid(
    value: Grid,
    container: Container
) -> Container:
    """ remove item from container """
    temp = gridToTT(value)
    value = temp
    return type(container)(e for e in container if e != value)

@dsl.primitive
def removeIndicesSet(
    value: IndicesSet,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)

@dsl.primitive
def removeObjects(
    value: Objects,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)

@dsl.primitive
def otherInt(
    container: Container,
    value: Integer
) -> Integer:
    """ other value in the container """
    return firstInt(removeInt(value, container))

@dsl.primitive
def otherBool(
    container: Container,
    value: Boolean
) -> Boolean:
    """ other value in the container """
    return firstBool(removeBool(value, container))

@dsl.primitive
def otherCord(
    container: Container,
    value: Cord
) -> Cord:
    """ other value in the container """
    return firstCord(removeCord(value, container))

@dsl.primitive
def otherContainer(
    container: Container,
    value: Container
) -> Container:
    """ other value in the container """
    return firstContainer(removeContainer(value, container))

@dsl.primitive
def otherFrozenSet(
    container: Container,
    value: FrozenSet
) -> FrozenSet:
    """ other value in the container """
    return firstFrozenSet(removeFrozenSet(value, container))

@dsl.primitive
def otherTuple(
    container: Container,
    value: Tuple
) -> Tuple:
    """ other value in the container """
    return firstTuple(removeTuple(value, container))

@dsl.primitive
def otherCallable(
    container: Container,
    value: Callable
) -> Callable:
    """ other value in the container """
    return firstCallable(removeCallable(value, container))

@dsl.primitive
def otherCC(
    container: Container,
    value: ContainerContainer
) -> ContainerContainer:
    """ other value in the container """
    return firstCC(removeCC(value, container))
#worked so far

@dsl.primitive
def otherIntegerSet(
    container: Container,
    value: IntegerSet
) -> IntegerSet:
    """ other value in the container """
    return firstIntegerSet(removeIntegerSet(value, container))

@dsl.primitive
def otherTT(
    container: Container,
    value: TupleTuple
) -> TupleTuple:
    """ other value in the container """
    return firstTT(removeTT(value, container))

@dsl.primitive
def otherCell(
    container: Container,
    value: Cell
) -> Cell:
    """ other value in the container """
    return firstCell(removeCell(value, container))

@dsl.primitive
def otherObject(
    container: Container,
    value: Object
) -> Object:
    """ other value in the container """
    return firstObject(removeObject(value, container))

@dsl.primitive
def otherIndices(
    container: Container,
    value: Indices
) -> Indices:
    """ other value in the container """
    return firstIndices(removeIndices(value, container))

@dsl.primitive
def otherdslGrid(
    container: Container,
    value: Grid
) -> Grid:
    """ other value in the container """
    return firstdslGrid(removedslGrid(value, container))

@dsl.primitive
def otherIndicesSet(
    container: Container,
    value: IndicesSet
) -> IndicesSet:
    """ other value in the container """
    return firstIndicesSet(removeIntegerSet(value, container))

@dsl.primitive
def otherObjects(
    container: Container,
    value: Objects
) -> Objects:
    """ other value in the container """
    return firstObjects(removeObjects(value, container))

@dsl.primitive
def interval(
    start: Integer,
    stop: Integer,
    step: Integer
) -> Tuple:
    """ range """
    return tuple(range(start, stop, step))

@dsl.primitive
def astuple(
    a: Integer,
    b: Integer
) -> Cord:
    """ constructs a tuple """
    return (a, b)

@dsl.primitive
def product(
    a: Container,
    b: Container
) -> FrozenSet:
    """ cartesian product """
    return frozenset((i, j) for j in b for i in a)

@dsl.primitive
def pair(
    a: Tuple,
    b: Tuple
) -> TupleTuple:
    """ zipping of two tuples """
    return tuple(zip(a, b))

@dsl.primitive
def branchInt(
    condition: Boolean,
    a: Integer,
    b: Integer
) -> Integer:
    """ if else branching """
    return a if condition else b

@dsl.primitive
def branchBool(
    condition: Boolean,
    a: Boolean,
    b: Boolean
) -> Boolean:
    """ if else branching """
    return a if condition else b

@dsl.primitive
def branchCord(
    condition: Boolean,
    a: Cord,
    b: Cord
) -> Cord:
    """ if else branching """
    return a if condition else b

@dsl.primitive
def branchContainer(
    condition: Boolean,
    a: Container,
    b: Container
) -> Container:
    """ if else branching """
    return a if condition else b

@dsl.primitive
def branchFrozenSet(
    condition: Boolean,
    a: FrozenSet,
    b: FrozenSet
) -> FrozenSet:
    """ if else branching """
    return a if condition else b

@dsl.primitive
def branchTuple(
    condition: Boolean,
    a: Tuple,
    b: Tuple
) -> Tuple:
    """ if else branching """
    return a if condition else b

@dsl.primitive
def branchCallable(
    condition: Boolean,
    a: Callable,
    b: Callable
) -> Callable:
    """ if else branching """
    return a if condition else b

@dsl.primitive
def branchCC(
    condition: Boolean,
    a: ContainerContainer,
    b: ContainerContainer
) -> ContainerContainer:
    """ if else branching """
    return a if condition else b

@dsl.primitive
def branchIntegerSet(
    condition: Boolean,
    a: IntegerSet,
    b: IntegerSet
) -> IntegerSet:
    """ if else branching """
    return a if condition else b

@dsl.primitive
def branchTT(
    condition: Boolean,
    a: TupleTuple,
    b: TupleTuple
) -> TupleTuple:
    """ if else branching """
    return a if condition else b

@dsl.primitive
def branchCell(
    condition: Boolean,
    a: Cell,
    b: Cell
) -> Cell:
    """ if else branching """
    return a if condition else b

@dsl.primitive
def branchObject(
    condition: Boolean,
    a: Object,
    b: Object
) -> Object:
    """ if else branching """
    return a if condition else b

@dsl.primitive
def branchIndices(
    condition: Boolean,
    a: Indices,
    b: Indices
) -> Indices:
    """ if else branching """
    return a if condition else b
#worked so far 2
@dsl.primitive
def branchdslGrid(
    condition: Boolean,
    a: Grid,
    b: Grid
) -> Grid:
    """ if else branching """
    return a if condition else b

@dsl.primitive
def branchIndicesSet(
    condition: Boolean,
    a: IndicesSet,
    b: IndicesSet
) -> IndicesSet:
    """ if else branching """
    return a if condition else b

@dsl.primitive
def branchObjects(
    condition: Boolean,
    a: Objects,
    b: Objects
) -> Objects:
    """ if else branching """
    return a if condition else b

@dsl.primitive
def compose(
    outer: Callable,
    inner: Callable
) -> Callable:
    """ function composition """
    return lambda x: outer(inner(x))

@dsl.primitive
def chaiN(
    h: Callable,
    g: Callable,
    f: Callable,
) -> Callable:
    """ function composition with three functions """
    return lambda x: h(g(f(x)))

@dsl.primitive
def matcherInt(
    function: Callable,
    target: Integer
) -> Callable:
    """ construction of equality function """
    return lambda x: function(x) == target

@dsl.primitive
def matcherBool(
    function: Callable,
    target: Boolean
) -> Callable:
    """ construction of equality function """
    return lambda x: function(x) == target

@dsl.primitive
def matcherCord(
    function: Callable,
    target: Cord
) -> Callable:
    """ construction of equality function """
    return lambda x: function(x) == target

@dsl.primitive
def matcherContainer(
    function: Callable,
    target: Container
) -> Callable:
    """ construction of equality function """
    return lambda x: function(x) == target

@dsl.primitive
def matcherFrozenSet(
    function: Callable,
    target: FrozenSet
) -> Callable:
    """ construction of equality function """
    return lambda x: function(x) == target

@dsl.primitive
def matcherTuple(
    function: Callable,
    target: Tuple
) -> Callable:
    """ construction of equality function """
    return lambda x: function(x) == target

@dsl.primitive
def matcherCallable(
    function: Callable,
    target: Callable
) -> Callable:
    """ construction of equality function """
    return lambda x: function(x) == target

@dsl.primitive
def matcherCC(
    function: Callable,
    target: ContainerContainer
) -> Callable:
    """ construction of equality function """
    return lambda x: function(x) == target

@dsl.primitive
def matcherIntegerSet(
    function: Callable,
    target: IntegerSet
) -> Callable:
    """ construction of equality function """
    return lambda x: function(x) == target

@dsl.primitive
def matcherTT(
    function: Callable,
    target: TupleTuple
) -> Callable:
    """ construction of equality function """
    return lambda x: function(x) == target

@dsl.primitive
def matcherCell(
    function: Callable,
    target: Cell
) -> Callable:
    """ construction of equality function """
    return lambda x: function(x) == target

@dsl.primitive
def matcherObject(
    function: Callable,
    target: Object
) -> Callable:
    """ construction of equality function """
    return lambda x: function(x) == target

@dsl.primitive
def matcherIndices(
    function: Callable,
    target: Indices
) -> Callable:
    """ construction of equality function """
    return lambda x: function(x) == target

@dsl.primitive
def matcherdslGrid(
    function: Callable,
    target: Grid
) -> Callable:
    """ construction of equality function """
    return lambda x: function(x) == target

@dsl.primitive
def matcherIndicesSet(
    function: Callable,
    target: IndicesSet
) -> Callable:
    """ construction of equality function """
    return lambda x: function(x) == target
#worked so far 3, 338primitives
@dsl.primitive
def matcherObjects(
    function: Callable,
    target: Objects
) -> Callable:
    """ construction of equality function """
    return lambda x: function(x) == target
    
@dsl.primitive
def rbindInt(
    function: Callable,
    fixed: Integer
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda x: function(x, fixed)
    elif n == 3:
        return lambda x, y: function(x, y, fixed)
    else:
        return lambda x, y, z: function(x, y, z, fixed)
    
@dsl.primitive
def rbindBool(
    function: Callable,
    fixed: Boolean
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda x: function(x, fixed)
    elif n == 3:
        return lambda x, y: function(x, y, fixed)
    else:
        return lambda x, y, z: function(x, y, z, fixed)

@dsl.primitive
def rbindCord(
    function: Callable,
    fixed: Cord
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda x: function(x, fixed)
    elif n == 3:
        return lambda x, y: function(x, y, fixed)
    else:
        return lambda x, y, z: function(x, y, z, fixed)
    
@dsl.primitive
def rbindContainer(
    function: Callable,
    fixed: Container
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda x: function(x, fixed)
    elif n == 3:
        return lambda x, y: function(x, y, fixed)
    else:
        return lambda x, y, z: function(x, y, z, fixed)
    
@dsl.primitive
def rbindFrozenSet(
    function: Callable,
    fixed: FrozenSet
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda x: function(x, fixed)
    elif n == 3:
        return lambda x, y: function(x, y, fixed)
    else:
        return lambda x, y, z: function(x, y, z, fixed)
    
@dsl.primitive
def rbindTuple(
    function: Callable,
    fixed: Tuple
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda x: function(x, fixed)
    elif n == 3:
        return lambda x, y: function(x, y, fixed)
    else:
        return lambda x, y, z: function(x, y, z, fixed)
    
@dsl.primitive
def rbindCallable(
    function: Callable,
    fixed: Callable
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda x: function(x, fixed)
    elif n == 3:
        return lambda x, y: function(x, y, fixed)
    else:
        return lambda x, y, z: function(x, y, z, fixed)
    
@dsl.primitive
def rbindCC(
    function: Callable,
    fixed: ContainerContainer
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda x: function(x, fixed)
    elif n == 3:
        return lambda x, y: function(x, y, fixed)
    else:
        return lambda x, y, z: function(x, y, z, fixed)
#worked 347 primitives
@dsl.primitive
def rbindIntegerSet(
    function: Callable,
    fixed: IntegerSet
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda x: function(x, fixed)
    elif n == 3:
        return lambda x, y: function(x, y, fixed)
    else:
        return lambda x, y, z: function(x, y, z, fixed)
    
@dsl.primitive
def rbindTT(
    function: Callable,
    fixed: TupleTuple
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda x: function(x, fixed)
    elif n == 3:
        return lambda x, y: function(x, y, fixed)
    else:
        return lambda x, y, z: function(x, y, z, fixed)
    
@dsl.primitive
def rbindCell(
    function: Callable,
    fixed: Cell
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda x: function(x, fixed)
    elif n == 3:
        return lambda x, y: function(x, y, fixed)
    else:
        return lambda x, y, z: function(x, y, z, fixed)
    
@dsl.primitive
def rbindObject(
    function: Callable,
    fixed: Object
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda x: function(x, fixed)
    elif n == 3:
        return lambda x, y: function(x, y, fixed)
    else:
        return lambda x, y, z: function(x, y, z, fixed)
    
@dsl.primitive
def rbindIndices(
    function: Callable,
    fixed: Indices
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda x: function(x, fixed)
    elif n == 3:
        return lambda x, y: function(x, y, fixed)
    else:
        return lambda x, y, z: function(x, y, z, fixed)
    
@dsl.primitive
def rbinddslGrid(
    function: Callable,
    fixed: Grid
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda x: function(x, fixed)
    elif n == 3:
        return lambda x, y: function(x, y, fixed)
    else:
        return lambda x, y, z: function(x, y, z, fixed)
    
@dsl.primitive
def rbindIndicesSet(
    function: Callable,
    fixed: IndicesSet
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda x: function(x, fixed)
    elif n == 3:
        return lambda x, y: function(x, y, fixed)
    else:
        return lambda x, y, z: function(x, y, z, fixed)
    
@dsl.primitive
def rbindObjects(
    function: Callable,
    fixed: Objects
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda x: function(x, fixed)
    elif n == 3:
        return lambda x, y: function(x, y, fixed)
    else:
        return lambda x, y, z: function(x, y, z, fixed)

@dsl.primitive
def lbindInt(
    function: Callable,
    fixed: Integer
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda y: function(fixed, y)
    elif n == 3:
        return lambda y, z: function(fixed, y, z)
    else:
        return lambda y, z, a: function(fixed, y, z, a)
    
@dsl.primitive
def lbindBool(
    function: Callable,
    fixed: Boolean
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda y: function(fixed, y)
    elif n == 3:
        return lambda y, z: function(fixed, y, z)
    else:
        return lambda y, z, a: function(fixed, y, z, a)

@dsl.primitive
def lbindCord(
    function: Callable,
    fixed: Cord
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda y: function(fixed, y)
    elif n == 3:
        return lambda y, z: function(fixed, y, z)
    else:
        return lambda y, z, a: function(fixed, y, z, a)
    
@dsl.primitive
def lbindContainer(
    function: Callable,
    fixed: Container
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda y: function(fixed, y)
    elif n == 3:
        return lambda y, z: function(fixed, y, z)
    else:
        return lambda y, z, a: function(fixed, y, z, a)
    
@dsl.primitive
def lbindFrozenSet(
    function: Callable,
    fixed: FrozenSet
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda y: function(fixed, y)
    elif n == 3:
        return lambda y, z: function(fixed, y, z)
    else:
        return lambda y, z, a: function(fixed, y, z, a)
    
@dsl.primitive
def lbindTuple(
    function: Callable,
    fixed: Tuple
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda y: function(fixed, y)
    elif n == 3:
        return lambda y, z: function(fixed, y, z)
    else:
        return lambda y, z, a: function(fixed, y, z, a)
    
@dsl.primitive
def lbindCallable(
    function: Callable,
    fixed: Callable
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda y: function(fixed, y)
    elif n == 3:
        return lambda y, z: function(fixed, y, z)
    else:
        return lambda y, z, a: function(fixed, y, z, a)
    
@dsl.primitive
def lbindCC(
    function: Callable,
    fixed: ContainerContainer
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda y: function(fixed, y)
    elif n == 3:
        return lambda y, z: function(fixed, y, z)
    else:
        return lambda y, z, a: function(fixed, y, z, a)

@dsl.primitive
def lbindIntegerSet(
    function: Callable,
    fixed: IntegerSet
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda y: function(fixed, y)
    elif n == 3:
        return lambda y, z: function(fixed, y, z)
    else:
        return lambda y, z, a: function(fixed, y, z, a)
#worked 364 primitives
@dsl.primitive
def lbindTT(
    function: Callable,
    fixed: TupleTuple
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda y: function(fixed, y)
    elif n == 3:
        return lambda y, z: function(fixed, y, z)
    else:
        return lambda y, z, a: function(fixed, y, z, a)
    
@dsl.primitive
def lbindCell(
    function: Callable,
    fixed: Cell
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda y: function(fixed, y)
    elif n == 3:
        return lambda y, z: function(fixed, y, z)
    else:
        return lambda y, z, a: function(fixed, y, z, a)
    
@dsl.primitive
def lbindObject(
    function: Callable,
    fixed: Object
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda y: function(fixed, y)
    elif n == 3:
        return lambda y, z: function(fixed, y, z)
    else:
        return lambda y, z, a: function(fixed, y, z, a)
    
@dsl.primitive
def lbindIndices(
    function: Callable,
    fixed: Indices
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda y: function(fixed, y)
    elif n == 3:
        return lambda y, z: function(fixed, y, z)
    else:
        return lambda y, z, a: function(fixed, y, z, a)
#worked 368 primitives
@dsl.primitive
def lbinddslGrid(
    function: Callable,
    fixed: Grid
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda y: function(fixed, y)
    elif n == 3:
        return lambda y, z: function(fixed, y, z)
    else:
        return lambda y, z, a: function(fixed, y, z, a)
    
@dsl.primitive
def lbindIndicesSet(
    function: Callable,
    fixed: IndicesSet
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda y: function(fixed, y)
    elif n == 3:
        return lambda y, z: function(fixed, y, z)
    else:
        return lambda y, z, a: function(fixed, y, z, a)
    
@dsl.primitive
def lbindObjects(
    function: Callable,
    fixed: Objects
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda y: function(fixed, y)
    elif n == 3:
        return lambda y, z: function(fixed, y, z)
    else:
        return lambda y, z, a: function(fixed, y, z, a)
#worked 371 primitives

@dsl.primitive
def fun_power(
    #power already in use
    function: Callable,
    n: Integer
) -> Callable:
    """ power of function """
    if n == 1:
        return function
    return compose(function, fun_power(function, n - 1))

#not woking 372 primitives

@dsl.primitive
def fork(
    outer: Callable,
    a: Callable,
    b: Callable
) -> Callable:
    """ creates a wrapper function """
    return lambda x: outer(a(x), b(x))

@dsl.primitive
def apply(
    function: Callable,
    container: Container
) -> Container:
    """ apply function to each item in container """
    return type(container)(function(e) for e in container)
#not working 374primitives
@dsl.primitive
def rapplyInt(
    functions: Container,
    value: Integer
) -> Container:
    """ apply each function in container to value """
    return type(functions)(function(value) for function in functions)

@dsl.primitive
def rapplyBool(
    functions: Container,
    value: Boolean
) -> Container:
    """ apply each function in container to value """
    return type(functions)(function(value) for function in functions)

@dsl.primitive
def rapplyCord(
    functions: Container,
    value: Cord
) -> Container:
    """ apply each function in container to value """
    return type(functions)(function(value) for function in functions)

@dsl.primitive
def rapplyContainer(
    functions: Container,
    value: Container
) -> Container:
    """ apply each function in container to value """
    return type(functions)(function(value) for function in functions)

@dsl.primitive
def rapplyFrozenSet(
    functions: Container,
    value: FrozenSet
) -> Container:
    """ apply each function in container to value """
    return type(functions)(function(value) for function in functions)

@dsl.primitive
def rapplyTuple(
    functions: Container,
    value: Tuple
) -> Container:
    """ apply each function in container to value """
    return type(functions)(function(value) for function in functions)

@dsl.primitive
def rapplyCallable(
    functions: Container,
    value: Callable
) -> Container:
    """ apply each function in container to value """
    return type(functions)(function(value) for function in functions)

@dsl.primitive
def rapplyCC(
    functions: Container,
    value: ContainerContainer
) -> Container:
    """ apply each function in container to value """
    return type(functions)(function(value) for function in functions)

@dsl.primitive
def rapplyIntegerSet(
    functions: Container,
    value: IntegerSet
) -> Container:
    """ apply each function in container to value """
    return type(functions)(function(value) for function in functions)

@dsl.primitive
def rapplyTT(
    functions: Container,
    value: TupleTuple
) -> Container:
    """ apply each function in container to value """
    return type(functions)(function(value) for function in functions)

@dsl.primitive
def rapplyCell(
    functions: Container,
    value: Cell
) -> Container:
    """ apply each function in container to value """
    return type(functions)(function(value) for function in functions)

@dsl.primitive
def rapplyObject(
    functions: Container,
    value: Object
) -> Container:
    """ apply each function in container to value """
    return type(functions)(function(value) for function in functions)

@dsl.primitive
def rapplyIndices(
    functions: Container,
    value: Indices
) -> Container:
    """ apply each function in container to value """
    return type(functions)(function(value) for function in functions)
# not working
@dsl.primitive
def rapplydslGrid(
    functions: Container,
    value: Grid
) -> Container:
    """ apply each function in container to value """
    temp = gridToTT(value)
    value = temp
    return type(functions)(function(value) for function in functions)

@dsl.primitive
def rapplyIndicesSet(
    functions: Container,
    value: IndicesSet
) -> Container:
    """ apply each function in container to value """
    return type(functions)(function(value) for function in functions)

@dsl.primitive
def rapplyObjects(
    functions: Container,
    value: Objects
) -> Container:
    """ apply each function in container to value """
    return type(functions)(function(value) for function in functions)

@dsl.primitive
def mapply(
    function: Callable,
    container: ContainerContainer
) -> FrozenSet:
    """ apply and merge """
    return merge(apply(function, container))

@dsl.primitive
def papply(
    function: Callable,
    a: Tuple,
    b: Tuple
) -> Tuple:
    """ apply function on two vectors """
    return tuple(function(i, j) for i, j in zip(a, b))

@dsl.primitive
def mpapply(
    function: Callable,
    a: Tuple,
    b: Tuple
) -> Tuple:
    """ apply function on two vectors and merge """
    return merge(papply(function, a, b))

@dsl.primitive
def prapply(
    function: Callable,
    a: Container,
    b: Container
) -> FrozenSet:
    """ apply function on cartesian product """
    return frozenset(function(i, j) for j in b for i in a)


    
@dsl.primitive
def leastcolorObject(
    element: Object
) -> Integer:
    """ least common color """
    values = [v for r in element for v in r]
    return min(set(values), key=values.count)

@dsl.primitive
def leastcolordslGrid(
    element: Grid
) -> Integer:
    """ least common color """
    temp = gridToTT(element)
    element = temp
    values = [v for v, _ in element]
    return min(set(values), key=values.count)

@dsl.primitive
def heightdslGrid(
    piece: Grid
) -> Integer:
    """ height of grid or patch """
    temp = gridToTT(piece)
    piece = temp
    if len(piece) == 0:
        return 0
    return len(piece)

@dsl.primitive
def heightObject(
    piece: Object
) -> Integer:
    """ height of grid or patch """
    if len(piece) == 0:
        return 0
    return lowermostObject(piece) - uppermostObject(piece) + 1

@dsl.primitive
def heightIndices(
    piece: Indices
) -> Integer:
    """ height of grid or patch """
    if len(piece) == 0:
        return 0
    return lowermostIndices(piece) - uppermostIndices(piece) + 1

@dsl.primitive
def widthdslGrid(
    piece: Grid
) -> Integer:
    """ width of grid or patch """
    temp = gridToTT(piece)
    piece = temp
    if len(piece) == 0:
        return 0
    return len(piece[0])

@dsl.primitive
def widthObject(
    piece: Object
) -> Integer:
    """ width of grid or patch """
    if len(piece) == 0:
        return 0
    return rightmostObject(piece) - leftmostObject(piece) + 1

@dsl.primitive
def widthIndices(
    piece: Indices
) -> Integer:
    """ width of grid or patch """
    if len(piece) == 0:
        return 0
    return rightmostIndices(piece) - leftmostIndices(piece) + 1

@dsl.primitive
def shapedslGrid(
    piece: Grid
) -> Cord:
    """ height and width of grid or patch """
    return (heightdslGrid(piece), widthdslGrid(piece))

@dsl.primitive
def shapeObject(
    piece: Object
) -> Cord:
    """ height and width of grid or patch """
    return (heightObject(piece), widthObject(piece))

@dsl.primitive
def shapeIndices(
    piece: Indices
) -> Cord:
    """ height and width of grid or patch """
    return (heightIndices(piece), widthIndices(piece))

@dsl.primitive
def portraitdslGrid(
    piece: Grid
) -> Boolean:
    """ whether height is greater than width """
    return heightdslGrid(piece) > widthdslGrid(piece)

@dsl.primitive
def portraitObject(
    piece: Object
) -> Boolean:
    """ whether height is greater than width """
    return heightObject(piece) > widthObject(piece)

@dsl.primitive
def portraitIndices(
    piece: Indices
) -> Boolean:
    """ whether height is greater than width """
    return heightIndices(piece) > widthIndices(piece)

@dsl.primitive
def colorcountObject(
    element: Object,
    value: Integer
) -> Integer:
    """ number of cells with color """
    return sum(v == value for v, _ in element)

@dsl.primitive
def colorcountdslGrid(
    element: Grid,
    value: Integer
) -> Integer:
    """ number of cells with color """
    temp = gridToTT(element)
    element = temp
    return sum(row.count(value) for row in element)

@dsl.primitive
def colorfilter(
    objs: Objects,
    value: Integer
) -> Objects:
    """ filter objects by color """
    return frozenset(obj for obj in objs if next(iter(obj))[0] == value)

@dsl.primitive
def sizefilter(
    container: Container,
    n: Integer
) -> FrozenSet:
    """ filter items by size """
    return frozenset(item for item in container if len(item) == n)

@dsl.primitive
def asindices(
    grid: Grid
) -> Indices:
    """ indices of all grid cells """
    temp = gridToTT(grid)
    grid = temp
    return frozenset((i, j) for i in range(len(grid)) for j in range(len(grid[0])))

@dsl.primitive
def ofcolor(
    grid: Grid,
    value: Integer
) -> Indices:
    """ indices of all grid cells with value """
    temp = gridToTT(grid)
    grid = temp
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)

@dsl.primitive
def urcornerObject(
    patch: Object
) -> Cord:
    """ index of upper right corner """
    return tuple(map(lambda ix: {0: min, 1: max}[ix[0]](ix[1]), enumerate(zip(*toindicesObject(patch)))))

@dsl.primitive
def urcornerIndices(
    patch: Indices
) -> Cord:
    """ index of upper right corner """
    return tuple(map(lambda ix: {0: min, 1: max}[ix[0]](ix[1]), enumerate(zip(*toindicesIndices(patch)))))

@dsl.primitive
def llcornerObject(
    patch: Object
) -> Cord:
    """ index of lower left corner """
    return tuple(map(lambda ix: {0: max, 1: min}[ix[0]](ix[1]), enumerate(zip(*toindicesObject(patch)))))

@dsl.primitive
def llcornerIndices(
    patch: Indices
) -> Cord:
    """ index of lower left corner """
    return tuple(map(lambda ix: {0: max, 1: min}[ix[0]](ix[1]), enumerate(zip(*toindicesIndices(patch)))))






@dsl.primitive
def recolorObject(
    value: Integer,
    patch: Object
) -> Object:
    """ recolor patch """
    return frozenset((value, index) for index in toindicesObject(patch))

@dsl.primitive
def recolorIndices(
    value: Integer,
    patch: Indices
) -> Object:
    """ recolor patch """
    return frozenset((value, index) for index in toindicesIndices(patch))



@dsl.primitive
def shiftIndices(
    patch: Indices,
    directions: Cord
) -> Indices:
    """ shift patch """
    if len(patch) == 0:
        return patch
    di, dj = directions
    return frozenset((i + di, j + dj) for i, j in patch)

@dsl.primitive
def normalizeObject(
    patch: Object
) -> Object:
    """ moves upper left corner to origin """
    if len(patch) == 0:
        return patch
    return shiftObject(patch, (-uppermostObject(patch), -leftmostObject(patch)))

@dsl.primitive
def normalizeIndices(
    patch: Indices
) -> Indices:
    """ moves upper left corner to origin """
    if len(patch) == 0:
        return patch
    return shiftIndices(patch, (-uppermostIndices(patch), -leftmostIndices(patch)))

@dsl.primitive
def dneighbors(
    loc: Cord
) -> Indices:
    """ directly adjacent indices """
    return frozenset({(loc[0] - 1, loc[1]), (loc[0] + 1, loc[1]), (loc[0], loc[1] - 1), (loc[0], loc[1] + 1)})

@dsl.primitive
def ineighbors(
    loc: Cord
) -> Indices:
    """ diagonally adjacent indices """
    return frozenset({(loc[0] - 1, loc[1] - 1), (loc[0] - 1, loc[1] + 1), (loc[0] + 1, loc[1] - 1), (loc[0] + 1, loc[1] + 1)})

@dsl.primitive
def neighbors(
    loc: Cord
) -> Indices:
    """ adjacent indices """
    return dneighbors(loc) | ineighbors(loc)

@dsl.primitive
def objectS(
    grid: Grid,
    univalued: Boolean,
    diagonal: Boolean,
    without_bg: Boolean
) -> Objects:
    """ objects occurring on the grid """
    temp = gridToTT(grid)
    grid = temp
    bg = mostcolordslGrid(ttToGrid(grid)) if without_bg else None
    objs = set()
    occupied = set()
    h, w = len(grid), len(grid[0])
    unvisited = asindices(ttToGrid(grid))
    diagfun = neighbors if diagonal else dneighbors
    for loc in unvisited:
        if loc in occupied:
            continue
        val = grid[loc[0]][loc[1]]
        if val == bg:
            continue
        obj = {(val, loc)}
        cands = {loc}
        while len(cands) > 0:
            neighborhood = set()
            for cand in cands:
                v = grid[cand[0]][cand[1]]
                if (val == v) if univalued else (v != bg):
                    obj.add((v, cand))
                    occupied.add(cand)
                    neighborhood |= {
                        (i, j) for i, j in diagfun(cand) if 0 <= i < h and 0 <= j < w
                    }
            cands = neighborhood - occupied
        objs.add(frozenset(obj))
    return frozenset(objs)

@dsl.primitive
def partition(
    grid: Grid
) -> Objects:
    """ each cell with the same value part of the same object """
    temp = gridToTT(grid)
    grid = temp
    return frozenset(
        frozenset(
            (v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value
        ) for value in palettedslGrid(ttToGrid(grid))
    )

@dsl.primitive
def fgpartition(
    grid: Grid
) -> Objects:
    """ each cell with the same value part of the same object without background """
    temp = gridToTT(grid)
    grid = temp
    return frozenset(
        frozenset(
            (v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value
        ) for value in palettedslGrid(ttToGrid(grid)) - {mostcolordslGrid(ttToGrid(grid))}
    )

@dsl.primitive
def uppermostObject(
    patch: Object
) -> Integer:
    """ row index of uppermost occupied cell """
    return min(i for i, j in toindicesObject(patch))

@dsl.primitive
def uppermostIndices(
    patch: Indices
) -> Integer:
    """ row index of uppermost occupied cell """
    return min(i for i, j in toindicesIndices(patch))

@dsl.primitive
def lowermostObject(
    patch: Object
) -> Integer:
    """ row index of lowermost occupied cell """
    return max(i for i, j in toindicesObject(patch))

@dsl.primitive
def lowermostIndices(
    patch: Indices
) -> Integer:
    """ row index of lowermost occupied cell """
    return max(i for i, j in toindicesIndices(patch))

@dsl.primitive
def leftmostObject(
    patch: Object
) -> Integer:
    """ column index of leftmost occupied cell """
    return min(j for i, j in toindicesObject(patch))

@dsl.primitive
def leftmostIndices(
    patch: Indices
) -> Integer:
    """ column index of leftmost occupied cell """
    return min(j for i, j in toindicesIndices(patch))

@dsl.primitive
def rightmostObject(
    patch: Object
) -> Integer:
    """ column index of rightmost occupied cell """
    return max(j for i, j in toindicesObject(patch))

@dsl.primitive
def rightmostIndices(
    patch: Indices
) -> Integer:
    """ column index of rightmost occupied cell """
    return max(j for i, j in toindicesIndices(patch))

@dsl.primitive
def squaredslGrid(
    piece: Grid
) -> Boolean:
    """ whether the piece forms a square """
    temp = gridToTT(piece)
    piece = temp
    return len(piece) == len(piece[0]) if isinstance(piece, tuple) else heightdslGrid(ttToGrid(piece)) * widthdslGrid(ttToGrid(piece)) == len(piece) and heightdslGrid(ttToGrid(piece)) == widthdslGrid(ttToGrid(piece))

@dsl.primitive
def squareObject(
    piece: Object
) -> Boolean:
    """ whether the piece forms a square """
    return len(piece) == len(piece[0]) if isinstance(piece, tuple) else heightObject(piece) * widthObject(piece) == len(piece) and heightObject(piece) == widthObject(piece)

@dsl.primitive
def squareIndices(
    piece: Indices
) -> Boolean:
    """ whether the piece forms a square """
    return len(piece) == len(piece[0]) if isinstance(piece, tuple) else heightIndices(piece) * widthIndices(piece) == len(piece) and heightIndices(piece) == widthIndices(piece)

@dsl.primitive
def vlineObject(
    patch: Object
) -> Boolean:
    """ whether the piece forms a vertical line """
    return heightObject(patch) == len(patch) and widthObject(patch) == 1

@dsl.primitive
def vlineIndices(
    patch: Indices
) -> Boolean:
    """ whether the piece forms a vertical line """
    return heightIndices(patch) == len(patch) and widthIndices(patch) == 1

#tobecontinued

@dsl.primitive
def hlineObject(
    patch: Object
) -> Boolean:
    """ whether the piece forms a horizontal line """
    return widthObject(patch) == len(patch) and heightObject(patch) == 1

@dsl.primitive
def hlineIndices(
    patch: Indices
) -> Boolean:
    """ whether the piece forms a horizontal line """
    return widthIndices(patch) == len(patch) and heightIndices(patch) == 1

@dsl.primitive
def hmatchingOO(
    a: Object,
    b: Object
) -> Boolean:
    """ whether there exists a row for which both patches have cells """
    return len(set(i for i, j in toindicesObject(a)) & set(i for i, j in toindicesObject(b))) > 0

@dsl.primitive
def hmatchingOI(
    a: Object,
    b: Indices
) -> Boolean:
    """ whether there exists a row for which both patches have cells """
    return len(set(i for i, j in toindicesObject(a)) & set(i for i, j in toindicesIndices(b))) > 0

@dsl.primitive
def hmatchingIO(
    a: Indices,
    b: Object
) -> Boolean:
    """ whether there exists a row for which both patches have cells """
    return len(set(i for i, j in toindicesIndices(a)) & set(i for i, j in toindicesObject(b))) > 0

@dsl.primitive
def hmatchingII(
    a: Indices,
    b: Indices
) -> Boolean:
    """ whether there exists a row for which both patches have cells """
    return len(set(i for i, j in toindicesIndices(a)) & set(i for i, j in toindicesIndices(b))) > 0

@dsl.primitive
def vmatchingOO(
    a: Object,
    b: Object
) -> Boolean:
    """ whether there exists a row for which both patches have cells """
    return len(set(j for i, j in toindicesObject(a)) & set(j for i, j in toindicesObject(b))) > 0

@dsl.primitive
def vmatchingOI(
    a: Object,
    b: Indices
) -> Boolean:
    """ whether there exists a row for which both patches have cells """
    return len(set(j for i, j in toindicesObject(a)) & set(j for i, j in toindicesIndices(b))) > 0

@dsl.primitive
def vmatchingIO(
    a: Indices,
    b: Object
) -> Boolean:
    """ whether there exists a row for which both patches have cells """
    return len(set(j for i, j in toindicesIndices(a)) & set(j for i, j in toindicesObject(b))) > 0

@dsl.primitive
def vmatchingII(
    a: Indices,
    b: Indices
) -> Boolean:
    """ whether there exists a row for which both patches have cells """
    return len(set(j for i, j in toindicesIndices(a)) & set(j for i, j in toindicesIndices(b))) > 0

@dsl.primitive
def manhattanOO(
    a: Object,
    b: Object
) -> Integer:
    """ closest manhattan distance between two patches """
    return min(abs(ai - bi) + abs(aj - bj) for ai, aj in toindicesObject(a) for bi, bj in toindicesObject(b))

@dsl.primitive
def manhattanOI(
    a: Object,
    b: Indices
) -> Integer:
    """ closest manhattan distance between two patches """
    return min(abs(ai - bi) + abs(aj - bj) for ai, aj in toindicesObject(a) for bi, bj in toindicesIndices(b))

@dsl.primitive
def manhattanIO(
    a: Indices,
    b: Object
) -> Integer:
    """ closest manhattan distance between two patches """
    return min(abs(ai - bi) + abs(aj - bj) for ai, aj in toindicesIndices(a) for bi, bj in toindicesObject(b))

@dsl.primitive
def manhattanII(
    a: Indices,
    b: Indices
) -> Integer:
    """ closest manhattan distance between two patches """
    return min(abs(ai - bi) + abs(aj - bj) for ai, aj in toindicesIndices(a) for bi, bj in toindicesIndices(b))

@dsl.primitive
def adjacentOO(
    a: Object,
    b: Object
) -> Boolean:
    """ whether two patches are adjacent """
    return manhattanOO(a, b) == 1

@dsl.primitive
def adjacentOI(
    a: Object,
    b: Indices
) -> Boolean:
    """ whether two patches are adjacent """
    return manhattanOI(a, b) == 1

@dsl.primitive
def adjacentIO(
    a: Indices,
    b: Object
) -> Boolean:
    """ whether two patches are adjacent """
    return manhattanIO(a, b) == 1

@dsl.primitive
def adjacentII(
    a: Indices,
    b: Indices
) -> Boolean:
    """ whether two patches are adjacent """
    return manhattanII(a, b) == 1

@dsl.primitive
def borderingObject(
    patch: Object,
    grid: Grid
) -> Boolean:
    """ whether a patch is adjacent to a grid border """
    temp = gridToTT(grid)
    grid = temp
    return uppermostObject(patch) == 0 or leftmostObject(patch) == 0 or lowermostObject(patch) == len(grid) - 1 or rightmostObject(patch) == len(grid[0]) - 1

@dsl.primitive
def borderingIndices(
    patch: Indices,
    grid: Grid
) -> Boolean:
    """ whether a patch is adjacent to a grid border """
    temp = gridToTT(grid)
    grid = temp
    return uppermostIndices(patch) == 0 or leftmostIndices(patch) == 0 or lowermostIndices(patch) == len(grid) - 1 or rightmostIndices(patch) == len(grid[0]) - 1

@dsl.primitive
def centerofmassObject(
    patch: Object
) -> Cord:
    """ center of mass """
    return tuple(map(lambda x: sum(x) // len(patch), zip(*toindicesObject(patch))))

@dsl.primitive
def centerofmassIndices(
    patch: Indices
) -> Cord:
    """ center of mass """
    return tuple(map(lambda x: sum(x) // len(patch), zip(*toindicesIndices(patch))))

@dsl.primitive
def paletteObject(
    element: Object
) -> IntegerSet:
    """ colors occurring in object or grid """
    return frozenset({v for v, _ in element})

@dsl.primitive
def palettedslGrid(
    element: Grid
) -> IntegerSet:
    """ colors occurring in object or grid """
    temp = gridToTT(element)
    element = temp
    return frozenset({v for r in element for v in r})

@dsl.primitive
def numcolorsObject(
    element: Object
) -> IntegerSet:
    """ number of colors occurring in object or grid """
    return len(paletteObject(element))

@dsl.primitive
def numcolorsdslGrid(
    element: Grid
) -> IntegerSet:
    """ number of colors occurring in object or grid """
    return len(gridToTT(palettedslGrid(element)))

@dsl.primitive
def color(
    obj: Object
) -> Integer:
    """ color of object """
    return next(iter(obj))[0]

@dsl.primitive
def toobjectObject(
    patch: Object,
    grid: Grid
) -> Object:
    """ object from patch and grid """
    temp = gridToTT(grid)
    grid = temp
    h, w = len(grid), len(grid[0])
    return frozenset((grid[i][j], (i, j)) for i, j in toindicesObject(patch) if 0 <= i < h and 0 <= j < w)

@dsl.primitive
def toobjectIndices(
    patch: Indices,
    grid: Grid
) -> Object:
    """ object from patch and grid """
    temp = gridToTT(grid)
    grid = temp
    h, w = len(grid), len(grid[0])
    return frozenset((grid[i][j], (i, j)) for i, j in toindicesIndices(patch) if 0 <= i < h and 0 <= j < w)

@dsl.primitive
def asobject(
    grid: Grid
) -> Object:
    """ conversion of grid to object """
    temp = gridToTT(grid)
    grid = temp
    return frozenset((v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r))

@dsl.primitive
def rot90(
    grid: Grid
) -> Grid:
    """ quarter clockwise rotation """
    temp = gridToTT(grid)
    grid = temp
    return ttToGrid(tuple(row for row in zip(*grid[::-1])))








@dsl.primitive
def cmirrordslGrid(
    piece: Grid
) -> Grid:
    """ mirroring along counterdiagonal """
    temp = gridToTT(piece)
    piece = temp
    return ttToGrid(tuple(zip(*(r[::-1] for r in piece[::-1]))))

@dsl.primitive
def cmirrorObject(
    piece: Object
) -> Object:
    """ mirroring along counterdiagonal """
    return vmirrorObject(dmirrorObject(vmirrorObject(piece)))

@dsl.primitive
def cmirrorIndices(
    piece: Indices
) -> Indices:
    """ mirroring along counterdiagonal """
    return vmirrorIndices(dmirrorIndices(vmirrorIndices(piece)))

@dsl.primitive
def fillObject(
    grid: Grid,
    value: Integer,
    patch: Object
) -> Grid:
    """ fill value at indices """
    temp = gridToTT(grid)
    grid = temp
    h, w = len(grid), len(grid[0])
    grid_filled = list(list(row) for row in grid)
    for i, j in toindicesObject(patch):
        if 0 <= i < h and 0 <= j < w:
            grid_filled[i][j] = value
    return ttToGrid(tuple(tuple(row) for row in grid_filled))

@dsl.primitive
def fillIndices(
    grid: Grid,
    value: Integer,
    patch: Indices
) -> Grid:
    """ fill value at indices """
    temp = gridToTT(grid)
    grid = temp
    h, w = len(grid), len(grid[0])
    grid_filled = list(list(row) for row in grid)
    for i, j in toindicesIndices(patch):
        if 0 <= i < h and 0 <= j < w:
            grid_filled[i][j] = value
    return ttToGrid(tuple(tuple(row) for row in grid_filled))

@dsl.primitive
def paint(
    grid: Grid,
    obj: Object
) -> Grid:
    """ paint object to grid """
    temp = gridToTT(grid)
    grid = temp
    h, w = len(grid), len(grid[0])
    grid_painted = list(list(row) for row in grid)
    for value, (i, j) in obj:
        if 0 <= i < h and 0 <= j < w:
            grid_painted[i][j] = value
    return ttToGrid(tuple(tuple(row) for row in grid_painted))

@dsl.primitive
def underfillObject(
    grid: Grid,
    value: Integer,
    patch: Object
) -> Grid:
    """ fill value at indices that are background """
    temp = gridToTT(grid)
    grid = temp
    h, w = len(grid), len(grid[0])
    bg = mostcolordslGrid(ttToGrid(grid))
    g = list(list(r) for r in grid)
    for i, j in toindicesObject(patch):
        if 0 <= i < h and 0 <= j < w:
            if g[i][j] == bg:
                g[i][j] = value
    return ttToGrid(tuple(tuple(r) for r in g))

@dsl.primitive
def underfillIndices(
    grid: Grid,
    value: Integer,
    patch: Indices
) -> Grid:
    """ fill value at indices that are background """
    temp = gridToTT(grid)
    grid = temp
    h, w = len(grid), len(grid[0])
    bg = mostcolordslGrid(ttToGrid(grid))
    g = list(list(r) for r in grid)
    for i, j in toindicesIndices(patch):
        if 0 <= i < h and 0 <= j < w:
            if g[i][j] == bg:
                g[i][j] = value
    return ttToGrid(tuple(tuple(r) for r in g))

@dsl.primitive
def underpaint(
    grid: Grid,
    obj: Object
) -> Grid:
    """ paint object to grid where there is background """
    temp = gridToTT(grid)
    grid = temp
    h, w = len(grid), len(grid[0])
    bg = mostcolordslGrid(ttToGrid(grid))
    g = list(list(r) for r in grid)
    for value, (i, j) in obj:
        if 0 <= i < h and 0 <= j < w:
            if g[i][j] == bg:
                g[i][j] = value
    return ttToGrid(tuple(tuple(r) for r in g))

@dsl.primitive
def hupscale(
    grid: Grid,
    factor: Integer
) -> Grid:
    """ upscale grid horizontally """
    temp = gridToTT(grid)
    grid = temp
    g = tuple()
    for row in grid:
        r = tuple()
        for value in row:
            r = r + tuple(value for num in range(factor))
        g = g + (r,)
    return ttToGrid(g)

@dsl.primitive
def vupscale(
    grid: Grid,
    factor: Integer
) -> Grid:
    """ upscale grid vertically """
    temp = gridToTT(grid)
    grid = temp
    g = tuple()
    for row in grid:
        g = g + tuple(row for num in range(factor))
    return ttToGrid(g)


    


@dsl.primitive




@dsl.primitive
def subgridObject(
    patch: Object,
    grid: Grid
) -> Grid:
    """ smallest subgrid containing object """
    return crop(grid, ulcornerObject(patch), shapeObject(patch))

@dsl.primitive
def subgridIndices(
    patch: Indices,
    grid: Grid
) -> Grid:
    """ smallest subgrid containing object """
    return crop(grid, ulcornerIndices(patch), shapeIndices(patch))



@dsl.primitive
def vsplit(
    grid: Grid,
    n: Integer
) -> Tuple:
    temp = gridToTT(grid)
    grid = temp
    if n == 0:
        primitive_assert(False, 'Function not designed for 0, n')
    """ split grid vertically """
    h, w = len(grid) // n, len(grid[0])
    offset = len(grid) % n != 0
    return tuple(crop(ttToGrid(grid), (h * i + i * offset, 0), (h, w)) for i in range(n))

@dsl.primitive
def cellwise(
    a: Grid,
    b: Grid,
    fallback: Integer
) -> Grid:
    """ cellwise match of two grids """
    x = gridToTT(a)
    a = x
    y = gridToTT(b)
    b = y
    h, w = len(a), len(a[0])
    resulting_grid = tuple()
    for i in range(h):
        row = tuple()
        for j in range(w):
            a_value = a[i][j]
            value = a_value if a_value == b[i][j] else fallback
            row = row + (value,)
        resulting_grid = resulting_grid + (row, )
    return ttToGrid(resulting_grid)





@dsl.primitive
def centerObject(
    patch: Object
) -> Cord:
    """ center of the patch """
    return (uppermostObject(patch) + heightObject(patch) // 2, leftmostObject(patch) + widthObject(patch) // 2)

@dsl.primitive
def centerIndices(
    patch: Indices
) -> Cord:
    """ center of the patch """
    return (uppermostIndices(patch) + heightIndices(patch) // 2, leftmostIndices(patch) + widthIndices(patch) // 2)

@dsl.primitive
def positionOO(
    a: Object,
    b: Object
) -> Cord:
    """ relative position between two patches """
    ia, ja = centerObject(toindicesObject(a))
    ib, jb = centerObject(toindicesObject(b))
    if ia == ib:
        return (0, 1 if ja < jb else -1)
    elif ja == jb:
        return (1 if ia < ib else -1, 0)
    elif ia < ib:
        return (1, 1 if ja < jb else -1)
    elif ia > ib:
        return (-1, 1 if ja < jb else -1)

@dsl.primitive
def positionIO(
    a: Indices,
    b: Object
) -> Cord:
    """ relative position between two patches """
    ia, ja = centerIndices(toindicesIndices(a))
    ib, jb = centerObject(toindicesObject(b))
    if ia == ib:
        return (0, 1 if ja < jb else -1)
    elif ja == jb:
        return (1 if ia < ib else -1, 0)
    elif ia < ib:
        return (1, 1 if ja < jb else -1)
    elif ia > ib:
        return (-1, 1 if ja < jb else -1)

@dsl.primitive
def positionOI(
    a: Object,
    b: Indices
) -> Cord:
    """ relative position between two patches """
    ia, ja = centerObject(toindicesObject(a))
    ib, jb = centerIndices(toindicesIndices(b))
    if ia == ib:
        return (0, 1 if ja < jb else -1)
    elif ja == jb:
        return (1 if ia < ib else -1, 0)
    elif ia < ib:
        return (1, 1 if ja < jb else -1)
    elif ia > ib:
        return (-1, 1 if ja < jb else -1)

@dsl.primitive
def positionII(
    a: Indices,
    b: Indices
) -> Cord:
    """ relative position between two patches """
    ia, ja = centerIndices(toindicesIndices(a))
    ib, jb = centerIndices(toindicesIndices(b))
    if ia == ib:
        return (0, 1 if ja < jb else -1)
    elif ja == jb:
        return (1 if ia < ib else -1, 0)
    elif ia < ib:
        return (1, 1 if ja < jb else -1)
    elif ia > ib:
        return (-1, 1 if ja < jb else -1)

@dsl.primitive
def index(
    grid: Grid,
    loc: Cord
) -> Integer:
    """ color at location """
    temp = gridToTT(grid)
    grid = temp
    i, j = loc
    h, w = len(grid), len(grid[0])
    if not (0 <= i < h and 0 <= j < w):
        return None
    return grid[loc[0]][loc[1]] 



@dsl.primitive
def cornersObject(
    patch: Object
) -> Indices:
    """ indices of corners """
    return frozenset({ulcornerObject(patch), urcornerObject(patch), llcornerObject(patch), lrcornerObject(patch)})

@dsl.primitive
def cornersIndices(
    patch: Indices
) -> Indices:
    """ indices of corners """
    return frozenset({ulcornerIndices(patch), urcornerIndices(patch), llcornerIndices(patch), lrcornerIndices(patch)})

@dsl.primitive
def connect(
    a: Cord,
    b: Cord
) -> Indices:
    """ line between two points """
    ai, aj = a
    bi, bj = b
    si = min(ai, bi)
    ei = max(ai, bi) + 1
    sj = min(aj, bj)
    ej = max(aj, bj) + 1
    if ai == bi:
        return frozenset((ai, j) for j in range(sj, ej))
    elif aj == bj:
        return frozenset((i, aj) for i in range(si, ei))
    elif bi - ai == bj - aj:
        return frozenset((i, j) for i, j in zip(range(si, ei), range(sj, ej)))
    elif bi - ai == aj - bj:
        return frozenset((i, j) for i, j in zip(range(si, ei), range(ej - 1, sj - 1, -1)))
    return frozenset()

@dsl.primitive
def coverObject(
    grid: Grid,
    patch: Object
) -> Grid:
    """ remove object from grid """
    return fillIndices(grid, mostcolordslGrid(grid), toindicesObject(patch))

@dsl.primitive
def coverIndices(
    grid: Grid,
    patch: Indices
) -> Grid:
    """ remove object from grid """
    return fillIndices(grid, mostcolordslGrid(grid), toindicesIndices(patch))

@dsl.primitive
def trim(
    grid: Grid
) -> Grid:
    """ trim border of grid """
    temp = gridToTT(grid)
    grid = temp
    return ttToGrid(tuple(r[1:-1] for r in grid[1:-1]))

@dsl.primitive
def move(
    grid: Grid,
    obj: Object,
    offset: Cord
) -> Grid:
    """ move object on grid """
    return paint(coverObject(grid, obj), shiftObject(obj, offset))

@dsl.primitive
def tophalf(
    grid: Grid
) -> Grid:
    """ upper half of grid """
    temp = gridToTT(grid)
    grid = temp
    return ttToGrid(grid[:len(grid) // 2])

@dsl.primitive
def bottomhalf(
    grid: Grid
) -> Grid:
    """ lower half of grid """
    temp = gridToTT(grid)
    grid = temp
    return ttToGrid(grid[len(grid) // 2 + len(grid) % 2:])

@dsl.primitive
def lefthalf(
    grid: Grid
) -> Grid:
    """ left half of grid """
    return rot270(tophalf(rot90(grid)))

@dsl.primitive
def righthalf(
    grid: Grid
) -> Grid:
    """ right half of grid """
    return rot270(bottomhalf(rot90(grid)))

@dsl.primitive
def vfrontier(
    location: Cord
) -> Indices:
    """ vertical frontier """
    return frozenset((i, location[1]) for i in range(30))

@dsl.primitive
def hfrontier(
    location: Cord
) -> Indices:
    """ horizontal frontier """
    return frozenset((location[0], j) for j in range(30))

#tbc

@dsl.primitive
def backdropObject(
    patch: Object
) -> Indices:
    """ indices in bounding box of patch """
    if len(patch) == 0:
        return frozenset({})
    indices = toindicesObject(patch)
    si, sj = ulcornerIndices(indices)
    ei, ej = lrcornerObject(patch)
    return frozenset((i, j) for i in range(si, ei + 1) for j in range(sj, ej + 1))

@dsl.primitive
def backdropIndices(
    patch: Indices
) -> Indices:
    """ indices in bounding box of patch """
    if len(patch) == 0:
        return frozenset({})
    indices = toindicesIndices(patch)
    si, sj = ulcornerIndices(indices)
    ei, ej = lrcornerIndices(patch)
    return frozenset((i, j) for i in range(si, ei + 1) for j in range(sj, ej + 1))

@dsl.primitive
def deltaObject(
    patch: Object
) -> Indices:
    """ indices in bounding box but not part of patch """
    if len(patch) == 0:
        return frozenset({})
    return backdropObject(patch) - toindicesObject(patch)

@dsl.primitive
def deltaIndices(
    patch: Indices
) -> Indices:
    """ indices in bounding box but not part of patch """
    if len(patch) == 0:
        return frozenset({})
    return backdropIndices(patch) - toindicesIndices(patch)

@dsl.primitive
def gravitateOO(
    source: Object,
    destination: Object
) -> Cord:
    """ direction to move source until adjacent to destination """
    si, sj = centerObject(source)
    di, dj = centerObject(destination)
    i, j = 0, 0
    if vmatchingOO(source, destination):
        i = 1 if si < di else -1
    else:
        j = 1 if sj < dj else -1
    gi, gj = i, j
    c = 0
    while not adjacentOO(source, destination) and c < 42:
        c += 1
        gi += i
        gj += j
        source = shiftObject(source, (i, j))
    return (gi - i, gj - j)

@dsl.primitive
def gravitateOI(
    source: Object,
    destination: Indices
) -> Cord:
    """ direction to move source until adjacent to destination """
    si, sj = centerObject(source)
    di, dj = centerIndices(destination)
    i, j = 0, 0
    if vmatchingOI(source, destination):
        i = 1 if si < di else -1
    else:
        j = 1 if sj < dj else -1
    gi, gj = i, j
    c = 0
    while not adjacentOI(source, destination) and c < 42:
        c += 1
        gi += i
        gj += j
        source = shiftObject(source, (i, j))
    return (gi - i, gj - j)

@dsl.primitive
def gravitateIO(
    source: Indices,
    destination: Object
) -> Cord:
    """ direction to move source until adjacent to destination """
    si, sj = centerIndices(source)
    di, dj = centerObject(destination)
    i, j = 0, 0
    if vmatchingIO(source, destination):
        i = 1 if si < di else -1
    else:
        j = 1 if sj < dj else -1
    gi, gj = i, j
    c = 0
    while not adjacentIO(source, destination) and c < 42:
        c += 1
        gi += i
        gj += j
        source = shiftIndices(source, (i, j))
    return (gi - i, gj - j)

@dsl.primitive
def gravitateII(
    source: Indices,
    destination: Indices
) -> Cord:
    """ direction to move source until adjacent to destination """
    si, sj = centerIndices(source)
    di, dj = centerIndices(destination)
    i, j = 0, 0
    if vmatchingII(source, destination):
        i = 1 if si < di else -1
    else:
        j = 1 if sj < dj else -1
    gi, gj = i, j
    c = 0
    while not adjacentII(source, destination) and c < 42:
        c += 1
        gi += i
        gj += j
        source = shiftIndices(source, (i, j))
    return (gi - i, gj - j)

@dsl.primitive
def inboxObject(
    patch: Object
) -> Indices:
    """ inbox for patch """
    ai, aj = uppermostObject(patch) + 1, leftmostObject(patch) + 1
    bi, bj = lowermostObject(patch) - 1, rightmostObject(patch) - 1
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)

@dsl.primitive
def inboxIndices(
    patch: Indices
) -> Indices:
    """ inbox for patch """
    ai, aj = uppermostIndices(patch) + 1, leftmostIndices(patch) + 1
    bi, bj = lowermostIndices(patch) - 1, rightmostIndices(patch) - 1
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)

@dsl.primitive
def outboxObject(
    patch: Object
) -> Indices:
    """ outbox for patch """
    ai, aj = uppermostObject(patch) - 1, leftmostObject(patch) - 1
    bi, bj = lowermostObject(patch) + 1, rightmostObject(patch) + 1
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)

@dsl.primitive
def outboxIndices(
    patch: Indices
) -> Indices:
    """ outbox for patch """
    ai, aj = uppermostIndices(patch) - 1, leftmostIndices(patch) - 1
    bi, bj = lowermostIndices(patch) + 1, rightmostIndices(patch) + 1
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)

@dsl.primitive
def boxObject(
    patch: Object
) -> Indices:
    """ outline of patch """
    if len(patch) == 0:
        return patch
    ai, aj = ulcornerObject(patch)
    bi, bj = lrcornerObject(patch)
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)

@dsl.primitive
def boxIndices(
    patch: Indices
) -> Indices:
    """ outline of patch """
    if len(patch) == 0:
        return patch
    ai, aj = ulcornerIndices(patch)
    bi, bj = lrcornerIndices(patch)
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)

@dsl.primitive
def shoot(
    start: Cord,
    direction: Cord
) -> Indices:
    """ line from starting point and direction """
    return connect(start, (start[0] + 42 * direction[0], start[1] + 42 * direction[1]))

@dsl.primitive
def occurrences(
    grid: Grid,
    obj: Object
) -> Indices:
    """ locations of occurrences of object in grid """
    temp = gridToTT(grid)
    grid = temp
    occs = set()
    normed = normalizeObject(obj)
    h, w = len(grid), len(grid[0])
    oh, ow = shapeObject(obj)
    h2, w2 = h - oh + 1, w - ow + 1
    for i in range(h2):
        for j in range(w2):
            occurs = True
            for v, (a, b) in shiftObject(normed, (i, j)):
                if not (0 <= a < h and 0 <= b < w and grid[a][b] == v):
                    occurs = False
                    break
            if occurs:
                occs.add((i, j))
    return frozenset(occs)

@dsl.primitive
def frontiers(
    grid: Grid
) -> Objects:
    """ set of frontiers """
    temp = gridToTT(grid)
    grid = temp
    h, w = len(grid), len(grid[0])
    row_indices = tuple(i for i, r in enumerate(grid) if len(set(r)) == 1)
    column_indices = tuple(j for j, c in enumerate(dmirrordslGrid(ttToGrid(grid))) if len(set(c)) == 1)
    hfrontiers = frozenset({frozenset({(grid[i][j], (i, j)) for j in range(w)}) for i in row_indices})
    vfrontiers = frozenset({frozenset({(grid[i][j], (i, j)) for i in range(h)}) for j in column_indices})
    return hfrontiers | vfrontiers

@dsl.primitive
def compress(
    grid: Grid
) -> Grid:
    """ removes frontiers from grid """
    temp = gridToTT(grid)
    grid = temp
    ri = tuple(i for i, r in enumerate(grid) if len(set(r)) == 1)
    ci = tuple(j for j, c in enumerate(dmirrordslGrid(ttToGrid(grid))) if len(set(c)) == 1)
    return ttToGrid(tuple(tuple(v for j, v in enumerate(r) if j not in ci) for i, r in enumerate(grid) if i not in ri))

@dsl.primitive
def hperiod(
    obj: Object
) -> Integer:
    """ horizontal periodicity """
    normalized = normalizeObject(obj)
    w = widthObject(normalized)
    for p in range(1, w):
        offsetted = shiftObject(normalized, (0, -p))
        pruned = frozenset({(c, (i, j)) for c, (i, j) in offsetted if j >= 0})
        if pruned.issubset(normalized):
            return p
    return w

@dsl.primitive
def vperiod(
    obj: Object
) -> Integer:
    """ vertical periodicity """
    normalized = normalizeObject(obj)
    h = heightObject(normalized)
    for p in range(1, h):
        offsetted = shiftObject(normalized, (-p, 0))
        pruned = frozenset({(c, (i, j)) for c, (i, j) in offsetted if i >= 0})
        if pruned.issubset(normalized):
            return p
    return h





def check_ragged(x:dslGrid) -> Boolean:
    t = len(x[0])
    for row in x:
        if len(row) != t:
            return True
    return False

def ttToGrid(x:dslGrid) -> Grid:
    if check_ragged(x):
        primitive_assert(False, "This is not of the right shape for Grid")
    return Grid(np.array(x))
        

print(f"Registered {len(dsl.primitives)} total primitives.")

p = dsl # backwards compatibility
'''