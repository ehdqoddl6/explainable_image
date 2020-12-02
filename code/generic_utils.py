import sys
import inspect
import types

def has_arg(fn, arg_name):
    
    if sys.version_info < (3,):
        if isinstance(fn, types.FunctionType) or isinstance(fn, types.MethodType):
            arg_spec = inspect.getargspec(fn)
        else:
            try:
                arg_spec = inspect.getargspec(fn.__call__)
            except AttributeError:
                return False
        return (arg_name in arg_spec.args)
    elif sys.version_info < (3, 6):
        arg_spec = inspect.getfullargspec(fn)
        return (arg_name in arg_spec.args or
                arg_name in arg_spec.kwonlyargs)
    else:
        try:
            signature = inspect.signature(fn)
        except ValueError:
            # handling Cython
            signature = inspect.signature(fn.__call__)
        parameter = signature.parameters.get(arg_name)
        if parameter is None:
            return False
        return (parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                   inspect.Parameter.KEYWORD_ONLY))
