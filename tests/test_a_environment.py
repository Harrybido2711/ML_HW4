import os


def test_versions():
    import scipy
    version = scipy.__version__
    msg = ("This won't affect your grade, but we strongly recommend "
           f"you use scipy>=1.15.1, not {version}")
    a, b, c = version.split('.')
    assert int(b) >= 15, msg

    import numpy
    version = numpy.__version__
    msg = ("This won't affect your grade, but we strongly recommend "
           f"you use numpy>=2.2.0, not {version}")
    a, b, c = version.split('.')
    assert int(a) == 2, msg
    assert int(b) >= 2, msg


def test_imports():
    """
    Please don't import fairlearn, sklearn, or scipy to solve any of the
    problems in this assignment.  If you fail this test, we will give you a
    zero for this assignment, regardless of how the package was used in your
    code.

    the 'a' in the file name is so this test is run first.
    """
    import src
    import importlib
    import pkgutil
    # Checking that these package names only appear in the warnings
    #   about how you should not use them.
    disallowed = ["fairlearn", "scipy", "sklearn"]
    disallowed += ["sys", "importlib"]
    disallowed_str = ', '.join(disallowed)
    docstring = f"Do not import or use these packages: {disallowed_str}."
    sparse_url = "https://docs.scipy.org/doc/scipy/reference/sparse.html"
    exceptions = [docstring, sparse_url]
    print(docstring)
    for fn in os.listdir(src.__path__[0]):
        if fn.endswith(".py"):
            fn = os.path.join(src.__path__[0], fn)
            with open(fn, encoding="utf-8") as inf:
                for i, line in enumerate(inf):
                    for key in disallowed:
                        if key in line:
                            msg = f"Don't use {key} in line {i+1} of {fn}"
                            assert line.strip() in exceptions, msg

    # Checking that these function names only appear in the warnings
    #   about how you should not use them.
    disallowed_funcs = ["polynomial", "polyfit", "polyval", "getattr", "globals"]
    funcs_str = ", ".join(disallowed_funcs)
    docstring = f"Do not use these numpy or internal functions: {funcs_str}"
    print(docstring)
    for fn in os.listdir(src.__path__[0]):
        if fn.endswith(".py"):
            fn = os.path.join(src.__path__[0], fn)
            with open(fn, encoding="utf-8") as inf:
                for i, line in enumerate(inf):
                    for key in disallowed_funcs:
                        if key in line:
                            msg = f"Don't use {key} in line {i+1} of {fn}"
                            assert line.strip() == docstring, msg
