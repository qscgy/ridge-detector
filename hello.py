from nest import register

@register(author='sam ehrenstein', version='1.0.0')
def hello_nest(name: str) -> str:
    """My first Nest module!"""

    return 'Hello ' + name