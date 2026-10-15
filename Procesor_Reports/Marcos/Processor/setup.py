from setuptools import setup

setup(
    name="ProcessorCli",
    version="0.1",
    # IMPORTANTE: Incluir ProcessorV2 para que ProcessorCli pueda importarlo
    py_modules=["ProcessorCli", "ProcessorV2"],
    install_requires=[
        "click>=8.0.0",
        "click-pathlib>=2020.3.13.0",
        "basictdf>=0.1.14",
        "colorlog>=5.0.0",
        "simplemdx",  # Asegurar que simplemdx esté disponible
    ],
    entry_points="""
        [console_scripts]
        ProcessorCli=ProcessorCli:cli
    """,
)
