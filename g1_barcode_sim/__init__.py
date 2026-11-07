"""g1_barcode_sim: G1 barcode-scanning simulation + test suites.

`from g1_barcode_sim import G1BarcodeEnv` is resolved lazily so that pure-Python users (e.g. the test-suite
generator) do not import the simulator module.
"""


def __getattr__(name):
    if name == "G1BarcodeEnv":
        from .env import G1BarcodeEnv
        return G1BarcodeEnv
    raise AttributeError(name)
