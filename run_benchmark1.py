import sys
sys.path.insert(0, "cactus/python/src")
import main as _m
sys.modules["main"] = _m
import benchmark
