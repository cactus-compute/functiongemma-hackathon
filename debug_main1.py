import sys
sys.path.insert(0, "cactus/python/src")
print("step 1: importing main1...")
import main1
print("step 2: main1 loaded, attributes:", [a for a in dir(main1) if not a.startswith("_")])
print("step 3: hasattr generate_hybrid:", hasattr(main1, "generate_hybrid"))
