import sys
sys.path.insert(0, '/Users/saikumarkatteramini/Downloads/functiongemma-hackathon')
from handsfree.location import is_location_share, is_location_query

tests = [
    ('what is my current location',   False, True),
    ('where am I',                    False, True),
    ('what is my address',            False, True),
    ('send my location to Mom',       True,  False),
    ('share my location with John',   True,  False),
    ('tell Sarah where I am',         True,  False),
    ('get directions to Golden Gate', False, False),
    ('play Bohemian Rhapsody',        False, False),
]
ok = True
for cmd, exp_share, exp_query in tests:
    share = is_location_share(cmd)
    query = is_location_query(cmd)
    match = (share == exp_share) and (query == exp_query)
    status = "OK  " if match else "FAIL"
    print(f"  {status} share={str(share):5} query={str(query):5} | {cmd}")
    ok = ok and match
print()
print("All passed!" if ok else "SOME FAILED")
