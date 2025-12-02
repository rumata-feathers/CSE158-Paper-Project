from pyalex import Works, config
import time
import json

from pathlib import Path

target_count = 2000
seed = 67

targ_file = Path(__file__).resolve().parent.parent / f"dataset/openalex_works_dataset_{target_count}_{seed}.json"
if targ_file.exists():
    print(f"Dataset file for {target_count} works with seed {seed} already exists. Skipping data loading.")
    exit(0)

print(f"Loading works from OpenAlex API: {target_count} works with seed {seed}")

config.email = "sunglassesguy05@gmail.com"

cur_page = 1
ds = dict() # by id

while len(ds) < target_count:
    batch = Works().sample(target_count, seed=seed).get(per_page=200, page=cur_page)

    if not batch:
        print("No more works returned, stopping.")
        break

    for work in batch:
        ds[work['id']] = work # filter out duplicates

        if len(ds) >= target_count:
            break
    
    print(f"Collected {len(ds)} unique works so far...")

    cur_page += 1
    time.sleep(0.5) # to avoid rate limiting

ds = list(ds.values())


print(f"Successfully loaded {len(ds)} works")

# write to file so we don't have to wait for API requests again
with open(targ_file, "w") as f:
    f.write(json.dumps(ds))