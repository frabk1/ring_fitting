import ray

for node in ray.nodes():
    if not node.get('Alive', False):  # pragma: no cover
        continue
    cores += node.get('Resources', {}).get('CPU', 0)
    print(cores)
print(cores)
