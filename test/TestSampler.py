

from samplers.Sampler import BasicSampler

bs = BasicSampler(capacity=5, variable_names= ["a","b"])

for p in range(10):
    bs.add(p+0,p+100)
    print(p)
    print(bs.get())
    print("a={}".format(bs.get().a))
    print("b={}".format(bs.get().b))

