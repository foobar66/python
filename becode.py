import numpy

def oper(a,b,name):
    dict = { "add": lambda x, y: x + y,
             "subtract": lambda x, y: x - y,
             "mutiply": lambda x, y: x * y,
             "divide" : lambda x, y: x / y
    }
    return dict[name](a,b)

def remove_outliers(sample, cutoff):
    cont = 1
    final_list = sample
    while cont:
      length = len(final_list)
      mean = numpy.mean(final_list, axis=0)
      standarddeviation = numpy.std(final_list, axis=0)
      final_list = [x for x in final_list if (x > mean - cutoff * standarddeviation)]
      final_list = [x for x in final_list if (x < mean + cutoff * standarddeviation)]
      if length == len(final_list):
          cont = 0
    return final_list

def clean_mean(sample, cutoff):
    final_list = remove_outliers(sample, cutoff)
    return round(numpy.mean(final_list, axis=0), 2)

sample = [1, 2, 3, 4]
cutoff = 2
sample = [1.01, 0.99, 1.02, 1.01, 0.99, 0.97, 1.03, 0.99, 1.02, 0.99, 3]
test = clean_mean(sample, cutoff)
print("result = " + str(test))
print("oper(3,4,\"add\") = " + str(oper(3,4,"add")))
print("oper(7,8,\"mutiply\") = " + str(oper(7,8,"mutiply")))

