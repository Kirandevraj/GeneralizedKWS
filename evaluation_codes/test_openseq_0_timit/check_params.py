import tensorflow as tf
total_parameters = 0
count = 0
for name, shape in tf.train.list_variables("/ssd_scratch/cvit/kiran/OpenSeq_pretrained/ds2_large"):
    # vars_in_checkpoint[var_name] = var_shape
    #if "student" in name:
      print(name, shape)
      variable_parameters = 1
      for dim in shape:
          # print(dim)
          variable_parameters *= dim
      print("current: ", variable_parameters)
      if "opaque_kernel" not in name:
        total_parameters += variable_parameters
      print("total: ", total_parameters)
      count += 1
    # if count == 11:
    # break
