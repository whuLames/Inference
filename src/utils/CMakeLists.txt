add_library(weightutils STATIC weights_utils.cu)
set_property(TARGET weightutils PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET weightutils PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET weightutils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)