# Test for embedding kernel
add_executable(embedding
    test_input_embedding.cu
)

target_link_libraries(
    embedding PUBLIC
    -lcudart
    -lcudadevrt
    embeddingFunctor)
    add_executable(embedding
    test_input_embedding.cu
)

add_executable(rmsnorm
        test_rmsnorm.cu)
        
target_link_libraries(
    rmsnorm_test PUBLIC
    -lcudart
    -lcudadevrt
    rmsnorm
)
    
# add_executable(rms_norm
#     test_rmsnorm.cu
# )
# target_link_libraries(
#     rms_norm PUBLIC
#     -lcudart
#     -lcudadevrt
#     rmsnorm)

# add_executable(paddingoffset
#     test_cal_paddingoffset.cu
# )
# target_link_libraries(
#   paddingoffset PUBLIC
#     -lcudart
#     -lcudadevrt
#     cal_paddingoffset) 

# add_executable(causalmask
#     test_casual_mask.cu
# )
# target_link_libraries(  # Libs for test_build_casual_mask
#    causalmask PUBLIC
#     -lcudart
#     -lcudadevrt
#     build_casual_mask) 

# add_executable(testlinear
#     test_linear.cu
# )
# target_link_libraries(  # Libs for test_build_casual_mask
#     testlinear PUBLIC
#     -lcudart
#     -lcudadevrt
#     linear) 
