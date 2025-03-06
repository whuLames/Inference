#include <src/kernels/calculate_paddingoffset.h>

__global__ void calPaddingOffset(int* padding_offet, int* cum_seq_len, const int* input_length, int seq_size, int max_seq_len)
{
    // 计算前缀和

    // version1: 
    // cum_seq_len[0] = 0;
    // int tmp = 0;
    // for(int i = 1; i < seq_size; i ++)
    // {
    //     cum_seq_len[i] = tmp + input_length[i];
    //     tmp += input_length[i];
    // }

    // int cur_padding_num = 0;
    // for(int i = 0; i < seq_size; i ++)
    // {
    //     for(int j = 0; j < max_seq_len; j ++)
    //     {
    //         padding_offet[i * max_seq_len + j] = cur_padding_num;
    //         if(j >= input_length[i]) cur_padding_num ++;
    //     }
    // }
    
    // version 2:

    // 最后index会小于 padding_offset 的总长度，即其余位置均置为zero 
    int index = 0; 
    int cur_padding_num = 0;
    int total_seq_len = 0;
    cum_seq_len[0] = 0; 
    for(int i = 0; i < seq_size; i ++)
    {
        int seq_len = input_length[i];
        for(int j = 0; j < seq_len; j ++)
        {
            padding_offet[index] = cur_padding_num;
            index ++;  
        }
        cur_padding_num += max_seq_len - seq_len;
        total_seq_len += input_length[i];
        cum_seq_len[i + 1] = total_seq_len;
    }

}

void launchCalPaddingOffset(TensorWrapper<int>* padding_offset, TensorWrapper<int>* cum_seq_len, TensorWrapper<int>* input_length)
{
    int seq_size = padding_offset -> shape[0];
    int max_seq_len = padding_offset -> shape[1];   

    calPaddingOffset<<<1,1>>>(padding_offset -> data, cum_seq_len -> data, input_length -> data, seq_size, max_seq_len);
}