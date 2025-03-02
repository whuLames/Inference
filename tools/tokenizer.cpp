#include <stdio.h>
#include <iostream>
#include <map>
#include <unordered_map>
#include <vector>
#include <string>
#include <queue>
#include <iostream>

struct FileBuffer {
    FILE *f;

    FileBuffer (const std::string &fileName) {
        f = fopen(fileName.c_str(), "rb");
    }

    int ReadInt() {
        int v;
        // fread(buf,1,sizeof(buf),fp)：表示每个数据的大小为1，读了4次，一共4b，返回值为实际读取的数据个数即4
        if (fread(&v, 1, 4, f) != 4) { 
            std::cout << "FileBuffer.ReadInt error." << "\n";
        };
        return v;
    }

    float ReadFloat() {
        float v;
        if (fread(&v, 1, 4, f) != 4) {
            std::cout << "FileBuffer.ReadFloat error." << "\n";
        };
        return v;
    }

    std::string ReadString() {
        int len = ReadInt();
        std::string ret = "";
        char *v = new char[len + 5];
        v[len] = 0;
        if (fread(v, 1, len, f) != len) {
            std::cout << "FileBuffer.ReadString error." << "\n";
        }
        return v;
    }

    void ReadBytes(uint8_t *buffer, uint64_t bytes) {
        if (fread(buffer, 1, bytes, f) != bytes) {
            std::cout << "FileBuffer.ReadBytes error." << "\n";
        }
    }

    ~FileBuffer() {
        fclose(f);
    }
};

void readFile(std::string file) {
    FileBuffer buffer(file); //这里的filename就是读取的weight文件，读了这个之后才能用tokenizer
        int versionId = buffer.ReadInt();

        if (versionId >= 1) {
            // versionId >= 1, 前置了一个key-value表
            int keyValueLen = buffer.ReadInt();
            for (int i = 0; i < keyValueLen; i++) {
                std::string key = buffer.ReadString();
                std::string value = buffer.ReadString();
                printf("key = %s, value = %s\n", key.c_str(), value.c_str());
            }
        }

        int vocab_len = buffer.ReadInt();
        printf("The length of vocab is %d \n", vocab_len);
        // string-id-score
        for (int i = 0; i < vocab_len; i++) {
            int len = buffer.ReadInt();
            std::string x = "";
            for (int j = 0; j < len; j++) {
                x += buffer.ReadInt();  // encode内容，对应torch2flm.py#160
            }
            int id = buffer.ReadInt();
            // float score = useScore ? buffer.ReadFloat() : -i;
            float score = buffer.ReadFloat();
            if(i >= 20000 && i <= 20100) std::cout << "id: " << id << " content: " << x << " score: " << score << std::endl;
        }

}

int main(int argc, char** argv) {

    if(argc != 2) return -1;
    std::string file_name = argv[1];
    readFile(file_name);
}