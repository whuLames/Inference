#include <string>

struct Weight {
    // read the weights from binary file
    virtual void loadWeights(std::string weight_path) = 0;
};

