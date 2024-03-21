//training of signle nuron 


#include <iostream>
#include <cmath>
#include <ctime>

using namespace std;

float train[][2] = {
    {0, 0},
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8},
};

float randomWeight() {
    std::srand(static_cast<unsigned int>(std::time(0)));
    return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * 1.0f + 1.0f;
}

#define size (float)sizeof(train) / (float)sizeof(train[0])

float cost(float w) {
    float result = 0;
    for (int i = 0; i < size; i++) {
        float x = train[i][0];
        float y = x * w;
        float diff = y - train[i][1];
        result += diff * diff;
    }
    return result; 
}

int main() {
    std::srand(static_cast<unsigned int>(std::time(0)));
    float w = randomWeight();

    // formula y = x * w
    float eps = 1e-3;
    float rate = 1e-3;

    for (int i = 0; i < 1000; i++) {
        float dcost = (cost(w + eps) - cost(w)) / eps;  
        w -= dcost * rate;
        cout << "Iteration " << i + 1 << ": Cost = " << cost(w) << ", w = " << w << endl;
    }
    return 0;
}
