class MLAlgorithm {
public:
    virtual void train(const char *data);
    virtual float predict(int user, float rating);
};

class RBM : public MLAlgorithm {
public:
    RBM(int hidden);
    void train(const char *data);
    float predict(int user, float rating);
};
